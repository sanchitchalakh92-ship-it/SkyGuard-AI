"""
Unit & Integration Test Suite for SkyGuard AI (SIH 26073).
Tests:
1. WMO Rule QC (Layer 1)
2. Temporal Diurnal & GRU Models (Layer 2)
3. Multivariate Thermodynamic Consistency (Layer 3)
4. Spatial IDW & Elevation Lapse Rate (Layer 4)
5. Decision Fusion & 55°C Demo Scenario (Layer 5)
6. Self-Healing Imputation
7. Sensor Health Tracking
"""
import pytest
import datetime
from src.ingestion.config_loader import load_stations
from src.ingestion.simulator import RealisticSimulator
from src.ingestion.injector import AnomalyInjector
from src.models.pipeline import SkyGuardPipeline
from src.models.rules import RuleBasedQC
from src.models.multivariate import MultivariateConsistencyModel
from src.models.spatial import SpatialConsistencyModel
from database.db import init_db


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()


@pytest.fixture
def stations():
    return load_stations()


@pytest.fixture
def pipeline(stations):
    return SkyGuardPipeline(stations)


@pytest.fixture
def simulator():
    return RealisticSimulator(seed=42)


@pytest.fixture
def injector():
    return AnomalyInjector()


def test_stations_loaded(stations):
    assert len(stations) == 24
    ids = [s.id for s in stations]
    assert "AWS_PUN" in ids
    assert "AWS_NGP" in ids
    assert "AWS_BOM" in ids


def test_layer1_rules_qc(stations, simulator, injector):
    rule_qc = RuleBasedQC()
    pune = stations[0]
    now = datetime.datetime.now()
    clean = simulator._calculate_physical_reading(pune, now)

    # Clean reading should not violate rules
    eval_clean = rule_qc.evaluate(clean, pune)
    assert eval_clean["score"] < 0.5
    assert not eval_clean["is_anomalous"]

    # Injected out of range RH > 100%
    corrupt = injector.inject_out_of_range(clean)
    eval_corrupt = rule_qc.evaluate(corrupt, pune)
    assert eval_corrupt["score"] >= 0.7
    assert eval_corrupt["is_anomalous"]

    # Injected sentinel -999
    dropout = injector.inject_dropout(clean, mode="sentinel")
    eval_drop = rule_qc.evaluate(dropout, pune)
    assert eval_drop["score"] == 1.0
    assert eval_drop["is_anomalous"]


def test_layer3_multivariate_physics(stations, simulator, injector):
    mv_model = MultivariateConsistencyModel()
    nagpur = next(s for s in stations if s.id == "AWS_NGP")
    now = datetime.datetime.now()
    clean = simulator._calculate_physical_reading(nagpur, now)

    # Clean reading
    eval_clean = mv_model.evaluate(clean, nagpur)
    assert len(eval_clean["physics_violations"]) == 0

    # Injected 55°C with 94% RH (impossible vapor pressure and supersaturation)
    injected_55c = injector.inject_55C_scenario(clean)
    eval_55c = mv_model.evaluate(injected_55c, nagpur)
    assert eval_55c["score"] >= 0.85
    assert len(eval_55c["physics_violations"]) > 0


def test_layer4_spatial_consistency(stations, simulator, injector):
    spatial_model = SpatialConsistencyModel(k_neighbors=4)
    now = datetime.datetime.now()
    network_map = {s.id: simulator._calculate_physical_reading(s, now) for s in stations}

    pune = stations[0]
    eval_clean = spatial_model.evaluate(network_map[pune.id], pune, stations, network_map)
    assert eval_clean["score"] < 0.4
    assert not eval_clean["is_spatial_outlier"]

    # Spike Pune to 55°C while neighbors are ~28°C
    corrupted_pune = injector.inject_spike(network_map[pune.id], magnitude=25.0)
    eval_spike = spatial_model.evaluate(corrupted_pune, pune, stations, network_map)
    assert eval_spike["score"] > 0.6
    assert eval_spike["is_spatial_outlier"]


def test_end_to_end_55c_scenario(pipeline, stations, simulator, injector):
    now = datetime.datetime.now()
    network_map = {s.id: simulator._calculate_physical_reading(s, now) for s in stations}

    # Prime
    for s in stations:
        pipeline.process_reading(network_map[s.id], network_map)

    # Inject 55°C into Nagpur
    nagpur = next(s for s in stations if s.id == "AWS_NGP")
    corrupt_reading = injector.inject_55C_scenario(network_map[nagpur.id])

    result = pipeline.process_reading(corrupt_reading, network_map)

    assert result["is_anomaly"] is True
    assert result["severity"] in ["High", "Critical"]
    assert result["confidence"] >= 0.85
    assert result["processing_latency_ms"] < 50.0  # Real-time requirement < 100ms
    assert result["imputation"] is not None
    assert result["imputation"]["imputed_temperature"] < 42.0  # Corrected to physical range
    assert len(result["shap_features"]) > 0


def test_genuine_thunderstorm_suppression(pipeline, stations, simulator, injector):
    now = datetime.datetime.now()
    network_map = {s.id: simulator._calculate_physical_reading(s, now) for s in stations}

    # Prime
    for s in stations:
        pipeline.process_reading(network_map[s.id], network_map)

    # Regional event across Maharashtra
    network_map["AWS_PUN"] = injector.inject_genuine_event(network_map["AWS_PUN"], "thunderstorm")
    network_map["AWS_KLP"] = injector.inject_genuine_event(network_map["AWS_KLP"], "thunderstorm")

    pune = stations[0]
    result = pipeline.process_reading(network_map["AWS_PUN"], network_map)

    # Should not trigger critical sensor alert
    assert result["severity"] != "Critical"


def test_cross_station_verification_single_fault_vs_multiple_event(stations, simulator, injector):
    """
    1. Cross-Station Verification:
    - If only one station shows an extreme value, classify it as a possible sensor fault;
    - If multiple nearby stations show the same abnormal pattern, flag it as a possible real weather event.
    """
    spatial_model = SpatialConsistencyModel(k_neighbors=4)
    now = datetime.datetime.now()
    network_map = {s.id: simulator._calculate_physical_reading(s, now) for s in stations}
    nagpur = next(s for s in stations if s.id == "AWS_NGP")

    # Case A: Only one station (Nagpur) shows extreme value (+25°C spike), neighbors nominal
    corrupt_nagpur = injector.inject_spike(network_map[nagpur.id], magnitude=25.0)
    cs_isolated = spatial_model.verify_cross_station(corrupt_nagpur, nagpur, stations, network_map)

    assert cs_isolated["verdict"] == "POSSIBLE_SENSOR_FAULT"
    assert cs_isolated["is_sensor_fault"] is True
    assert cs_isolated["is_real_weather_event"] is False
    assert "Possible Sensor Fault" in cs_isolated["classification"]

    # Case B: Multiple nearby stations show the same abnormal pattern (e.g. Pune & Kolhapur)
    pune = next(s for s in stations if s.id == "AWS_PUN")
    # Both stations experience a similar sudden temperature shift (e.g. drop by 10°C or spike)
    shifted_pune = simulator._calculate_physical_reading(pune, now)
    shifted_pune.temperature = 18.0
    network_map["AWS_PUN"] = shifted_pune

    # Update neighboring stations in Western Maharashtra to also exhibit similar shift
    for nbr_id in ["AWS_KLP", "AWS_BOM", "AWS_NSK"]:
        stn_obj = next(s for s in stations if s.id == nbr_id)
        nbr_reading = simulator._calculate_physical_reading(stn_obj, now)
        nbr_reading.temperature = 18.5  # nearby stations also report same abnormal cold pattern
        network_map[nbr_id] = nbr_reading

    cs_multiple = spatial_model.verify_cross_station(shifted_pune, pune, stations, network_map)
    assert cs_multiple["verdict"] == "POSSIBLE_REAL_WEATHER_EVENT"
    assert cs_multiple["is_real_weather_event"] is True
    assert cs_multiple["is_sensor_fault"] is False
    assert "Possible Real Weather Event" in cs_multiple["classification"]


def test_explainable_ai_alert_fields(pipeline, stations, simulator, injector):
    """
    2. Explainable AI Alert:
    Instead of only showing 'Anomaly Detected', shows:
    what changed, expected vs actual value, deviation %, confidence, possible cause, recommended action.
    """
    now = datetime.datetime.now()
    network_map = {s.id: simulator._calculate_physical_reading(s, now) for s in stations}
    for s in stations:
        pipeline.process_reading(network_map[s.id], network_map)

    nagpur = next(s for s in stations if s.id == "AWS_NGP")
    corrupt = injector.inject_55C_scenario(network_map[nagpur.id])
    res = pipeline.process_reading(corrupt, network_map)

    assert "explainable_alert" in res
    xai = res["explainable_alert"]

    # Verify all 6 required fields
    assert "what_changed" in xai and len(xai["what_changed"]) > 0
    assert "expected_value" in xai and xai["expected_value"] is not None
    assert "actual_value" in xai and xai["actual_value"] == 55.0
    assert "deviation_pct" in xai and xai["deviation_pct"] > 0
    assert "confidence" in xai and xai["confidence"] >= 0.8
    assert "possible_cause" in xai and len(xai["possible_cause"]) > 0
    assert "recommended_action" in xai and len(xai["recommended_action"]) > 0

    # Ensure badge is informative and not generic
    assert xai["badge_text"] != "Anomaly Detected"


def test_predictive_maintenance_repeated_abnormal_behavior(stations, simulator, injector):
    """
    3. Predictive Maintenance:
    Detects repeated abnormal sensor behaviour and warns:
    'Sensor may require inspection/calibration.'
    """
    from src.health.sensor_health import SensorHealthTracker
    health_tracker = SensorHealthTracker()
    stn = stations[0]

    # Cycle 1: Single anomaly
    state1 = health_tracker.update_sensor_health(stn, is_anomaly=True, root_cause="Spike/Sensor Fault", residual_t=8.0)
    assert state1["anomaly_count_24h"] == 1

    # Cycle 2: Repeated abnormal behavior (second consecutive anomaly)
    state2 = health_tracker.update_sensor_health(stn, is_anomaly=True, root_cause="Spike/Sensor Fault", residual_t=9.5)
    assert state2["repeated_abnormal_behavior"] is True
    assert state2["requires_calibration"] is True
    assert state2["maintenance_warning"] == "Sensor may require inspection/calibration."
    assert "Sensor may require inspection/calibration." in state2["recommended_maintenance"]


def test_context_aware_anomaly_detection(pipeline, stations, simulator):
    """
    4. Context-Aware Anomaly Detection:
    Analyzes readings against recent trends, historical behavior, and related sensor measurements.
    An abrupt temperature jump while other parameters remain stable indicates a possible sensor issue,
    but is checked against local weather conditions before final classification.
    """
    from src.ingestion.base import WeatherReading
    now = datetime.datetime.now()
    nagpur = next(s for s in stations if s.id == "AWS_NGP")

    # Step 1: Baseline history (warmup pipeline with nominal readings)
    network_map = {s.id: simulator._calculate_physical_reading(s, now) for s in stations}
    for s in stations:
        pipeline.process_reading(network_map[s.id], network_map)

    # Step 2: Abrupt temperature jump while RH, P, and solar remain stable
    nominal_ngp = network_map[nagpur.id]
    jump_reading = WeatherReading(
        station_id=nagpur.id,
        timestamp=now + datetime.timedelta(minutes=10),
        temperature=nominal_ngp.temperature + 16.0,  # abrupt jump +16°C
        pressure=nominal_ngp.pressure,               # stable
        humidity=nominal_ngp.humidity,               # stable
        wind_speed=nominal_ngp.wind_speed,           # stable
        rainfall=0.0,
        solar_radiation=nominal_ngp.solar_radiation   # stable
    )

    res_isolated = pipeline.process_reading(jump_reading, network_map)
    assert res_isolated["is_anomaly"] is True
    assert "context_aware" in res_isolated
    ctx = res_isolated["context_aware"]
    assert ctx["is_isolated_shift"] is True
    assert ctx["is_corroborated_by_weather"] is False
    assert ctx["related_parameters"]["temperature_jump_isolated"] is True
    assert "Abrupt temperature jump" in ctx["explanation"]

    # Step 3: Now simulate the same jump occurring during an actual hot front across nearby stations
    corroborated_network = dict(network_map)
    for nbr_id in ["AWS_BHO", "AWS_IXU", "AWS_HYD", "AWS_NSK"]:
        stn_obj = next(s for s in stations if s.id == nbr_id)
        nbr_rdg = simulator._calculate_physical_reading(stn_obj, now)
        nbr_rdg.temperature += 15.0  # neighbors also elevated
        corroborated_network[nbr_id] = nbr_rdg

    res_weather = pipeline.process_reading(jump_reading, corroborated_network)
    ctx_weather = res_weather["context_aware"]
    assert ctx_weather["is_corroborated_by_weather"] is True
    assert "weather event" in ctx_weather["explanation"].lower() or "nearby" in ctx_weather["explanation"].lower()


def test_sensor_fault_identification_categories(pipeline, stations, simulator):
    """
    Sensor Fault Identification:
    Distinguishes between drift, stuck readings, sudden spikes, missing data, and communication failure,
    with a confidence score and never claiming confirmed sensor faults without review.
    """
    from src.ingestion.base import WeatherReading
    now = datetime.datetime.now()
    nagpur = next(s for s in stations if s.id == "AWS_NGP")
    clean = simulator._calculate_physical_reading(nagpur, now)

    # 1. Sudden Spike
    spike_rdg = WeatherReading(
        station_id=nagpur.id,
        timestamp=now,
        temperature=clean.temperature + 22.0,
        pressure=clean.pressure,
        humidity=clean.humidity
    )
    res_spike = pipeline.process_reading(spike_rdg)
    fault_spike = res_spike["sensor_fault_identification"]
    assert fault_spike["fault_type"] in ["Possible Sensor Spike", "SUDDEN_SPIKE", "PHYSICAL_THERMODYNAMIC_SPIKE"]
    assert "Spike" in fault_spike["fault_category"]
    assert fault_spike["is_confirmed_fault"] is False
    assert fault_spike["confidence_score"] >= 0.85

    # 2. Missing Data (single sensor null)
    missing_rdg = WeatherReading(
        station_id=nagpur.id,
        timestamp=now + datetime.timedelta(minutes=1),
        temperature=None,
        pressure=clean.pressure,
        humidity=clean.humidity
    )
    res_missing = pipeline.process_reading(missing_rdg)
    fault_missing = res_missing["sensor_fault_identification"]
    assert fault_missing["fault_type"] in ["Missing Data", "MISSING_DATA"]
    assert "Missing Data" in fault_missing["fault_category"]
    assert fault_missing["is_confirmed_fault"] is False

    # 3. Communication Failure (multiple sensors null / dropped frame)
    coms_rdg = WeatherReading(
        station_id=nagpur.id,
        timestamp=now + datetime.timedelta(minutes=2),
        temperature=None,
        pressure=None,
        humidity=None
    )
    res_coms = pipeline.process_reading(coms_rdg)
    fault_coms = res_coms["sensor_fault_identification"]
    assert fault_coms["fault_type"] in ["Communication Failure", "COMMUNICATION_FAILURE"]
    assert "Communication Failure" in fault_coms["fault_category"]
    assert fault_coms["is_confirmed_fault"] is False


def test_explainable_ai_checklist_and_unconfirmed_disclaimer(pipeline, stations, simulator, injector):
    """
    Explainable AI:
    Shows why an anomaly was flagged, which measurements contributed to it,
    and what an engineer could check. Avoids presenting predictions as confirmed sensor faults.
    """
    now = datetime.datetime.now()
    network_map = {s.id: simulator._calculate_physical_reading(s, now) for s in stations}
    nagpur = next(s for s in stations if s.id == "AWS_NGP")
    corrupt = injector.inject_55C_scenario(network_map[nagpur.id])
    res = pipeline.process_reading(corrupt, network_map)

    # 1. Contributing measurements
    assert "contributing_measurements" in res
    contribs = res["contributing_measurements"]
    assert len(contribs) > 0
    temp_contrib = next((c for c in contribs if c["parameter"] == "Temperature"), None)
    assert temp_contrib is not None
    assert temp_contrib["observed_value"] == 55.0
    assert temp_contrib["contribution_pct"] > 50

    # 2. Engineer checklist
    assert "engineer_inspection_checklist" in res
    checklist = res["engineer_inspection_checklist"]
    assert len(checklist) >= 2
    assert any("wiring" in item.lower() or "shield" in item.lower() or "rtd" in item.lower() for item in checklist)

    # 3. Unconfirmed disclaimer
    assert "unconfirmed_fault_disclaimer" in res
    disclaimer = res["unconfirmed_fault_disclaimer"]
    assert "probabilistic" in disclaimer.lower() or "unconfirmed" in disclaimer.lower()
    assert "confirmed" not in res["root_cause"].lower()


def test_continuous_model_evaluation_and_gated_retraining():
    """
    Continuous Model Evaluation:
    Compares model predictions against reviewed and verified events,
    tracks false alarms and missed anomalies, and retrains models ONLY when validated data is available.
    """
    from src.models.evaluation import ContinuousModelEvaluator
    from database.db import get_db_connection

    # Clean up any previously created test evaluation events
    conn = get_db_connection()
    conn.execute("DELETE FROM model_evaluations WHERE event_id LIKE 'test_eval_%'")
    conn.commit()
    conn.close()

    evaluator = ContinuousModelEvaluator(min_validated_threshold=20)

    # 1. Initial seeded metrics should exist
    metrics = evaluator.get_evaluation_metrics()
    assert "precision" in metrics
    assert "recall" in metrics
    assert "false_alarm_rate" in metrics
    assert "missed_anomaly_rate" in metrics
    assert metrics["retraining_gated"] is True  # Seeded with 12 reviews, threshold is 20

    # 2. Attempt retraining when under threshold -> Must be rejected
    rejected = evaluator.trigger_gated_retraining(triggered_by="Junior Analyst")
    assert rejected["success"] is False
    assert rejected["status"] == "REJECTED"
    assert rejected["reason_code"] == "INSUFFICIENT_VALIDATED_DATA"
    assert "20" in rejected["reason"] and "validated" in rejected["reason"].lower()

    # 3. Record expert reviews until threshold (>= 20) is met
    for i in range(8):
        evaluator.review_event(
            event_id=f"test_eval_{i}_{datetime.datetime.now().timestamp()}",
            station_id="AWS_NGP",
            model_prediction="Possible Sensor Spike",
            model_confidence=0.92,
            verified_status="VERIFIED_TRUE_POSITIVE",
            reviewed_by="Senior Quality Meteorologist",
            review_notes="Field team calibrated thermistor and confirmed spike."
        )

    # 4. Check updated metrics: validated sample count is now >= 20
    updated_metrics = evaluator.get_evaluation_metrics()
    assert updated_metrics["validated_training_samples"] >= 20
    assert updated_metrics["retraining_gated"] is False

    # 5. Retraining should now SUCCEED and record new version in audit log
    success_retrain = evaluator.trigger_gated_retraining(triggered_by="Lead Data Scientist")
    assert success_retrain["success"] is True
    assert success_retrain["status"] == "SUCCESS"
    assert "model_version" in success_retrain
    assert success_retrain["validated_samples_used"] >= 20


def test_ai_sensor_fault_classification_isolated_spike(stations, simulator, injector):
    """
    Test that an isolated unphysical 55°C spike is classified as SENSOR_FAULT:
    - Verdict: SENSOR_FAULT
    - Severity: CRITICAL
    - 4 Distinguishing Factors (Spatial isolated, Thermodynamic violation, Unphysical rate of change, Incoherent multivariate)
    - Detailed Anomaly Explanation
    - Operational Impact for immediate hardware isolation
    """
    from src.models.fusion import DecisionFusionEngine
    from src.models.rules import RuleBasedQC
    from src.models.temporal import TemporalModel
    from src.models.multivariate import MultivariateConsistencyModel
    from src.models.spatial import SpatialConsistencyModel

    nagpur = next(s for s in stations if s.id == "AWS_NGP")
    now = datetime.datetime.now()
    clean = simulator._calculate_physical_reading(nagpur, now)
    injected_55c = injector.inject_55C_scenario(clean)

    rule_qc = RuleBasedQC()
    temp_qc = TemporalModel()
    mv_qc = MultivariateConsistencyModel()
    spatial_qc = SpatialConsistencyModel(k_neighbors=4)
    fusion = DecisionFusionEngine()

    rule_eval = rule_qc.evaluate(injected_55c, nagpur)
    temporal_eval = temp_qc.evaluate(injected_55c, nagpur)
    mv_eval = mv_qc.evaluate(injected_55c, nagpur)
    spatial_eval = spatial_qc.evaluate(injected_55c, nagpur, stations, {nagpur.id: injected_55c})

    classification = fusion.classify_fault_vs_weather(
        reading=injected_55c,
        station=nagpur,
        rule_eval=rule_eval,
        temporal_eval=temporal_eval,
        multivariate_eval=mv_eval,
        spatial_eval=spatial_eval
    )

    # 1. Verdict & Label
    assert classification["verdict"] == "SENSOR_FAULT"
    assert classification["is_sensor_fault"] is True
    assert classification["is_real_weather_event"] is False
    assert "Sensor Fault" in classification["classification_label"]
    assert classification["confidence"] >= 0.85

    # 2. Severity & Score
    assert classification["severity"] == "CRITICAL"
    assert classification["severity_score"] >= 0.85
    assert classification["severity_color"] == "rose"
    assert "isolation" in classification["operational_impact"].lower() or "nwp" in classification["operational_impact"].lower()

    # 3. Anomaly Explanation
    explanation = classification["anomaly_explanation"]
    assert "Sensor Fault" in explanation
    assert "55.0°C" in explanation or "55°C" in explanation or "excursion" in explanation
    assert "Distinguishing Evidence" in explanation

    # 4. Distinguishing Factors Matrix
    factors = classification["distinguishing_factors"]
    assert "spatial_consensus" in factors
    assert "thermodynamic_validity" in factors
    assert "rate_of_change_physical" in factors
    assert "multivariate_coherence" in factors

    assert factors["spatial_consensus"]["is_weather_supportive"] is False
    assert factors["thermodynamic_validity"]["is_weather_supportive"] is False
    assert "violation" in factors["thermodynamic_validity"]["status"].lower()
    assert len(classification["evidence_summary"]) >= 3


def test_ai_real_weather_event_classification_multi_station_consensus(stations, simulator):
    """
    Test that a regional atmospheric transition (e.g. convective cold front)
    observed across multiple neighboring AWS stations is classified as REAL_WEATHER_EVENT:
    - Verdict: REAL_WEATHER_EVENT
    - Severity: graded by convective intensity (MODERATE / HIGH / CRITICAL)
    - 4 Distinguishing Factors (Spatial consensus, Thermodynamic valid, Coherent multivariate)
    - Suppressed from sensor maintenance queues and routed to nowcasting
    """
    from src.models.fusion import DecisionFusionEngine
    from src.models.rules import RuleBasedQC
    from src.models.temporal import TemporalModel
    from src.models.multivariate import MultivariateConsistencyModel
    from src.models.spatial import SpatialConsistencyModel
    from src.ingestion.base import WeatherReading

    pune = next(s for s in stations if s.id == "AWS_PUN")
    now = datetime.datetime.now()

    # Create coordinated cold front reading across Pune and surrounding stations:
    # Pune drops to 21°C with 88% RH and 950 hPa (nominal for 560m elevation)
    pune_reading = WeatherReading(
        station_id=pune.id,
        timestamp=now,
        temperature=21.0,
        pressure=950.0,
        humidity=88.0,
        wind_speed=18.0,
        rainfall=12.0
    )

    # Neighboring AWS stations in Maharashtra also observed the cold front
    readings_map = {
        pune.id: pune_reading,
        "AWS_BOM": WeatherReading(station_id="AWS_BOM", timestamp=now, temperature=22.0, pressure=1005.0, humidity=85.0),
        "AWS_NSK": WeatherReading(station_id="AWS_NSK", timestamp=now, temperature=20.5, pressure=1003.5, humidity=90.0),
        "AWS_SAT": WeatherReading(station_id="AWS_SAT", timestamp=now, temperature=21.5, pressure=1004.0, humidity=86.0),
        "AWS_KLP": WeatherReading(station_id="AWS_KLP", timestamp=now, temperature=22.5, pressure=1004.5, humidity=84.0),
    }

    rule_qc = RuleBasedQC()
    temp_qc = TemporalModel()
    mv_qc = MultivariateConsistencyModel()
    spatial_qc = SpatialConsistencyModel(k_neighbors=4)
    fusion = DecisionFusionEngine()

    rule_eval = rule_qc.evaluate(pune_reading, pune)
    temporal_eval = temp_qc.evaluate(pune_reading, pune)
    mv_eval = mv_qc.evaluate(pune_reading, pune)
    spatial_eval = spatial_qc.evaluate(pune_reading, pune, stations, readings_map)

    # Pune's reading has high spatial agreement with neighbors
    classification = fusion.classify_fault_vs_weather(
        reading=pune_reading,
        station=pune,
        rule_eval=rule_eval,
        temporal_eval=temporal_eval,
        multivariate_eval=mv_eval,
        spatial_eval=spatial_eval,
        is_genuine_event=True
    )

    # 1. Verdict & Label
    assert classification["verdict"] == "REAL_WEATHER_EVENT"
    assert classification["is_real_weather_event"] is True
    assert classification["is_sensor_fault"] is False
    assert "Weather Event" in classification["classification_label"]
    assert classification["confidence"] >= 0.85

    # 2. Distinguishing Factors
    factors = classification["distinguishing_factors"]
    assert factors["spatial_consensus"]["is_weather_supportive"] is True
    assert factors["thermodynamic_validity"]["is_weather_supportive"] is True
    assert factors["multivariate_coherence"]["is_weather_supportive"] is True
    assert "consensus" in factors["spatial_consensus"]["status"].lower() or "multi-station" in factors["spatial_consensus"]["status"].lower()

    # 3. Severity & Operational Impact
    assert classification["severity"] in ["CRITICAL", "HIGH", "MODERATE"]
    assert "weather" in classification["operational_impact"].lower() or "nowcasting" in classification["operational_impact"].lower() or "front" in classification["operational_impact"].lower()


def test_pipeline_integration_event_classification(pipeline, stations, simulator, injector):
    """
    Test full end-to-end pipeline execution returns rich event_classification
    and explainable alert fields.
    """
    nagpur = next(s for s in stations if s.id == "AWS_NGP")
    now = datetime.datetime.now()
    clean = simulator._calculate_physical_reading(nagpur, now)
    injected_55c = injector.inject_55C_scenario(clean)

    res = pipeline.process_reading(injected_55c)

    # 1. Top-level pipeline payload
    assert "event_classification" in res
    evt = res["event_classification"]
    assert evt["verdict"] == "SENSOR_FAULT"
    assert evt["severity"] == "CRITICAL"
    assert "distinguishing_factors" in evt
    assert "operational_impact" in evt
    assert "anomaly_explanation" in evt
    assert res["is_sensor_fault"] is True
    assert res["is_real_weather_event"] is False

    # 2. Explainable Alert Integration
    assert "explainable_alert" in res
    exp_alert = res["explainable_alert"]
    assert "event_classification" in exp_alert
    assert "operational_impact" in exp_alert
    assert "distinguishing_factors" in exp_alert
    assert exp_alert["event_classification"]["verdict"] == "SENSOR_FAULT"
    assert exp_alert["severity"] == "Critical"
    assert any(term in exp_alert["badge_text"] for term in ["Possible Sensor", "Spike", "Fault", "Thermodynamic"])

    # 3. Alert Queue Record
    assert len(pipeline.recent_alerts) > 0
    top_alert = pipeline.recent_alerts[0]
    assert "event_classification" in top_alert
    assert top_alert["event_classification"]["verdict"] == "SENSOR_FAULT"
    assert "distinguishing_factors" in top_alert
    assert "operational_impact" in top_alert



