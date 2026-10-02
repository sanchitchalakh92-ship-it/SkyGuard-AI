"""
Core Multi-Layer Detection & Processing Pipeline for SkyGuard AI.
Orchestrates Layers 1-5, Imputation, Explainability, and Health Monitoring.
"""
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict, deque

from src.ingestion.base import WeatherReading, StationInfo
from src.models.rules import RuleBasedQC
from src.models.temporal import TemporalModel
from src.models.multivariate import MultivariateConsistencyModel
from src.models.spatial import SpatialConsistencyModel
from src.models.fusion import DecisionFusionEngine
from src.models.imputer import WeatherImputer
from src.explain.explainer import SkyGuardExplainer
from src.health.sensor_health import SensorHealthTracker


class SkyGuardPipeline:
    """
    Real-time high-throughput stateful pipeline for processing AWS observations.
    Executes in < 15 ms per reading on standard CPU.
    """

    def __init__(self, stations: List[StationInfo]):
        self.stations = stations
        self.stations_map = {s.id: s for s in stations}

        # Initialize layers
        self.layer1_rules = RuleBasedQC()
        self.layer2_temporal = TemporalModel()
        self.layer3_multivariate = MultivariateConsistencyModel()
        self.layer4_spatial = SpatialConsistencyModel()
        self.layer5_fusion = DecisionFusionEngine()
        self.imputer = WeatherImputer()
        self.explainer = SkyGuardExplainer()
        self.health_tracker = SensorHealthTracker()

        # Rolling history buffers: station_id -> deque of recent WeatherReadings (max 48 steps)
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=48))
        # Latest reading per station - auto-primed with nominal baseline
        from src.ingestion.simulator import RealisticSimulator
        _sim = RealisticSimulator(seed=42)
        _now = datetime.now()
        self.latest_readings: Dict[str, WeatherReading] = {
            s.id: _sim._calculate_physical_reading(s, _now) for s in stations
        }
        # Recent alerts buffer
        self.recent_alerts: deque = deque(maxlen=100)

    def process_reading(
        self,
        reading: WeatherReading,
        current_network_readings: Optional[Dict[str, WeatherReading]] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        stn = self.stations_map.get(reading.station_id)
        if not stn:
            raise KeyError(f"Unknown station {reading.station_id}")

        station_history = list(self.history[stn.id])
        current_map = current_network_readings or self.latest_readings

        # Layer 1: Rules
        rule_eval = self.layer1_rules.evaluate(reading, stn, station_history)

        # Layer 2: Temporal
        temporal_eval = self.layer2_temporal.evaluate(reading, stn, station_history)

        # Layer 4: Spatial (run before multivariate so cross-station consensus is available for context-aware check)
        spatial_eval = self.layer4_spatial.evaluate(reading, stn, self.stations, current_map)

        # Layer 3: Multivariate (Context-Aware Anomaly Detection)
        multivariate_eval = self.layer3_multivariate.evaluate(
            reading, stn, station_history, spatial_eval, temporal_eval
        )

        # Layer 5: Decision Fusion
        fusion_result = self.layer5_fusion.fuse(
            reading, stn, rule_eval, temporal_eval, multivariate_eval, spatial_eval
        )

        is_anomaly = fusion_result["is_anomaly"]
        severity = fusion_result["severity"]
        root_cause = fusion_result["root_cause"]

        # Imputation (if anomalous or missing)
        imputation_result = None
        if is_anomaly or reading.temperature is None:
            imputation_result = self.imputer.impute_reading(
                reading, stn, spatial_eval, temporal_eval
            )

        # Explainability: SHAP values + Natural language reasoning + Explainable AI Alert
        shap_contributions = self.explainer.compute_shap_values(reading, spatial_eval, temporal_eval)
        explanation_text = self.explainer.generate_natural_language_explanation(
            reading, stn, fusion_result, spatial_eval, temporal_eval, multivariate_eval, rule_eval
        )
        recommended_action = self.explainer.get_recommended_action(root_cause, severity)
        explainable_alert = self.explainer.generate_explainable_alert(
            reading, stn, fusion_result, spatial_eval, temporal_eval, multivariate_eval, rule_eval
        )

        # Update sensor health
        res_t = temporal_eval.get("temp_residual", 0.0)
        health_state = self.health_tracker.update_sensor_health(stn, is_anomaly, root_cause, res_t)

        # Update state buffers
        self.history[stn.id].append(reading)
        self.latest_readings[stn.id] = reading

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        result_payload = {
            "station_id": stn.id,
            "station_name": stn.name,
            "state": stn.state,
            "timestamp": reading.timestamp.isoformat() if hasattr(reading.timestamp, "isoformat") else str(reading.timestamp),
            "reading": {
                "temperature": reading.temperature,
                "pressure": reading.pressure,
                "humidity": reading.humidity,
                "wind_speed": reading.wind_speed,
                "rainfall": reading.rainfall,
                "solar_radiation": reading.solar_radiation
            },
            "is_anomaly": is_anomaly,
            "severity": severity,
            "confidence": fusion_result["confidence"],
            "anomaly_probability": fusion_result["anomaly_probability"],
            "root_cause": root_cause,
            "is_genuine_event": fusion_result.get("is_genuine_event", False),
            "layer_scores": fusion_result["layer_scores"],
            "explanation": explanation_text,
            "recommended_action": recommended_action,
            "explainable_alert": explainable_alert,
            "cross_station_verification": spatial_eval.get("cross_station_verification", {}),
            "sensor_fault_identification": fusion_result.get("sensor_fault_identification", {}),
            "event_classification": fusion_result.get("event_classification", {}),
            "is_sensor_fault": fusion_result.get("is_sensor_fault", False),
            "is_real_weather_event": fusion_result.get("is_real_weather_event", False),
            "operational_impact": explainable_alert.get("operational_impact", ""),
            "distinguishing_factors": explainable_alert.get("distinguishing_factors", {}),
            "evidence_summary": explainable_alert.get("evidence_summary", []),
            "context_aware": multivariate_eval.get("context_aware", {}),
            "contributing_measurements": explainable_alert.get("contributing_measurements", []),
            "engineer_inspection_checklist": explainable_alert.get("engineer_inspection_checklist", []),
            "unconfirmed_fault_disclaimer": explainable_alert.get("unconfirmed_fault_disclaimer", ""),
            "predictive_maintenance": {
                "maintenance_warning": health_state.get("maintenance_warning"),
                "requires_calibration": health_state.get("requires_calibration", False),
                "repeated_abnormal_behavior": health_state.get("repeated_abnormal_behavior", False),
                "affected_sensors": health_state.get("affected_sensors", [])
            },
            "shap_features": shap_contributions,
            "imputation": imputation_result,
            "sensor_health": {
                "composite": health_state["composite_health"],
                "temperature": round(health_state["temperature_health"], 1),
                "pressure": round(health_state["pressure_health"], 1),
                "humidity": round(health_state["humidity_health"], 1),
                "priority": health_state["priority"],
                "maintenance_warning": health_state.get("maintenance_warning"),
                "requires_calibration": health_state.get("requires_calibration", False),
                "repeated_abnormal_behavior": health_state.get("repeated_abnormal_behavior", False)
            },
            "processing_latency_ms": elapsed_ms
        }

        # If alert and not suppressed, append to alert stream
        if is_anomaly and not fusion_result.get("suppressed_by_cooldown"):
            self.recent_alerts.appendleft({
                "id": f"alert_{stn.id}_{int(time.time()*1000)}",
                "station_id": stn.id,
                "station_name": stn.name,
                "timestamp": result_payload["timestamp"],
                "severity": severity,
                "root_cause": root_cause,
                "message": explanation_text,
                "reading": result_payload["reading"],
                "imputation": imputation_result,
                "explainable_alert": explainable_alert,
                "what_changed": explainable_alert["what_changed"],
                "expected_value": explainable_alert["expected_value"],
                "actual_value": explainable_alert["actual_value"],
                "deviation_pct": explainable_alert["deviation_pct"],
                "deviation_pct_formatted": explainable_alert["deviation_pct_formatted"],
                "confidence": explainable_alert["confidence"],
                "confidence_pct": explainable_alert["confidence_pct"],
                "possible_cause": explainable_alert["possible_cause"],
                "recommended_action": explainable_alert["recommended_action"],
                "badge_text": explainable_alert["badge_text"],
                "badge_color": explainable_alert["badge_color"],
                "cross_station_verification": spatial_eval.get("cross_station_verification", {}),
                "sensor_fault_identification": fusion_result.get("sensor_fault_identification", {}),
                "event_classification": fusion_result.get("event_classification", {}),
                "is_sensor_fault": fusion_result.get("is_sensor_fault", False),
                "is_real_weather_event": fusion_result.get("is_real_weather_event", False),
                "operational_impact": explainable_alert.get("operational_impact", ""),
                "distinguishing_factors": explainable_alert.get("distinguishing_factors", {}),
                "evidence_summary": explainable_alert.get("evidence_summary", []),
                "context_aware": multivariate_eval.get("context_aware", {}),
                "contributing_measurements": explainable_alert.get("contributing_measurements", []),
                "engineer_inspection_checklist": explainable_alert.get("engineer_inspection_checklist", []),
                "unconfirmed_fault_disclaimer": explainable_alert.get("unconfirmed_fault_disclaimer", ""),
                "maintenance_warning": health_state.get("maintenance_warning")
            })

        return result_payload
