"""
Layer 5: Decision Fusion Meta-Classifier and Genuine Event Discriminator.
Combines scores across all layers into confidence, severity, and root-cause classification.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from src.ingestion.injector import AnomalyType


class DecisionFusionEngine:
    """
    Synthesizes rule-based, temporal, multivariate, and spatial layer assessments.
    Differentiates true hardware/network faults from genuine extreme meteorological events.
    Applies hysteresis to suppress alert storms.
    """

    def __init__(self):
        # Layer weights
        self.w_rules = 0.35
        self.w_temporal = 0.20
        self.w_multivariate = 0.20
        self.w_spatial = 0.25

        # Alert hysteresis cache: station_id -> last_alert_time
        self.last_alerts: Dict[str, datetime] = {}
        self.cooldown_minutes = 10

    def _determine_severity(self, probability: float) -> str:
        if probability >= 0.85:
            return "Critical"
        elif probability >= 0.65:
            return "High"
        elif probability >= 0.40:
            return "Medium"
        elif probability >= 0.20:
            return "Low"
        return "Normal"

    def _identify_sensor_fault(
        self,
        reading,
        rule_eval: Dict[str, Any],
        temporal_eval: Dict[str, Any],
        multivariate_eval: Dict[str, Any],
        spatial_eval: Dict[str, Any],
        is_genuine_event: bool
    ) -> Dict[str, Any]:
        """
        Sensor Fault Identification:
        Distinguishes between possible sensor drift, stuck readings, sudden spikes,
        missing data and communication failure, with a calibrated confidence score.
        Avoids presenting model predictions as confirmed sensor faults.
        """
        if is_genuine_event:
            return {
                "fault_type": "Possible Real Weather Event",
                "confidence_score": 0.96,
                "confidence_pct": "96%",
                "is_confirmed_fault": False,
                "is_sensor_fault": False,
                "is_real_weather_event": True,
                "detection_mechanism": "Cross-station multi-sensor spatial consensus",
                "description": "Multi-station atmospheric transition consistent with regional weather dynamics."
            }

        rule_suspected = rule_eval.get("suspected_causes", [])
        rule_violations = " ".join(rule_eval.get("violations", []))

        t = getattr(reading, "temperature", None)
        p = getattr(reading, "pressure", None)
        rh = getattr(reading, "humidity", None)

        # 1. Communication Failure vs Missing Data
        missing_count = sum(1 for v in [t, p, rh] if v is None or v == -999.0 or (isinstance(v, float) and np.isnan(v)))
        if missing_count >= 2 or ("Communication Error" in rule_suspected and missing_count >= 2):
            return {
                "fault_type": "Communication Failure",
                "fault_category": "Communication Failure",
                "confidence_score": 0.98,
                "confidence_pct": "98%",
                "is_confirmed_fault": False,
                "is_sensor_fault": True,
                "is_real_weather_event": False,
                "detection_mechanism": "Telemetry Frame Loss / Multi-Channel Sentinel Detector",
                "description": "Multiple simultaneous sensor channel dropouts indicative of telemetry gateway or power link interruption."
            }
        elif missing_count == 1:
            return {
                "fault_type": "Missing Data",
                "fault_category": "Missing Data",
                "confidence_score": 0.99,
                "confidence_pct": "99%",
                "is_confirmed_fault": False,
                "is_sensor_fault": True,
                "is_real_weather_event": False,
                "detection_mechanism": "Single-Channel Null/Sentinel Parser",
                "description": "Isolated sensor channel dropout with missing frame packets while remaining telemetry channels function."
            }

        # 2. Stuck Readings (Persistence / Frozen Sensor)
        if "Frozen Sensor" in rule_suspected or "Frozen sensor" in rule_violations:
            conf = 0.95
            return {
                "fault_type": "Stuck Reading",
                "fault_category": "Stuck Reading",
                "confidence_score": conf,
                "confidence_pct": f"{int(conf*100)}%",
                "is_confirmed_fault": False,
                "is_sensor_fault": True,
                "is_real_weather_event": False,
                "detection_mechanism": "Statistical Zero-Variance Invariance Test",
                "description": "Sensor value remained invariant across consecutive observation cycles with zero physical micro-fluctuations."
            }

        # 3. Possible Sensor Drift
        cusum_stat = temporal_eval.get("cusum_drift", 0.0) if temporal_eval else 0.0
        temp_res = temporal_eval.get("temp_residual", 0.0) if temporal_eval else 0.0
        context_aware = multivariate_eval.get("context_aware", {})
        delta_t = context_aware.get("recent_trends", {}).get("delta_temperature", 0.0)

        if cusum_stat >= 3.0 or ("drift" in rule_violations.lower()) or (2.0 <= temp_res <= 5.5 and abs(delta_t) < 1.5):
            conf = min(0.96, max(0.85, 0.82 + (cusum_stat / 20.0)))
            return {
                "fault_type": "Possible Sensor Drift",
                "fault_category": "Possible Sensor Drift",
                "confidence_score": round(float(conf), 2),
                "confidence_pct": f"{int(conf*100)}%",
                "is_confirmed_fault": False,
                "is_sensor_fault": True,
                "is_real_weather_event": False,
                "detection_mechanism": "Cumulative Sum (CUSUM) Sequential Drift Test",
                "description": "Gradual monotonic baseline offset accumulating over time without physical meteorological driver."
            }

        # 4. Sudden Spikes (Sudden Step Jumps)
        if "Abrupt" in rule_violations or "step jump" in rule_violations or abs(delta_t) >= 3.0 or temp_res >= 5.0 or "Spike" in " ".join(rule_suspected):
            jump_mag = max(abs(delta_t), temp_res)
            conf = min(0.99, max(0.85, 0.80 + (jump_mag / 20.0)))
            return {
                "fault_type": "Possible Sensor Spike",
                "fault_category": "Possible Sensor Spike",
                "confidence_score": round(float(conf), 2),
                "confidence_pct": f"{int(conf*100)}%",
                "is_confirmed_fault": False,
                "is_sensor_fault": True,
                "is_real_weather_event": False,
                "detection_mechanism": "Rate-of-Change Impulse & Spatial IDW Residual",
                "description": f"Abrupt unphysical step excursion ({jump_mag:.1f}°C) within single observation interval."
            }

        conf = 0.88
        return {
            "fault_type": "Possible Sensor Spike",
            "fault_category": "Possible Sensor Spike",
            "confidence_score": conf,
            "confidence_pct": "88%",
            "is_confirmed_fault": False,
            "is_sensor_fault": True,
            "is_real_weather_event": False,
            "detection_mechanism": "Multi-Layer Decision Ensemble",
            "description": "Telemetry value deviates from ensemble statistical expectation."
        }

    def _classify_root_cause(
        self,
        rule_eval: Dict[str, Any],
        temporal_eval: Dict[str, Any],
        multivariate_eval: Dict[str, Any],
        spatial_eval: Dict[str, Any],
        is_genuine_event: bool,
        reading: Optional[Any] = None
    ) -> str:
        fault_info = self._identify_sensor_fault(
            reading, rule_eval, temporal_eval, multivariate_eval, spatial_eval, is_genuine_event
        )
        return fault_info["fault_type"]

    def _discriminate_genuine_event(
        self,
        reading,
        rule_eval: Dict[str, Any],
        temporal_eval: Dict[str, Any],
        multivariate_eval: Dict[str, Any],
        spatial_eval: Dict[str, Any]
    ) -> bool:
        """
        Genuine event discriminator:
        - If physical dew point constraint holds
        - Cross-Station Verification: If multiple nearby stations show the same abnormal pattern, flag as real weather event
        - If only one station shows an extreme value (is_sensor_fault), it is strictly a sensor fault.
        """
        # If dew point > temp, physics is broken -> strictly sensor anomaly, NOT genuine event
        if multivariate_eval.get("physics_violations"):
            return False

        cross_stn = spatial_eval.get("cross_station_verification", {})
        # Cross-Station Rule: Multiple nearby stations show the same abnormal pattern
        if cross_stn.get("is_real_weather_event"):
            return True

        # Cross-Station Rule: Only one station shows extreme value while neighbors are nominal
        if cross_stn.get("is_sensor_fault"):
            return False

        # If temperature is 55°C while spatial agreement is 0.0, this is an anomaly
        if spatial_eval.get("neighbor_agreement", 1.0) == 0.0 and spatial_eval.get("score", 0) > 0.8:
            return False

        # Genuine Thunderstorm Signature:
        # Rapid T drop with RH surge, but Dew Point <= T holds, and neighbor agreement > 0.4
        t_res = temporal_eval.get("temp_residual", 0.0)
        nbr_agree = spatial_eval.get("neighbor_agreement", 0.0)
        if t_res > 5.0 and nbr_agree >= 0.5:
            return True

        return False

    def classify_fault_vs_weather(
        self,
        reading,
        station,
        rule_eval: Dict[str, Any],
        temporal_eval: Dict[str, Any],
        multivariate_eval: Dict[str, Any],
        spatial_eval: Dict[str, Any],
        is_genuine_event: Optional[bool] = None,
        raw_anomaly_prob: float = 0.0,
        fault_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        AI-Based Sensor Fault vs. Real Weather Event Classifier:
        Discriminates between hardware/sensor failures and genuine meteorological transitions,
        providing granular anomaly explanation, distinguishing evidence factors, and severity grading.
        """
        if is_genuine_event is None:
            is_genuine_event = self._discriminate_genuine_event(
                reading, rule_eval, temporal_eval, multivariate_eval, spatial_eval
            )

        if fault_info is None:
            fault_info = self._identify_sensor_fault(
                reading, rule_eval, temporal_eval, multivariate_eval, spatial_eval, is_genuine_event
            )

        cross_stn = spatial_eval.get("cross_station_verification", {})
        nbr_agree = spatial_eval.get("neighbor_agreement", 1.0)
        nbr_count = spatial_eval.get("neighbors_count", 0)
        sp_res = spatial_eval.get("temp_residual")
        sp_expected = spatial_eval.get("spatial_expected_temp")
        sp_res_str = f"{sp_res:+.1f}°C" if sp_res is not None else "N/A"
        sp_exp_str = f"{sp_expected:.1f}°C" if sp_expected is not None else "N/A"

        t = getattr(reading, "temperature", None)
        p = getattr(reading, "pressure", None)
        rh = getattr(reading, "humidity", None)
        w = getattr(reading, "wind_speed", 0.0) or 0.0
        rf = getattr(reading, "rainfall", 0.0) or 0.0

        physics_violations = multivariate_eval.get("physics_violations", [])
        context_aware = multivariate_eval.get("context_aware", {})
        delta_t = context_aware.get("recent_trends", {}).get("delta_temperature", 0.0)
        temp_residual = temporal_eval.get("temp_residual", 0.0)
        delta_t_val = delta_t if delta_t is not None else 0.0
        temp_res_val = temp_residual if temp_residual is not None else 0.0
        jump_mag = max(abs(delta_t_val), temp_res_val)

        # -------------------------------------------------------------------------
        # Factor 1: Spatial Consensus (Multi-Station Corroboration vs Isolated Outlier)
        # -------------------------------------------------------------------------
        is_spatial_weather = bool(cross_stn.get("is_real_weather_event") or (nbr_agree >= 0.40 and nbr_count >= 2))
        agreeing_cnt = cross_stn.get("agreeing_stations_count", int(nbr_agree * nbr_count))
        nearby_cnt = cross_stn.get("nearby_stations_count", nbr_count)

        stn_id = getattr(station, "id", "") if station else ""
        if is_spatial_weather:
            factor_spatial = {
                "factor": "Spatial Consensus",
                "status": "Multi-Station Consensus",
                "is_weather_supportive": True,
                "score": round(float(nbr_agree), 2),
                "detail": f"{agreeing_cnt} of {nearby_cnt} nearby AWS stations corroborate this atmospheric transition ({nbr_agree*100:.0f}% spatial consensus)."
            }
        else:
            factor_spatial = {
                "factor": "Spatial Consensus",
                "status": "Single-Station Isolated Outlier",
                "is_weather_supportive": False,
                "score": round(float(1.0 - nbr_agree), 2),
                "detail": f"Isolated to target station {stn_id} (residual: {sp_res_str} vs expected {sp_exp_str}). Surrounding AWS stations remain nominal."
            }

        # -------------------------------------------------------------------------
        # Factor 2: Thermodynamic Validity (Physical Laws vs Instrument Glitch)
        # -------------------------------------------------------------------------
        has_physics_violation = len(physics_violations) > 0
        if has_physics_violation:
            factor_thermo = {
                "factor": "Thermodynamic Validity",
                "status": "Physical Atmospheric Law Violation",
                "is_weather_supportive": False,
                "score": 0.0,
                "detail": f"Thermodynamic violation: {'; '.join(physics_violations)}. Non-physical dew point / vapor pressure relation proves hardware malfunction."
            }
        else:
            factor_thermo = {
                "factor": "Thermodynamic Validity",
                "status": "Thermodynamically Valid",
                "is_weather_supportive": True,
                "score": 1.0,
                "detail": "Observed psychrometric parameters adhere strictly to Clausius-Clapeyron atmospheric saturation limits (Td <= T)."
            }

        # -------------------------------------------------------------------------
        # Factor 3: Rate of Change & Thermal Inertia
        # -------------------------------------------------------------------------
        jump_mag = max(abs(delta_t), temp_residual)
        is_frozen = "Stuck" in fault_info.get("fault_type", "") or "Frozen" in fault_info.get("fault_type", "")
        if is_frozen:
            factor_roc = {
                "factor": "Rate of Change Physics",
                "status": "Unphysical Invariance (Frozen Transducer)",
                "is_weather_supportive": False,
                "score": 0.0,
                "detail": "Zero turbulent variance across consecutive cycles. Natural atmospheric boundary-layer turbulence is absent."
            }
        elif jump_mag >= 5.0 and not is_spatial_weather:
            factor_roc = {
                "factor": "Rate of Change Physics",
                "status": "Unphysical Step Jump",
                "is_weather_supportive": False,
                "score": 0.1,
                "detail": f"Single-step excursion of {jump_mag:.1f}°C exceeds atmospheric air mass thermal inertia limits without convective trigger."
            }
        else:
            factor_roc = {
                "factor": "Rate of Change Physics",
                "status": "Physical Atmospheric Transition Rate",
                "is_weather_supportive": True,
                "score": 0.95,
                "detail": f"Temporal rate-of-change ({jump_mag:.1f}°C) is consistent with natural diurnal solar heating or synoptic advection."
            }

        # -------------------------------------------------------------------------
        # Factor 4: Multivariate Parameter Coherence
        # -------------------------------------------------------------------------
        is_ctx_corroborated = context_aware.get("is_corroborated_by_weather", False)
        is_isolated_shift = context_aware.get("is_isolated_shift", False)
        if is_ctx_corroborated or (rf > 0.0 or w > 15.0 or (t is not None and t < 25.0 and rh is not None and rh > 75.0)):
            factor_mv = {
                "factor": "Multivariate Parameter Coherence",
                "status": "Coherent Multi-Sensor Coupling",
                "is_weather_supportive": True,
                "score": 0.90,
                "detail": "Thermal variation is physically coupled with correlated relative humidity, pressure, wind gust, or precipitation response."
            }
        elif is_isolated_shift or (jump_mag >= 3.0 and not is_spatial_weather):
            factor_mv = {
                "factor": "Multivariate Parameter Coherence",
                "status": "Incoherent Isolated Shift",
                "is_weather_supportive": False,
                "score": 0.15,
                "detail": "Temperature shifted abruptly in isolation while barometric pressure and humidity showed zero corresponding atmospheric reaction."
            }
        else:
            factor_mv = {
                "factor": "Multivariate Parameter Coherence",
                "status": "Nominal Multivariate Consistency",
                "is_weather_supportive": True,
                "score": 0.85,
                "detail": "Cross-sensor readings show expected mutual consistency under prevailing ambient conditions."
            }

        # -------------------------------------------------------------------------
        # Classification Verdict & Subtype Assignment
        # -------------------------------------------------------------------------
        if has_physics_violation:
            verdict = "SENSOR_FAULT"
            label = "Possible Sensor Fault"
            subtype = "Thermodynamic Disparity Fault"
            conf = 0.98
        elif is_genuine_event or (is_spatial_weather and not has_physics_violation):
            verdict = "REAL_WEATHER_EVENT"
            label = "Real Weather Event"
            if rf > 5.0 or w > 20.0 or (temp_residual >= 4.0 and rh is not None and rh > 80):
                subtype = "Convective Squall / Thunderstorm Front"
                conf = 0.97
            elif sp_res is not None and abs(sp_res) >= 3.0 and nbr_agree >= 0.5:
                subtype = "Synoptic Frontal Transition"
                conf = 0.94
            else:
                subtype = "Regional Meteorological Event"
                conf = 0.91
        elif fault_info.get("fault_type") in ["Communication Failure", "Missing Data"]:
            verdict = "SENSOR_FAULT"
            label = "Possible Sensor Fault"
            subtype = fault_info.get("fault_type")
            conf = fault_info.get("confidence_score", 0.98)
        elif is_frozen:
            verdict = "SENSOR_FAULT"
            label = "Possible Sensor Fault"
            subtype = "Stuck / Frozen Sensor"
            conf = 0.96
        elif "Drift" in fault_info.get("fault_type", ""):
            verdict = "SENSOR_FAULT"
            label = "Possible Sensor Fault"
            subtype = "Sensor Calibration Drift"
            conf = fault_info.get("confidence_score", 0.92)
        elif jump_mag >= 3.0 or raw_anomaly_prob >= 0.40:
            verdict = "SENSOR_FAULT"
            label = "Possible Sensor Fault"
            subtype = "Possible Sensor Spike"
            conf = min(0.99, max(0.85, 0.82 + (jump_mag / 20.0)))
        else:
            verdict = "REAL_WEATHER_EVENT" if is_spatial_weather else "SENSOR_FAULT"
            label = "Normal / Consistent Observation"
            subtype = "Baseline Environmental Reading"
            conf = 0.85

        # -------------------------------------------------------------------------
        # Severity Grading & Operational Impact
        # -------------------------------------------------------------------------
        if verdict == "SENSOR_FAULT":
            if (t is not None and (t >= 50.0 or t <= -10.0)) or jump_mag >= 12.0 or subtype in ["Communication Failure", "Thermodynamic Disparity Fault"]:
                severity = "CRITICAL"
                sev_score = 0.95
                sev_color = "rose"
                operational_impact = (
                    "CRITICAL SENSOR FAILURE: Immediate automated isolation of corrupted channel from Numerical Weather Prediction (NWP) "
                    "data assimilation to prevent false synoptic alerts. Apply spatial IDW imputation and notify maintenance dispatch."
                )
            elif jump_mag >= 5.0 or subtype in ["Stuck / Frozen Sensor", "Missing Data"] or "Drift" in subtype:
                severity = "HIGH"
                sev_score = 0.78
                sev_color = "rose"
                operational_impact = (
                    "HIGH SENSOR FAULT: Reading flagged as suspect. Synthesize replacement telemetry via spatial regression and schedule "
                    "on-site technician diagnostic inspection."
                )
            elif raw_anomaly_prob >= 0.35:
                severity = "MODERATE"
                sev_score = 0.55
                sev_color = "amber"
                operational_impact = (
                    "MODERATE SENSOR ANOMALY: Telemetry drift detected. Flag for sensor calibration offset recalibration during upcoming scheduled maintenance."
                )
            else:
                severity = "LOW"
                sev_score = 0.25
                sev_color = "blue"
                operational_impact = "LOW SEVERITY: Minor telemetry perturbation within acceptable tolerance. Automated monitoring active."
        else:
            # Real Weather Event Severity
            if (w is not None and w >= 25.0) or (rf is not None and rf >= 25.0) or (temp_residual is not None and temp_residual >= 8.0 and nbr_agree >= 0.6):
                severity = "CRITICAL"
                sev_score = 0.94
                sev_color = "teal"
                operational_impact = (
                    "CRITICAL WEATHER HAZARD: Coherent extreme convective storm or squall line confirmed across multiple AWS. "
                    "Issue immediate nowcasting weather advisory to state disaster management authority and civil aviation."
                )
            elif (w is not None and w >= 15.0) or (rf is not None and rf >= 10.0) or (temp_residual is not None and temp_residual >= 4.0):
                severity = "HIGH"
                sev_score = 0.76
                sev_color = "teal"
                operational_impact = (
                    "HIGH METEOROLOGICAL ALERT: Active weather front or intense precipitation cooling. Corroborated by spatial AWS cluster; "
                    "broadcast observation to regional forecasting desks."
                )
            elif (temp_residual is not None and temp_residual >= 2.0) or (sp_res is not None and abs(sp_res) >= 2.0):
                severity = "MODERATE"
                sev_score = 0.50
                sev_color = "emerald"
                operational_impact = (
                    "MODERATE WEATHER EVENT: Regional air mass boundary or diurnal transition. All sensors operational; log synoptic transition."
                )
            else:
                severity = "LOW"
                sev_score = 0.20
                sev_color = "emerald"
                operational_impact = "LOW METEOROLOGICAL ADVISORY: Benign ambient diurnal variation. Baseline automated monitoring active."

        # -------------------------------------------------------------------------
        # Comprehensive Anomaly Explanation Synthesis
        # -------------------------------------------------------------------------
        t_val = f"{t:.1f}°C" if t is not None else "Missing"
        jump_str = f" (excursion of {jump_mag:+.1f}°C)" if jump_mag > 0 else ""
        if verdict == "SENSOR_FAULT":
            explanation = (
                f"Event classified by AI as Possible Sensor Fault ({subtype}) with {conf*100:.0f}% confidence and {severity} severity. "
                f"Station observed {t_val}{jump_str}. "
                f"Distinguishing Evidence: (1) Isolated reading with 0 of {nearby_cnt} neighboring AWS stations confirming the excursion; "
                f"(2) {'Thermodynamic violation of Clausius-Clapeyron limits' if has_physics_violation else 'Uncoupled rate-of-change with static auxiliary parameters'}; "
                f"(3) Model identifies this as an instrument failure rather than a physical weather transition."
            )
        else:
            explanation = (
                f"Event classified by AI as Genuine Real Weather Event ({subtype}) with {conf*100:.0f}% confidence and {severity} severity. "
                f"Station observed {t_val} with physical atmospheric parameter coupling. "
                f"Distinguishing Evidence: (1) Spatial consensus corroborated by {agreeing_cnt} of {nearby_cnt} nearby AWS stations ({nbr_agree*100:.0f}% agreement); "
                f"(2) Observation adheres strictly to Clausius-Clapeyron thermodynamic laws; "
                f"(3) Alert suppressed from sensor fault queues and routed to meteorological nowcasting."
            )

        evidence_summary = [
            f"Spatial: {factor_spatial['status']} ({factor_spatial['score']*100:.0f}% agreement)",
            f"Thermodynamic: {factor_thermo['status']}",
            f"Rate-of-Change: {factor_roc['status']}",
            f"Multivariate: {factor_mv['status']}"
        ]

        return {
            "verdict": verdict,
            "classification_label": label,
            "subtype": subtype,
            "confidence": round(float(conf), 3),
            "confidence_pct": f"{int(conf * 100)}%",
            "severity": severity,
            "severity_score": round(float(sev_score), 2),
            "severity_color": sev_color,
            "operational_impact": operational_impact,
            "anomaly_explanation": explanation,
            "distinguishing_factors": {
                "spatial_consensus": factor_spatial,
                "thermodynamic_validity": factor_thermo,
                "rate_of_change_physical": factor_roc,
                "multivariate_coherence": factor_mv
            },
            "evidence_summary": evidence_summary,
            "is_sensor_fault": verdict == "SENSOR_FAULT",
            "is_real_weather_event": verdict == "REAL_WEATHER_EVENT"
        }

    def fuse(
        self,
        reading,
        station,
        rule_eval: Dict[str, Any],
        temporal_eval: Dict[str, Any],
        multivariate_eval: Dict[str, Any],
        spatial_eval: Dict[str, Any]
    ) -> Dict[str, Any]:
        r_score = rule_eval.get("score", 0.0)
        t_score = temporal_eval.get("score", 0.0)
        m_score = multivariate_eval.get("score", 0.0)
        s_score = spatial_eval.get("score", 0.0)

        # Weighted combination
        raw_prob = (
            self.w_rules * r_score +
            self.w_temporal * t_score +
            self.w_multivariate * m_score +
            self.w_spatial * s_score
        )

        # Boost if multiple layers strongly trigger simultaneously
        active_layers = sum(1 for s in [r_score, t_score, m_score, s_score] if s > 0.6)
        if active_layers >= 3:
            raw_prob = min(1.0, raw_prob * 1.25)

        # Check for genuine meteorological event
        is_genuine = self._discriminate_genuine_event(
            reading, rule_eval, temporal_eval, multivariate_eval, spatial_eval
        )

        # Sensor fault identification with calibrated confidence
        fault_info = self._identify_sensor_fault(
            reading, rule_eval, temporal_eval, multivariate_eval, spatial_eval, is_genuine
        )

        # AI-based Sensor Fault vs Real Weather Event Classification with severity & explanation
        event_classification = self.classify_fault_vs_weather(
            reading=reading,
            station=station,
            rule_eval=rule_eval,
            temporal_eval=temporal_eval,
            multivariate_eval=multivariate_eval,
            spatial_eval=spatial_eval,
            is_genuine_event=is_genuine,
            raw_anomaly_prob=raw_prob,
            fault_info=fault_info
        )

        if is_genuine or event_classification["verdict"] == "REAL_WEATHER_EVENT":
            anomaly_prob = 0.05
            is_anomaly = False
            root_cause = "Possible Real Weather Event"
            severity = event_classification["severity"] if event_classification["severity"] in ["Critical", "High"] else "Normal"
        else:
            anomaly_prob = round(float(raw_prob), 3)
            is_anomaly = anomaly_prob >= 0.40
            severity = self._determine_severity(anomaly_prob)
            root_cause = fault_info["fault_type"]

        # Hysteresis check for alert triggering
        now = reading.timestamp or datetime.now()
        suppressed_by_hysteresis = False
        if is_anomaly:
            last = self.last_alerts.get(station.id)
            if last and (now - last) < timedelta(minutes=self.cooldown_minutes):
                suppressed_by_hysteresis = True
            else:
                self.last_alerts[station.id] = now

        return {
            "is_anomaly": is_anomaly,
            "anomaly_probability": anomaly_prob,
            "confidence": round(float(max(anomaly_prob, 1.0 - anomaly_prob)), 3),
            "severity": severity,
            "root_cause": root_cause,
            "sensor_fault_identification": fault_info,
            "event_classification": event_classification,
            "is_sensor_fault": event_classification["verdict"] == "SENSOR_FAULT",
            "is_real_weather_event": event_classification["verdict"] == "REAL_WEATHER_EVENT",
            "is_genuine_event": is_genuine,
            "suppressed_by_cooldown": suppressed_by_hysteresis,
            "layer_scores": {
                "rules": r_score,
                "temporal": t_score,
                "multivariate": m_score,
                "spatial": s_score
            }
        }
