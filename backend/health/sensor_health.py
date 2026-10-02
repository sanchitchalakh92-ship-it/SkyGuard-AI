"""
Sensor Health and Predictive Maintenance Tracking for SkyGuard AI.
Calculates continuous sensor degradation index (0-100), CUSUM drift detection,
and prioritizes maintenance requirements across the AWS network.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from src.ingestion.base import StationInfo


class SensorHealthTracker:
    """
    Maintains health telemetry for Temperature, Pressure, and Humidity sensors:
    - Drift trend tracking via CUSUM
    - Anomaly rate penalization
    - Degradation trajectory forecasting
    - Maintenance priority categorization (OK, Watch, Service Soon, Urgent)
    """

    def __init__(self):
        # station_id -> { "temperature": score, "pressure": score, "humidity": score, ... }
        self.health_states: Dict[str, Dict[str, Any]] = {}

    def _init_station_state(self, station_id: str) -> Dict[str, Any]:
        return {
            "temperature_health": 100.0,
            "pressure_health": 100.0,
            "humidity_health": 100.0,
            "composite_health": 100.0,
            "cusum_drift_t": 0.0,
            "anomaly_count_24h": 0,
            "temp_anomaly_count_24h": 0,
            "pressure_anomaly_count_24h": 0,
            "humidity_anomaly_count_24h": 0,
            "consecutive_anomalies": 0,
            "frozen_count_24h": 0,
            "missing_count_24h": 0,
            "repeated_abnormal_behavior": False,
            "requires_calibration": False,
            "maintenance_warning": None,
            "affected_sensors": [],
            "last_updated": datetime.now(),
            "priority": "OK",
            "recommended_maintenance": "Sensor operating nominally."
        }

    def update_sensor_health(
        self,
        station: StationInfo,
        is_anomaly: bool,
        root_cause: str,
        residual_t: float = 0.0
    ) -> Dict[str, Any]:
        state = self.health_states.get(station.id)
        if not state:
            state = self._init_station_state(station.id)
            self.health_states[station.id] = state

        # 1. CUSUM drift tracking for temperature
        k_slack = 1.0  # slack allowance (°C)
        h_threshold = 4.5
        drift_delta = max(0.0, abs(residual_t) - k_slack)
        state["cusum_drift_t"] = round(max(0.0, state["cusum_drift_t"] * 0.95 + drift_delta * 0.2), 2)

        # 2. Update penalties if anomaly occurred
        if is_anomaly:
            state["anomaly_count_24h"] += 1
            state["consecutive_anomalies"] += 1

            if root_cause in ["Frozen Sensor", "Frozen"]:
                state["frozen_count_24h"] += 1
                state["temp_anomaly_count_24h"] += 1
                state["temperature_health"] = max(10.0, state["temperature_health"] - 12.0)
            elif root_cause in ["Spike/Sensor Fault", "Possible Sensor Fault", "Spike"]:
                state["temp_anomaly_count_24h"] += 1
                state["temperature_health"] = max(15.0, state["temperature_health"] - 8.0)
            elif root_cause in ["Communication Error", "Dropout"]:
                state["missing_count_24h"] += 1
                state["pressure_anomaly_count_24h"] += 1
                state["pressure_health"] = max(20.0, state["pressure_health"] - 6.0)
            elif root_cause in ["Inconsistent Multivariate"]:
                state["humidity_anomaly_count_24h"] += 1
                state["humidity_health"] = max(20.0, state["humidity_health"] - 10.0)
            elif root_cause in ["Calibration Drift", "Sensor Drift"]:
                state["temp_anomaly_count_24h"] += 1
                state["temperature_health"] = max(20.0, state["temperature_health"] - 7.0)
            else:
                state["temp_anomaly_count_24h"] += 1
                state["temperature_health"] = max(20.0, state["temperature_health"] - 5.0)
        else:
            state["consecutive_anomalies"] = 0
            # Gradual health recovery over uninterrupted clean operation (up to 100)
            state["temperature_health"] = min(100.0, state["temperature_health"] + 0.3)
            state["pressure_health"] = min(100.0, state["pressure_health"] + 0.3)
            state["humidity_health"] = min(100.0, state["humidity_health"] + 0.3)

        # Deduct if CUSUM drift is elevated
        if state["cusum_drift_t"] > h_threshold:
            state["temperature_health"] = max(15.0, state["temperature_health"] - 5.0)

        # Composite score
        comp = (
            0.45 * state["temperature_health"] +
            0.30 * state["humidity_health"] +
            0.25 * state["pressure_health"]
        )
        state["composite_health"] = round(float(comp), 1)

        # 3. Predictive Maintenance: Detect repeated abnormal sensor behavior
        # Criteria: >= 3 anomalies in 24h, or consecutive anomalies >= 2, or high CUSUM drift > 3.5, or frozen >= 2
        is_repeated_abnormal = (
            state["anomaly_count_24h"] >= 3 or
            state["consecutive_anomalies"] >= 2 or
            state["temp_anomaly_count_24h"] >= 2 or
            state["cusum_drift_t"] >= 3.5 or
            state["frozen_count_24h"] >= 1
        )

        affected_sensors = []
        if state["temp_anomaly_count_24h"] >= 2 or state["cusum_drift_t"] >= 3.5 or state["frozen_count_24h"] >= 1:
            affected_sensors.append("Temperature Sensor")
        if state["pressure_anomaly_count_24h"] >= 2:
            affected_sensors.append("Pressure Sensor")
        if state["humidity_anomaly_count_24h"] >= 2:
            affected_sensors.append("Humidity Sensor")
        if not affected_sensors and is_repeated_abnormal:
            affected_sensors.append("Primary Transducer")

        state["affected_sensors"] = affected_sensors

        if is_repeated_abnormal:
            state["repeated_abnormal_behavior"] = True
            state["requires_calibration"] = True
            state["maintenance_warning"] = "Sensor may require inspection/calibration."
        else:
            state["repeated_abnormal_behavior"] = False
            state["requires_calibration"] = False
            state["maintenance_warning"] = None

        # Maintenance Priority
        if comp >= 85.0 and not is_repeated_abnormal:
            priority = "OK"
            maint_note = "All sensors operating within standard calibration limits."
        elif comp >= 70.0 and not is_repeated_abnormal:
            priority = "Watch"
            maint_note = "Mild calibration drift or intermittent fluctuations observed. Monitor telemetry."
        elif is_repeated_abnormal or comp >= 50.0:
            priority = "Service Soon" if comp >= 50.0 else "Urgent"
            sensor_str = ", ".join(affected_sensors) if affected_sensors else "Sensor"
            maint_note = f"Sensor may require inspection/calibration. Repeated abnormal behaviour detected in {sensor_str} ({state['anomaly_count_24h']} anomalies in 24h, CUSUM drift {state['cusum_drift_t']}°C)."
        else:
            priority = "Urgent"
            maint_note = "Critical sensor failure or persistent freeze. Immediate physical servicing required."

        state["priority"] = priority
        state["recommended_maintenance"] = maint_note
        state["last_updated"] = datetime.now()

        # Update station object health_score
        station.health_score = state["composite_health"]
        station.status = "Normal" if priority == "OK" else ("Alert" if is_anomaly else "Maintenance")

        return state

    def get_ranked_maintenance_list(self, stations: List[StationInfo]) -> List[Dict[str, Any]]:
        """Returns stations ranked by urgent maintenance priority."""
        ranked = []
        for s in stations:
            state = self.health_states.get(s.id) or self._init_station_state(s.id)
            ranked.append({
                "station_id": s.id,
                "station_name": s.name,
                "state": s.state,
                "composite_health": state["composite_health"],
                "temperature_health": round(state["temperature_health"], 1),
                "pressure_health": round(state["pressure_health"], 1),
                "humidity_health": round(state["humidity_health"], 1),
                "priority": state["priority"],
                "maintenance_warning": state.get("maintenance_warning"),
                "requires_calibration": state.get("requires_calibration", False),
                "repeated_abnormal_behavior": state.get("repeated_abnormal_behavior", False),
                "affected_sensors": state.get("affected_sensors", []),
                "recommended_action": state["recommended_maintenance"],
                "anomaly_count_24h": state["anomaly_count_24h"]
            })

        ranked.sort(key=lambda x: x["composite_health"])
        return ranked
