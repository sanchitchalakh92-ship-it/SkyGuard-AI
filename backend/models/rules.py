"""
Layer 1: Fast Rule-Based Meteorological Quality Control (WMO standard).
Runs O(1) checks per reading for physical ranges, rate of change, persistence, and dew point constraints.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.ingestion.base import WeatherReading, StationInfo
from src.features.feature_engineering import compute_dew_point


class RuleBasedQC:
    """
    WMO-compliant real-time deterministic quality control checks:
    1. Physical Bounds Test
    2. Climatological/Station Range Test
    3. Step / Rate-of-Change Test
    4. Persistence / Frozen Sensor Test
    5. Missing / Sentinel Value Test (-999, NaN)
    6. Dew Point Constraint (Td <= T)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.temp_min = -40.0
        self.temp_max = 60.0
        self.press_min = 500.0
        self.press_max = 1085.0
        self.rh_min = 0.0
        self.rh_max = 100.0

        # Step limits (per observation interval)
        self.max_dt_step = 6.0    # °C
        self.max_dp_step = 4.0    # hPa
        self.max_drh_step = 35.0  # %

        # Persistence
        self.max_frozen_count = 5
        self.frozen_eps = 0.001

    def evaluate(
        self,
        reading: WeatherReading,
        station: StationInfo,
        recent_history: Optional[List[WeatherReading]] = None
    ) -> Dict[str, Any]:
        violations = []
        rule_score = 0.0
        suspected_causes = []

        t = reading.temperature
        p = reading.pressure
        rh = reading.humidity

        # 1. Missing / Sentinel Value Check
        if t is None or p is None or rh is None or np.isnan(t) or np.isnan(p) or np.isnan(rh):
            violations.append("Missing or NaN sensor values")
            suspected_causes.append("Communication Error")
            return {
                "score": 1.0,
                "is_anomalous": True,
                "violations": violations,
                "suspected_causes": suspected_causes
            }

        if t == -999.0 or p == -999.0 or rh == -999.0:
            violations.append("Sentinel missing value (-999) detected")
            suspected_causes.append("Communication Error")
            return {
                "score": 1.0,
                "is_anomalous": True,
                "violations": violations,
                "suspected_causes": suspected_causes
            }

        # 2. Physical Range Check
        if t < self.temp_min or t > self.temp_max:
            violations.append(f"Temperature {t}°C outside physical bounds [{self.temp_min}, {self.temp_max}]")
            suspected_causes.append("Out-of-Range/Data Corruption")
            rule_score = max(rule_score, 0.95)

        if p < self.press_min or p > self.press_max:
            violations.append(f"Pressure {p} hPa outside physical bounds [{self.press_min}, {self.press_max}]")
            suspected_causes.append("Out-of-Range/Data Corruption")
            rule_score = max(rule_score, 0.95)

        if rh < self.rh_min or rh > self.rh_max:
            violations.append(f"Relative humidity {rh}% outside physical bounds [{self.rh_min}, {self.rh_max}]")
            suspected_causes.append("Out-of-Range/Data Corruption")
            rule_score = max(rule_score, 0.95)

        # 3. Station Climatological Normal Range Check
        if station and station.normal_temp_range and len(station.normal_temp_range) == 2:
            c_min, c_max = station.normal_temp_range
            # Allow 4°C margin for genuine records before flagging severe
            if t < (c_min - 4.0) or t >= (c_max + 4.0):
                violations.append(f"Temperature {t}°C exceeds station climatological limit [{c_min}, {c_max}]")
                suspected_causes.append("Spike/Sensor Fault")
                rule_score = max(rule_score, 0.90)

        # 4. Thermodynamic Dew Point Check
        td = compute_dew_point(t, rh)
        if td is not None and not np.isnan(td):
            if td > (t + 0.15):
                violations.append(f"Thermodynamic violation: Dew point ({td}°C) exceeds ambient temperature ({t}°C)")
                suspected_causes.append("Inconsistent Multivariate")
                rule_score = max(rule_score, 0.90)

        # 5. Temporal Rate-of-Change / Step Test
        if recent_history and len(recent_history) > 0:
            last = recent_history[-1]
            if last.temperature is not None:
                diff_t = abs(t - last.temperature)
                if diff_t > self.max_dt_step:
                    violations.append(f"Abrupt temperature step jump: ΔT={diff_t:.1f}°C > {self.max_dt_step}°C")
                    suspected_causes.append("Spike/Sensor Fault")
                    rule_score = max(rule_score, min(1.0, 0.6 + (diff_t / self.max_dt_step) * 0.2))

            if last.pressure is not None:
                diff_p = abs(p - last.pressure)
                if diff_p > self.max_dp_step:
                    violations.append(f"Abrupt pressure step jump: ΔP={diff_p:.1f}hPa > {self.max_dp_step}hPa")
                    suspected_causes.append("Spike/Sensor Fault")
                    rule_score = max(rule_score, 0.70)

            if last.humidity is not None:
                diff_rh = abs(rh - last.humidity)
                if diff_rh > self.max_drh_step:
                    violations.append(f"Abrupt humidity jump: ΔRH={diff_rh:.1f}% > {self.max_drh_step}%")
                    suspected_causes.append("Spike/Sensor Fault")
                    rule_score = max(rule_score, 0.70)

        # 6. Persistence / Frozen Sensor Test
        if recent_history and len(recent_history) >= self.max_frozen_count:
            recent_readings = recent_history[-self.max_frozen_count:] + [reading]
            recent_temps = [r.temperature for r in recent_readings if r.temperature is not None]
            if len(recent_temps) == (self.max_frozen_count + 1):
                std_t = float(np.std(recent_temps))
                if std_t < self.frozen_eps:
                    violations.append(f"Frozen sensor: Temperature stuck at {t}°C for {len(recent_temps)} steps")
                    suspected_causes.append("Frozen Sensor")
                    rule_score = max(rule_score, 0.92)

        is_anom = rule_score >= 0.65 or len(violations) > 0
        return {
            "score": round(rule_score, 3),
            "is_anomalous": is_anom,
            "violations": violations,
            "suspected_causes": list(set(suspected_causes))
        }
