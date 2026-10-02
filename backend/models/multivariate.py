"""
Layer 3: Multivariate Consistency Model.
Combines Isolation Forest with atmospheric physics consistency (Clausius-Clapeyron, barometric MSLP, vapor pressure).
"""
import math
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.ensemble import IsolationForest

from src.ingestion.base import WeatherReading, StationInfo
from src.features.feature_engineering import (
    compute_dew_point,
    compute_actual_vapor_pressure,
    compute_saturation_vapor_pressure,
    compute_sea_level_pressure,
    compute_potential_temperature
)


class MultivariateConsistencyModel:
    """
    Evaluates physical correlation and statistical anomaly between T, P, RH.
    Detects unphysical thermodynamic combinations (e.g. 50°C with 98% RH).
    """

    FEATURE_NAMES = [
        "temperature", "pressure", "humidity",
        "dew_point", "vapor_pressure", "dew_point_depression", "mslp"
    ]

    def __init__(self):
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.03,
            random_state=42
        )
        self.is_fitted = False
        self._fit_default_distribution()

    def _fit_default_distribution(self):
        """Fit Isolation Forest on a diverse synthetic population of physical Indian weather points."""
        synthetic_points = []
        for t in np.linspace(5.0, 45.0, 30):
            for rh in np.linspace(15.0, 95.0, 20):
                for p in np.linspace(850.0, 1020.0, 10):
                    es = compute_saturation_vapor_pressure(t)
                    e = (rh / 100.0) * es
                    td = compute_dew_point(t, rh)
                    if td <= t:
                        mslp = p + 50.0
                        synthetic_points.append([t, p, rh, td, e, t - td, mslp])

        X = np.array(synthetic_points)
        self.iso_forest.fit(X)
        self.is_fitted = True

    def extract_vector(self, reading: WeatherReading, station: StationInfo) -> np.ndarray:
        t = reading.temperature if reading.temperature is not None else 28.0
        p = reading.pressure if reading.pressure is not None else 1005.0
        rh = reading.humidity if reading.humidity is not None else 60.0

        es = compute_saturation_vapor_pressure(t)
        e = compute_actual_vapor_pressure(t, rh)
        td = compute_dew_point(t, rh)
        dep = t - td if td is not None else 0.0
        mslp = compute_sea_level_pressure(p, station.elevation_m if station else 0.0, t)

        return np.array([t, p, rh, td if td is not None else t, e, dep, mslp], dtype=np.float32)

    def evaluate(
        self,
        reading: WeatherReading,
        station: StationInfo,
        recent_history: Optional[List[WeatherReading]] = None,
        spatial_eval: Optional[Dict[str, Any]] = None,
        temporal_eval: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        t = reading.temperature
        p = reading.pressure
        rh = reading.humidity

        if t is None or p is None or rh is None:
            return {
                "score": 1.0,
                "is_anomalous": True,
                "physics_violation": True,
                "reason": "Missing primary parameters",
                "context_aware": {
                    "verdict": "POSSIBLE_SENSOR_FAULT",
                    "classification": "Missing Data",
                    "is_sensor_issue": True,
                    "is_real_weather_event": False,
                    "context_reason": "Missing critical parameters (temperature, pressure, or humidity)."
                }
            }

        physics_penalty = 0.0
        violations = []

        # 1. Clausius-Clapeyron limit: Dew point must not exceed ambient temperature
        td = compute_dew_point(t, rh)
        if td is not None and td > (t + 0.1):
            physics_penalty += 0.85
            violations.append(f"Supersaturation anomaly: Dew Point ({td}°C) > Temp ({t}°C)")

        # 2. Extreme vapor pressure check: 50°C + 95% RH yields unphysical e > 115 hPa at surface
        es = compute_saturation_vapor_pressure(t)
        e = (rh / 100.0) * es
        if e > 65.0:  # Earth surface max vapor pressure rarely exceeds ~50-60 hPa (wet-bulb > 35°C lethal limit)
            physics_penalty += 0.90
            violations.append(f"Unphysical thermodynamic vapor pressure {e:.1f} hPa at {t}°C / {rh}%")

        # 3. Barometric MSLP plausibility: Sea level pressure should fall within [950, 1050] hPa
        mslp = compute_sea_level_pressure(p, station.elevation_m if station else 0.0, t)
        if mslp < 930.0 or mslp > 1060.0:
            physics_penalty += 0.70
            violations.append(f"Elevation-adjusted MSLP {mslp} hPa outside realistic meteorological bounds")

        # 4. Statistical Isolation Forest Score
        feat_vec = self.extract_vector(reading, station)
        raw_score = self.iso_forest.score_samples([feat_vec])[0]
        stat_score = float(np.clip((-raw_score - 0.42) / 0.35, 0.0, 1.0))

        # 5. Context-Aware Anomaly Evaluation (Trends + Historical + Related Sensors + Local Weather)
        context_aware = self.evaluate_context_aware(
            reading=reading,
            station=station,
            recent_history=recent_history,
            spatial_eval=spatial_eval,
            temporal_eval=temporal_eval
        )

        context_boost = 0.35 if context_aware.get("is_sensor_issue") else 0.0
        combined_score = min(1.0, max(stat_score, physics_penalty, context_boost))

        return {
            "score": round(float(combined_score), 3),
            "isolation_forest_score": round(stat_score, 3),
            "physics_violations": violations,
            "feature_vector": feat_vec.tolist(),
            "dew_point": td,
            "mslp": mslp,
            "vapor_pressure": round(e, 2),
            "context_aware": context_aware
        }

    def evaluate_context_aware(
        self,
        reading: WeatherReading,
        station: StationInfo,
        recent_history: Optional[List[WeatherReading]] = None,
        spatial_eval: Optional[Dict[str, Any]] = None,
        temporal_eval: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Context-Aware Anomaly Detection:
        Analyzes readings against recent trends, historical behavior and related sensor measurements.
        For example, an abrupt temperature jump while other parameters remain stable may indicate a sensor issue,
        but should be checked against local weather conditions before classification.
        """
        t = reading.temperature
        p = reading.pressure
        rh = reading.humidity
        solar = reading.solar_radiation
        wind = reading.wind_speed
        rain = reading.rainfall

        # 1. Recent Trends (Rates of Change from last observation)
        delta_t = 0.0
        delta_p = 0.0
        delta_rh = 0.0
        delta_solar = 0.0

        if recent_history and len(recent_history) > 0:
            last_reading = recent_history[-1]
            if t is not None and last_reading.temperature is not None:
                delta_t = round(float(t - last_reading.temperature), 2)
            if p is not None and last_reading.pressure is not None:
                delta_p = round(float(p - last_reading.pressure), 2)
            if rh is not None and last_reading.humidity is not None:
                delta_rh = round(float(rh - last_reading.humidity), 2)
            if solar is not None and last_reading.solar_radiation is not None:
                delta_solar = round(float(solar - last_reading.solar_radiation), 2)

        # 2. Historical Diurnal Behavior
        exp_t = None
        temp_residual = 0.0
        if temporal_eval:
            exp_t = temporal_eval.get("expected_temperature") or temporal_eval.get("spatial_expected_temp")
            temp_residual = temporal_eval.get("temp_residual", 0.0)

        # 3. Related Sensor Measurements Analysis
        # An abrupt temperature jump (e.g. >= 3.0°C in one step or >= 4.5°C vs historical diurnal)
        is_abrupt_temp_jump = abs(delta_t) >= 3.0 or temp_residual >= 4.5

        # Check if other parameters remained stable
        p_stable = abs(delta_p) <= 0.8
        rh_stable = abs(delta_rh) <= 5.0
        solar_stable = abs(delta_solar) <= 40.0
        related_parameters_stable = p_stable and rh_stable and solar_stable

        # 4. Check against Local Weather Conditions before final classification
        cross_stn = spatial_eval.get("cross_station_verification", {}) if spatial_eval else {}
        is_local_weather_event = cross_stn.get("is_real_weather_event", False)
        agreeing_stns = cross_stn.get("agreeing_stations_count", 0)
        nearby_count = cross_stn.get("nearby_stations_count", 0)
        has_severe_storm = (rain is not None and rain > 10.0) and (wind is not None and wind > 18.0)

        # Synthesis & Classification:
        if is_abrupt_temp_jump and related_parameters_stable:
            # Abrupt temperature jump while other parameters remain stable
            # Before classifying, check against local weather conditions:
            if is_local_weather_event or (agreeing_stns >= 2 and agreeing_stns > nearby_count * 0.4) or has_severe_storm:
                context_verdict = "POSSIBLE_REAL_WEATHER_EVENT"
                classification = "Possible Real Weather Event"
                is_sensor_issue = False
                is_real_weather_event = True
                context_reason = (
                    f"Abrupt temperature change ({delta_t:+.1f}°C) observed while local parameters were relatively steady, "
                    f"but cross-station verification confirms {agreeing_stns}/{nearby_count} nearby AWS stations show the same pattern. "
                    f"Local weather conditions validate a possible real weather event (convective gust front/microclimate)."
                )
            else:
                context_verdict = "POSSIBLE_SENSOR_FAULT"
                classification = "Possible Sensor Fault"
                is_sensor_issue = True
                is_real_weather_event = False
                context_reason = (
                    f"Abrupt temperature jump of {delta_t:+.1f}°C detected while related parameters remain stable "
                    f"(ΔP={delta_p:+.1f} hPa, ΔRH={delta_rh:+.1f}%, ΔSolar={delta_solar:+.0f} W/m²). "
                    f"Nearby AWS stations report nominal atmospheric conditions with no regional front. "
                    f"Classifying as a possible sensor issue (suspect reading)."
                )
        elif is_local_weather_event:
            context_verdict = "POSSIBLE_REAL_WEATHER_EVENT"
            classification = "Possible Real Weather Event"
            is_sensor_issue = False
            is_real_weather_event = True
            context_reason = "Multi-station spatial consensus indicates a genuine regional meteorological event."
        else:
            context_verdict = "NOMINAL"
            classification = "Consistent with Context"
            is_sensor_issue = False
            is_real_weather_event = False
            context_reason = "Readings align with recent trends, diurnal history, and related sensor channels."

        return {
            "verdict": context_verdict,
            "classification": classification,
            "is_sensor_issue": is_sensor_issue,
            "is_real_weather_event": is_real_weather_event,
            "is_isolated_shift": (is_abrupt_temp_jump and related_parameters_stable and not is_real_weather_event),
            "is_corroborated_by_weather": is_real_weather_event,
            "is_abrupt_temp_jump": is_abrupt_temp_jump,
            "related_parameters_stable": related_parameters_stable,
            "local_weather_checked": True,
            "local_weather_verdict": cross_stn.get("verdict", "NOMINAL"),
            "context_reason": context_reason,
            "explanation": context_reason,
            "related_parameters": {
                "temperature_jump_isolated": (is_abrupt_temp_jump and related_parameters_stable and not is_real_weather_event),
                "all_related_stable": related_parameters_stable
            },
            "recent_trends": {
                "delta_temperature": delta_t,
                "delta_pressure": delta_p,
                "delta_humidity": delta_rh,
                "delta_solar": delta_solar,
                "history_length": len(recent_history) if recent_history else 0
            },
            "related_measurements_check": {
                "pressure_stable": p_stable,
                "humidity_stable": rh_stable,
                "solar_stable": solar_stable,
                "all_related_stable": related_parameters_stable
            },
            "historical_behavior": {
                "expected_temperature": exp_t,
                "temperature_residual": round(float(temp_residual), 2)
            }
        }

