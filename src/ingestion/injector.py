"""
Anomaly Injector for SkyGuard AI.
Generates controlled synthetic anomalies and genuine extreme meteorological events
for evaluation, validation benchmarks, and interactive live demonstration scenarios.
"""
import copy
import random
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.ingestion.base import WeatherReading


class AnomalyType:
    SPIKE = "Spike/Sensor Fault"
    FROZEN = "Frozen Sensor"
    DRIFT = "Calibration Drift"
    DROPOUT = "Communication Error"
    NOISE_BURST = "Power Fluctuation Pattern"
    OUT_OF_RANGE = "Out-of-Range/Data Corruption"
    INCONSISTENT_MULTIVARIATE = "Inconsistent Multivariate"
    SPATIAL_OUTLIER = "Spatial Outlier"
    GENUINE_METEOROLOGICAL_EVENT = "Genuine Meteorological Event"


class AnomalyInjector:
    """
    Injects synthetic anomalies into real or simulated weather readings while
    retaining ground-truth labels for precision/recall and benchmark evaluation.
    """

    def __init__(self):
        self.frozen_cache: Dict[str, WeatherReading] = {}
        self.drift_accumulators: Dict[str, float] = {}

    def inject_55C_scenario(self, reading: WeatherReading) -> WeatherReading:
        """
        Specific SIH 26073 Example Scenario:
        Station suddenly reports 55°C with high humidity (94%) and abnormal pressure fluctuations,
        while neighbors remain normal.
        """
        injected = copy.deepcopy(reading)
        injected.temperature = 55.0
        injected.humidity = 94.0
        injected.pressure = round((reading.pressure or 1008.0) - 18.5, 1)
        injected.raw_payload = {
            **(injected.raw_payload or {}),
            "injected": True,
            "ground_truth_label": AnomalyType.SPIKE,
            "ground_truth_is_anomaly": True,
            "scenario": "demo_55C_scenario"
        }
        return injected

    def inject_spike(self, reading: WeatherReading, parameter: str = "temperature", magnitude: float = 25.0) -> WeatherReading:
        injected = copy.deepcopy(reading)
        current_val = getattr(injected, parameter) or 25.0
        setattr(injected, parameter, round(current_val + magnitude, 2))
        injected.raw_payload = {
            **(injected.raw_payload or {}),
            "injected": True,
            "ground_truth_label": AnomalyType.SPIKE,
            "ground_truth_is_anomaly": True,
            "injected_param": parameter
        }
        return injected

    def inject_frozen(self, reading: WeatherReading, station_id: str, fixed_reading: Optional[WeatherReading] = None) -> WeatherReading:
        injected = copy.deepcopy(reading)
        if station_id not in self.frozen_cache:
            self.frozen_cache[station_id] = fixed_reading or reading

        cached = self.frozen_cache[station_id]
        injected.temperature = cached.temperature
        injected.pressure = cached.pressure
        injected.humidity = cached.humidity
        injected.raw_payload = {
            **(injected.raw_payload or {}),
            "injected": True,
            "ground_truth_label": AnomalyType.FROZEN,
            "ground_truth_is_anomaly": True
        }
        return injected

    def clear_frozen(self, station_id: str):
        if station_id in self.frozen_cache:
            del self.frozen_cache[station_id]

    def inject_drift(self, reading: WeatherReading, station_id: str, parameter: str = "temperature", rate_per_step: float = 0.5) -> WeatherReading:
        injected = copy.deepcopy(reading)
        accum = self.drift_accumulators.get(station_id, 0.0) + rate_per_step
        self.drift_accumulators[station_id] = accum

        current_val = getattr(injected, parameter) or 25.0
        setattr(injected, parameter, round(current_val + accum, 2))
        injected.raw_payload = {
            **(injected.raw_payload or {}),
            "injected": True,
            "ground_truth_label": AnomalyType.DRIFT,
            "ground_truth_is_anomaly": True,
            "drift_accumulated": accum
        }
        return injected

    def inject_dropout(self, reading: WeatherReading, mode: str = "sentinel") -> WeatherReading:
        """
        mode: 'sentinel' (-999.0) or 'nan' (None)
        """
        injected = copy.deepcopy(reading)
        if mode == "sentinel":
            injected.temperature = -999.0
            injected.pressure = -999.0
            injected.humidity = -999.0
        else:
            injected.temperature = None
            injected.humidity = None

        injected.raw_payload = {
            **(injected.raw_payload or {}),
            "injected": True,
            "ground_truth_label": AnomalyType.DROPOUT,
            "ground_truth_is_anomaly": True
        }
        return injected

    def inject_out_of_range(self, reading: WeatherReading) -> WeatherReading:
        injected = copy.deepcopy(reading)
        # Humidity > 100% or physical impossibility
        choice = random.choice(["rh_overflow", "temp_extreme", "rh_negative"])
        if choice == "rh_overflow":
            injected.humidity = 125.0
        elif choice == "temp_extreme":
            injected.temperature = 74.5
        else:
            injected.humidity = -15.0

        injected.raw_payload = {
            **(injected.raw_payload or {}),
            "injected": True,
            "ground_truth_label": AnomalyType.OUT_OF_RANGE,
            "ground_truth_is_anomaly": True
        }
        return injected

    def inject_multivariate_inconsistency(self, reading: WeatherReading) -> WeatherReading:
        """
        Injects a physical contradiction:
        50°C temperature combined with 98% relative humidity at inland surface pressure,
        which would require impossible atmospheric vapor pressure.
        """
        injected = copy.deepcopy(reading)
        injected.temperature = 49.5
        injected.humidity = 97.0
        injected.raw_payload = {
            **(injected.raw_payload or {}),
            "injected": True,
            "ground_truth_label": AnomalyType.INCONSISTENT_MULTIVARIATE,
            "ground_truth_is_anomaly": True
        }
        return injected

    def inject_genuine_event(self, reading: WeatherReading, event_type: str = "thunderstorm") -> WeatherReading:
        """
        GENUINE METEOROLOGICAL EVENT (Negative class - should NOT be flagged as anomaly!)
        - Thunderstorm outflow: rapid temp drop (-4°C), sharp humidity rise (+25%), pressure surge (+2 hPa)
        """
        injected = copy.deepcopy(reading)
        if event_type == "thunderstorm":
            injected.temperature = round((reading.temperature or 30.0) - 5.5, 1)
            injected.humidity = round(min(98.0, (reading.humidity or 55.0) + 30.0), 1)
            injected.pressure = round((reading.pressure or 1005.0) + 2.5, 1)
            injected.wind_speed = 38.0
            injected.rainfall = 14.2
        elif event_type == "heatwave":
            injected.temperature = round((reading.temperature or 32.0) + 6.0, 1)
            injected.humidity = round(max(15.0, (reading.humidity or 40.0) - 15.0), 1)

        injected.raw_payload = {
            **(injected.raw_payload or {}),
            "injected": True,
            "ground_truth_label": AnomalyType.GENUINE_METEOROLOGICAL_EVENT,
            "ground_truth_is_anomaly": False,
            "genuine_event": event_type
        }
        return injected
