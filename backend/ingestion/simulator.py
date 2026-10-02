"""
Physics-accurate Realistic Automatic Weather Station (AWS) Stream Simulator.
Simulates diurnal cycles, elevation-adjusted pressure/temperature, and atmospheric physics.
"""
import math
import random
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import numpy as np

from src.ingestion.base import DataSource, StationInfo, WeatherReading


class RealisticSimulator(DataSource):
    """
    Simulates high-fidelity AWS observations respecting atmospheric physics:
    - Diurnal solar heating cycle
    - Hypsometric barometric equation for elevation
    - Clausius-Clapeyron T-RH inverse relationship
    - Semi-diurnal atmospheric pressure tides
    - Micro-turbulent sensor noise
    """

    def __init__(self, seed: Optional[int] = 42):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _calculate_physical_reading(self, station: StationInfo, dt: datetime) -> WeatherReading:
        # Solar hour calculation
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        # 1. Temperature Calculation
        # Diurnal peak around 14:30 (2.5 PM), minimum around 05:30 (dawn)
        solar_phase = 2 * math.pi * (hour - 9.0) / 24.0
        diurnal_temp_swing = 6.5 * math.sin(solar_phase)

        # Baseline sea-level temperature based on latitude
        lat_factor = math.cos(math.radians(station.latitude))
        sea_level_base = 28.0 + 4.0 * lat_factor

        # Environmental Lapse Rate: 6.5°C per 1000m
        elevation_temp_drop = 6.5 * (station.elevation_m / 1000.0)
        temp_expected = sea_level_base + diurnal_temp_swing - elevation_temp_drop
        # Micro-fluctuation
        temp_val = round(temp_expected + random.gauss(0, 0.3), 1)

        # 2. Surface Pressure Calculation
        # Barometric formula: P = P0 * (1 - L*h / T0) ^ (g*M / (R*L))
        # Atmospheric thermal tide: semi-diurnal wave peaking at 10:00 and 22:00 (~1.2 hPa)
        tide_phase = 4 * math.pi * (hour - 4.0) / 24.0
        pressure_tide = 1.2 * math.sin(tide_phase)

        p0 = 1013.25 + pressure_tide
        # Standard barometric elevation reduction
        lapse_rate = 0.0065
        t0 = 288.15
        pressure_expected = p0 * ((1.0 - (lapse_rate * station.elevation_m) / t0) ** 5.255)
        pressure_val = round(pressure_expected + random.gauss(0, 0.4), 1)

        # 3. Relative Humidity Calculation
        # Inversely correlated with temperature diurnal swing
        # Minimum RH at peak temperature, maximum at dawn
        rh_swing = -22.0 * math.sin(solar_phase)
        # Coastal stations (low elevation + near ocean like Mumbai/Goa/Chennai) have higher baseline RH
        is_coastal = station.elevation_m < 30 and (station.longitude < 73.5 or station.longitude > 80.0)
        base_rh = 75.0 if is_coastal else 60.0
        rh_expected = base_rh + rh_swing
        rh_val = round(max(10.0, min(98.0, rh_expected + random.gauss(0, 1.2))), 1)

        # 4. Auxiliary Metrics (Wind, Rain, Solar Radiation)
        # Wind peaks slightly in afternoon
        wind_base = 10.0 + 4.0 * max(0.0, math.sin(solar_phase))
        wind_val = round(max(0.5, wind_base + random.gauss(0, 1.5)), 1)

        # Solar radiation: 0 at night, parabolic peak at 12:30 PM up to ~850 W/m²
        if 6.0 <= hour <= 18.5:
            sun_angle = math.sin(math.pi * (hour - 6.0) / 12.5)
            solar_val = round(max(0.0, 850.0 * sun_angle + random.gauss(0, 20.0)), 1)
        else:
            solar_val = 0.0

        rain_val = 0.0

        return WeatherReading(
            station_id=station.id,
            timestamp=dt,
            temperature=temp_val,
            pressure=pressure_val,
            humidity=rh_val,
            wind_speed=wind_val,
            rainfall=rain_val,
            solar_radiation=solar_val,
            is_imputed=False,
            raw_payload={
                "source": "RealisticSimulator",
                "station_name": station.name,
                "elevation_m": station.elevation_m
            }
        )

    async def fetch_current(self, station: StationInfo) -> WeatherReading:
        now = datetime.now()
        return self._calculate_physical_reading(station, now)

    async def fetch_current_all(self, stations: List[StationInfo]) -> List[WeatherReading]:
        now = datetime.now()
        return [self._calculate_physical_reading(s, now) for s in stations]

    async def fetch_history(self, station: StationInfo, days: int = 7) -> List[WeatherReading]:
        now = datetime.now()
        readings = []
        total_hours = days * 24
        for i in range(total_hours, 0, -1):
            dt = now - timedelta(hours=i)
            readings.append(self._calculate_physical_reading(station, dt))
        return readings
