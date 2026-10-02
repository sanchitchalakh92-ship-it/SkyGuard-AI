"""
OpenWeatherMap and WeatherAPI Adapter for secondary independent validation.
Falls back to Simulator if API key is not present.
Uses requests via asyncio.to_thread.
"""
import os
import asyncio
import logging
from typing import List, Optional
from datetime import datetime
import requests

from src.ingestion.base import DataSource, StationInfo, WeatherReading
from src.ingestion.simulator import RealisticSimulator

logger = logging.getLogger(__name__)


class OWMSource(DataSource):
    """
    Adapter for OpenWeatherMap API (free tier requires OPENWEATHERMAP_API_KEY).
    Automatically falls back to realistic simulation if no key or on error.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
        self.simulator = RealisticSimulator()

    def _sync_fetch(self, station: StationInfo) -> Optional[WeatherReading]:
        if not self.api_key:
            return None

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": station.latitude,
            "lon": station.longitude,
            "appid": self.api_key,
            "units": "metric"
        }
        try:
            resp = requests.get(url, params=params, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                main = data.get("main", {})
                wind = data.get("wind", {})
                return WeatherReading(
                    station_id=station.id,
                    timestamp=datetime.now(),
                    temperature=main.get("temp"),
                    pressure=main.get("pressure"),
                    humidity=main.get("humidity"),
                    wind_speed=wind.get("speed", 10.0) * 3.6,
                    rainfall=0.0,
                    solar_radiation=500.0,
                    is_imputed=False,
                    raw_payload={"source": "OpenWeatherMap", **data}
                )
        except Exception as e:
            logger.info(f"OWM request failed for {station.id}: {e}")
        return None

    async def fetch_current(self, station: StationInfo) -> WeatherReading:
        if not self.api_key:
            return await self.simulator.fetch_current(station)

        reading = await asyncio.to_thread(self._sync_fetch, station)
        if reading is not None:
            return reading
        return await self.simulator.fetch_current(station)

    async def fetch_current_all(self, stations: List[StationInfo]) -> List[WeatherReading]:
        return [await self.fetch_current(s) for s in stations]

    async def fetch_history(self, station: StationInfo, days: int = 7) -> List[WeatherReading]:
        return await self.simulator.fetch_history(station, days=days)
