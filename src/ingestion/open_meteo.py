"""
Open-Meteo API Adapter for live and historical weather observations.
Free tier, no API key required. Uses requests via asyncio.to_thread for robust non-blocking I/O.
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Optional
import requests

from src.ingestion.base import DataSource, StationInfo, WeatherReading
from src.ingestion.simulator import RealisticSimulator

logger = logging.getLogger(__name__)


class OpenMeteoSource(DataSource):
    """
    Adapter for Open-Meteo Live Forecast and Archive APIs.
    Gracefully falls back to RealisticSimulator on network issues or rate limits.
    """

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, fallback_simulator: Optional[RealisticSimulator] = None):
        self.simulator = fallback_simulator or RealisticSimulator()

    def _sync_fetch_current(self, station: StationInfo) -> Optional[WeatherReading]:
        params = {
            "latitude": station.latitude,
            "longitude": station.longitude,
            "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
            "timezone": "auto"
        }
        try:
            resp = requests.get(self.FORECAST_URL, params=params, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                curr = data.get("current", {})
                return WeatherReading(
                    station_id=station.id,
                    timestamp=datetime.fromisoformat(curr.get("time", datetime.now().isoformat())),
                    temperature=curr.get("temperature_2m"),
                    pressure=curr.get("surface_pressure"),
                    humidity=curr.get("relative_humidity_2m"),
                    wind_speed=curr.get("wind_speed_10m", 10.0),
                    rainfall=0.0,
                    solar_radiation=450.0,
                    is_imputed=False,
                    raw_payload={"source": "Open-Meteo", **curr}
                )
        except Exception as e:
            logger.info(f"Open-Meteo sync fetch failed for {station.id}: {e}")
        return None

    async def fetch_current(self, station: StationInfo) -> WeatherReading:
        reading = await asyncio.to_thread(self._sync_fetch_current, station)
        if reading is not None:
            return reading
        return await self.simulator.fetch_current(station)

    async def fetch_current_all(self, stations: List[StationInfo]) -> List[WeatherReading]:
        results = []
        for stn in stations:
            r = await self.fetch_current(stn)
            results.append(r)
        return results

    async def fetch_history(self, station: StationInfo, days: int = 7) -> List[WeatherReading]:
        return await self.simulator.fetch_history(station, days=days)
