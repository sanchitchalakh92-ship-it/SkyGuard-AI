"""
Base models and abstract interface for SkyGuard AI data sources.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class WeatherReading(BaseModel):
    """Core AWS observation reading containing primary T, P, RH parameters."""
    station_id: str
    timestamp: datetime
    temperature: Optional[float] = Field(None, description="Air temperature in °C")
    pressure: Optional[float] = Field(None, description="Atmospheric surface pressure in hPa")
    humidity: Optional[float] = Field(None, description="Relative humidity in %")
    wind_speed: Optional[float] = Field(12.0, description="Wind speed in km/h (auxiliary)")
    rainfall: Optional[float] = Field(0.0, description="Hourly rainfall in mm (auxiliary)")
    solar_radiation: Optional[float] = Field(500.0, description="Solar radiation in W/m² (auxiliary)")
    is_imputed: bool = False
    imputed_fields: List[str] = Field(default_factory=list)
    raw_payload: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StationInfo(BaseModel):
    """Metadata for an Automatic Weather Station (AWS)."""
    id: str
    name: str
    state: str
    latitude: float
    longitude: float
    elevation_m: float
    normal_temp_range: List[float] = Field(default_factory=lambda: [10.0, 45.0])
    status: str = "Normal"  # Normal, Alert, Maintenance, Offline
    health_score: float = 100.0


class DataSource(ABC):
    """Abstract interface for pluggable weather data sources."""

    @abstractmethod
    async def fetch_current(self, station: StationInfo) -> WeatherReading:
        """Fetch current weather reading for a given station."""
        pass

    @abstractmethod
    async def fetch_current_all(self, stations: List[StationInfo]) -> List[WeatherReading]:
        """Fetch current readings for all stations."""
        pass

    @abstractmethod
    async def fetch_history(self, station: StationInfo, days: int = 7) -> List[WeatherReading]:
        """Fetch historical readings for model training or baseline analysis."""
        pass
