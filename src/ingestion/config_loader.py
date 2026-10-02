"""
Config loader for SkyGuard AI.
Loads YAML configuration and builds StationInfo list and threshold lookups.
"""
import os
import yaml
from typing import Dict, List, Any
from pathlib import Path
from src.ingestion.base import StationInfo

_CONFIG_CACHE: Dict[str, Any] = {}

def get_config_path() -> Path:
    env_path = os.getenv("SKYGUARD_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    # Default relative to project root
    base = Path(__file__).resolve().parent.parent.parent
    return base / "config" / "config.yaml"

def load_config(reload: bool = False) -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE and not reload:
        return _CONFIG_CACHE

    cfg_path = get_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f)

    return _CONFIG_CACHE

def load_stations() -> List[StationInfo]:
    cfg = load_config()
    stations_data = cfg.get("stations", [])
    stations = []
    for s in stations_data:
        station = StationInfo(
            id=s["id"],
            name=s["name"],
            state=s["state"],
            latitude=float(s["latitude"]),
            longitude=float(s["longitude"]),
            elevation_m=float(s["elevation_m"]),
            normal_temp_range=s.get("normal_temp_range", [10.0, 45.0]),
            status="Normal",
            health_score=100.0
        )
        stations.append(station)
    return stations

def get_station_by_id(station_id: str) -> StationInfo:
    stations = load_stations()
    for s in stations:
        if s.id == station_id:
            return s
    raise KeyError(f"Station {station_id} not found in configuration.")
