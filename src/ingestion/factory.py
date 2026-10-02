"""
Factory function to build the active data source based on environment/configuration.
"""
import os
from src.ingestion.base import DataSource
from src.ingestion.simulator import RealisticSimulator
from src.ingestion.open_meteo import OpenMeteoSource
from src.ingestion.owm import OWMSource


def get_data_source(source_type: str = None) -> DataSource:
    mode = (source_type or os.getenv("DEFAULT_DATA_SOURCE", "simulator")).lower()
    if mode == "open_meteo":
        return OpenMeteoSource()
    elif mode == "owm":
        return OWMSource()
    else:
        return RealisticSimulator()
