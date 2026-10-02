"""
Meteorological Feature Engineering for SkyGuard AI.
Computes thermodynamics and temporal features from strictly T, P, RH observations.
"""
import math
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_saturation_vapor_pressure(temperature_c: float) -> float:
    """
    Magnus-Tetens formula for saturation vapor pressure over water (in hPa).
    Valid for -40°C to +50°C.
    """
    if temperature_c is None or np.isnan(temperature_c):
        return np.nan
    return 6.112 * math.exp((17.67 * temperature_c) / (temperature_c + 243.5))


def compute_actual_vapor_pressure(temperature_c: float, relative_humidity: float) -> float:
    """Computes actual vapor pressure (e in hPa)."""
    if temperature_c is None or relative_humidity is None:
        return np.nan
    if np.isnan(temperature_c) or np.isnan(relative_humidity):
        return np.nan
    es = compute_saturation_vapor_pressure(temperature_c)
    return max(0.0, (relative_humidity / 100.0) * es)


def compute_dew_point(temperature_c: float, relative_humidity: float) -> float:
    """
    Calculates dew point temperature in °C using Magnus approximation.
    Physical rule: Dew point MUST be <= Temperature.
    """
    if temperature_c is None or relative_humidity is None:
        return np.nan
    if np.isnan(temperature_c) or np.isnan(relative_humidity) or relative_humidity <= 0:
        return np.nan

    rh = min(100.0, max(0.01, relative_humidity))
    a = 17.67
    b = 243.5
    alpha = ((a * temperature_c) / (b + temperature_c)) + math.log(rh / 100.0)
    td = (b * alpha) / (a - alpha)
    return round(td, 2)


def compute_sea_level_pressure(surface_pressure_hpa: float, elevation_m: float, temperature_c: float) -> float:
    """
    Reduces station surface pressure to Mean Sea Level Pressure (MSLP) in hPa
    using the barometric formula.
    """
    if surface_pressure_hpa is None or np.isnan(surface_pressure_hpa):
        return np.nan
    temp_k = (temperature_c if temperature_c is not None and not np.isnan(temperature_c) else 25.0) + 273.15
    # Hypsometric reduction: P0 = P * exp( (g0 * M * h) / (R * T_avg) )
    # With g0*M/R ~ 0.034163 K/m
    factor = math.exp((0.034163 * elevation_m) / temp_k)
    return round(surface_pressure_hpa * factor, 1)


def compute_potential_temperature(temperature_c: float, surface_pressure_hpa: float) -> float:
    """Computes Poisson potential temperature theta in °C (conserved in dry adiabatic motions)."""
    if temperature_c is None or surface_pressure_hpa is None or surface_pressure_hpa <= 0:
        return np.nan
    temp_k = temperature_c + 273.15
    theta_k = temp_k * ((1000.0 / surface_pressure_hpa) ** 0.286)
    return round(theta_k - 273.15, 2)


def extract_features_record(
    reading: Dict[str, any],
    recent_history: Optional[List[Dict[str, any]]] = None,
    elevation_m: float = 0.0
) -> Dict[str, float]:
    """
    Extracts complete thermodynamic, rate-of-change, and residual features for one observation.
    """
    t = reading.get("temperature")
    p = reading.get("pressure")
    rh = reading.get("humidity")

    features = {
        "temperature": t if t is not None else np.nan,
        "pressure": p if p is not None else np.nan,
        "humidity": rh if rh is not None else np.nan,
    }

    # Thermodynamics
    es = compute_saturation_vapor_pressure(t) if t is not None else np.nan
    e = compute_actual_vapor_pressure(t, rh) if (t is not None and rh is not None) else np.nan
    td = compute_dew_point(t, rh) if (t is not None and rh is not None) else np.nan
    mslp = compute_sea_level_pressure(p, elevation_m, t) if (p is not None and t is not None) else np.nan
    theta = compute_potential_temperature(t, p) if (t is not None and p is not None) else np.nan

    features["sat_vapor_pressure"] = es
    features["vapor_pressure"] = e
    features["dew_point"] = td
    features["dew_point_depression"] = (t - td) if (t is not None and td is not None and not np.isnan(td)) else np.nan
    features["mslp"] = mslp
    features["potential_temperature"] = theta

    # Temporal rate of change derivatives (dT/dt, dP/dt, dRH/dt)
    dt_val = np.nan
    dp_val = np.nan
    drh_val = np.nan

    if recent_history and len(recent_history) > 0:
        prev = recent_history[-1]
        prev_t = prev.get("temperature")
        prev_p = prev.get("pressure")
        prev_rh = prev.get("humidity")

        if t is not None and prev_t is not None:
            dt_val = round(t - prev_t, 2)
        if p is not None and prev_p is not None:
            dp_val = round(p - prev_p, 2)
        if rh is not None and prev_rh is not None:
            drh_val = round(rh - prev_rh, 2)

    features["dt_step"] = dt_val
    features["dp_step"] = dp_val
    features["drh_step"] = drh_val

    # Rolling window stats (mean, std, median absolute deviation) if history available
    if recent_history and len(recent_history) >= 3:
        hist_temps = [r.get("temperature") for r in recent_history if r.get("temperature") is not None]
        if len(hist_temps) >= 3 and t is not None:
            median_t = float(np.median(hist_temps))
            mad_t = float(np.median(np.abs(np.array(hist_temps) - median_t))) + 1e-4
            features["t_robust_zscore"] = round((t - median_t) / (1.4826 * mad_t), 2)
        else:
            features["t_robust_zscore"] = 0.0
    else:
        features["t_robust_zscore"] = 0.0

    return features
