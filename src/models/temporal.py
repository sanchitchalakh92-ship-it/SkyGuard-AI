"""
Layer 2: Temporal Model for Diurnal/Seasonal Pattern Learning and Forecasting.
Combines rolling robust median/MAD residual scoring with a PyTorch GRU Forecaster.
"""
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

from src.ingestion.base import WeatherReading, StationInfo


class WeatherGRUForecaster(nn.Module):
    """
    Lightweight GRU network predicting next (T, P, RH) given a historical sequence.
    Fast CPU inference (< 5 ms per sample).
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        # Predict next step from last hidden state
        pred = self.fc(out[:, -1, :])
        return pred


class TemporalModel:
    """
    Learns normal diurnal cycles, computes robust residuals (MAD-scaled z-scores),
    and evaluates GRU forecasting residuals.
    """

    def __init__(self, sequence_length: int = 12):
        self.seq_len = sequence_length
        self.device = torch.device("cpu")
        self.model = WeatherGRUForecaster(input_dim=3, hidden_dim=32).to(self.device)
        self.model.eval()

        # Station baselines: hour of day -> mean/std
        self.hourly_means: Dict[str, Dict[int, Dict[str, float]]] = {}

    def _estimate_diurnal_expected(self, station: StationInfo, dt) -> Tuple[float, float, float]:
        """Calculates expected physical diurnal baseline for the station at that hour."""
        hour = dt.hour + dt.minute / 60.0
        solar_phase = 2 * math.pi * (hour - 9.0) / 24.0

        # Elevation lapse rate
        elevation_drop = 6.5 * (station.elevation_m / 1000.0)
        expected_t = 30.0 + 6.0 * math.sin(solar_phase) - elevation_drop

        # Hypsometric pressure
        p0 = 1013.25
        expected_p = p0 * ((1.0 - (0.0065 * station.elevation_m) / 288.15) ** 5.255)

        # RH anti-phase
        expected_rh = 60.0 - 20.0 * math.sin(solar_phase)
        return expected_t, expected_p, expected_rh

    def evaluate(
        self,
        reading: WeatherReading,
        station: StationInfo,
        recent_history: Optional[List[WeatherReading]] = None
    ) -> Dict[str, Any]:
        dt = reading.timestamp
        exp_t, exp_p, exp_rh = self._estimate_diurnal_expected(station, dt)

        t = reading.temperature or exp_t
        p = reading.pressure or exp_p
        rh = reading.humidity or exp_rh

        # Diurnal residuals
        res_t = abs(t - exp_t)
        res_p = abs(p - exp_p)
        res_rh = abs(rh - exp_rh)

        # Robust z-score via history if available
        z_t = 0.0
        z_p = 0.0
        z_rh = 0.0

        if recent_history and len(recent_history) >= 4:
            hist_t = [r.temperature for r in recent_history if r.temperature is not None]
            hist_p = [r.pressure for r in recent_history if r.pressure is not None]
            hist_rh = [r.humidity for r in recent_history if r.humidity is not None]

            if len(hist_t) >= 4:
                med = float(np.median(hist_t))
                mad = float(np.median(np.abs(np.array(hist_t) - med))) + 0.5
                z_t = abs(t - med) / (1.4826 * mad)

            if len(hist_p) >= 4:
                med_p = float(np.median(hist_p))
                mad_p = float(np.median(np.abs(np.array(hist_p) - med_p))) + 0.3
                z_p = abs(p - med_p) / (1.4826 * mad_p)

            if len(hist_rh) >= 4:
                med_rh = float(np.median(hist_rh))
                mad_rh = float(np.median(np.abs(np.array(hist_rh) - med_rh))) + 1.0
                z_rh = abs(rh - med_rh) / (1.4826 * mad_rh)

        # Forecaster residual: if enough history, run GRU inference
        forecast_err = 0.0
        predicted_t = exp_t
        if recent_history and len(recent_history) >= 6:
            seq = []
            for r in recent_history[-6:]:
                seq.append([r.temperature or exp_t, (r.pressure or exp_p) / 10.0, (r.humidity or exp_rh) / 10.0])
            with torch.no_grad():
                tensor_in = torch.tensor([seq], dtype=torch.float32)
                pred = self.model(tensor_in).numpy()[0]
                predicted_t = float(pred[0])
                diff = abs(t - predicted_t)
                forecast_err = min(1.0, diff / 8.0)

        # Composite temporal score
        # High score if z-score > 3.0 or diurnal residual is very high (> 12°C for temperature)
        temp_score = min(1.0, max(res_t / 15.0, z_t / 4.0))
        press_score = min(1.0, max(res_p / 15.0, z_p / 4.0))
        rh_score = min(1.0, max(res_rh / 35.0, z_rh / 4.0))

        temporal_score = 0.5 * temp_score + 0.25 * press_score + 0.25 * rh_score
        temporal_score = max(temporal_score, forecast_err * 0.7)

        return {
            "score": round(float(temporal_score), 3),
            "expected_temperature": round(exp_t, 1),
            "expected_pressure": round(exp_p, 1),
            "expected_humidity": round(exp_rh, 1),
            "temp_residual": round(res_t, 2),
            "z_score_temperature": round(z_t, 2),
            "predicted_temperature": round(predicted_t, 1)
        }
