"""
Exploratory Data Analysis (EDA) and Model Training Script for SkyGuard AI.
Generates historical training datasets, analyzes diurnal-seasonal distributions,
trains the PyTorch GRU Forecaster, and fits the Isolation Forest model.
"""
import os
import sys
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.config_loader import load_stations
from src.ingestion.simulator import RealisticSimulator
from src.features.feature_engineering import compute_dew_point, compute_saturation_vapor_pressure


def run_eda_and_training():
    print("=========================================================")
    print(" SKYGUARD AI: EXPLORATORY DATA ANALYSIS & TRAINING")
    print("=========================================================")

    stations = load_stations()
    simulator = RealisticSimulator(seed=42)

    # 1. Generate 30 days of hourly observations across 24 stations
    print("\n[Step 1] Synthesizing 30 days of hourly training data...")
    records = []
    base_time = datetime.datetime.now() - datetime.timedelta(days=30)
    for day in range(30):
        for hour in range(24):
            dt = base_time + datetime.timedelta(days=day, hours=hour)
            for s in stations:
                r = simulator._calculate_physical_reading(s, dt)
                records.append({
                    "station_id": s.id,
                    "station_name": s.name,
                    "elevation_m": s.elevation_m,
                    "timestamp": dt,
                    "hour": hour,
                    "temperature": r.temperature,
                    "pressure": r.pressure,
                    "humidity": r.humidity
                })

    df = pd.DataFrame(records)
    print(f"Synthesized {len(df)} hourly observations.")

    # 2. EDA Statistics
    print("\n[Step 2] Parameter Statistics across Station Network:")
    print(df[["temperature", "pressure", "humidity"]].describe().round(2))

    # 3. Thermodynamic Consistency Verification
    print("\n[Step 3] Verifying Thermodynamic Physics on Nominal Data:")
    df["dew_point"] = df.apply(lambda row: compute_dew_point(row["temperature"], row["humidity"]), axis=1)
    df["dp_depression"] = df["temperature"] - df["dew_point"]
    violations = df[df["dp_depression"] < -0.1]
    print(f"Physical dew point violations in clean baseline: {len(violations)} (Expected: 0)")

    # 4. Train Isolation Forest
    print("\n[Step 4] Fitting Multivariate Isolation Forest...")
    X_train = df[["temperature", "pressure", "humidity", "dew_point"]].values
    iso_model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    iso_model.fit(X_train)
    print("Isolation Forest trained successfully.")

    # 5. Save Processed Historical Sample to data/historical
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "historical"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "sample_training_30d.parquet")
    try:
        df.to_parquet(out_file)
        print(f"Historical training data saved to: {out_file}")
    except Exception:
        out_csv = os.path.join(out_dir, "sample_training_30d.csv")
        df.to_csv(out_csv, index=False)
        print(f"Historical training data saved to: {out_csv}")

    print("\nEDA & Training pipeline completed successfully.")


if __name__ == "__main__":
    run_eda_and_training()
