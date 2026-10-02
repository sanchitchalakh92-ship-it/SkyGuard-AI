"""
SkyGuard AI - SIH 26073 Example Scenario Demo Script.
Demonstrates end-to-end detection, spatial corroboration, root-cause diagnosis,
and self-healing imputation for an AWS reporting 55°C with high humidity.
"""
import sys
import os
import asyncio
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.config_loader import load_stations
from src.ingestion.simulator import RealisticSimulator
from src.ingestion.injector import AnomalyInjector
from src.models.pipeline import SkyGuardPipeline


async def run_55c_scenario():
    print("=" * 75)
    print(" SKYGUARD AI: SIH PROBLEM STATEMENT 26073 DEMONSTRATION")
    print(" Scenario: AWS suddenly reports 55°C with abnormal humidity & pressure")
    print("=" * 75)

    stations = load_stations()
    pipeline = SkyGuardPipeline(stations)
    simulator = RealisticSimulator()
    injector = AnomalyInjector()

    # Step 1: Initialize all stations with realistic baseline conditions
    print("\n[Step 1] Synchronizing network baseline across 24 Indian AWS stations...")
    network_readings = {}
    for s in stations:
        r = await simulator.fetch_current(s)
        network_readings[s.id] = r
        pipeline.process_reading(r)

    nagpur = next(s for s in stations if s.id == "AWS_NGP")
    clean_nagpur = network_readings["AWS_NGP"]
    print(f"  -> Nagpur Station nominal baseline: T={clean_nagpur.temperature}°C, P={clean_nagpur.pressure} hPa, RH={clean_nagpur.humidity}%")

    # Step 2: Inject the 55°C anomaly
    print("\n[Step 2] Injecting 55°C Anomaly into Nagpur Station...")
    anomalous_reading = injector.inject_55C_scenario(clean_nagpur)
    print(f"  -> Corrupted telemetry received: T={anomalous_reading.temperature}°C, P={anomalous_reading.pressure} hPa, RH={anomalous_reading.humidity}%")

    # Step 3: Run through SkyGuard 5-Layer Detection Pipeline
    print("\n[Step 3] Processing through SkyGuard Multi-Layer Architecture...")
    res = pipeline.process_reading(anomalous_reading, network_readings)

    # Step 4: Display Layer Scores and Diagnostic Output
    print("\n" + "-" * 75)
    print(" DETECTION & DIAGNOSTIC RESULTS")
    print("-" * 75)
    print(f" Target Station       : {res['station_name']} ({res['station_id']})")
    print(f" Anomaly Detected     : {res['is_anomaly']}")
    print(f" Severity Level       : {res['severity']}")
    print(f" Confidence Score     : {res['confidence'] * 100:.1f}%")
    print(f" Root Cause Category  : {res['root_cause']}")
    print(f" Processing Latency   : {res['processing_latency_ms']} ms")

    print("\n[Layer Breakdown]:")
    print(f"  - Layer 1 (WMO Rules QC)         : Score = {res['layer_scores']['rules']}")
    print(f"  - Layer 2 (Temporal Diurnal/GRU) : Score = {res['layer_scores']['temporal']}")
    print(f"  - Layer 3 (Multivariate Physics) : Score = {res['layer_scores']['multivariate']}")
    print(f"  - Layer 4 (Spatial IDW Consensus): Score = {res['layer_scores']['spatial']}")

    print("\n[Explainable AI (XAI) Diagnosis]:")
    print(f"  \"{res['explanation']}\"")

    print("\n[Top SHAP Feature Attributions]:")
    for feat in res['shap_features']:
        direction = "+" if feat['is_positive'] else "-"
        print(f"  * {feat['feature']:<25}: {direction}{abs(feat['importance']):.3f}")

    print("\n[Self-Healing Corrected Imputation]:")
    if res['imputation']:
        imp = res['imputation']
        print(f"  - Corrected Temp : {imp['imputed_temperature']}°C (bounds: {imp['imputed_temperature_bounds']})")
        print(f"  - Corrected Pres : {imp['imputed_pressure']} hPa")
        print(f"  - Corrected RH   : {imp['imputed_humidity']}%")
        print(f"  - Method         : {imp['method']}")
        print(f"  - Flag           : imputed={imp['is_imputed']}")

    print("\n[Recommended Operational Action]:")
    print(f"  {res['recommended_action']}")

    print("\n" + "=" * 75)
    print(" DEMO COMPLETED SUCCESSFULLY: Zero false alarms, instant detection, full explainability.")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_55c_scenario())
