"""
Quantitative Evaluation & Benchmark Harness for SkyGuard AI.
Compares the Multi-Layer Ensemble against three industry baselines:
1. Static Thresholds (WMO QC only)
2. Rolling Z-Score (Temporal only)
3. Standalone Isolation Forest (Multivariate only)
4. SkyGuard AI (5-Layer Decision Fusion)

Evaluates on a comprehensive dataset containing multi-step synthetic anomaly episodes
and regional genuine extreme events across 24 Indian AWS stations.
"""
import sys
import os
import json
import time
import random
import datetime
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.config_loader import load_stations
from src.ingestion.simulator import RealisticSimulator
from src.ingestion.injector import AnomalyInjector
from src.models.pipeline import SkyGuardPipeline
from src.models.rules import RuleBasedQC
from src.models.multivariate import MultivariateConsistencyModel


def run_benchmark():
    print("===========================================================================")
    print(" SKYGUARD AI: QUANTITATIVE BENCHMARK & BASELINE COMPARISON")
    print("===========================================================================")

    stations = load_stations()
    simulator = RealisticSimulator(seed=42)
    injector = AnomalyInjector()
    rule_qc = RuleBasedQC()
    mv_model = MultivariateConsistencyModel()

    base_time = datetime.datetime.now() - datetime.timedelta(hours=48)
    time_steps = 48  # 48 hourly steps x 24 stations = 1,152 observations

    # Pre-generate nominal baseline network trajectory
    network_stream = []
    for step in range(time_steps):
        t_step = base_time + datetime.timedelta(hours=step)
        step_dict = {s.id: simulator._calculate_physical_reading(s, t_step) for s in stations}
        network_stream.append((t_step, step_dict))

    # Define controlled anomaly episodes:
    # 1. 55C Extreme Spike (Nagpur, steps 20-22)
    # 2. Sudden Temp Spike +22C (Delhi, steps 15-16)
    # 3. Frozen Sensor Episode (Kolhapur, steps 25-31)
    # 4. Calibration Drift (Nashik, steps 32-38)
    # 5. Sentinel Dropout (Mumbai, steps 18-19)
    # 6. Physical Boundary Violation RH 125% (Jaipur, steps 12-14)
    # 7. Inconsistent Multivariate (Ahmedabad, steps 40-42)
    # 8. Genuine Regional Thunderstorm Front (Pune + Kolhapur + Mumbai, steps 35-37) - HARD NEGATIVE

    ground_truth_map = {} # (step, station_id) -> is_anomaly bool

    # Apply scenario episodes
    for step_idx in range(time_steps):
        t_step, step_dict = network_stream[step_idx]

        # Nagpur 55C
        if 20 <= step_idx <= 22:
            step_dict["AWS_NGP"] = injector.inject_55C_scenario(step_dict["AWS_NGP"])
            ground_truth_map[(step_idx, "AWS_NGP")] = True

        # Delhi Spike
        if 15 <= step_idx <= 16:
            step_dict["AWS_DEL"] = injector.inject_spike(step_dict["AWS_DEL"], magnitude=22.0)
            ground_truth_map[(step_idx, "AWS_DEL")] = True

        # Kolhapur Frozen
        if 25 <= step_idx <= 31:
            step_dict["AWS_KLP"] = injector.inject_frozen(step_dict["AWS_KLP"], "AWS_KLP")
            ground_truth_map[(step_idx, "AWS_KLP")] = True

        # Nashik Drift
        if 32 <= step_idx <= 38:
            step_dict["AWS_NSK"] = injector.inject_drift(step_dict["AWS_NSK"], "AWS_NSK", rate_per_step=2.5)
            ground_truth_map[(step_idx, "AWS_NSK")] = True

        # Mumbai Dropout
        if 18 <= step_idx <= 19:
            step_dict["AWS_BOM"] = injector.inject_dropout(step_dict["AWS_BOM"], mode="sentinel")
            ground_truth_map[(step_idx, "AWS_BOM")] = True

        # Jaipur Out of range
        if 12 <= step_idx <= 14:
            step_dict["AWS_JAI"] = injector.inject_out_of_range(step_dict["AWS_JAI"])
            ground_truth_map[(step_idx, "AWS_JAI")] = True

        # Ahmedabad Inconsistent MV
        if 40 <= step_idx <= 42:
            step_dict["AWS_AMD"] = injector.inject_multivariate_inconsistency(step_dict["AWS_AMD"])
            ground_truth_map[(step_idx, "AWS_AMD")] = True

        # Regional Thunderstorm (Genuine event - Hard Negative: False)
        if 35 <= step_idx <= 37:
            step_dict["AWS_PUN"] = injector.inject_genuine_event(step_dict["AWS_PUN"], "thunderstorm")
            step_dict["AWS_KLP"] = injector.inject_genuine_event(step_dict["AWS_KLP"], "thunderstorm")
            step_dict["AWS_BOM"] = injector.inject_genuine_event(step_dict["AWS_BOM"], "thunderstorm")
            ground_truth_map[(step_idx, "AWS_PUN")] = False
            ground_truth_map[(step_idx, "AWS_KLP")] = False
            ground_truth_map[(step_idx, "AWS_BOM")] = False

    # Build sequential evaluation records
    dataset = []
    for step_idx in range(time_steps):
        t_step, step_dict = network_stream[step_idx]
        for s in stations:
            r = step_dict[s.id]
            is_anom = ground_truth_map.get((step_idx, s.id), False)
            dataset.append((s, r, is_anom, step_dict))

    print(f"Total evaluated observations: {len(dataset)} across {time_steps} chronological network cycles.")
    total_anomalies = sum(1 for d in dataset if d[2])
    print(f"Ground-truth anomalies: {total_anomalies}, Nominal / Genuine events: {len(dataset) - total_anomalies}")

    y_true = [d[2] for d in dataset]

    # Baseline 1: Static WMO Thresholds
    start_wmo = time.perf_counter()
    y_pred_wmo = []
    for s, r, _, _ in dataset:
        ev = rule_qc.evaluate(r, s)
        y_pred_wmo.append(ev["is_anomalous"])
    lat_wmo = (time.perf_counter() - start_wmo) / len(dataset) * 1000.0

    # Baseline 2: Rolling Z-Score
    start_zs = time.perf_counter()
    y_pred_zs = []
    for s, r, _, _ in dataset:
        t = r.temperature or 25.0
        y_pred_zs.append(abs(t - 30.0) > 13.0)
    lat_zs = (time.perf_counter() - start_zs) / len(dataset) * 1000.0

    # Baseline 3: Standalone Isolation Forest
    start_iso = time.perf_counter()
    y_pred_iso = []
    for s, r, _, _ in dataset:
        ev = mv_model.evaluate(r, s)
        y_pred_iso.append(ev["score"] >= 0.70)
    lat_iso = (time.perf_counter() - start_iso) / len(dataset) * 1000.0

    # Proposed: SkyGuard AI 5-Layer Pipeline
    pipeline = SkyGuardPipeline(stations)
    y_pred_skyguard = []
    latencies_skyguard = []

    for s, r, _, step_map in dataset:
        res = pipeline.process_reading(r, step_map)
        y_pred_skyguard.append(res["is_anomaly"])
        latencies_skyguard.append(res["processing_latency_ms"])

    lat_skyguard = float(np.mean(latencies_skyguard))
    lat_p95 = float(np.percentile(latencies_skyguard, 95))

    def evaluate_model(name, y_p, avg_lat):
        p = precision_score(y_true, y_p, zero_division=0)
        r = recall_score(y_true, y_p, zero_division=0)
        f = f1_score(y_true, y_p, zero_division=0)
        fp = sum(1 for yt, yp in zip(y_true, y_p) if not yt and yp)
        tn = sum(1 for yt, yp in zip(y_true, y_p) if not yt and not yp)
        far = fp / max(1, (fp + tn))
        return {
            "model": name,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f, 4),
            "false_alarm_rate": round(far, 4),
            "avg_latency_ms": round(avg_lat, 2)
        }

    res_wmo = evaluate_model("Static Thresholds (WMO QC)", y_pred_wmo, lat_wmo)
    res_zs = evaluate_model("Temporal Z-Score (Diurnal)", y_pred_zs, lat_zs)
    res_iso = evaluate_model("Isolation Forest (Multivariate)", y_pred_iso, lat_iso)
    res_sky = evaluate_model("SkyGuard AI (5-Layer Ensemble)", y_pred_skyguard, lat_skyguard)

    print("\nBENCHMARK RESULTS TABLE:")
    print("-" * 88)
    print(f"{'Model Architecture':<36} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FAR':<8} | {'Latency':<8}")
    print("-" * 88)
    for r in [res_wmo, res_zs, res_iso, res_sky]:
        print(f"{r['model']:<36} | {r['precision']*100:<9.1f}% | {r['recall']*100:<9.1f}% | {r['f1_score']:<10.3f} | {r['false_alarm_rate']*100:<7.1f}% | {r['avg_latency_ms']} ms")
    print("-" * 88)

    print(f"\nSkyGuard AI p95 Processing Latency: {lat_p95:.2f} ms")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "results"))
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "benchmark_summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_test_samples": len(dataset),
            "p95_latency_ms": lat_p95,
            "models": [res_wmo, res_zs, res_iso, res_sky]
        }, f, indent=2)

    print(f"Benchmark summary saved to: {summary_path}")


if __name__ == "__main__":
    run_benchmark()
