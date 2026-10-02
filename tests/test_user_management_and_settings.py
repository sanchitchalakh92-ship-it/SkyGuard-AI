"""
Automated Unit and Integration Tests for User Management and System Settings.
Covers:
- User count statistics (Total, Active, Pending)
- Admin Add User & Edit User
- Account Approval & Status Toggling
- Activity Audit Logs retrieval
- System Settings retrieval & updates
- Database Snapshot Backup
- CSV Audit Export
"""
import time
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from database.db import init_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    init_db()

def test_get_users_and_metrics():
    resp = client.get("/api/users")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "users" in data
    assert "counts" in data
    assert data["counts"]["total"] >= 24
    assert data["counts"]["pending"] >= 1

def test_add_and_edit_user():
    test_email = f"officer_{int(time.time()*1000)}@imd.gov.in"
    add_resp = client.post("/api/users/add", json={
        "name": "Deputy Director Verma",
        "email": test_email,
        "role": "HEAD_OFFICER",
        "department": "National Weather Forecasting Centre",
        "station_access": "ALL"
    })
    assert add_resp.status_code == 200
    user = add_resp.json()["user"]
    user_id = user["id"]

    # Update user details
    update_resp = client.put(f"/api/users/{user_id}", json={
        "name": "Director Verma (Promoted)",
        "role": "HEAD_OFFICER",
        "department": "IMD Headquarters",
        "station_access": "ALL",
        "status": "Active"
    })
    assert update_resp.status_code == 200
    assert update_resp.json()["user"]["name"] == "Director Verma (Promoted)"

def test_user_status_and_approval():
    test_email = f"pending_{int(time.time()*1000)}@imd.gov.in"
    add_resp = client.post("/api/users/add", json={
        "name": "Applicant Joshi",
        "email": test_email,
        "role": "DATA_ANALYST",
        "department": "Regional Climatology",
        "station_access": "AWS_PUN"
    })
    user_id = add_resp.json()["user"]["id"]

    # Suspend user
    status_resp = client.post(f"/api/users/{user_id}/status", json={"status": "Suspended"})
    assert status_resp.status_code == 200

    # Reactivate user
    activate_resp = client.post(f"/api/users/{user_id}/status", json={"status": "Active"})
    assert activate_resp.status_code == 200

def test_system_settings_and_backup():
    # 1. Fetch settings
    settings_resp = client.get("/api/settings")
    assert settings_resp.status_code == 200
    categories = settings_resp.json()["settings"]
    assert "GENERAL" in categories
    assert "MONITORING" in categories
    assert "AIML" in categories

    # 2. Update settings
    update_resp = client.post("/api/settings", json={
        "settings": {
            "refresh_interval": "45",
            "confidence_threshold": "0.88"
        }
    })
    assert update_resp.status_code == 200
    assert update_resp.json()["success"] is True

    # 3. Create database snapshot backup
    backup_resp = client.post("/api/settings/backup")
    assert backup_resp.status_code == 200
    assert "backup" in backup_resp.json()
    assert backup_resp.json()["backup"]["filename"].startswith("skyguard_backup_")

def test_activity_logs_and_export():
    # Fetch logs
    logs_resp = client.get("/api/users/activity-logs")
    assert logs_resp.status_code == 200
    logs = logs_resp.json()["logs"]
    assert len(logs) > 0

    # Export CSV
    export_resp = client.get("/api/settings/export-audit")
    assert export_resp.status_code == 200
    assert "ID,Timestamp" in export_resp.text

def test_weather_api_endpoints():
    # 1. Test Weather Radar Metadata Endpoint
    radar_resp = client.get("/api/weather/radar-meta")
    assert radar_resp.status_code == 200
    rdata = radar_resp.json()
    assert rdata["success"] is True
    assert "radar_tile_url" in rdata
    assert "tilecache.rainviewer.com" in rdata["radar_tile_url"]

    # 2. Test Live Station Weather API Endpoint
    weather_resp = client.get("/api/weather/live/AWS_PUN")
    assert weather_resp.status_code == 200
    wdata = weather_resp.json()
    assert wdata["success"] is True
    assert wdata["station_id"] == "AWS_PUN"
    assert "temperature" in wdata
    assert "humidity" in wdata
    assert "provider" in wdata

    # 3. Test Invalid Station 404
    invalid_resp = client.get("/api/weather/live/NON_EXISTENT")
    assert invalid_resp.status_code == 404

def test_anomaly_detection_endpoints():
    # 1. Test Anomaly Dashboard Telemetry (Isolation Forest & Multi-Model)
    resp = client.get("/api/anomalies/dashboard?station_id=AWS_NGP&algorithm=hybrid&parameter=temperature")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["station_id"] == "AWS_NGP"
    assert data["anomaly_rate_pct"] == 4.85
    assert len(data["base_curve"]) == 24
    assert len(data["observed_curve"]) == 24
    assert len(data["anomaly_points"]) >= 5
    assert len(data["type_counts"]) == 5
    assert len(data["detected_anomalies"]) >= 5
    assert len(data["root_causes"]) >= 4

    # Verify each required anomaly type is covered
    types = [a["anomaly_type"] for a in data["detected_anomalies"]]
    assert any("Spike" in t for t in types)
    assert any("Drift" in t for t in types)
    assert any("Stuck" in t for t in types)
    assert any("Missing" in t for t in types)
    assert any("Communication" in t for t in types)

    # Verify severity, confidence, station, and reasons
    for anom in data["detected_anomalies"]:
        assert "severity" in anom
        assert "confidence" in anom
        assert "station_id" in anom
        assert "reason" in anom
        assert len(anom["reason"]) > 10

    # 2. Test Live Inference Trigger
    inf_resp = client.post("/api/anomalies/run-inference", json={
        "station_id": "AWS_KLP",
        "algorithm": "isolation_forest",
        "parameter": "pressure"
    })
    assert inf_resp.status_code == 200
    inf_data = inf_resp.json()
    assert inf_data["success"] is True
    assert inf_data["station_id"] == "AWS_KLP"


def test_realtime_weather_monitoring_endpoints():
    # 1. Test Real-time Station Telemetry (Meadowfield UI Schema)
    resp = client.get("/api/stations/AWS_PUN/live")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["station"]["id"] == "AWS_PUN"
    assert data["station"]["status"] in ["Online", "Alert", "Maintenance", "Offline"]

    tel = data["telemetry"]
    # 6 Top Metric Cards
    assert "temperature" in tel and "current" in tel["temperature"]
    assert "wind" in tel and "speed_kmh" in tel["wind"] and "direction_deg" in tel["wind"]
    assert "humidity" in tel and "current" in tel["humidity"] and "dew_point" in tel["humidity"]
    assert "pressure" in tel and "current" in tel["pressure"] and "trend_3h_hpa" in tel["pressure"]
    assert "rainfall" in tel and "today_mm" in tel["rainfall"] and "year_mm" in tel["rainfall"]
    assert "solar" in tel and "radiation_wm2" in tel["solar"] and "uv_index" in tel["solar"]

    # 3 Bottom Panels
    assert tel["wind"]["direction_cardinal"] is not None
    assert tel["wind"]["max_gust_kmh"] is not None
    assert tel["rainfall"]["last_hour_mm"] == 0.0
    assert tel["rainfall"]["year_mm"] == 307.4
    assert "comfort" in tel
    assert "heat_index" in tel["comfort"]
    assert "wind_chill" in tel["comfort"]
    assert "wet_bulb" in tel["comfort"]

    # 2. Test Ingestion via FastAPI / MQTT / Simulated Streams
    ingest_payload = {
        "station_id": "AWS_PUN",
        "temperature": 33.1,
        "humidity": 42.0,
        "pressure": 1009.2,
        "wind_speed": 2.4,
        "rainfall": 0.0,
        "solar_radiation": 750.0,
        "source": "MQTT Protocol Broker"
    }
    ingest_resp = client.post("/api/telemetry/ingest", json=ingest_payload)
    assert ingest_resp.status_code == 200
    ingest_data = ingest_resp.json()
    assert ingest_data["success"] is True
    assert ingest_data["station_id"] == "AWS_PUN"

    # 3. Test Historical Readings from SQLite DB
    hist_resp = client.get("/api/history/AWS_PUN?limit=10")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["success"] is True
    assert hist_data["count"] > 0
    assert len(hist_data["readings"]) > 0

    # 4. Test Telemetry CSV Export
    csv_resp = client.get("/api/telemetry/export/AWS_PUN")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers.get("content-type", "")
    assert "StationID,Timestamp,Temperature" in csv_resp.text


def test_dashboard_improvements_metrics_and_separation():
    """
    1. Correct metric calculations: 23 operational out of 24 is 95.8%.
    2. Separate anomalies from maintenance: maintenance stations are distinct from alerts.
    """
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    summary = data["summary"]

    total = summary["total_stations"]
    op = summary["operational"]
    expected_pct = round((op / total) * 100, 1)

    assert summary["operational_pct"] == expected_pct
    assert summary["operational_pct"] == 95.8
    assert summary["network_health_pct"] == 95.8
    assert summary["operational"] == 23
    assert summary["maintenance"] >= 1
    assert summary["under_alert"] == 1  # Nagpur is under alert (anomaly), distinct from maintenance


def test_anomaly_summary_endpoint():
    """
    3. Anomaly summary: today's detections, unresolved critical, false-alarm rate, avg resolution time.
    """
    resp = client.get("/api/anomalies/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "today_detections" in data and data["today_detections"] >= 1
    assert "unresolved_critical" in data and data["unresolved_critical"] >= 1
    assert "false_alarm_rate_pct" in data and data["false_alarm_rate_pct"] == 0.08
    assert "avg_resolution_time_min" in data and data["avg_resolution_time_min"] == 18.5
    assert "resolution_rate_pct" in data and data["resolution_rate_pct"] > 90


def test_station_data_freshness():
    """
    4. Data freshness: every station displays its last received reading and delayed status.
    """
    resp = client.get("/stations")
    assert resp.status_code == 200
    stations = resp.json()
    assert len(stations) >= 20

    for stn in stations:
        assert "last_received" in stn
        assert "freshness_label" in stn
        assert "freshness_status" in stn
        assert stn["freshness_status"] in ["Fresh", "Delayed", "Stale"]
        assert "is_delayed" in stn
        assert isinstance(stn["is_delayed"], bool)


def test_model_status_endpoint():
    """
    5. Model status: active model version, last evaluation date, precision, recall, false-positive rate.
    """
    resp = client.get("/api/models/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["active_version"] == "v2.4.1-prod"
    assert "2026" in data["last_evaluation_date"]
    assert data["precision_pct"] == 96.8
    assert data["recall_pct"] == 97.4
    assert data["false_positive_rate_pct"] == 1.8
    assert data["f1_score"] == 0.971
    assert data["auc_roc"] == 0.991


def test_transparent_demo_scenario():
    """
    6. Demo transparency: clearly label simulated data and demo triggers for evaluators.
    """
    resp = client.post("/inject", json={
        "station_id": "AWS_NGP",
        "anomaly_type": "55c_scenario"
    }, headers={"X-User-Role": "OPERATIONS_ENGINEER"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data.get("is_simulated_demo") is True
    assert "synthetic" in data.get("simulation_label", "").lower()
    assert "simulated" in data.get("demo_disclaimer", "").lower() or "evaluat" in data.get("demo_disclaimer", "").lower()


def test_strengthened_role_based_access_control():
    """
    7. Strengthen access control: enforce role-based permissions on the backend (403 Forbidden).
    """
    # DATA_ANALYST cannot save system settings
    analyst_settings_resp = client.post("/api/settings", json={
        "settings": {"refresh_interval": "10"}
    }, headers={"X-User-Role": "DATA_ANALYST", "X-User-Email": "analyst@skyguard.imd.gov.in"})
    assert analyst_settings_resp.status_code == 403
    assert "Access Denied" in analyst_settings_resp.json()["detail"]

    # DATA_ANALYST cannot inject simulated anomalies
    analyst_inject_resp = client.post("/inject", json={
        "station_id": "AWS_NGP",
        "anomaly_type": "55c_scenario"
    }, headers={"X-User-Role": "DATA_ANALYST", "X-User-Email": "analyst@skyguard.imd.gov.in"})
    assert analyst_inject_resp.status_code == 403
    assert "Access Denied" in analyst_inject_resp.json()["detail"]

    # OPERATIONS_ENGINEER cannot add new user accounts
    engineer_user_resp = client.post("/api/users/add", json={
        "name": "Unauthorized User",
        "email": "unauth@skyguard.imd.gov.in",
        "role": "DATA_ANALYST",
        "department": "Weather Analysis",
        "station_access": "AWS_PUN"
    }, headers={"X-User-Role": "OPERATIONS_ENGINEER", "X-User-Email": "engineer@skyguard.imd.gov.in"})
    assert engineer_user_resp.status_code == 403
    assert "Access Denied" in engineer_user_resp.json()["detail"]

    # HEAD_OFFICER can successfully save settings
    officer_settings_resp = client.post("/api/settings", json={
        "settings": {"refresh_interval": "30"}
    }, headers={"X-User-Role": "HEAD_OFFICER", "X-User-Email": "head@skyguard.imd.gov.in"})
    assert officer_settings_resp.status_code == 200
    assert officer_settings_resp.json()["success"] is True




