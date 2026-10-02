-- SkyGuard AI Database Schema (SQLite / PostgreSQL compatible)
-- Problem Statement: SIH 26073 - MoES / IMD AWS Anomaly Detection

-- 0. Users & Authentication Table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(128) NOT NULL,
    email VARCHAR(128) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    role VARCHAR(32) NOT NULL, -- 'HEAD_OFFICER', 'OPERATIONS_ENGINEER', 'DATA_ANALYST'
    department VARCHAR(64) DEFAULT 'IMD Meteorological Division',
    status VARCHAR(32) DEFAULT 'Active', -- 'Active', 'Pending', 'Suspended'
    station_access VARCHAR(256) DEFAULT 'ALL', -- 'ALL' or comma-separated station IDs
    is_verified BOOLEAN DEFAULT 1,
    last_login TIMESTAMP,
    failed_attempts INTEGER DEFAULT 0,
    mfa_enabled BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 0.1 Email OTP Verifications Table (For Registration & Password Reset)
CREATE TABLE IF NOT EXISTS email_otps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email VARCHAR(128) NOT NULL,
    otp_code VARCHAR(8) NOT NULL,
    purpose VARCHAR(32) NOT NULL, -- 'REGISTRATION', 'PASSWORD_RESET'
    temp_data TEXT, -- stores JSON encoded name, pw_hash, role during pending reg
    expires_at TIMESTAMP NOT NULL,
    is_used BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 0.2 User Activity & Audit Logs Table
CREATE TABLE IF NOT EXISTS activity_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_name VARCHAR(128) NOT NULL,
    user_email VARCHAR(128) NOT NULL,
    role VARCHAR(32) NOT NULL,
    action VARCHAR(64) NOT NULL, -- 'LOGIN', 'FAILED_LOGIN', 'ADD_USER', 'ROLE_CHANGE', 'STATUS_CHANGE', 'CONFIG_UPDATE', 'DATA_EXPORT', 'TEST_INJECTION'
    details TEXT,
    ip_address VARCHAR(45) DEFAULT '127.0.0.1',
    status VARCHAR(16) DEFAULT 'SUCCESS' -- 'SUCCESS', 'FAILED', 'WARNING'
);

-- 0.3 System & Application Settings Table
CREATE TABLE IF NOT EXISTS system_settings (
    key VARCHAR(64) PRIMARY KEY,
    value TEXT NOT NULL,
    category VARCHAR(64) NOT NULL, -- 'GENERAL', 'MONITORING', 'AIML', 'NOTIFICATIONS', 'SECURITY', 'DATABASE', 'APPEARANCE'
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 1. Stations Table
CREATE TABLE IF NOT EXISTS stations (
    station_id VARCHAR(32) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    state VARCHAR(64) NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    elevation_m REAL NOT NULL,
    status VARCHAR(32) DEFAULT 'Normal',
    health_score REAL DEFAULT 100.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Telemetry Observations Table
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id VARCHAR(32) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    temperature REAL,
    pressure REAL,
    humidity REAL,
    wind_speed REAL,
    rainfall REAL,
    solar_radiation REAL,
    is_anomaly BOOLEAN DEFAULT 0,
    severity VARCHAR(16) DEFAULT 'Normal',
    confidence REAL DEFAULT 1.0,
    is_imputed BOOLEAN DEFAULT 0,
    imputed_temperature REAL,
    imputed_pressure REAL,
    imputed_humidity REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

-- 3. Anomaly Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    alert_id VARCHAR(64) PRIMARY KEY,
    station_id VARCHAR(32) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    severity VARCHAR(16) NOT NULL,
    root_cause VARCHAR(64) NOT NULL,
    confidence REAL NOT NULL,
    explanation TEXT,
    recommended_action TEXT,
    is_genuine_event BOOLEAN DEFAULT 0,
    what_changed TEXT,
    expected_value REAL,
    actual_value REAL,
    deviation_pct REAL,
    cross_station_verdict VARCHAR(64),
    maintenance_warning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

-- 4. Sensor Health History Table
CREATE TABLE IF NOT EXISTS sensor_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id VARCHAR(32) NOT NULL,
    composite_health REAL NOT NULL,
    temperature_health REAL NOT NULL,
    pressure_health REAL NOT NULL,
    humidity_health REAL NOT NULL,
    cusum_drift_t REAL DEFAULT 0.0,
    priority VARCHAR(16) DEFAULT 'OK',
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

-- 5. Maintenance & Troubleshooting History Table (Operations Engineer)
CREATE TABLE IF NOT EXISTS maintenance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id VARCHAR(32) NOT NULL,
    technician VARCHAR(128) NOT NULL,
    action_type VARCHAR(64) NOT NULL, -- 'CALIBRATION', 'REPAIR', 'CLEANING', 'REPLACEMENT', 'DIAGNOSTIC'
    notes TEXT NOT NULL,
    status VARCHAR(32) DEFAULT 'Completed',
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

-- 6. Analytical Reports Governance Table (Head Officer & Data Analyst)
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title VARCHAR(256) NOT NULL,
    report_type VARCHAR(64) NOT NULL, -- 'MONTHLY_AUDIT', 'SENSOR_DRIFT', 'HEATWAVE_ABLATION', 'MODEL_PERFORMANCE'
    author VARCHAR(128) NOT NULL,
    status VARCHAR(32) DEFAULT 'Pending Approval', -- 'Pending Approval', 'Approved', 'Rejected'
    approved_by VARCHAR(128),
    approved_at TIMESTAMP,
    key_findings TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Continuous Model Evaluation & Human-in-the-Loop Review Table
CREATE TABLE IF NOT EXISTS model_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id VARCHAR(64) NOT NULL,
    station_id VARCHAR(32) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    model_prediction VARCHAR(64) NOT NULL,
    model_confidence REAL NOT NULL,
    verified_status VARCHAR(32) NOT NULL, -- 'VERIFIED_TRUE_POSITIVE', 'VERIFIED_FALSE_ALARM', 'VERIFIED_MISSED_ANOMALY', 'VERIFIED_GENUINE_WEATHER'
    reviewed_by VARCHAR(128) NOT NULL,
    review_notes TEXT,
    is_validated_for_training BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

-- 8. Validated Model Retraining Runs History Table
CREATE TABLE IF NOT EXISTS model_retraining_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version VARCHAR(32) NOT NULL,
    validated_samples_count INTEGER NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    f1_score REAL NOT NULL,
    false_alarm_rate REAL NOT NULL,
    missed_anomaly_rate REAL NOT NULL,
    triggered_by VARCHAR(128) NOT NULL,
    status VARCHAR(32) DEFAULT 'COMPLETED',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for high-speed queries
CREATE INDEX IF NOT EXISTS idx_readings_station_time ON readings(station_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_station_time ON alerts(station_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_email_otps ON email_otps(email, purpose);
CREATE INDEX IF NOT EXISTS idx_activity_logs ON activity_logs(timestamp, user_email);
CREATE INDEX IF NOT EXISTS idx_system_settings_cat ON system_settings(category);
CREATE INDEX IF NOT EXISTS idx_maintenance_station ON maintenance_records(station_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);
CREATE INDEX IF NOT EXISTS idx_evaluations_event ON model_evaluations(event_id, verified_status);
CREATE INDEX IF NOT EXISTS idx_retraining_version ON model_retraining_runs(model_version);


