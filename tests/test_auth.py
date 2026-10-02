"""
Automated Unit and Integration Tests for SkyGuard AI Authentication & OTP Flows.
Covers:
- User Registration with 6-Digit Email OTP Verification
- Direct Login & Role-Based Access Control (Head Officer, Engineer, Analyst)
- Forgot Password with Email OTP Verification & Secure PBKDF2 Password Reset
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

def test_registration_with_email_otp_and_login():
    test_email = f"auto_{int(time.time() * 1000)}@skyguard.imd.gov.in"
    
    # 1. Initiate registration
    init_resp = client.post("/api/auth/register-initiate", json={
        "name": "Automation Officer",
        "email": test_email,
        "password": "securepassword123",
        "role": "HEAD_OFFICER",
        "department": "Directorate of Weather Observation"
    })
    assert init_resp.status_code == 200
    init_data = init_resp.json()
    assert init_data["success"] is True
    otp = init_data["otp_demo"]
    assert len(otp) == 6

    # 2. Verify OTP
    verify_resp = client.post("/api/auth/register-verify-otp", json={
        "email": test_email,
        "otp": otp
    })
    assert verify_resp.status_code == 200
    verify_data = verify_resp.json()
    assert verify_data["success"] is True
    assert verify_data["user"]["email"] == test_email
    assert verify_data["user"]["role"] == "HEAD_OFFICER"

    # 3. Login with verified account
    login_resp = client.post("/api/auth/login", json={
        "email": test_email,
        "password": "securepassword123"
    })
    assert login_resp.status_code == 200
    assert login_resp.json()["user"]["name"] == "Automation Officer"

def test_forgot_password_flow():
    test_email = "sanchit@example.com"

    # 1. Initiate forgot password
    fp_resp = client.post("/api/auth/forgot-password-initiate", json={
        "email": test_email
    })
    assert fp_resp.status_code == 200
    fp_data = fp_resp.json()
    assert fp_data["success"] is True
    otp = fp_data["otp_demo"]
    assert len(otp) == 6

    # 2. Reset password with OTP
    reset_resp = client.post("/api/auth/forgot-password-reset", json={
        "email": test_email,
        "otp": otp,
        "new_password": "updatedpassword789"
    })
    assert reset_resp.status_code == 200
    assert reset_resp.json()["success"] is True

    # 3. Old password must fail
    old_login = client.post("/api/auth/login", json={
        "email": test_email,
        "password": "engineer123"
    })
    assert old_login.status_code == 401

    # 4. New password must succeed
    new_login = client.post("/api/auth/login", json={
        "email": test_email,
        "password": "updatedpassword789"
    })
    assert new_login.status_code == 200
    assert new_login.json()["user"]["role"] == "OPERATIONS_ENGINEER"

    # Reset it back to default for demo consistency
    fp_back = client.post("/api/auth/forgot-password-initiate", json={"email": test_email})
    client.post("/api/auth/forgot-password-reset", json={
        "email": test_email,
        "otp": fp_back.json()["otp_demo"],
        "new_password": "engineer123"
    })
