"""
SkyGuard AI Email Dispatcher Module.
Supports real email delivery via standard SMTP (e.g. Gmail, Outlook, AWS SES)
with an automatic graceful fallback to instant on-screen / console simulation
for offline judging and evaluation.
"""
import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Tuple

logger = logging.getLogger("skyguard_mailer")

def send_otp_email(to_email: str, otp_code: str, purpose: str = "REGISTRATION", recipient_name: str = "Officer") -> Tuple[bool, str]:
    """
    Sends an OTP verification email to the recipient.
    
    Returns:
        (success: bool, status_message: str)
    """
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM", "SkyGuard AI <no-reply@skyguard.imd.gov.in>").strip()

    # If SMTP credentials are not configured, use local instant delivery simulation
    if not smtp_host or not smtp_user or not smtp_pass:
        msg = f"[OFFLINE / SIMULATION MODE] OTP for {to_email}: {otp_code} (SMTP credentials not set in .env)"
        logger.info(msg)
        return False, "Simulated delivery (SMTP credentials not configured in .env)"

    try:
        subject = f"[SkyGuard AI] Your Verification Code: {otp_code}"
        if purpose == "PASSWORD_RESET":
            subject = f"[SkyGuard AI] Password Reset Code: {otp_code}"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f1f5f9; margin: 0; padding: 20px; }}
            .card {{ max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }}
            .header {{ background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: #ffffff; padding: 28px 24px; text-align: center; }}
            .title {{ font-size: 22px; font-weight: 800; margin: 0; }}
            .subtitle {{ font-size: 13px; opacity: 0.9; margin-top: 4px; }}
            .body {{ padding: 28px 24px; color: #1e293b; font-size: 14px; line-height: 1.6; }}
            .otp-box {{ background: #f0fdf4; border: 2px dashed #22c55e; border-radius: 12px; padding: 18px; text-align: center; margin: 24px 0; }}
            .otp-code {{ font-family: 'Courier New', monospace; font-size: 32px; font-weight: 900; letter-spacing: 6px; color: #15803d; }}
            .footer {{ background: #f8fafc; padding: 16px 24px; text-align: center; font-size: 12px; color: #64748b; border-top: 1px solid #e2e8f0; }}
          </style>
        </head>
        <body>
          <div class="card">
            <div class="header">
              <h1 class="title">SkyGuard AI</h1>
              <p class="subtitle">Ministry of Earth Sciences (MoES) &bull; IMD AWS Network</p>
            </div>
            <div class="body">
              <p>Hello <strong>{recipient_name}</strong>,</p>
              <p>You requested a 6-digit verification code to authenticate your account with official email: <strong>{to_email}</strong>.</p>
              
              <div class="otp-box">
                <div style="font-size: 12px; font-weight: bold; color: #166534; text-transform: uppercase; margin-bottom: 6px;">Your One-Time Verification Code</div>
                <div class="otp-code">{otp_code}</div>
                <div style="font-size: 11px; color: #15803d; margin-top: 6px;">Valid for 10 minutes &bull; Do not share with anyone</div>
              </div>

              <p style="font-size: 13px; color: #475569;">If you did not initiate this request, you can safely ignore this email.</p>
            </div>
            <div class="footer">
              SkyGuard AI Automated Security Dispatch &bull; SIH 2026 Problem Statement 26073
            </div>
          </div>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = to_email

        # Attach HTML
        msg.attach(MIMEText(html_content, "html"))

        # Connect to SMTP server
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
            server.starttls()

        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, [to_email], msg.as_string())
        server.quit()

        logger.info(f"[REAL EMAIL DELIVERED] Successfully dispatched OTP to {to_email}")
        return True, f"Real email delivered to {to_email}"
    except Exception as e:
        logger.error(f"[EMAIL DELIVERY ERROR] Failed to send email to {to_email}: {e}")
        return False, f"SMTP Error: {str(e)}"
