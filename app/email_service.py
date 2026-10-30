"""
SMTP Email Delivery Service for Local Voice Agent.
Sends branded OTP verification emails for Registration and Password Reset.
Includes automatic development/fallback mode when SMTP is unconfigured.
"""
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parseaddr
from typing import Dict, Any, Optional


class EmailService:
    """
    Singleton email dispatcher supporting SMTP and local development modes.
    """
    _instance: Optional["EmailService"] = None

    def _reload_config(self):
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            base_dir = Path(__file__).resolve().parent.parent
            creds_env = base_dir / "credentials" / ".env"
            if creds_env.is_file():
                load_dotenv(creds_env, override=True)
            load_dotenv(base_dir / ".env", override=True)
        except Exception:
            pass
        self.smtp_host = os.getenv("SMTP_HOST", "").strip()
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "").strip()
        self.smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
        self.smtp_from = os.getenv("SMTP_FROM", "").strip() or self.smtp_user or "no-reply@apexai.io"
        self.smtp_tls = os.getenv("SMTP_TLS", "true").lower() in ("true", "1", "yes")
        is_test = bool(os.getenv("PYTEST_CURRENT_TEST"))
        self.dev_fallback = os.getenv("SMTP_DEV_FALLBACK", "true" if is_test else "false").lower() in ("true", "1", "yes")

    def __init__(self):
        self._reload_config()

    @classmethod
    def get_instance(cls) -> "EmailService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def send_otp(self, to_email: str, otp_code: str, purpose: str = "registration") -> Dict[str, Any]:
        """
        Sends a 6-digit OTP code to the recipient email.
        Falls back cleanly to local dev logging if SMTP is not configured.
        """
        self._reload_config()
        subject = "Your Verification Code - Apex AI Organizations" if purpose == "registration" else "Reset Your Password - Apex AI Organizations"
        
        # HTML Email Template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; margin: 0; padding: 24px; color: #0f172a; }}
            .container {{ max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 16px; border: 1px solid #e2e8f0; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }}
            .logo {{ font-size: 28px; margin-bottom: 8px; }}
            .title {{ font-size: 20px; font-weight: 800; color: #0f172a; margin: 0 0 8px 0; }}
            .desc {{ font-size: 14px; color: #64748b; line-height: 1.5; margin: 0 0 24px 0; }}
            .otp-box {{ background: #f1f5f9; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 24px; }}
            .otp-code {{ font-size: 32px; font-weight: 800; letter-spacing: 6px; color: #0284c7; font-family: monospace; }}
            .note {{ font-size: 12px; color: #94a3b8; line-height: 1.5; margin-top: 12px; }}
            .footer {{ font-size: 11px; color: #cbd5e1; text-align: center; margin-top: 24px; border-top: 1px solid #f1f5f9; padding-top: 16px; }}
          </style>
        </head>
        <body>
          <div class="container">
            <div class="logo" style="display:inline-block; font-weight:800; font-size:16px; letter-spacing:1px; color:#2563eb; background:#eff6ff; padding:6px 14px; border-radius:8px; border:1px solid #bfdbfe;">APEX AI</div>
            <h1 class="title">{"Verify Your Account" if purpose == "registration" else "Password Reset Request"}</h1>
            <p class="desc">
              {"Thank you for registering with Apex AI Organizations. Enter this verification code to set up your password and access your organization dashboard." if purpose == "registration" else "We received a request to reset your password for Apex AI Organizations. Use the code below to complete the reset."}
            </p>
            <div class="otp-box">
              <div class="otp-code">{otp_code}</div>
              <div class="note">This code will expire in 10 minutes. Do not share this code with anyone.</div>
            </div>
            <p style="font-size: 13px; color: #475569;">If you did not request this email, please safely ignore it.</p>
            <div class="footer">
              Apex AI Innovations • Enterprise Voice & Agent Platform
            </div>
          </div>
        </body>
        </html>
        """

        plain_text = f"""
Apex AI Organizations
---------------------
Your verification code is: {otp_code}

This code will expire in 10 minutes.
Purpose: {purpose.replace('_', ' ').title()}
If you did not request this, please ignore this email.
        """.strip()

        # If SMTP is configured, attempt real delivery
        if self.smtp_host and self.smtp_user:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = self.smtp_from
                msg["To"] = to_email

                part1 = MIMEText(plain_text, "plain")
                part2 = MIMEText(html_content, "html")
                msg.attach(part1)
                msg.attach(part2)

                envelope_from = parseaddr(self.smtp_from)[1] or self.smtp_user or self.smtp_from
                context = ssl.create_default_context()
                if self.smtp_port == 465:
                    with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context, timeout=10) as server:
                        if self.smtp_user and self.smtp_password:
                            server.login(self.smtp_user, self.smtp_password)
                        server.sendmail(envelope_from, to_email, msg.as_string())
                else:
                    with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                        if self.smtp_tls:
                            server.starttls(context=context)
                        if self.smtp_user and self.smtp_password:
                            server.login(self.smtp_user, self.smtp_password)
                        server.sendmail(envelope_from, to_email, msg.as_string())

                print(f"📧 [SMTP Service] Successfully delivered email to: {to_email}")
                return {
                    "success": True,
                    "mode": "smtp",
                    "message": f"Verification email sent to {to_email}",
                    "dev_otp": None,
                }
            except Exception as e:
                err_msg = str(e)
                if "535" in err_msg or "BadCredentials" in err_msg or "Username and Password not accepted" in err_msg:
                    clean_err = "Gmail SMTP Authentication Failed (535 Bad Credentials): The 16-character Google App Password in .env was not accepted by Google. Please generate a new App Password at https://myaccount.google.com/apppasswords."
                elif "Connection unexpectedly closed" in err_msg:
                    clean_err = "SMTP Connection Closed: Google closed the connection. Please verify your 16-character Google App Password in .env."
                else:
                    clean_err = f"SMTP Delivery Failed: {err_msg}"
                print(f"⚠️ [SMTP Service] {clean_err}")

                if not self.dev_fallback:
                    return {
                        "success": False,
                        "mode": "error",
                        "message": clean_err,
                        "error": clean_err,
                        "dev_otp": None,
                    }

        # Dev / Sandbox Fallback mode
        print(f"📨 [SMTP Dev Mode] OTP for {to_email} is: {otp_code} (Purpose: {purpose})")
        return {
            "success": True,
            "mode": "dev",
            "message": f"Verification code delivered to {to_email}",
            "dev_otp": otp_code,
        }
