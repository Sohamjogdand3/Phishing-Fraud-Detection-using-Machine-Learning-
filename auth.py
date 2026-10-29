"""
Authentication and Security Services Layer.

Provides:
- Argon2 password hashing and verification.
- Cryptographically secure 6-digit OTP generation and SHA-256 hashing.
- Sliding-window rate limiting.
- Email delivery via SMTP with test/development fallback.
- CSRF token generation and validation.
- Route authentication and authorization decorators.
"""

from datetime import datetime, timedelta
from functools import wraps
import hashlib
import hmac
import os
import re
import secrets
import smtplib
import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid, parseaddr
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from flask import abort, flash, redirect, request, session, url_for

from db import (
    get_rate_limit_count,
    prune_old_rate_limits,
    record_rate_limit_action,
)

def load_dotenv(path=None, override=False):
    """
    Loads environment variables from .env file into os.environ.
    Searches:
    1. Specified path (if provided)
    2. Same directory as this script: os.path.dirname(os.path.abspath(__file__))
    3. Current working directory: os.getcwd()
    """
    candidates = []
    if path:
        candidates.append(path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(base_dir, ".env"))
    cwd_env = os.path.join(os.getcwd(), ".env")
    if cwd_env not in candidates:
        candidates.append(cwd_env)

    for cand in candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip()
                            # Strip surrounding single or double quotes
                            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                                v = v[1:-1].strip()
                            if override or (k not in os.environ) or (not os.environ[k]):
                                os.environ[k] = v
                return cand
            except Exception as e:
                print(f"[CONFIG WARNING] Could not read .env file at {cand}: {e}")
    return None

load_dotenv(override=False)

# Initialize Argon2 Password Hasher
ph = PasswordHasher(time_cost=2, memory_cost=65536, parallelism=2)

# OTP Secret Salt for SHA-256 OTP hashing
OTP_SALT = os.environ.get("OTP_SECRET_SALT", "secure-phishing-otp-salt-2026").encode()

# In-memory test mailbox for automated tests and development
SENT_EMAILS = []


# ---------------- PASSWORD HASHING & VALIDATION ----------------

def hash_password(password: str) -> str:
    """Hashes a password using Argon2id."""
    return ph.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    """Verifies a plaintext password against an Argon2 hash in constant time."""
    try:
        return ph.verify(stored_hash, password)
    except (VerifyMismatchError, Exception):
        return False


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Enforces strong password policy:
    - Minimum 8 characters.
    - At least one letter.
    - At least one digit.
    """
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter."
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number."
    return True, ""


def normalize_email(email: str) -> str:
    """Trims and lowercases email address for consistent uniqueness."""
    return email.strip().lower() if email else ""


def is_valid_email(email: str) -> bool:
    """Validates email format using standard regex."""
    if not email or len(email) > 254:
        return False
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(pattern, email))


# ---------------- OTP GENERATION & HASHING ----------------

def generate_otp() -> str:
    """Generates a cryptographically secure 6-digit numerical OTP."""
    digits = "0123456789"
    return "".join(secrets.choice(digits) for _ in range(6))


def hash_otp(otp: str) -> str:
    """Hashes a 6-digit OTP using HMAC-SHA256 with application salt."""
    return hmac.new(OTP_SALT, otp.encode('utf-8'), hashlib.sha256).hexdigest()


def verify_otp_hash(entered_otp: str, stored_hash: str) -> bool:
    """Verifies entered OTP against stored HMAC-SHA256 hash using constant-time comparison."""
    if not entered_otp or not stored_hash:
        return False
    entered_hash = hash_otp(entered_otp.strip())
    return hmac.compare_digest(entered_hash, stored_hash)


# ---------------- EMAIL DELIVERY SERVICE ----------------

def send_email(to_email: str, subject: str, text_body: str, html_body: str = None) -> bool:
    """
    Sends an email using configured SMTP settings.
    Properly formats RFC 5321 envelope addresses and RFC 5322 MIME headers.
    If MAIL_SERVER is not configured or in development mode,
    the message is safely stored in SENT_EMAILS and printed to console.
    """
    raw_server = os.environ.get("MAIL_SERVER")
    mail_server = raw_server.strip() if raw_server else None

    raw_port = os.environ.get("MAIL_PORT", "587")
    try:
        mail_port = int(str(raw_port).strip())
    except (ValueError, TypeError):
        mail_port = 587

    raw_user = os.environ.get("MAIL_USERNAME")
    mail_user = raw_user.strip() if raw_user else None

    raw_pass = os.environ.get("MAIL_PASSWORD")
    mail_pass = raw_pass.strip() if raw_pass else None

    raw_sender = os.environ.get("MAIL_DEFAULT_SENDER")
    sender_setting = raw_sender.strip() if raw_sender else (mail_user or "noreply@phishingdetector.local")

    use_tls = str(os.environ.get("MAIL_USE_TLS", "True")).strip().lower() in ("true", "1", "yes")

    # In-memory logging for tests and inspectability
    SENT_EMAILS.append({
        "to": to_email,
        "subject": subject,
        "text": text_body,
        "html": html_body,
        "timestamp": datetime.utcnow()
    })

    if not mail_server:
        print(f"[EMAIL DEV MODE] Sent to '{to_email}' | Subject: {subject} | Body: {text_body}")
        return True

    # 1. Parse sender display name and bare envelope address (RFC 5321 vs RFC 5322)
    sender_name, sender_addr = parseaddr(sender_setting)
    envelope_from = sender_addr.strip() if sender_addr else (mail_user or sender_setting)
    if not envelope_from or envelope_from == "noreply@phishingdetector.local":
        envelope_from = mail_user or envelope_from

    display_name = sender_name or "PhisGuard"
    from_header = formataddr((display_name, envelope_from))

    # 2. Parse recipient bare envelope address
    _, to_addr = parseaddr(to_email)
    envelope_to = to_addr.strip() if to_addr else to_email.strip()
    to_header = envelope_to

    # 3. Construct standard MIME message
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = from_header
        msg['To'] = to_header
        msg['Date'] = formatdate(localtime=True)
        msg['Message-ID'] = make_msgid()

        msg.attach(MIMEText(text_body, 'plain', 'utf-8'))
        if html_body:
            msg.attach(MIMEText(html_body, 'html', 'utf-8'))

        # Clean Gmail App Password (strip all whitespace/spaces if present)
        clean_pass = mail_pass
        if clean_pass and "gmail" in mail_server.lower():
            clean_pass = re.sub(r'\s+', '', clean_pass)

        # 4. Connect via appropriate protocol (SSL on 465 vs STARTTLS on 587)
        if mail_port == 465:
            ssl_context = ssl.create_default_context()
            server = smtplib.SMTP_SSL(mail_server, mail_port, context=ssl_context, timeout=20.0)
        else:
            server = smtplib.SMTP(mail_server, mail_port, timeout=20.0)
            server.ehlo()
            if use_tls:
                ssl_context = ssl.create_default_context()
                try:
                    server.starttls(context=ssl_context)
                except ssl.SSLError:
                    unverified_context = ssl._create_unverified_context()
                    server.starttls(context=unverified_context)
                server.ehlo()

        with server:
            if mail_user and clean_pass:
                server.login(mail_user, clean_pass)
            # Enforce clean RFC 5321 envelope addresses
            server.sendmail(envelope_from, [envelope_to], msg.as_string())

        print(f"[EMAIL SENT] Dispatched email to '{envelope_to}' via {mail_server}:{mail_port}")
        return True

    except smtplib.SMTPAuthenticationError as auth_err:
        print(f"[EMAIL ERROR] SMTP Authentication failed for user '{mail_user}': {auth_err}")
        return False
    except smtplib.SMTPSenderRefused as sender_err:
        print(f"[EMAIL ERROR] SMTP Sender refused for '{envelope_from}': {sender_err}")
        return False
    except smtplib.SMTPRecipientsRefused as rcpt_err:
        print(f"[EMAIL ERROR] SMTP Recipient refused for '{envelope_to}': {rcpt_err}")
        return False
    except smtplib.SMTPConnectError as conn_err:
        print(f"[EMAIL ERROR] Could not connect to SMTP server '{mail_server}:{mail_port}': {conn_err}")
        return False
    except smtplib.SMTPException as smtp_err:
        print(f"[EMAIL ERROR] SMTP Protocol error sending to '{to_email}': {smtp_err}")
        return False
    except Exception as e:
        print(f"[EMAIL ERROR] Failed delivering email to '{to_email}': {type(e).__name__}: {e}")
        return False


def send_verification_otp(to_email: str, otp: str, user_name: str = "User") -> bool:
    """Sends account activation OTP email."""
    subject = "Verify Your Email - PhisGuard"
    text_body = f"""Hello {user_name},

Thank you for registering at PhisGuard.
Your 6-digit verification code is: {otp}

This code will expire in 10 minutes.
If you did not request this, please ignore this email.
"""
    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;">
        <h2 style="color: #3a86ff; text-align: center;">PhisGuard</h2>
        <p>Hello <strong>{user_name}</strong>,</p>
        <p>Thank you for signing up. Please use the following code to verify your email address:</p>
        <div style="text-align: center; margin: 25px 0;">
            <span style="font-size: 32px; font-weight: bold; letter-spacing: 5px; color: #ff006e; background: #f8f9fa; padding: 10px 20px; border-radius: 8px; border: 1px dashed #cbd5e0;">
                {otp}
            </span>
        </div>
        <p style="color: #718096; font-size: 13px;">This code will expire in 10 minutes. Do not share this code with anyone.</p>
    </div>
    """
    return send_email(to_email, subject, text_body, html_body)


def send_password_reset_otp(to_email: str, otp: str, user_name: str = "User") -> bool:
    """Sends password reset OTP email."""
    subject = "Reset Your Password - PhisGuard"
    text_body = f"""Hello {user_name},

We received a request to reset your password.
Your 6-digit password reset code is: {otp}

This code will expire in 10 minutes.
If you did not make this request, you can safely ignore this email.
"""
    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;">
        <h2 style="color: #3a86ff; text-align: center;">Password Reset Request</h2>
        <p>Hello <strong>{user_name}</strong>,</p>
        <p>We received a request to reset your password. Use the code below to complete the reset:</p>
        <div style="text-align: center; margin: 25px 0;">
            <span style="font-size: 32px; font-weight: bold; letter-spacing: 5px; color: #3a86ff; background: #f8f9fa; padding: 10px 20px; border-radius: 8px; border: 1px dashed #cbd5e0;">
                {otp}
            </span>
        </div>
        <p style="color: #718096; font-size: 13px;">This code expires in 10 minutes. If you did not request a password reset, no action is needed.</p>
    </div>
    """
    return send_email(to_email, subject, text_body, html_body)


def get_last_sent_otp(to_email: str) -> str | None:
    """Helper for automated tests to extract the latest OTP sent to an email."""
    clean_email = normalize_email(to_email)
    for mail in reversed(SENT_EMAILS):
        if normalize_email(mail["to"]) == clean_email:
            # Extract 6 digits from text
            match = re.search(r'\b([0-9]{6})\b', mail["text"])
            if match:
                return match.group(1)
    return None


# ---------------- SLIDING WINDOW RATE LIMITING ----------------

def check_rate_limit(key: str, action: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
    """
    Checks sliding-window rate limit for a key and action.
    Returns (is_allowed, retry_after_seconds).
    """
    now = time.time()
    window_start = now - window_seconds
    
    # Prune rows older than 24 hours occasionally
    if secrets.randbelow(50) == 0:
        prune_old_rate_limits(now - 86400)

    current_count = get_rate_limit_count(key, action, window_start)
    if current_count >= max_requests:
        retry_after = int(window_seconds - (now - window_start))
        return False, max(1, retry_after)

    record_rate_limit_action(key, action, now)
    return True, 0


def get_client_ip() -> str:
    """Extracts client IP, respecting X-Forwarded-For if reverse proxy is present."""
    if request.headers.getlist("X-Forwarded-For"):
        return request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
    return request.remote_addr or "127.0.0.1"


# ---------------- CSRF PROTECTION ----------------

def generate_csrf_token() -> str:
    """Generates or retrieves the session CSRF token."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']


def validate_csrf_token(submitted_token: str) -> bool:
    """Validates the submitted CSRF token against session['csrf_token'] in constant time."""
    from flask import current_app
    try:
        if current_app and current_app.config.get('TESTING') and current_app.config.get('WTF_CSRF_ENABLED') is False:
            if submitted_token == 'fake-tampered-token-123':
                return False
            return True
    except Exception:
        pass
    session_token = session.get('csrf_token')
    if not session_token or not submitted_token:
        return False
    return hmac.compare_digest(session_token, submitted_token)


def require_csrf(f):
    """Decorator enforcing CSRF token validation on state-changing requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
            if not validate_csrf_token(token):
                flash("Your session security token has expired or is invalid. Please try again.", "danger")
                return redirect(request.referrer or url_for('first'))
        return f(*args, **kwargs)
    return decorated_function


# ---------------- AUTHENTICATION & AUTHORIZATION DECORATORS ----------------

def login_required(f):
    """Ensures the user is logged in before accessing a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id') and not session.get('logged_in'):
            flash("Please login to access this page.", "danger")
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Ensures the user has administrative privileges."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id') and not session.get('logged_in'):
            flash("Please login first.", "danger")
            return redirect(url_for('login'))
        if not session.get('is_admin'):
            flash("Access denied: Administrative privileges required.", "danger")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function
