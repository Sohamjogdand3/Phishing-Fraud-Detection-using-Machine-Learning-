"""
Flask Application for Phishing URL Detection Website.

Serves the user interface, manages complete email-based authentication
(Signup, Email OTP Verification, Login, Forgot & Reset Password),
processes URL submissions, and coordinates feature extraction and machine learning inference.
"""

from datetime import datetime, timedelta
import os
import pickle
import warnings

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
                            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                                v = v[1:-1].strip()
                            if override or (k not in os.environ) or (not os.environ[k]):
                                os.environ[k] = v
                return cand
            except Exception:
                pass
    return None

load_dotenv(override=False)
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
import numpy as np

warnings.filterwarnings('ignore')

from feature import FeatureExtraction, normalize_url
from db import (
    create_user,
    decrement_otp_attempts,
    get_active_otp,
    get_user_by_email,
    get_user_by_id,
    mark_otp_used,
    store_otp,
    update_user_password,
    update_user_profile,
    update_user_verification,
)
from auth import (
    admin_required,
    check_rate_limit,
    generate_csrf_token,
    generate_otp,
    get_client_ip,
    hash_otp,
    hash_password,
    is_valid_email,
    login_required,
    normalize_email,
    send_password_reset_otp,
    send_verification_otp,
    validate_csrf_token,
    validate_password_strength,
    verify_otp_hash,
    verify_password,
)

app = Flask(__name__)

# Security & Session Configuration
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "phishing-detector-secure-session-key-2026")
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=4)

# Admin credentials from environment (defaults to admin/admin if not set)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")

# Context Processor for CSRF token in templates
@app.context_processor
def inject_template_globals():
    return dict(csrf_token=generate_csrf_token)


# Safe Model Loading with Schema Contract Verification
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
gbc = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            gbc = pickle.load(f)
            # Verify contract
            if hasattr(gbc, 'n_features_in_') and gbc.n_features_in_ != 30:
                print(f"WARNING: Model expects {gbc.n_features_in_} features, expected 30.")
            if hasattr(gbc, 'classes_'):
                print(f"Model loaded successfully with classes: {gbc.classes_}")
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Failed to load model artifact: {e}")
    gbc = None


# ---------------- NAVIGATION & CORE ROUTES ----------------

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')


@app.route('/performance')
@login_required
def performance():
    return render_template('performance.html')


@app.route('/chart')
@login_required
def chart():
    return render_template('chart.html')


@app.route('/index')
@login_required
def index():
    return render_template('index.html')


# ---------------- USER AUTHENTICATION: SIGNUP ----------------

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if session.get('user_id') or session.get('logged_in'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        # 1. CSRF Token Validation
        csrf = request.form.get('csrf_token')
        if not validate_csrf_token(csrf):
            flash("Your session security token has expired or is invalid. Please try again.", "danger")
            return render_template('signup.html', full_name=request.form.get('full_name'), email=request.form.get('email'))

        # 2. Rate Limiting on IP
        client_ip = get_client_ip()
        allowed, retry_sec = check_rate_limit(client_ip, "signup", max_requests=10, window_seconds=3600)
        if not allowed:
            flash(f"Too many registration attempts. Please try again after {retry_sec} seconds.", "danger")
            return render_template('signup.html', full_name=request.form.get('full_name'), email=request.form.get('email'))

        full_name = request.form.get('full_name', '').strip()
        email = normalize_email(request.form.get('email', ''))
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # 3. Input Validation
        if not full_name or len(full_name) < 2:
            flash("Please enter a valid full name (minimum 2 characters).", "danger")
            return render_template('signup.html', full_name=full_name, email=email)

        if not is_valid_email(email):
            flash("Please enter a valid email address.", "danger")
            return render_template('signup.html', full_name=full_name, email=email)

        if password != confirm_password:
            flash("Passwords do not match. Please re-enter your password.", "danger")
            return render_template('signup.html', full_name=full_name, email=email)

        is_strong, err_msg = validate_password_strength(password)
        if not is_strong:
            flash(err_msg, "danger")
            return render_template('signup.html', full_name=full_name, email=email)

        # 4. Check for Duplicate / Existing User
        existing_user = get_user_by_email(email)
        if existing_user:
            if existing_user['is_verified']:
                flash("An account with that email address already exists. Please log in.", "warning")
                return redirect(url_for('login', email=email))
            else:
                # Update full name & password hash for existing unverified registration and send fresh OTP
                pw_hash = hash_password(password)
                update_user_profile(existing_user['id'], full_name, pw_hash)
                user_id = existing_user['id']
        else:
            # 5. Create Inactive / Unverified User Record
            pw_hash = hash_password(password)
            user_id = create_user(full_name=full_name, email=email, password_hash=pw_hash, is_verified=0)

        # 6. Generate, Hash, and Store 6-Digit Verification OTP
        otp = generate_otp()
        otp_hash = hash_otp(otp)
        expires_at = datetime.utcnow() + timedelta(minutes=10)
        store_otp(user_id=user_id, otp_hash=otp_hash, purpose="email_verification", expires_at=expires_at)

        # 7. Dispatch Verification Email
        email_sent = send_verification_otp(to_email=email, otp=otp, user_name=full_name)
        session['pending_verification_email'] = email

        if email_sent:
            flash("Registration successful! A 6-digit verification code has been sent to your email.", "success")
        else:
            flash(
                "Your account was created, but we could not deliver the 6-digit verification code to your email. "
                "Please check your server SMTP settings or click 'Resend Verification Code' below.",
                "warning"
            )

        return redirect(url_for('verify_otp', email=email))

    return render_template('signup.html')


# ---------------- USER AUTHENTICATION: EMAIL OTP VERIFICATION ----------------

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    if session.get('user_id') or session.get('logged_in'):
        return redirect(url_for('index'))

    email = normalize_email(request.args.get('email') or session.get('pending_verification_email', ''))

    if request.method == 'POST':
        # 1. CSRF Token Validation
        csrf = request.form.get('csrf_token')
        if not validate_csrf_token(csrf):
            flash("Your session security token has expired or is invalid. Please try again.", "danger")
            return render_template('verify_otp.html', email=email)

        # 2. Rate Limiting on IP
        client_ip = get_client_ip()
        allowed, retry_sec = check_rate_limit(client_ip, "verify_otp", max_requests=20, window_seconds=900)
        if not allowed:
            flash(f"Too many verification attempts. Please wait {retry_sec} seconds before trying again.", "danger")
            return render_template('verify_otp.html', email=email)

        submitted_email = normalize_email(request.form.get('email', '') or email)
        otp_code = request.form.get('otp', '').strip()

        if not submitted_email or not otp_code:
            flash("Please enter both your email address and the 6-digit verification code.", "danger")
            return render_template('verify_otp.html', email=submitted_email)

        user = get_user_by_email(submitted_email)
        if not user:
            flash("No registration found with that email. Please create an account.", "danger")
            return redirect(url_for('signup'))

        if user['is_verified']:
            flash("Your account is already verified! Please log in.", "info")
            return redirect(url_for('login', email=submitted_email))

        active_otp = get_active_otp(user['id'], "email_verification")
        if not active_otp:
            flash("No active verification code found. Please request a new code below.", "danger")
            return render_template('verify_otp.html', email=submitted_email)

        # 3. Check OTP Expiry
        try:
            exp_time = datetime.fromisoformat(active_otp['expires_at'])
            if exp_time < datetime.utcnow():
                mark_otp_used(active_otp['id'])
                flash("Your verification code has expired. Please request a new code.", "danger")
                return render_template('verify_otp.html', email=submitted_email)
        except Exception:
            pass

        # 4. Verify OTP Constant-Time Hash
        if not verify_otp_hash(otp_code, active_otp['otp_hash']):
            attempts_left = decrement_otp_attempts(active_otp['id'])
            if attempts_left > 0:
                flash(f"Invalid verification code. You have {attempts_left} attempts remaining.", "danger")
            else:
                flash("Maximum verification attempts reached. This code is now invalid. Please request a new code.", "danger")
            return render_template('verify_otp.html', email=submitted_email)

        # 5. Successful Verification: Activate Account & Establish Session
        mark_otp_used(active_otp['id'])
        update_user_verification(user['id'], 1)
        session.pop('pending_verification_email', None)

        session.clear()
        session['user_id'] = user['id']
        session['user_name'] = user['full_name']
        session['user_email'] = user['email']
        session['logged_in'] = True
        session['is_admin'] = bool(user['is_admin'])

        flash("Account verified successfully! Welcome to PhisGuard.", "success")
        return redirect(url_for('index'))

    return render_template('verify_otp.html', email=email)


# ---------------- USER AUTHENTICATION: RESEND OTP ----------------

@app.route('/resend-otp', methods=['POST'])
def resend_otp():
    csrf = request.form.get('csrf_token')
    if not validate_csrf_token(csrf):
        flash("Your session security token has expired or is invalid. Please try again.", "danger")
        return redirect(request.referrer or url_for('first'))

    email = normalize_email(request.form.get('email', ''))
    purpose = request.form.get('purpose', 'email_verification')

    if not email:
        flash("Please provide an email address to resend the code.", "warning")
        return redirect(url_for('login'))

    # Rate limiting on resend requests per email & purpose (max 10 per 15 minutes)
    rate_key = f"{email}:resend:{purpose}"
    allowed, retry_sec = check_rate_limit(rate_key, "resend_otp", max_requests=10, window_seconds=900)
    if not allowed:
        flash(f"Please wait {retry_sec} seconds before requesting another code.", "warning")
        if purpose == "email_verification":
            return redirect(url_for('verify_otp', email=email))
        return redirect(url_for('reset_password', email=email))

    user = get_user_by_email(email)
    if user:
        if purpose == "email_verification" and user['is_verified']:
            flash("Your account is already verified! Please log in.", "info")
            return redirect(url_for('login', email=email))

        # Generate, Hash, and Store New OTP (supersedes any active previous OTPs)
        otp = generate_otp()
        otp_hash = hash_otp(otp)
        expires_at = datetime.utcnow() + timedelta(minutes=10)
        store_otp(user_id=user['id'], otp_hash=otp_hash, purpose=purpose, expires_at=expires_at)

        if purpose == "email_verification":
            email_sent = send_verification_otp(email, otp, user['full_name'])
            if email_sent:
                flash("A new 6-digit verification code has been sent to your email.", "info")
            else:
                flash("Failed to send verification code due to an SMTP delivery error. Please check server settings.", "danger")
            return redirect(url_for('verify_otp', email=email))
        else:
            email_sent = send_password_reset_otp(email, otp, user['full_name'])
            if email_sent:
                flash("A new password reset code has been sent to your email.", "info")
            else:
                flash("Failed to send password reset code due to an SMTP delivery error. Please check server settings.", "danger")
            return redirect(url_for('reset_password', email=email))

    # Generic response if user does not exist
    flash("If an account exists for that email, a new code has been sent.", "info")
    if purpose == "email_verification":
        return redirect(url_for('verify_otp', email=email))
    return redirect(url_for('reset_password', email=email))


# ---------------- USER AUTHENTICATION: LOGIN ----------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id') or session.get('logged_in'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        # 1. CSRF Token Validation
        csrf = request.form.get('csrf_token')
        if not validate_csrf_token(csrf):
            flash("Your session security token has expired or is invalid. Please try again.", "danger")
            return render_template('login.html', email=request.form.get('email') or request.form.get('uname'))

        # 2. Rate Limiting on IP (10 attempts per 5 minutes)
        client_ip = get_client_ip()
        allowed, retry_sec = check_rate_limit(client_ip, "login", max_requests=10, window_seconds=300)
        if not allowed:
            flash(f"Too many failed login attempts. Please wait {retry_sec} seconds before trying again.", "danger")
            return render_template('login.html', email=request.form.get('email') or request.form.get('uname'))

        login_identifier = (request.form.get('email', '') or request.form.get('uname', '')).strip()
        password = request.form.get('pwd', '').strip()

        if not login_identifier or not password:
            flash("Please enter both your email/username and password.", "danger")
            return render_template('login.html', email=login_identifier)

        # 3. Legacy Admin Fallback (preserves backward compatibility with existing tests/admin scripts)
        if login_identifier == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session.clear()
            session['logged_in'] = True
            session['is_admin'] = True
            session['user_name'] = "Administrator"
            session['user_email'] = "admin@system.local"
            flash('Login successful!', 'success')
            next_url = request.args.get('next')
            return redirect(next_url if next_url and next_url.startswith('/') else url_for('index'))

        # 4. Email-Based User Authentication
        user = get_user_by_email(login_identifier)
        if not user or not verify_password(user['password_hash'], password):
            flash("Invalid email or password. Please check your credentials.", "danger")
            return render_template('login.html', email=login_identifier)

        # 5. Check Email Verification Status
        if not user['is_verified']:
            # Generate and dispatch fresh verification OTP
            otp = generate_otp()
            otp_hash = hash_otp(otp)
            expires_at = datetime.utcnow() + timedelta(minutes=10)
            store_otp(user_id=user['id'], otp_hash=otp_hash, purpose="email_verification", expires_at=expires_at)
            email_sent = send_verification_otp(user['email'], otp, user['full_name'])

            session['pending_verification_email'] = user['email']
            if email_sent:
                flash("Your account email has not been verified yet. We have sent a new verification code to your email.", "warning")
            else:
                flash("Your account email is unverified, and sending a verification code failed due to an SMTP error. Please check server settings.", "warning")
            return redirect(url_for('verify_otp', email=user['email']))

        # 6. Establish Authenticated Session
        session.clear()
        session['user_id'] = user['id']
        session['user_name'] = user['full_name']
        session['user_email'] = user['email']
        session['logged_in'] = True
        session['is_admin'] = bool(user['is_admin'])

        flash(f"Welcome back, {user['full_name']}!", "success")
        next_url = request.args.get('next')
        return redirect(next_url if next_url and next_url.startswith('/') else url_for('index'))

    return render_template('login.html', email=request.args.get('email', ''))


# ---------------- USER AUTHENTICATION: FORGOT PASSWORD ----------------

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if session.get('user_id') or session.get('logged_in'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        # 1. CSRF Token Validation
        csrf = request.form.get('csrf_token')
        if not validate_csrf_token(csrf):
            flash("Your session security token has expired or is invalid. Please try again.", "danger")
            return render_template('forgot_password.html')

        # 2. Rate Limiting on IP (15 requests per 15 minutes)
        client_ip = get_client_ip()
        allowed, retry_sec = check_rate_limit(client_ip, "forgot_password", max_requests=15, window_seconds=900)
        if not allowed:
            flash(f"Too many password reset requests. Please wait {retry_sec} seconds before trying again.", "danger")
            return render_template('forgot_password.html')

        email = normalize_email(request.form.get('email', ''))

        if email and is_valid_email(email):
            user = get_user_by_email(email)
            if user:
                # Generate, Hash, and Store 6-Digit Reset OTP
                otp = generate_otp()
                otp_hash = hash_otp(otp)
                expires_at = datetime.utcnow() + timedelta(minutes=10)
                store_otp(user_id=user['id'], otp_hash=otp_hash, purpose="password_reset", expires_at=expires_at)
                email_sent = send_password_reset_otp(email, otp, user['full_name'])
                if not email_sent:
                    print(f"[AUTH WARNING] Failed to deliver password reset email to '{email}' via SMTP")

        # 3. Generic Response to Prevent Account Enumeration
        flash("If an account exists with that email address, a password reset code has been sent.", "info")
        return redirect(url_for('reset_password', email=email))

    return render_template('forgot_password.html')


# ---------------- USER AUTHENTICATION: RESET PASSWORD ----------------

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if session.get('user_id') or session.get('logged_in'):
        return redirect(url_for('index'))

    email = normalize_email(request.args.get('email', ''))

    if request.method == 'POST':
        # 1. CSRF Token Validation
        csrf = request.form.get('csrf_token')
        if not validate_csrf_token(csrf):
            flash("Your session security token has expired or is invalid. Please try again.", "danger")
            return render_template('reset_password.html', email=email)

        # 2. Rate Limiting on IP
        client_ip = get_client_ip()
        allowed, retry_sec = check_rate_limit(client_ip, "reset_password", max_requests=15, window_seconds=900)
        if not allowed:
            flash(f"Too many reset attempts. Please wait {retry_sec} seconds.", "danger")
            return render_template('reset_password.html', email=email)

        submitted_email = normalize_email(request.form.get('email', '') or email)
        otp_code = request.form.get('otp', '').strip()
        new_password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # 3. Form Validation
        if not submitted_email or not otp_code or not new_password:
            flash("All fields are required to reset your password.", "danger")
            return render_template('reset_password.html', email=submitted_email)

        if new_password != confirm_password:
            flash("Passwords do not match. Please re-enter your new password.", "danger")
            return render_template('reset_password.html', email=submitted_email)

        is_strong, err_msg = validate_password_strength(new_password)
        if not is_strong:
            flash(err_msg, "danger")
            return render_template('reset_password.html', email=submitted_email)

        user = get_user_by_email(submitted_email)
        if not user:
            flash("Invalid password reset request or account not found.", "danger")
            return render_template('reset_password.html', email=submitted_email)

        active_otp = get_active_otp(user['id'], "password_reset")
        if not active_otp:
            flash("No active password reset request found. Please request a new code.", "danger")
            return render_template('reset_password.html', email=submitted_email)

        # 4. Check OTP Expiry
        try:
            exp_time = datetime.fromisoformat(active_otp['expires_at'])
            if exp_time < datetime.utcnow():
                mark_otp_used(active_otp['id'])
                flash("Your password reset code has expired. Please request a new code.", "danger")
                return render_template('reset_password.html', email=submitted_email)
        except Exception:
            pass

        # 5. Verify OTP Constant-Time Hash
        if not verify_otp_hash(otp_code, active_otp['otp_hash']):
            attempts_left = decrement_otp_attempts(active_otp['id'])
            if attempts_left > 0:
                flash(f"Invalid reset code. You have {attempts_left} attempts remaining.", "danger")
            else:
                flash("Maximum attempts reached. This reset code is now invalid. Please request a new code.", "danger")
            return render_template('reset_password.html', email=submitted_email)

        # 6. Apply New Argon2 Password Hash & Invalidate OTP and Existing Sessions
        mark_otp_used(active_otp['id'])
        new_pw_hash = hash_password(new_password)
        update_user_password(user['id'], new_pw_hash)
        session.clear()

        flash("Your password has been successfully reset! Please log in with your new password.", "success")
        return redirect(url_for('login', email=submitted_email))

    return render_template('reset_password.html', email=email)


# ---------------- USER AUTHENTICATION: LOGOUT ----------------

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('first'))


# ---------------- PHISHING URL INFERENCE PIPELINE ----------------

@app.route("/predict", methods=["POST"])
def predict():
    raw_url = request.form.get("url", "").strip()

    # Input validation
    if not raw_url:
        flash("Please enter a valid URL to analyze.", "warning")
        return redirect(url_for('index') if (session.get('user_id') or session.get('logged_in')) else url_for('first'))

    if len(raw_url) > 2048:
        flash("Submitted URL exceeds maximum allowed length of 2048 characters.", "danger")
        return redirect(url_for('index') if (session.get('user_id') or session.get('logged_in')) else url_for('first'))

    if gbc is None:
        flash("Model artifact is unavailable. Please check server configuration.", "danger")
        return redirect(url_for('index') if (session.get('user_id') or session.get('logged_in')) else url_for('first'))

    # Normalize URL
    target_url = normalize_url(raw_url)

    try:
        # 1. Feature Extraction & Schema Contract
        extractor = FeatureExtraction(target_url)
        features = extractor.getFeaturesList()

        if len(features) != 30:
            raise ValueError(f"Schema mismatch: expected 30 features, got {len(features)}")

        # Validate allowed values {-1, 0, 1}
        for idx, val in enumerate(features):
            if val not in (-1, 0, 1):
                raise ValueError(f"Feature at index {idx} has invalid value {val}. Expected in {{-1, 0, 1}}.")

        x = np.array(features, dtype=int).reshape(1, 30)

        # 2. Model Prediction
        y_pred = int(gbc.predict(x)[0])
        probabilities = gbc.predict_proba(x)[0]
        # classes_ are [-1, 1], so index 0 = Phishing (-1), index 1 = Non-Phishing / Safe (1)
        prob_phishing = float(probabilities[0])
        prob_safe = float(probabilities[1])

        # 3. Modern Heuristic & Defense-in-Depth Assessment
        risk_reasons = list(extractor.heuristics.get("risk_flags", []))

        # Check if strong heuristics override borderlines
        is_safe = (y_pred == 1) and (prob_safe >= 0.50)

        # High-confidence malicious signals override false-negative ML classifications
        high_risk_signals = (
            extractor.heuristics.get("is_brand_impersonation") or
            extractor.heuristics.get("is_ip_host") or
            extractor.heuristics.get("has_at_symbol")
        )
        if high_risk_signals and is_safe:
            is_safe = False
            # Cap safe probability below 0.5 so result.html reflects dangerous verdict
            prob_safe = min(prob_safe, 0.15)
            risk_reasons.append("Security Alert: Malicious structural patterns detected despite legitimate protocol.")

        # If model predicted phishing, provide reason if none yet listed
        if not is_safe and not risk_reasons:
            risk_reasons.append("URL structural anomalies and domain metadata matched known phishing signatures.")

        # Pass probability to template (xx >= 0.50 is Safe, xx < 0.50 is Phishing)
        xx_val = round(prob_safe, 4) if is_safe else min(round(prob_safe, 4), 0.49)

        return render_template(
            'result.html',
            xx=xx_val,
            url=target_url,
            risk_reasons=risk_reasons
        )

    except Exception as e:
        flash(f"Analysis failed: {str(e)}", "danger")
        return redirect(url_for('index') if (session.get('user_id') or session.get('logged_in')) else url_for('first'))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
