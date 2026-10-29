"""
Automated Test Suite for Email-Based Authentication System.

Covers:
1. User registration & input validation (duplicate prevention, password strength).
2. Cryptographic password hashing (Argon2id, no plaintext passwords).
3. 6-digit OTP email generation & HMAC-SHA256 hashing in database.
4. Email verification flow (valid OTP, wrong OTP, attempt decrementing, expiry).
5. Resend OTP cooldown and superseding of old codes.
6. Login authentication (verified vs unverified, wrong credentials, legacy admin).
7. Password reset / forgot password flow (anti-enumeration, OTP consumption, re-use prevention).
8. Route access protection (@login_required) and session clearing on logout.
9. CSRF protection and sliding-window rate limiting.
"""

from datetime import datetime, timedelta
import os
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure temporary test database
os.environ["DATABASE_PATH"] = os.path.join(PROJECT_ROOT, "test_phishing_app.db")

from app import app
from auth import (
    SENT_EMAILS,
    check_rate_limit,
    generate_csrf_token,
    generate_otp,
    get_last_sent_otp,
    hash_otp,
    hash_password,
    is_valid_email,
    normalize_email,
    validate_csrf_token,
    validate_password_strength,
    verify_otp_hash,
    verify_password,
)
from db import (
    create_user,
    decrement_otp_attempts,
    get_active_otp,
    get_db_connection,
    get_user_by_email,
    init_db,
    mark_otp_used,
    store_otp,
    update_user_password,
    update_user_verification,
)


class TestAuthSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

    def setUp(self):
        # Re-initialize clean test database and isolated client for each test
        if os.path.exists(os.environ["DATABASE_PATH"]):
            try:
                os.remove(os.environ["DATABASE_PATH"])
            except Exception:
                pass
        init_db()
        SENT_EMAILS.clear()
        self.client = app.test_client()

    def tearDown(self):
        if os.path.exists(os.environ["DATABASE_PATH"]):
            try:
                os.remove(os.environ["DATABASE_PATH"])
            except Exception:
                pass

    # Helper: retrieve valid CSRF token from client session
    def get_csrf(self):
        with self.client.session_transaction() as sess:
            if 'csrf_token' not in sess:
                sess['csrf_token'] = 'valid-test-csrf-token-2026'
            return sess['csrf_token']

    # ---------------- 1. SIGNUP & VALIDATION TESTS ----------------

    def test_01_successful_user_registration(self):
        """User can sign up with valid credentials; account created unverified."""
        token = self.get_csrf()
        res = self.client.post('/signup', data={
            'csrf_token': token,
            'full_name': 'Alice Smith',
            'email': 'alice@example.com',
            'password': 'SecurePassword123',
            'confirm_password': 'SecurePassword123'
        }, follow_redirects=False)

        # Redirects to verify-otp
        self.assertEqual(res.status_code, 302)
        self.assertIn('/verify-otp', res.headers.get('Location'))

        # Check DB
        user = get_user_by_email('alice@example.com')
        self.assertIsNotNone(user)
        self.assertEqual(user['full_name'], 'Alice Smith')
        self.assertEqual(user['is_verified'], 0)
        self.assertTrue(user['password_hash'].startswith('$argon2'))
        self.assertNotEqual(user['password_hash'], 'SecurePassword123')

        # Check OTP email was sent
        last_otp = get_last_sent_otp('alice@example.com')
        self.assertIsNotNone(last_otp)
        self.assertEqual(len(last_otp), 6)
        self.assertTrue(last_otp.isdigit())

    def test_02_password_validation_rules(self):
        """Rejects short passwords and passwords missing numbers or letters."""
        # Short password
        ok, msg = validate_password_strength("Short1")
        self.assertFalse(ok)
        self.assertIn("at least 8 characters", msg)

        # No numbers
        ok, msg = validate_password_strength("AllLettersOnly")
        self.assertFalse(ok)
        self.assertIn("at least one number", msg)

        # No letters
        ok, msg = validate_password_strength("1234567890")
        self.assertFalse(ok)
        self.assertIn("at least one letter", msg)

        # Valid strong password
        ok, msg = validate_password_strength("StrongPass2026")
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_03_prevent_duplicate_verified_email_registration(self):
        """Cannot register with an email that is already registered and verified."""
        # Pre-seed verified user
        pw_hash = hash_password("OldPassword123")
        create_user("Bob Jones", "bob@example.com", pw_hash, is_verified=1)

        token = self.get_csrf()
        res = self.client.post('/signup', data={
            'csrf_token': token,
            'full_name': 'Bob Impersonator',
            'email': 'bob@example.com',
            'password': 'NewPassword123',
            'confirm_password': 'NewPassword123'
        }, follow_redirects=True)

        self.assertIn(b"An account with that email address already exists", res.data)

    def test_04_email_normalization(self):
        """Email is lowercased and whitespace trimmed consistently."""
        self.assertEqual(normalize_email("  Test.User@Example.COM  "), "test.user@example.com")
        self.assertTrue(is_valid_email("user@domain.org"))
        self.assertFalse(is_valid_email("invalid-email-address"))

    # ---------------- 2. OTP VERIFICATION TESTS ----------------

    def test_05_otp_stored_as_hash_only(self):
        """OTP is stored as HMAC-SHA256 hash in SQLite; plaintext OTP is never persisted."""
        otp = "739201"
        otp_h = hash_otp(otp)
        user_id = create_user("Charlie", "charlie@test.com", hash_password("Pass1234"), is_verified=0)
        otp_id = store_otp(user_id, otp_h, "email_verification", datetime.utcnow() + timedelta(minutes=10))

        conn = get_db_connection()
        row = conn.execute("SELECT * FROM otps WHERE id = ?;", (otp_id,)).fetchone()
        conn.close()

        # Database contains hash, not plaintext
        self.assertNotEqual(row['otp_hash'], otp)
        self.assertEqual(row['otp_hash'], otp_h)
        self.assertTrue(verify_otp_hash(otp, row['otp_hash']))

    def test_06_successful_otp_verification_activates_account(self):
        """Entering correct OTP activates user and logs them in."""
        token = self.get_csrf()
        self.client.post('/signup', data={
            'csrf_token': token,
            'full_name': 'Diana Prince',
            'email': 'diana@example.com',
            'password': 'AmazonWarrior1',
            'confirm_password': 'AmazonWarrior1'
        })
        otp = get_last_sent_otp('diana@example.com')

        token = self.get_csrf()
        res = self.client.post('/verify-otp', data={
            'csrf_token': token,
            'email': 'diana@example.com',
            'otp': otp
        }, follow_redirects=True)

        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Account verified successfully", res.data)

        # User is now verified in DB
        user = get_user_by_email('diana@example.com')
        self.assertEqual(user['is_verified'], 1)

    def test_07_wrong_otp_decrements_attempts(self):
        """Entering incorrect OTP decrements attempts_left and rejects verification."""
        token = self.get_csrf()
        self.client.post('/signup', data={
            'csrf_token': token,
            'full_name': 'Eve Hacker',
            'email': 'eve@example.com',
            'password': 'Password123',
            'confirm_password': 'Password123'
        })

        token = self.get_csrf()
        res = self.client.post('/verify-otp', data={
            'csrf_token': token,
            'email': 'eve@example.com',
            'otp': '000000'
        }, follow_redirects=True)

        self.assertIn(b"Invalid verification code. You have 4 attempts remaining", res.data)
        user = get_user_by_email('eve@example.com')
        self.assertEqual(user['is_verified'], 0)

    def test_08_expired_otp_rejected(self):
        """Expired OTP is rejected."""
        pw_hash = hash_password("Password123")
        uid = create_user("Frank", "frank@test.com", pw_hash, is_verified=0)
        # Expired 5 minutes ago
        past_time = datetime.utcnow() - timedelta(minutes=5)
        store_otp(uid, hash_otp("112233"), "email_verification", past_time)

        token = self.get_csrf()
        res = self.client.post('/verify-otp', data={
            'csrf_token': token,
            'email': 'frank@test.com',
            'otp': '112233'
        }, follow_redirects=True)

        self.assertIn(b"expired", res.data)

    def test_09_resend_otp_supersedes_old_code(self):
        """Requesting resend generates new OTP and supersedes previous OTP."""
        token = self.get_csrf()
        self.client.post('/signup', data={
            'csrf_token': token,
            'full_name': 'Grace Hopper',
            'email': 'grace@example.com',
            'password': 'CompilerQueen1',
            'confirm_password': 'CompilerQueen1'
        })
        old_otp = get_last_sent_otp('grace@example.com')

        token = self.get_csrf()
        self.client.post('/resend-otp', data={
            'csrf_token': token,
            'email': 'grace@example.com',
            'purpose': 'email_verification'
        }, follow_redirects=True)

        new_otp = get_last_sent_otp('grace@example.com')
        self.assertIsNotNone(new_otp)

        # Trying old OTP fails
        token = self.get_csrf()
        res_old = self.client.post('/verify-otp', data={
            'csrf_token': token,
            'email': 'grace@example.com',
            'otp': old_otp
        }, follow_redirects=True)
        user = get_user_by_email('grace@example.com')
        self.assertEqual(user['is_verified'], 0)

        # Trying new OTP succeeds
        token = self.get_csrf()
        res_new = self.client.post('/verify-otp', data={
            'csrf_token': token,
            'email': 'grace@example.com',
            'otp': new_otp
        }, follow_redirects=True)
        self.assertIn(b"Account verified successfully", res_new.data)
        user = get_user_by_email('grace@example.com')
        self.assertEqual(user['is_verified'], 1)

    # ---------------- 3. LOGIN & SESSION TESTS ----------------

    def test_10_unverified_account_login_rejected(self):
        """Unverified accounts cannot log in and are prompted to verify."""
        pw_hash = hash_password("Password123")
        create_user("Unverified User", "unverified@test.com", pw_hash, is_verified=0)

        token = self.get_csrf()
        res = self.client.post('/login', data={
            'csrf_token': token,
            'email': 'unverified@test.com',
            'pwd': 'Password123'
        }, follow_redirects=True)

        self.assertIn(b"has not been verified yet", res.data)

    def test_11_verified_account_login_successful(self):
        """Verified user can log in and establish session."""
        pw_hash = hash_password("ValidPassword99")
        create_user("Verified User", "verified@test.com", pw_hash, is_verified=1)

        token = self.get_csrf()
        res = self.client.post('/login', data={
            'csrf_token': token,
            'email': 'verified@test.com',
            'pwd': 'ValidPassword99'
        }, follow_redirects=True)

        self.assertIn(b"Welcome back, Verified User!", res.data)
        with self.client.session_transaction() as sess:
            self.assertEqual(sess.get('user_email'), 'verified@test.com')
            self.assertTrue(sess.get('logged_in'))

    def test_12_invalid_credentials_rejected(self):
        """Wrong password or non-existent user rejected safely."""
        pw_hash = hash_password("RealPassword1")
        create_user("Real User", "real@test.com", pw_hash, is_verified=1)

        token = self.get_csrf()
        res = self.client.post('/login', data={
            'csrf_token': token,
            'email': 'real@test.com',
            'pwd': 'WrongPassword99'
        }, follow_redirects=True)
        self.assertIn(b"Invalid email or password", res.data)

        # Non-existent user
        token = self.get_csrf()
        res2 = self.client.post('/login', data={
            'csrf_token': token,
            'email': 'ghost@test.com',
            'pwd': 'SomePassword1'
        }, follow_redirects=True)
        self.assertIn(b"Invalid email or password", res2.data)

    def test_13_legacy_admin_login_works(self):
        """Legacy admin login credentials remain supported."""
        token = self.get_csrf()
        res = self.client.post('/login', data={
            'csrf_token': token,
            'uname': 'admin',
            'pwd': 'admin'
        }, follow_redirects=True)
        self.assertIn(b"Login successful!", res.data)
        with self.client.session_transaction() as sess:
            self.assertTrue(sess.get('is_admin'))
            self.assertTrue(sess.get('logged_in'))

    # ---------------- 4. FORGOT & RESET PASSWORD TESTS ----------------

    def test_14_forgot_password_anti_enumeration(self):
        """Forgot password response is generic for both existing and non-existing accounts."""
        pw_hash = hash_password("OriginalPassword1")
        create_user("Known User", "known@test.com", pw_hash, is_verified=1)

        token = self.get_csrf()
        # Existing email
        res1 = self.client.post('/forgot-password', data={
            'csrf_token': token,
            'email': 'known@test.com'
        }, follow_redirects=True)
        self.assertIn(b"If an account exists with that email address", res1.data)

        token = self.get_csrf()
        # Non-existent email
        res2 = self.client.post('/forgot-password', data={
            'csrf_token': token,
            'email': 'nobody@test.com'
        }, follow_redirects=True)
        self.assertIn(b"If an account exists with that email address", res2.data)

    def test_15_password_reset_flow(self):
        """Complete reset flow: request OTP -> set new password -> login with new password."""
        pw_hash = hash_password("OldPassword123")
        create_user("Reset User", "reset@test.com", pw_hash, is_verified=1)

        # 1. Request reset
        token = self.get_csrf()
        self.client.post('/forgot-password', data={
            'csrf_token': token,
            'email': 'reset@test.com'
        })
        reset_otp = get_last_sent_otp('reset@test.com')
        self.assertIsNotNone(reset_otp)

        # 2. Reset password
        token = self.get_csrf()
        res = self.client.post('/reset-password', data={
            'csrf_token': token,
            'email': 'reset@test.com',
            'otp': reset_otp,
            'password': 'BrandNewPassword99',
            'confirm_password': 'BrandNewPassword99'
        }, follow_redirects=True)
        self.assertIn(b"Your password has been successfully reset", res.data)

        # 3. Old password fails
        token = self.get_csrf()
        res_old = self.client.post('/login', data={
            'csrf_token': token,
            'email': 'reset@test.com',
            'pwd': 'OldPassword123'
        }, follow_redirects=True)
        self.assertIn(b"Invalid email or password", res_old.data)

        # 4. New password succeeds
        token = self.get_csrf()
        res_new = self.client.post('/login', data={
            'csrf_token': token,
            'email': 'reset@test.com',
            'pwd': 'BrandNewPassword99'
        }, follow_redirects=True)
        self.assertIn(b"Welcome back, Reset User!", res_new.data)

    def test_16_reset_otp_cannot_be_reused(self):
        """A consumed reset OTP cannot be used a second time."""
        pw_hash = hash_password("InitPassword123")
        create_user("Replay User", "replay@test.com", pw_hash, is_verified=1)

        token = self.get_csrf()
        self.client.post('/forgot-password', data={'csrf_token': token, 'email': 'replay@test.com'})
        otp = get_last_sent_otp('replay@test.com')

        # First use
        token = self.get_csrf()
        self.client.post('/reset-password', data={
            'csrf_token': token,
            'email': 'replay@test.com',
            'otp': otp,
            'password': 'NewPassword123',
            'confirm_password': 'NewPassword123'
        })

        # Replay attempt
        token = self.get_csrf()
        res = self.client.post('/reset-password', data={
            'csrf_token': token,
            'email': 'replay@test.com',
            'otp': otp,
            'password': 'AnotherPassword123',
            'confirm_password': 'AnotherPassword123'
        }, follow_redirects=True)
        self.assertIn(b"No active password reset request found", res.data)

    # ---------------- 5. ROUTE PROTECTION & LOGOUT ----------------

    def test_17_protected_routes_require_login(self):
        """Unauthenticated requests to /index, /performance, /chart redirect to login."""
        with self.client.session_transaction() as sess:
            sess.clear()

        for route in ['/index', '/performance', '/chart']:
            res = self.client.get(route, follow_redirects=False)
            self.assertEqual(res.status_code, 302)
            self.assertIn('/login', res.headers.get('Location'))

    def test_18_logout_clears_session(self):
        """Logging out destroys session state."""
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['logged_in'] = True
            sess['user_email'] = 'test@test.com'

        res = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"You have been logged out", res.data)

        with self.client.session_transaction() as sess:
            self.assertNotIn('user_id', sess)
            self.assertNotIn('logged_in', sess)

    # ---------------- 6. SECURITY & RATE LIMITING ----------------

    def test_19_csrf_protection(self):
        """Invalid CSRF token is rejected."""
        res = self.client.post('/login', data={
            'csrf_token': 'fake-tampered-token-123',
            'email': 'any@test.com',
            'pwd': 'any'
        }, follow_redirects=True)
        self.assertIn(b"security token has expired or is invalid", res.data)

    def test_20_sliding_window_rate_limiting(self):
        """Rate limit triggers after exceeding allowed threshold."""
        key = "test_ip_192.168.1.50"
        action = "test_login"
        max_req = 3
        window = 60

        # First 3 allowed
        for _ in range(3):
            allowed, _ = check_rate_limit(key, action, max_req, window)
            self.assertTrue(allowed)

        # 4th request blocked
        allowed, retry_after = check_rate_limit(key, action, max_req, window)
        self.assertFalse(allowed)
        self.assertGreater(retry_after, 0)


if __name__ == '__main__':
    unittest.main()
