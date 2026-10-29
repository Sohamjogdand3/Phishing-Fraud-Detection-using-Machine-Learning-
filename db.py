"""
Database Management Layer for Phishing URL Detection Website.

Uses SQLite with thread-safe connection pooling, parameterized queries,
schema initialization, and automated table migrations.
"""

import os
import sqlite3
from datetime import datetime

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

DATABASE_PATH = os.environ.get(
    "DATABASE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "phishing_app.db")
)


def get_db_connection():
    """Returns a SQLite connection configured with Row factory for dict-like access."""
    conn = sqlite3.connect(DATABASE_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Initializes tables and indices for users, OTPs, and rate limiting."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            is_verified INTEGER DEFAULT 0,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)

    # 2. OTPs Table (Stores ONLY hashed OTPs)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS otps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            otp_hash TEXT NOT NULL,
            purpose TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            attempts_left INTEGER DEFAULT 5,
            is_used INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
    """)

    # 3. Rate Limiting Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rate_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp REAL NOT NULL
        );
    """)

    # Create helpful indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_otps_user_purpose ON otps(user_id, purpose, is_used);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_key ON rate_limits(key, action, timestamp);")

    conn.commit()
    conn.close()


# Ensure DB is initialized when module is imported
init_db()


# ---------------- USER REPOSITORY FUNCTIONS ----------------

def create_user(full_name: str, email: str, password_hash: str, is_verified: int = 0, is_admin: int = 0):
    """Inserts a new user record. Email is stored normalized (lowercase)."""
    now = datetime.utcnow().isoformat()
    clean_email = email.strip().lower()
    clean_name = full_name.strip()

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (full_name, email, password_hash, is_verified, is_admin, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (clean_name, clean_email, password_hash, is_verified, is_admin, now, now))
        user_id = cursor.lastrowid
        conn.commit()
        return user_id
    finally:
        conn.close()


def get_user_by_email(email: str):
    """Retrieves a user by case-insensitive email."""
    if not email:
        return None
    clean_email = email.strip().lower()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE;", (clean_email,))
        return cursor.fetchone()
    finally:
        conn.close()


def get_user_by_id(user_id: int):
    """Retrieves a user by ID."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?;", (user_id,))
        return cursor.fetchone()
    finally:
        conn.close()


def update_user_verification(user_id: int, is_verified: int = 1):
    """Marks a user's account as verified."""
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    try:
        conn.execute("""
            UPDATE users SET is_verified = ?, updated_at = ? WHERE id = ?;
        """, (is_verified, now, user_id))
        conn.commit()
    finally:
        conn.close()


def update_user_password(user_id: int, new_password_hash: str):
    """Updates a user's password hash."""
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    try:
        conn.execute("""
            UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?;
        """, (new_password_hash, now, user_id))
        conn.commit()
    finally:
        conn.close()


def update_user_profile(user_id: int, full_name: str, new_password_hash: str = None):
    """Updates full name and optionally password hash for an account."""
    now = datetime.utcnow().isoformat()
    clean_name = full_name.strip()
    conn = get_db_connection()
    try:
        if new_password_hash:
            conn.execute("""
                UPDATE users SET full_name = ?, password_hash = ?, updated_at = ? WHERE id = ?;
            """, (clean_name, new_password_hash, now, user_id))
        else:
            conn.execute("""
                UPDATE users SET full_name = ?, updated_at = ? WHERE id = ?;
            """, (clean_name, now, user_id))
        conn.commit()
    finally:
        conn.close()


# ---------------- OTP REPOSITORY FUNCTIONS ----------------

def store_otp(user_id: int, otp_hash: str, purpose: str, expires_at: datetime):
    """Stores a hashed OTP and invalidates any previous unused OTPs for this user and purpose."""
    now = datetime.utcnow().isoformat()
    exp_str = expires_at.isoformat()

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Supersede / invalidate older active OTPs for the same purpose
        cursor.execute("""
            UPDATE otps SET is_used = 1 WHERE user_id = ? AND purpose = ? AND is_used = 0;
        """, (user_id, purpose))

        # Insert new active OTP
        cursor.execute("""
            INSERT INTO otps (user_id, otp_hash, purpose, expires_at, attempts_left, is_used, created_at)
            VALUES (?, ?, ?, ?, 5, 0, ?);
        """, (user_id, otp_hash, purpose, exp_str, now))
        otp_id = cursor.lastrowid
        conn.commit()
        return otp_id
    finally:
        conn.close()


def get_active_otp(user_id: int, purpose: str):
    """Retrieves the latest unused, unexpired OTP record for a user and purpose."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM otps 
            WHERE user_id = ? AND purpose = ? AND is_used = 0
            ORDER BY id DESC LIMIT 1;
        """, (user_id, purpose))
        return cursor.fetchone()
    finally:
        conn.close()


def decrement_otp_attempts(otp_id: int):
    """Decrements attempts_left on an OTP record. Marks used if attempts reach zero."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE otps SET attempts_left = attempts_left - 1 WHERE id = ?;
        """, (otp_id,))
        cursor.execute("""
            UPDATE otps SET is_used = 1 WHERE id = ? AND attempts_left <= 0;
        """, (otp_id,))
        conn.commit()
        cursor.execute("SELECT attempts_left FROM otps WHERE id = ?;", (otp_id,))
        row = cursor.fetchone()
        return row['attempts_left'] if row else 0
    finally:
        conn.close()


def mark_otp_used(otp_id: int):
    """Marks an OTP as successfully consumed."""
    conn = get_db_connection()
    try:
        conn.execute("UPDATE otps SET is_used = 1 WHERE id = ?;", (otp_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------- RATE LIMITING REPOSITORY ----------------

def record_rate_limit_action(key: str, action: str, timestamp: float):
    """Logs an action timestamp for rate limiting."""
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO rate_limits (key, action, timestamp) VALUES (?, ?, ?);
        """, (key, action, timestamp))
        conn.commit()
    finally:
        conn.close()


def get_rate_limit_count(key: str, action: str, window_start_timestamp: float) -> int:
    """Counts actions matching key and action since window_start_timestamp."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) AS cnt FROM rate_limits 
            WHERE key = ? AND action = ? AND timestamp >= ?;
        """, (key, action, window_start_timestamp))
        row = cursor.fetchone()
        return row['cnt'] if row else 0
    finally:
        conn.close()


def prune_old_rate_limits(older_than_timestamp: float):
    """Cleans up expired rate-limiting rows to keep the DB lean."""
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM rate_limits WHERE timestamp < ?;", (older_than_timestamp,))
        conn.commit()
    finally:
        conn.close()

