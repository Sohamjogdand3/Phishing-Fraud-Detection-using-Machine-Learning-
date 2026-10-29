# Phishing URL Detection Website

## 1. Prerequisites

- **Python Version**: Python 3.11 (tested on Python 3.11.9; compatible with Python 3.10 and 3.11).
- **Package Manager**: `pip` (Python package manager).
- **Operating System**: Windows (or compatible OS).
- **Gmail Account (Optional)**: A Gmail account with an App Password if email OTP delivery is required.

---

## 2. Create and Activate Virtual Environment (Windows)

Open a terminal in the project root directory:

### Command Prompt (cmd.exe)
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

Ensure your virtual environment is active, then upgrade `pip` and install project dependencies:

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create your local `.env` configuration file from the provided template:

### Command Prompt
```cmd
copy .env.example .env
```

### PowerShell
```powershell
Copy-Item .env.example .env
```

Open `.env` in a text editor and adjust values as needed:
- `FLASK_SECRET_KEY`: A secure random string used for session cookies and CSRF protection.
- `OTP_SECRET_SALT`: A secure random string used for hashing one-time passwords.
- `DATABASE_PATH`: SQLite database file path (defaults to `phishing_app.db`).
- `PORT`: Web server port (defaults to `5000`).

---

## 5. Configure SMTP Email Delivery (Gmail)

To enable email OTP delivery for account registration and password resets:

1. Enable **2-Step Verification** on your Google Account (`https://myaccount.google.com/security`).
2. Generate an **App Password** under **Security > 2-Step Verification > App Passwords**.
3. Update the email configuration section in `.env`:

```env
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_16_character_app_password
MAIL_DEFAULT_SENDER=your_email@gmail.com
```

*Note: If `MAIL_SERVER` or credentials are left unset, the application automatically runs in development fallback mode, outputting OTP codes to the terminal console so authentication remains functional during local testing.*

---

## 6. Initialize the Database

The SQLite database is initialized automatically with required tables upon starting the application. 

To initialize or verify the database schema manually before launching:

```cmd
python -c "import db; db.init_db(); print('Database initialized successfully.')"
```

---

## 7. Run the Application

Launch the Flask development server:

```cmd
python app.py
```

---

## 8. Open in Browser

Navigate to the local URL in your web browser:

```
http://127.0.0.1:5000
```
or
```
http://localhost:5000
```

---

## 9. Run Automated Tests

Execute the automated test suites using the active virtual environment:

### URL Feature Extraction and ML Model Pipeline Tests
```cmd
python -m unittest test_suite.py
```

### Authentication, OTP, and Security Rate-Limiting Tests
```cmd
python -m unittest test_auth_suite.py
```

### End-to-End Workflow Verification
```cmd
python e2e_verification.py
```
