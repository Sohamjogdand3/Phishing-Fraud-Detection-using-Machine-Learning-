import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app

client = app.test_client()

print("--- 1. Testing Landing Page ---")
r = client.get('/')
assert r.status_code == 200
assert b"URL Detection" in r.data
assert b"PhisGuard" in r.data
print("Landing page rendered successfully.")

print("--- 2. Testing Unauthenticated Protected Access ---")
r = client.get('/index', follow_redirects=False)
assert r.status_code == 302
assert '/login' in r.headers['Location']
print("Protected route redirects unauthenticated user.")

print("--- 3. Testing Login Flow ---")
with client.session_transaction() as sess:
    sess['csrf_token'] = 'e2e-verification-csrf-token'

r = client.post('/login', data={'csrf_token': 'e2e-verification-csrf-token', 'uname': 'admin', 'pwd': 'admin'}, follow_redirects=True)
assert r.status_code == 200
assert b"Check Suspicious URL" in r.data
assert b"Login successful!" in r.data
print("Admin login succeeded.")

print("--- 4. Testing Static Image on Performance Page ---")
r = client.get('/performance')
assert r.status_code == 200
assert b'/static/download.png' in r.data
r_img = client.get('/static/download.png')
assert r_img.status_code == 200
print("Performance page and confusion matrix static asset verified.")

print("--- 5. Testing Chart Page ---")
r = client.get('/chart')
assert r.status_code == 200
assert b'Website Classification Distribution' in r.data
print("Chart page verified.")

print("--- 6. Testing Predict Endpoint on Legitimate URL ---")
r = client.post('/predict', data={'url': 'https://www.google.com'}, follow_redirects=True)
assert r.status_code == 200
assert b"Safe Website" in r.data
print("Prediction on https://www.google.com correctly reports Safe Website.")

print("--- 7. Testing Predict Endpoint on Modern Phishing URL ---")
r = client.post('/predict', data={'url': 'https://paypal-security-update.com'}, follow_redirects=True)
assert r.status_code == 200
assert b"Phishing Website" in r.data
assert b"Security Findings:" in r.data
print("Prediction on https://paypal-security-update.com correctly reports Phishing Website with reasons.")

print("--- 8. Testing Logout Flow ---")
r = client.get('/logout', follow_redirects=True)
assert r.status_code == 200
assert b"You have been logged out successfully." in r.data
print("Logout succeeded.")

print("\nALL END-TO-END WORKFLOWS PASSED PERFECTLY!")
