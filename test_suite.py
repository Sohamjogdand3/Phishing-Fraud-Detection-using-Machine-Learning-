"""
Comprehensive Automated Test Suite for Phishing URL Detection Website.

Covers all 20 required testing scenarios:
1. Valid legitimate HTTPS URL.
2. Legitimate HTTP URL.
3. URL containing a raw IP address.
4. URL with suspicious brand impersonation.
5. URL containing @.
6. URL with excessive subdomains.
7. URL without a scheme.
8. Malformed URL.
9. Empty URL.
10. Unreachable domain.
11. Slow or timed-out HTTP response.
12. Website with malformed HTML.
13. Missing or invalid model artifact.
14. Missing or incorrectly ordered feature schema.
15. Unavailable external API.
16. Unauthenticated request to protected endpoints.
17. Attempt to fetch localhost or private-network resources (SSRF).
18. Legitimate URL not falsely flagged merely due to HTTP.
19. Known phishing test URL from safe dataset.
20. Modern HTTPS phishing URL from safe dataset.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import requests

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app, gbc
from feature import (
    BLOCKED_NETWORKS,
    FeatureExtraction,
    is_ssrf_safe,
    normalize_url,
)


class TestPhishingDetectionSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    # 1. Valid legitimate HTTPS URL
    def test_01_valid_legitimate_https(self):
        url = "https://www.google.com"
        fe = FeatureExtraction(url)
        features = fe.getFeaturesList()
        self.assertEqual(len(features), 30)
        self.assertTrue(all(f in (-1, 0, 1) for f in features))
        self.assertEqual(features[7], 1)  # Hppts: 1 (HTTPS)

        # Verify model prediction
        x = np.array(features).reshape(1, 30)
        pred = gbc.predict(x)[0]
        self.assertEqual(pred, 1)  # Safe

    # 2. Legitimate HTTP URL
    def test_02_legitimate_http(self):
        url = "http://example.com"
        fe = FeatureExtraction(url)
        features = fe.getFeaturesList()
        self.assertEqual(len(features), 30)
        self.assertEqual(features[7], -1)  # Hppts: -1 (HTTP)
        # Verify that despite HTTP, legitimate domain features classify as Safe
        x = np.array(features).reshape(1, 30)
        prob = gbc.predict_proba(x)[0]
        self.assertGreater(prob[1], 0.50)  # Prob(Safe) > 50%

    # 3. URL containing a raw IP address
    def test_03_raw_ip_address_url(self):
        url = "http://93.184.216.34/login.html"
        fe = FeatureExtraction(url)
        features = fe.getFeaturesList()
        self.assertEqual(features[0], -1)  # UsingIp must be -1
        self.assertTrue(fe.heuristics["is_ip_host"])

    # 4. URL with suspicious brand impersonation
    def test_04_brand_impersonation(self):
        url = "https://paypal-account-security-update.com"
        fe = FeatureExtraction(url)
        self.assertTrue(fe.heuristics["is_brand_impersonation"])
        self.assertEqual(fe.heuristics["impersonated_brand"], "paypal")
        self.assertIn("Potential brand impersonation detected for 'paypal'.", fe.heuristics["risk_flags"])

    # 5. URL containing @
    def test_05_url_with_at_symbol(self):
        url = "https://legit-site.com@phishing-target.xyz/login"
        fe = FeatureExtraction(url)
        features = fe.getFeaturesList()
        self.assertEqual(features[3], -1)  # symbol: -1
        self.assertTrue(fe.heuristics["has_at_symbol"])

    # 6. URL with excessive subdomains
    def test_06_excessive_subdomains(self):
        url = "https://login.account.verification.update.service.com"
        fe = FeatureExtraction(url)
        features = fe.getFeaturesList()
        self.assertEqual(features[6], -1)  # SubDomains: -1
        self.assertTrue(fe.heuristics["excessive_subdomains"])

    # 7. URL without a scheme
    def test_07_url_without_scheme(self):
        raw = "google.com"
        normalized = normalize_url(raw)
        self.assertTrue(normalized.startswith("https://"))
        fe = FeatureExtraction(raw)
        self.assertTrue(fe.url.startswith("https://"))
        self.assertEqual(len(fe.getFeaturesList()), 30)

    # 8. Malformed URL
    def test_08_malformed_url(self):
        malformed = "http://:::invalid-url:::"
        fe = FeatureExtraction(malformed)
        features = fe.getFeaturesList()
        # Must return valid 30 features without unhandled exception
        self.assertEqual(len(features), 30)
        self.assertTrue(len(fe.diagnostics["errors"]) > 0)

    # 9. Empty URL
    def test_09_empty_url(self):
        fe = FeatureExtraction("")
        features = fe.getFeaturesList()
        self.assertEqual(len(features), 30)
        # Test route submission with empty URL
        res = self.client.post('/predict', data={'url': ''}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

    # 10. Unreachable domain
    def test_10_unreachable_domain(self):
        unreachable = "http://this-domain-definitely-does-not-exist-987654321.org"
        fe = FeatureExtraction(unreachable)
        features = fe.getFeaturesList()
        self.assertEqual(len(features), 30)
        self.assertEqual(features[24], -1)  # DNSRecording: -1
        self.assertFalse(fe.diagnostics["dns_resolved"])

    # 11. Slow or timed-out HTTP response
    def test_11_timeout_handling(self):
        # Verify requests.get timeout parameter is safely caught without crash
        with patch('requests.get', side_effect=requests.exceptions.Timeout("Read timeout")):
            fe = FeatureExtraction("https://www.google.com")
            self.assertEqual(len(fe.getFeaturesList()), 30)
            self.assertIn("HTTP fetch failed: Read timeout", fe.diagnostics["errors"])

    # 12. Website with malformed HTML
    def test_12_malformed_html(self):
        # Mock requests.get returning malformed/broken HTML
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"<html><body><<>>broken<a href='#'>"]
        mock_resp.encoding = 'utf-8'
        mock_resp.history = []
        with patch('requests.get', return_value=mock_resp):
            fe = FeatureExtraction("https://example.com")
            features = fe.getFeaturesList()
            self.assertEqual(len(features), 30)

    # 13. Missing or invalid model artifact
    def test_13_invalid_model_artifact(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True

        with patch('app.gbc', None):
            res = self.client.post('/predict', data={'url': 'https://example.com'}, follow_redirects=True)
            self.assertEqual(res.status_code, 200)
            self.assertIn(b"Model artifact is unavailable", res.data)

    # 14. Missing or incorrectly ordered feature schema
    def test_14_schema_contract_validation(self):
        fe = FeatureExtraction("https://example.com")
        # Tamper with features length
        fe.features = [1] * 29
        with self.assertRaises(ValueError):
            fe.getFeaturesList()

    # 15. Unavailable external API
    def test_15_unavailable_external_api(self):
        # WHOIS failure
        with patch('whois.whois', side_effect=Exception("WHOIS server down")):
            fe = FeatureExtraction("https://example.com")
            features = fe.getFeaturesList()
            self.assertEqual(len(features), 30)
            self.assertIn(features[8], (-1, 0, 1))

    # 16. Unauthenticated request to protected endpoints
    def test_16_unauthenticated_protected_endpoints(self):
        with self.client.session_transaction() as sess:
            sess.clear()

        for path in ['/index', '/performance', '/chart']:
            res = self.client.get(path, follow_redirects=False)
            self.assertEqual(res.status_code, 302)
            self.assertIn('/login', res.headers.get('Location'))

    # 17. Attempt to fetch localhost or private-network resources (SSRF)
    def test_17_ssrf_protection(self):
        ssrf_targets = [
            "localhost",
            "127.0.0.1",
            "10.0.0.5",
            "192.168.1.100",
            "169.254.169.254",
            "0.0.0.0"
        ]
        for target in ssrf_targets:
            self.assertFalse(is_ssrf_safe(target), f"SSRF guard failed for {target}")

        # Live feature extraction should block direct SSRF
        fe = FeatureExtraction("http://127.0.0.1:5000/secret")
        self.assertFalse(fe.diagnostics["ssrf_safe"])
        self.assertFalse(fe.diagnostics["http_fetched"])

        # Test SSRF via HTTP redirect to internal resource
        with patch('requests.get') as mock_get:
            mock_redirect = MagicMock()
            mock_redirect.is_redirect = True
            mock_redirect.headers = {'Location': 'http://169.254.169.254/latest/meta-data/'}
            mock_get.return_value = mock_redirect
            fe_redirect = FeatureExtraction("http://example.com/redirect-to-metadata")
            self.assertFalse(fe_redirect.diagnostics["http_fetched"])
            self.assertTrue(any("Blocked SSRF attempt via redirect" in err for err in fe_redirect.diagnostics["errors"]))

    # 18. Legitimate URL not falsely flagged merely due to HTTP
    def test_18_http_legitimate_url_handling(self):
        fe = FeatureExtraction("http://example.com")
        x = np.array(fe.getFeaturesList()).reshape(1, 30)
        prob = gbc.predict_proba(x)[0]
        # Legitimate domain features must prevail over HTTP penalty
        self.assertGreater(prob[1], 0.50)

    # 19. Known phishing test URL from safe dataset
    def test_19_known_phishing_dataset_url(self):
        # Offline verified sample from upload.csv ground-truth phishing instance
        # Row 1 of upload.csv: Result = -1 (Phishing)
        sample_phishing_features = [-1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 0, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1]
        x = np.array(sample_phishing_features).reshape(1, 30)
        pred = gbc.predict(x)[0]
        self.assertEqual(pred, -1)  # Phishing

    # 20. Modern HTTPS phishing URL from safe dataset
    def test_20_modern_https_phishing_url(self):
        phishing_https_url = "https://metamask-crypto-wallet-verify.io"
        fe = FeatureExtraction(phishing_https_url)
        self.assertTrue(fe.heuristics["is_brand_impersonation"])
        # Web app predict endpoint must report phishing
        res = self.client.post('/predict', data={'url': phishing_https_url}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Phishing Website", res.data)
        self.assertIn(b"Potential brand impersonation detected", res.data)


if __name__ == '__main__':
    unittest.main()
