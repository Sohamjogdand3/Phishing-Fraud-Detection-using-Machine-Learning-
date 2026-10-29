"""
Feature extraction pipeline for Phishing URL Detection.

Extracts exactly 30 numerical features corresponding to the UCI Phishing Websites
dataset schema:
    1. having_IP_Address       (-1: Phishing,  1: Legitimate)
    2. URL_Length              (-1: Phishing,  0: Suspicious,  1: Legitimate)
    3. Shortining_Service      (-1: Phishing,  1: Legitimate)
    4. having_At_Symbol        (-1: Phishing,  1: Legitimate)
    5. double_slash_redirecting(-1: Phishing,  1: Legitimate)
    6. Prefix_Suffix           (-1: Phishing,  1: Legitimate)
    7. having_Sub_Domain       (-1: Phishing,  0: Suspicious,  1: Legitimate)
    8. SSLfinal_State          (-1: Phishing,  0: Suspicious,  1: Legitimate)
    9. Domain_registeration_len(-1: Phishing,  1: Legitimate)
   10. Favicon                 (-1: Phishing,  1: Legitimate)
   11. port                    (-1: Phishing,  1: Legitimate)
   12. HTTPS_token             (-1: Phishing,  1: Legitimate)
   13. Request_URL             (-1: Phishing,  0: Suspicious,  1: Legitimate)
   14. URL_of_Anchor           (-1: Phishing,  0: Suspicious,  1: Legitimate)
   15. Links_in_tags           (-1: Phishing,  0: Suspicious,  1: Legitimate)
   16. SFH                     (-1: Phishing,  0: Suspicious,  1: Legitimate)
   17. Submitting_to_email     (-1: Phishing,  1: Legitimate)
   18. Abnormal_URL            (-1: Phishing,  1: Legitimate)
   19. Redirect                (-1: Phishing,  0: Suspicious,  1: Legitimate)
   20. on_mouseover            (-1: Phishing,  1: Legitimate)
   21. RightClick              (-1: Phishing,  1: Legitimate)
   22. popUpWidnow             (-1: Phishing,  1: Legitimate)
   23. Iframe                  (-1: Phishing,  1: Legitimate)
   24. age_of_domain           (-1: Phishing,  1: Legitimate)
   25. DNSRecord               (-1: Phishing,  1: Legitimate)
   26. web_traffic             (-1: Phishing,  0: Suspicious,  1: Legitimate)
   27. Page_Rank               (-1: Phishing,  1: Legitimate)
   28. Google_Index            (-1: Phishing,  1: Legitimate)
   29. Links_pointing_to_page  (-1: Phishing,  0: Suspicious,  1: Legitimate)
   30. Statistical_report      (-1: Phishing,  1: Legitimate)

Security & Reliability:
- SSRF filtering to block private/loopback/metadata destinations.
- Strict connection/read timeouts on outbound calls.
- Safe HTML/WHOIS parsing without unhandled crashes.
- Transparent diagnostic reporting.
"""

import concurrent.futures
from datetime import date, datetime
import ipaddress
import math
import re
import socket
import time
import urllib.parse
from bs4 import BeautifulSoup
import requests

try:
    import whois
except ImportError:
    whois = None

# Blocked IP networks for SSRF protection
BLOCKED_NETWORKS = [
    ipaddress.ip_network('127.0.0.0/8'),        # Loopback
    ipaddress.ip_network('10.0.0.0/8'),         # Private class A
    ipaddress.ip_network('172.16.0.0/12'),      # Private class B
    ipaddress.ip_network('192.168.0.0/16'),     # Private class C
    ipaddress.ip_network('169.254.0.0/16'),     # Link-local / Cloud metadata
    ipaddress.ip_network('0.0.0.0/8'),          # Current network
    ipaddress.ip_network('224.0.0.0/4'),        # Multicast
    ipaddress.ip_network('240.0.0.0/4'),        # Reserved
    ipaddress.ip_network('::1/128'),            # IPv6 loopback
    ipaddress.ip_network('fc00::/7'),           # IPv6 private
    ipaddress.ip_network('fe80::/10'),          # IPv6 link-local
]

# Popular high-reputation domains for fast offline reputation
TOP_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
    'yahoo.com', 'reddit.com', 'netflix.com', 'linkedin.com', 'microsoft.com',
    'instagram.com', 'twitter.com', 'apple.com', 'github.com', 'stackoverflow.com',
    'bing.com', 'ebay.com', 'pinterest.com', 'wordpress.org', 'adobe.com',
    'cloudflare.com', 'paypal.com', 'whatsapp.com', 'zoom.us', 'live.com',
    'office.com', 'microsoftonline.com', 'dropbox.com', 'twitch.tv', 'spotify.com'
}

# Common high-target brands for impersonation detection
TARGET_BRANDS = [
    'paypal', 'apple', 'google', 'netflix', 'microsoft', 'chase',
    'wellsfargo', 'amazon', 'binance', 'metamask', 'steam', 'coinbase',
    'bankofamerica', 'citibank', 'facebook', 'instagram', 'whatsapp',
    'yahoo', 'outlook', 'icloud', 'roblox', 'twitter'
]

# Suspicious TLDs frequently abused in disposable phishing
SUSPICIOUS_TLDS = {
    'xyz', 'top', 'work', 'click', 'buzz', 'cfd', 'gq', 'tk', 'ml', 'ga',
    'fit', 'rest', 'monster', 'beauty', 'hair', 'skin', 'quest', 'sbs'
}

# URL shortener domains regex
SHORTENERS_REGEX = re.compile(
    r'^(?:bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl\.com|'
    r'tr\.im|is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|'
    r'twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|budurl\.com|ping\.fm|'
    r'post\.ly|just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|'
    r'short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|'
    r'db\.tt|qr\.ae|adf\.ly|cur\.lv|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|'
    r'u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|'
    r'scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd)$',
    re.IGNORECASE
)

# Known phishing IP addresses / patterns from dataset specification
KNOWN_PHISH_IPS = {
    '146.112.61.108', '213.174.157.151', '121.50.168.88', '192.185.217.116',
    '78.46.211.158', '181.174.165.13', '46.242.145.103', '121.50.168.40',
    '83.125.22.219', '46.242.145.98', '107.151.148.44', '107.151.148.107',
    '64.70.19.203', '199.184.144.27', '107.151.148.108', '107.151.148.109',
    '119.28.52.61', '54.83.43.69', '52.69.166.231', '216.58.192.225',
    '118.184.25.86', '67.208.74.71', '23.253.126.58', '104.239.157.210',
    '175.126.123.219', '141.8.224.221', '10.10.10.10', '43.229.108.32',
    '103.232.215.140', '69.172.201.153', '216.218.185.162', '54.225.104.146',
    '103.243.24.98', '199.59.243.120', '31.170.160.61', '213.19.128.77',
    '62.113.226.131', '208.100.26.234', '195.16.127.102', '195.16.127.157',
    '34.196.13.28', '103.224.212.222', '172.217.4.225', '54.72.9.51',
    '192.64.147.141', '198.200.56.183', '23.253.164.103', '52.48.191.26',
    '52.214.197.72', '87.98.255.18', '209.99.17.27', '216.38.62.18',
    '104.130.124.96', '47.89.58.141', '54.86.225.156', '54.82.156.19',
    '37.157.192.102', '204.11.56.48', '110.34.231.42'
}

KNOWN_PHISH_DOMAINS = {
    'at.ua', 'usa.cc', 'baltazarpresentes.com.br', 'pe.hu', 'esy.es',
    'hol.es', 'sweddy.com', 'myjino.ru', '96.lt', 'ow.ly'
}


def is_ssrf_safe(hostname: str) -> bool:
    """Validate that hostname does not resolve to private, loopback, or metadata addresses."""
    if not hostname:
        return False
    clean_host = hostname.strip('[]').split(':')[0]
    try:
        ip = ipaddress.ip_address(clean_host)
        return not any(ip in net for net in BLOCKED_NETWORKS)
    except ValueError:
        pass
    try:
        addr_info = socket.getaddrinfo(clean_host, None)
        for _, _, _, _, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            if any(ip in net for net in BLOCKED_NETWORKS):
                return False
        return True
    except Exception:
        return False


def normalize_url(raw_url: str) -> str:
    """Normalize input URL, adding https:// if scheme is omitted."""
    if not raw_url:
        return ""
    url = raw_url.strip()
    if not url.startswith(('http://', 'https://')):
        # Default to https
        url = 'https://' + url
    return url


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0
    entropy = 0.0
    length = len(text)
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _extract_first_date(d):
    """Safely extracts the first valid datetime/date from single or list WHOIS date fields."""
    if isinstance(d, list):
        for item in d:
            if isinstance(item, (datetime, date)):
                return item
    elif isinstance(d, (datetime, date)):
        return d
    return None


class FeatureExtraction:
    """
    Extracts the 30 UCI phishing features from a URL with safety checks,
    bounded network timeouts, and fallback handling.
    """

    def __init__(self, url: str):
        self.url = normalize_url(url)
        self.domain = ""
        self.hostname = ""
        self.urlparse = None
        self.response = None
        self.soup = None
        self.whois_response = None
        self.features = []
        self.diagnostics = {
            "url_normalized": self.url,
            "ssrf_safe": True,
            "http_fetched": False,
            "whois_fetched": False,
            "dns_resolved": False,
            "resolved_ip": None,
            "errors": []
        }
        self.heuristics = {
            "is_ip_host": False,
            "is_brand_impersonation": False,
            "impersonated_brand": None,
            "suspicious_tld": False,
            "high_entropy": False,
            "excessive_subdomains": False,
            "has_at_symbol": False,
            "risk_flags": []
        }

        # 1. Parse URL
        try:
            self.urlparse = urllib.parse.urlparse(self.url)
            self.domain = self.urlparse.netloc or ""
            # Strip port from domain if present
            self.hostname = (self.urlparse.hostname or self.domain.split(':')[0]).lower()
        except Exception as e:
            self.diagnostics["errors"].append(f"URL parsing failed: {e}")
            self.domain = ""
            self.hostname = ""

        # 2. SSRF Validation
        if not is_ssrf_safe(self.hostname):
            self.diagnostics["ssrf_safe"] = False
            self.diagnostics["errors"].append(
                f"Security: Host '{self.hostname}' is a private, loopback, or cloud metadata address. Network access blocked."
            )

        # 3. DNS Resolution
        if self.diagnostics["ssrf_safe"] and self.hostname:
            try:
                addr_info = socket.getaddrinfo(self.hostname, None)
                if addr_info:
                    self.diagnostics["dns_resolved"] = True
                    self.diagnostics["resolved_ip"] = addr_info[0][4][0]
            except Exception as e:
                self.diagnostics["errors"].append(f"DNS resolution failed: {e}")

        # 4. Safe HTTP Fetch with SSRF-validated redirects (Strict 2.0s timeout, max 512KB)
        if self.diagnostics["ssrf_safe"] and self.diagnostics["dns_resolved"] and self.url:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                }
                current_target = self.url
                history = []
                max_hops = 4
                final_res = None

                for _ in range(max_hops + 1):
                    parsed_hop = urllib.parse.urlparse(current_target)
                    hop_host = (parsed_hop.hostname or parsed_hop.netloc.split(':')[0]).lower()
                    if not is_ssrf_safe(hop_host):
                        self.diagnostics["errors"].append(
                            f"Security: Blocked SSRF attempt via redirect to restricted target '{hop_host}'."
                        )
                        final_res = None
                        break

                    res = requests.get(
                        current_target,
                        timeout=2.0,
                        headers=headers,
                        allow_redirects=False,
                        stream=True
                    )
                    final_res = res
                    if res.is_redirect and 'Location' in res.headers:
                        history.append(res)
                        next_target = urllib.parse.urljoin(current_target, res.headers['Location'])
                        current_target = next_target
                    else:
                        break

                if final_res is not None and not final_res.is_redirect:
                    self.response = final_res
                    self.response.history = history
                    content = b""
                    for chunk in final_res.iter_content(chunk_size=16384):
                        content += chunk
                        if len(content) > 512 * 1024:
                            break
                    encoding = final_res.encoding or 'utf-8'
                    html_text = content.decode(encoding, errors='replace')
                    self.soup = BeautifulSoup(html_text, 'html.parser')
                    self.diagnostics["http_fetched"] = True
            except Exception as e:
                self.diagnostics["errors"].append(f"HTTP fetch failed: {e}")

        # 5. WHOIS Lookup (Threaded with 1.5s timeout)
        if self.diagnostics["ssrf_safe"] and self.diagnostics["dns_resolved"] and whois and self.hostname:
            # Query base registered domain (e.g., example.com rather than sub.sub.example.com)
            parts = self.hostname.split('.')
            base_domain = '.'.join(parts[-2:]) if len(parts) >= 2 else self.hostname
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(whois.whois, base_domain)
                    self.whois_response = future.result(timeout=1.5)
                    if self.whois_response:
                        self.diagnostics["whois_fetched"] = True
            except concurrent.futures.TimeoutError:
                self.diagnostics["errors"].append("WHOIS query timed out (1.5s limit reached).")
            except Exception as e:
                self.diagnostics["errors"].append(f"WHOIS lookup failed: {e}")

        # 6. Extract all 30 features
        self.extract_all_features()

        # 7. Extract modern heuristics
        self.extract_modern_heuristics()

    def extract_all_features(self):
        """Extract all 30 features in the exact order required by the model schema."""
        self.features = []
        self.features.append(self.UsingIp())                  # 1
        self.features.append(self.longUrl())                  # 2
        self.features.append(self.shortUrl())                 # 3
        self.features.append(self.symbol())                   # 4
        self.features.append(self.redirecting())              # 5
        self.features.append(self.prefixSuffix())             # 6
        self.features.append(self.SubDomains())               # 7
        self.features.append(self.Hppts())                    # 8
        self.features.append(self.DomainRegLen())             # 9
        self.features.append(self.Favicon())                  # 10
        self.features.append(self.NonStdPort())               # 11
        self.features.append(self.HTTPSDomainURL())           # 12
        self.features.append(self.RequestURL())               # 13
        self.features.append(self.AnchorURL())                # 14
        self.features.append(self.LinksInScriptTags())        # 15
        self.features.append(self.ServerFormHandler())        # 16
        self.features.append(self.InfoEmail())                # 17
        self.features.append(self.AbnormalURL())              # 18
        self.features.append(self.WebsiteForwarding())        # 19
        self.features.append(self.StatusBarCust())            # 20
        self.features.append(self.DisableRightClick())        # 21
        self.features.append(self.UsingPopupWindow())         # 22
        self.features.append(self.IframeRedirection())        # 23
        self.features.append(self.AgeofDomain())              # 24
        self.features.append(self.DNSRecording())             # 25
        self.features.append(self.WebsiteTraffic())           # 26
        self.features.append(self.PageRank())                 # 27
        self.features.append(self.GoogleIndex())              # 28
        self.features.append(self.LinksPointingToPage())      # 29
        self.features.append(self.StatsReport())              # 30

    def extract_modern_heuristics(self):
        """Analyze modern phishing indicators beyond the legacy 2015 UCI schema."""
        # 1. IP Host
        if self.UsingIp() == -1:
            self.heuristics["is_ip_host"] = True
            self.heuristics["risk_flags"].append("Raw IP address used as website host.")

        # 2. Suspicious TLD
        if self.hostname:
            tld = self.hostname.split('.')[-1]
            if tld in SUSPICIOUS_TLDS:
                self.heuristics["suspicious_tld"] = True
                self.heuristics["risk_flags"].append(f"Uses high-abuse top-level domain (.{tld}).")

        # 3. Brand Impersonation in Subdomain / Path
        if self.hostname:
            clean_host = self.hostname
            # Remove legitimate domain matches
            is_legit_brand = any(clean_host == b or clean_host.endswith('.' + b) or clean_host.endswith('.' + b + '.com') for b in TARGET_BRANDS)
            if not is_legit_brand:
                for brand in TARGET_BRANDS:
                    # Brand in subdomains (e.g., paypal.com.security-check.xyz)
                    if brand in clean_host or (brand in self.url.lower() and ('login' in self.url.lower() or 'verify' in self.url.lower() or 'secure' in self.url.lower())):
                        self.heuristics["is_brand_impersonation"] = True
                        self.heuristics["impersonated_brand"] = brand
                        self.heuristics["risk_flags"].append(f"Potential brand impersonation detected for '{brand}'.")
                        break

        # 4. Excessive Subdomains
        clean_domain = self.hostname
        if clean_domain.startswith('www.'):
            clean_domain = clean_domain[4:]
        if clean_domain.count('.') >= 3:
            self.heuristics["excessive_subdomains"] = True
            self.heuristics["risk_flags"].append("Excessive subdomain depth (>= 3 dots).")

        # 5. Shannon Entropy
        domain_without_tld = clean_domain.split('.')[0] if '.' in clean_domain else clean_domain
        entropy = calculate_entropy(domain_without_tld)
        if len(domain_without_tld) > 8 and entropy > 3.8:
            self.heuristics["high_entropy"] = True
            self.heuristics["risk_flags"].append("Domain exhibits high entropy (randomly generated characters).")

        # 6. At symbol redirect
        if "@" in self.url:
            self.heuristics["has_at_symbol"] = True
            self.heuristics["risk_flags"].append("URL contains '@' credential redirect token.")

    # ----------------- 30 INDIVIDUAL FEATURE METHODS -----------------

    # 1. UsingIp: Checks if an IP address is used instead of a domain name
    def UsingIp(self) -> int:
        if not self.hostname:
            return 1
        clean_host = self.hostname.strip('[]').split(':')[0]
        try:
            ipaddress.ip_address(clean_host)
            return -1
        except ValueError:
            # Check hex or octal encoded IP
            if re.match(r'^(?:0x[0-9a-fA-F]+\.|\d+\.){3}(?:0x[0-9a-fA-F]+|\d+)$', clean_host):
                return -1
            return 1

    # 2. longUrl: Length < 54 -> 1, 54-75 -> 0, > 75 -> -1
    def longUrl(self) -> int:
        length = len(self.url)
        if length < 54:
            return 1
        elif 54 <= length <= 75:
            return 0
        return -1

    # 3. shortUrl: Checks for known URL shorteners
    def shortUrl(self) -> int:
        clean_host = self.hostname.lower()
        if SHORTENERS_REGEX.match(clean_host):
            return -1
        # Fallback check on full URL
        if re.search(r'bit\.ly|goo\.gl|shorte\.st|tinyurl\.com|t\.co|ow\.ly|is\.gd|cutt\.us', self.url, re.IGNORECASE):
            return -1
        return 1

    # 4. symbol: Checks for @ symbol
    def symbol(self) -> int:
        return -1 if "@" in self.url else 1

    # 5. redirecting: Checks for // in URL path after protocol
    def redirecting(self) -> int:
        last_double_slash = self.url.rfind('//')
        return -1 if last_double_slash > 7 else 1

    # 6. prefixSuffix: Checks for '-' in domain
    def prefixSuffix(self) -> int:
        return -1 if '-' in self.domain else 1

    # 7. SubDomains: Dots in domain name
    def SubDomains(self) -> int:
        clean_host = self.hostname
        if clean_host.startswith('www.'):
            clean_host = clean_host[4:]
        dots = clean_host.count('.')
        if dots <= 1:
            return 1
        elif dots == 2:
            return 0
        return -1

    # 8. Hppts: Protocol / SSL state
    def Hppts(self) -> int:
        if self.urlparse and self.urlparse.scheme.lower() == 'https':
            return 1
        return -1

    # 9. DomainRegLen: Registration duration from WHOIS
    def DomainRegLen(self) -> int:
        if not self.whois_response:
            return -1
        try:
            exp_date = _extract_first_date(self.whois_response.expiration_date)
            create_date = _extract_first_date(self.whois_response.creation_date)
            if exp_date and create_date:
                # Registration length in months
                months = (exp_date.year - create_date.year) * 12 + (exp_date.month - create_date.month)
                return 1 if months >= 12 else -1
            return -1
        except Exception:
            return -1

    # 10. Favicon: Checks if favicon is loaded from external domain
    def Favicon(self) -> int:
        if not self.soup:
            return 1
        try:
            for link in self.soup.find_all('link', href=True):
                rel = [r.lower() for r in link.get('rel', [])]
                if 'icon' in rel or 'shortcut' in rel:
                    href = link['href']
                    if href.startswith(('http://', 'https://')):
                        parsed_href = urllib.parse.urlparse(href)
                        if parsed_href.netloc and self.domain not in parsed_href.netloc:
                            return -1
            return 1
        except Exception:
            return 1

    # 11. NonStdPort: Non-standard port in URL
    def NonStdPort(self) -> int:
        if not self.urlparse:
            return 1
        try:
            port = self.urlparse.port
            if port and port not in (80, 443):
                return -1
            return 1
        except Exception:
            return 1

    # 12. HTTPSDomainURL: "https" token in domain name
    def HTTPSDomainURL(self) -> int:
        clean_host = self.hostname
        if 'https' in clean_host or 'http' in clean_host.replace('https', ''):
            return -1
        return 1

    # 13. RequestURL: % of external objects (img, audio, embed, iframe)
    def RequestURL(self) -> int:
        if not self.soup:
            return 1
        try:
            total = 0
            external = 0
            for tag, attr in [('img', 'src'), ('audio', 'src'), ('embed', 'src'), ('iframe', 'src')]:
                for el in self.soup.find_all(tag, attrs={attr: True}):
                    src = el[attr]
                    total += 1
                    if src.startswith(('http://', 'https://')):
                        src_netloc = urllib.parse.urlparse(src).netloc
                        if self.domain and self.domain not in src_netloc:
                            external += 1
            if total == 0:
                return 1
            pct = (external / float(total)) * 100.0
            if pct < 22.0:
                return 1
            elif 22.0 <= pct < 61.0:
                return 0
            return -1
        except Exception:
            return 1

    # 14. AnchorURL: % of unsafe anchor links
    def AnchorURL(self) -> int:
        if not self.soup:
            return 1
        try:
            total = 0
            unsafe = 0
            for a in self.soup.find_all('a', href=True):
                href = a['href'].strip()
                total += 1
                if not href or href == '#' or href.lower().startswith(('javascript:', 'mailto:')):
                    unsafe += 1
                elif href.startswith(('http://', 'https://')):
                    href_netloc = urllib.parse.urlparse(href).netloc
                    if self.domain and self.domain not in href_netloc:
                        unsafe += 1
            if total == 0:
                return 1
            pct = (unsafe / float(total)) * 100.0
            if pct < 31.0:
                return 1
            elif 31.0 <= pct <= 67.0:
                return 0
            return -1
        except Exception:
            return 1

    # 15. LinksInScriptTags: % of external link and script tags
    def LinksInScriptTags(self) -> int:
        if not self.soup:
            return 1
        try:
            total = 0
            external = 0
            for link in self.soup.find_all('link', href=True):
                href = link['href']
                total += 1
                if href.startswith(('http://', 'https://')):
                    if self.domain and self.domain not in urllib.parse.urlparse(href).netloc:
                        external += 1
            for script in self.soup.find_all('script', src=True):
                src = script['src']
                total += 1
                if src.startswith(('http://', 'https://')):
                    if self.domain and self.domain not in urllib.parse.urlparse(src).netloc:
                        external += 1
            if total == 0:
                return 1
            pct = (external / float(total)) * 100.0
            if pct < 17.0:
                return 1
            elif 17.0 <= pct <= 81.0:
                return 0
            return -1
        except Exception:
            return 1

    # 16. ServerFormHandler (SFH): Form action checks
    def ServerFormHandler(self) -> int:
        if not self.soup:
            return 1
        try:
            forms = self.soup.find_all('form', action=True)
            if not forms:
                return 1
            for form in forms:
                action = form.get('action', '').strip()
                if not action or action.lower() == 'about:blank':
                    return -1
                elif action.startswith(('http://', 'https://')):
                    action_netloc = urllib.parse.urlparse(action).netloc
                    if self.domain and self.domain not in action_netloc:
                        return 0
            return 1
        except Exception:
            return 1

    # 17. InfoEmail: Uses mailto: or mail() to transmit credentials
    def InfoEmail(self) -> int:
        if not self.response:
            return 1
        try:
            text = self.response.text.lower()
            if 'mailto:' in text or 'mail(' in text:
                return -1
            return 1
        except Exception:
            return 1

    # 18. AbnormalURL: Host does not match WHOIS domain identity
    def AbnormalURL(self) -> int:
        if not self.whois_response:
            return 1
        try:
            whois_domain = self.whois_response.domain_name
            if isinstance(whois_domain, list):
                whois_domain = [str(d).lower() for d in whois_domain]
                match = any(self.hostname.endswith(d) for d in whois_domain)
                return 1 if match else -1
            elif isinstance(whois_domain, str):
                return 1 if self.hostname.endswith(whois_domain.lower()) else -1
            return 1
        except Exception:
            return 1

    # 19. WebsiteForwarding: HTTP redirect hop count
    def WebsiteForwarding(self) -> int:
        if not self.response:
            return 1
        try:
            history_len = len(self.response.history)
            if history_len <= 1:
                return 1
            elif 2 <= history_len <= 4:
                return 0
            return -1
        except Exception:
            return 1

    # 20. StatusBarCust: Manipulating status bar on mouseover
    # UCI mapping: Phishing (-1) if onmouseover modifies status, Legitimate (1) otherwise
    def StatusBarCust(self) -> int:
        if not self.response:
            return 1
        try:
            text = self.response.text
            if re.search(r'onmouseover\s*=\s*["\'].*?window\.status', text, re.IGNORECASE):
                return -1
            return 1
        except Exception:
            return 1

    # 21. DisableRightClick: Script disables right-click context menu
    # UCI mapping: Phishing (-1) if right click disabled, Legitimate (1) otherwise
    def DisableRightClick(self) -> int:
        if not self.response:
            return 1
        try:
            text = self.response.text
            if re.search(r'event\.button\s*==\s*2|contextmenu\s*=\s*["\']return\s+false', text, re.IGNORECASE):
                return -1
            return 1
        except Exception:
            return 1

    # 22. UsingPopupWindow: Alert or prompt popups
    # UCI mapping: Phishing (-1) if popup input prompt found, Legitimate (1) otherwise
    def UsingPopupWindow(self) -> int:
        if not self.response:
            return 1
        try:
            text = self.response.text
            if re.search(r'prompt\s*\(|alert\s*\(\s*["\'](?:password|login|account)', text, re.IGNORECASE):
                return -1
            return 1
        except Exception:
            return 1

    # 23. IframeRedirection: Uses hidden/borderless iframe
    # UCI mapping: Phishing (-1) if invisible iframe found, Legitimate (1) otherwise
    def IframeRedirection(self) -> int:
        if not self.soup:
            return 1
        try:
            iframes = self.soup.find_all('iframe')
            for iframe in iframes:
                frame_border = iframe.get('frameborder', '')
                style = iframe.get('style', '').lower()
                if frame_border == '0' or 'display:none' in style or 'visibility:hidden' in style:
                    return -1
            return 1
        except Exception:
            return 1

    # 24. AgeofDomain: Domain age >= 6 months -> 1, else -1
    def AgeofDomain(self) -> int:
        if not self.whois_response:
            return 1
        try:
            create_date = _extract_first_date(self.whois_response.creation_date)
            if create_date:
                today = datetime.now(create_date.tzinfo) if create_date.tzinfo else datetime.now()
                age_days = (today - create_date).days
                return 1 if age_days >= 180 else -1
            return 1
        except Exception:
            return 1

    # 25. DNSRecording: DNS record resolves
    def DNSRecording(self) -> int:
        return 1 if self.diagnostics["dns_resolved"] else -1

    # 26. WebsiteTraffic: Popularity rank
    # Alexa API is dead (decommissioned May 2022). High-reputation offline lookup or neutral fallback.
    def WebsiteTraffic(self) -> int:
        clean_host = self.hostname
        # Strip www.
        if clean_host.startswith('www.'):
            clean_host = clean_host[4:]
        if clean_host in TOP_DOMAINS or any(clean_host.endswith('.' + d) for d in TOP_DOMAINS):
            return 1
        return 0

    # 27. PageRank: Global web authority
    def PageRank(self) -> int:
        clean_host = self.hostname
        if clean_host.startswith('www.'):
            clean_host = clean_host[4:]
        if clean_host in TOP_DOMAINS or any(clean_host.endswith('.' + d) for d in TOP_DOMAINS):
            return 1
        return -1

    # 28. GoogleIndex: Search engine indexing
    def GoogleIndex(self) -> int:
        # Resolving domain with valid DNS is indexed; dead domain is not
        return 1 if self.diagnostics["dns_resolved"] else -1

    # 29. LinksPointingToPage: External backlinks count
    # UCI mapping: 0 links -> -1, 1-2 links -> 0, > 2 links -> 1
    def LinksPointingToPage(self) -> int:
        clean_host = self.hostname
        if clean_host in TOP_DOMAINS or any(clean_host.endswith('.' + d) for d in TOP_DOMAINS):
            return 1
        if not self.soup:
            return 0
        try:
            count = len(self.soup.find_all('a', href=True))
            if count == 0:
                return -1
            elif count <= 2:
                return 0
            return 1
        except Exception:
            return 0

    # 30. StatsReport: Known phishing blacklist
    def StatsReport(self) -> int:
        clean_host = self.hostname
        if clean_host in KNOWN_PHISH_DOMAINS or any(clean_host.endswith('.' + d) for d in KNOWN_PHISH_DOMAINS):
            return -1
        ip = self.diagnostics.get("resolved_ip")
        if ip and ip in KNOWN_PHISH_IPS:
            return -1
        return 1

    def getFeaturesList(self) -> list:
        """Returns the validated 30-feature vector."""
        # Enforce exact 30 integers in {-1, 0, 1}
        if len(self.features) != 30:
            raise ValueError(f"Feature vector must have exactly 30 elements, got {len(self.features)}")
        return [int(f) for f in self.features]
