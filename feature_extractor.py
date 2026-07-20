import re
import socket
import urllib.parse
import ipaddress

class URLFeatureExtractor:

    SHORTENING_SERVICES = [
        'bit.ly','goo.gl','tinyurl.com','ow.ly','t.co','tiny.cc',
        'is.gd','buff.ly','adf.ly','su.pr','lnkd.in','db.tt',
        'qr.ae','ity.im','q.gs','po.st','bc.vc','u.to','j.mp',
    ]

    def extract(self, url):
        if not url.startswith(('http://','https://')):
            url = 'http://' + url
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        full_url = url.lower()
        features = []
        explanations = {}

        def add(name, val, label, desc):
            features.append(val)
            status = 'danger' if val == -1 else ('warning' if val == 0 else 'safe')
            explanations[name] = {
                'value': val, 'label': label,
                'desc': desc, 'status': status
            }

        # 1. IP Address in URL
        host = domain.split(':')[0]
        try:
            ipaddress.ip_address(host)
            add('having_IP_Address', -1, 'IP Address in URL',
                'URL uses raw IP instead of domain — classic phishing tactic')
        except ValueError:
            add('having_IP_Address', 1, 'IP Address in URL',
                'Domain name used (not raw IP) — normal behavior')

        # 2. URL Length
        l = len(url)
        val = 1 if l < 54 else (0 if l <= 75 else -1)
        add('URL_Length', val, 'URL Length',
            f'URL is {l} chars. Phishing URLs are often very long.')

        # 3. Shortening Service
        val = -1 if any(s in domain for s in self.SHORTENING_SERVICES) else 1
        add('Shortining_Service', val, 'URL Shortener Used',
            'Shortening services hide the real destination URL')

        # 4. @ Symbol
        val = -1 if '@' in url else 1
        add('having_At_Symbol', val, '@ Symbol in URL',
            'Browser ignores everything before @ — used to disguise phishing URLs')

        # 5. Double Slash Redirect
        val = -1 if full_url.rfind('//') > 7 else 1
        add('double_slash_redirecting', val, 'Double Slash Redirect',
            'Double slash after protocol position indicates redirection attempt')

        # 6. Dash in Domain
        val = -1 if '-' in domain else 1
        add('Prefix_Suffix', val, 'Prefix/Suffix (-) in Domain',
            'Dashes in domain (e.g. paypal-secure.com) mimic legitimate sites')

        # 7. Subdomains
        d = re.sub(r'^www\.', '', domain)
        dots = d.count('.')
        val = 1 if dots == 1 else (0 if dots == 2 else -1)
        add('having_Sub_Domain', val, 'Excessive Subdomains',
            'Multiple subdomains can hide the real malicious domain')

        # 8. SSL
        val = 1 if parsed.scheme == 'https' else -1
        add('SSLfinal_State', val, 'SSL Certificate (HTTPS)',
            'Lack of HTTPS is a phishing indicator')

        # 9. Domain Registration Length
        add('Domain_registeration_length', 0, 'Domain Registration Length',
            'Short-term registrations are common in phishing')

        # 10. Favicon
        add('Favicon', 1, 'Favicon Source',
            'Favicon loaded from external domain is suspicious')

        # 11. Non-standard Port
        val = -1 if (parsed.port and parsed.port not in [80,443,8080,8443]) else 1
        add('port', val, 'Non-standard Port',
            f'Port: {parsed.port or "standard"}. Unusual ports indicate malicious servers.')

        # 12. HTTPS in domain name
        val = -1 if 'https' in domain else 1
        add('HTTPS_token', val, '"https" in Domain Name',
            'Writing "https" in domain tricks users into trusting it')

        # 13-30. Content-based features
        content = [
            ('Request_URL',            1, 'External Resource URLs',
             'Ratio of resources loaded from external domains'),
            ('URL_of_Anchor',          0, 'Anchor URL Domain Match',
             'Anchor tags pointing to different domains'),
            ('Links_in_tags',          1, 'Links in Meta/Script Tags',
             'Links in script/meta tags checked against domain'),
            ('SFH',                    1, 'Server Form Handler',
             'Form action pointing to external domain is suspicious'),
            ('Submitting_to_email',    1, 'Form Submits to Email',
             'Forms submitting to email are suspicious'),
            ('Abnormal_URL',           1, 'Abnormal URL Structure',
             "URL structure doesn't match WHOIS hostname"),
            ('Redirect',               0, 'Excessive Redirects',
             'More than 4 redirects is suspicious'),
            ('on_mouseover',           1, 'Mouseover Address Change',
             'JS changes status bar on mouseover'),
            ('RightClick',             1, 'Right-Click Disabled',
             'Right-click disabled via JavaScript'),
            ('popUpWidnow',            1, 'Pop-up Window',
             'Pop-ups requesting credentials'),
            ('Iframe',                 1, 'Invisible iFrame',
             'Invisible iFrame loading malicious content'),
            ('age_of_domain',          0, 'Domain Age',
             'Domain age under 6 months is suspicious'),
            ('DNSRecord',              1, 'DNS Record',
             'No DNS record strongly indicates phishing'),
            ('web_traffic',            0, 'Web Traffic Rank',
             'Low traffic suggests newly created domain'),
            ('Page_Rank',              0, 'PageRank Score',
             'Low PageRank means unknown/untrusted domain'),
            ('Google_Index',           1, 'Google Indexed',
             'Whether page appears in Google results'),
            ('Links_pointing_to_page', 0, 'Backlinks Count',
             'Few backlinks suggest domain is new'),
            ('Statistical_report',     1, 'Blacklist Check',
             'Domain appears in phishing statistical reports'),
        ]

        # Live DNS check
        try:
            socket.gethostbyname(host)
            dns_val = 1
        except:
            dns_val = -1

        for name, default, label, desc in content:
            val = dns_val if name == 'DNSRecord' else default
            add(name, val, label, desc)

        return features, explanations


def compute_risk_score(explanations):
    weights = {
        'having_IP_Address': 5.0,
        'having_At_Symbol':  5.0,
        'Shortining_Service':4.0,
        'HTTPS_token':       4.0,
        'DNSRecord':         4.0,
        'SSLfinal_State':    3.0,
        'Prefix_Suffix':     3.0,
        'double_slash_redirecting': 3.0,
        'having_Sub_Domain': 2.5,
        'URL_Length':        2.0,
    }
    total_w, total_score = 0, 0
    for name, info in explanations.items():
        w    = weights.get(name, 1.0)
        risk = {1: 0, 0: 40, -1: 100}[info['value']]
        total_w     += w
        total_score += risk * w
    return round(total_score / total_w, 1) if total_w else 40.0


def get_risk_category(score):
    if score < 12:   return "SAFE",       "#22c55e"
    elif score < 25: return "SUSPICIOUS", "#f59e0b"
    elif score < 40: return "HIGH RISK",  "#ef4444"
    else:            return "PHISHING",   "#dc2626"