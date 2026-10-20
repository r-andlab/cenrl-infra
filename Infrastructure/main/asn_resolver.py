"""
asn_resolver.py

Resolves a target domain to its ASN, with per-run caching.

    domain  --DNS-->  IP  --pyasn-->  ASN

DNS is the slow, failure-prone step, so:
  - every domain is resolved at most once (cached for the object's lifetime)
  - DNS failures 'fail open': the resolver returns None, and callers treat
    an unknown ASN as 'no known collision' rather than dropping the target.

The DNS function is injectable so this class can be unit-tested with a fake
resolver (no real network calls). In production you pass the default, which
uses socket.gethostbyname.
"""

import logging
import socket
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _default_dns(domain: str) -> Optional[str]:
    """Resolve a domain to a single IPv4 string, or None on failure."""
    try:
        return socket.gethostbyname(domain)
    except (socket.gaierror, socket.timeout, UnicodeError, OSError):
        return None


class AsnResolver:
    def __init__(self, caida_dat: str, dns_func: Callable[[str], Optional[str]] = None):
        """
        caida_dat : path to a pyasn .dat file (built from CAIDA pfx2as via
                    Infrastructure/main/convert_caida.py)
        dns_func  : domain -> IP-string-or-None. Defaults to socket lookup.
                    Inject a fake for testing.
        """
        import pyasn  # imported here so tests that inject dns still need the db
        self._db = pyasn.pyasn(caida_dat)
        self._dns = dns_func or _default_dns
        # domain -> ASN (or None if unresolvable). None is cached too, so we
        # don't retry a failing domain over and over within a run.
        self._cache: Dict[str, Optional[int]] = {}
        self.stats = {"dns_fail": 0, "asn_fail": 0, "resolved": 0, "cache_hits": 0}

    def asn_for(self, domain: str) -> Optional[int]:
        """Return the ASN for a domain, or None if it can't be determined."""
        if domain in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[domain]

        ip = self._dns(domain)
        if ip is None:
            self.stats["dns_fail"] += 1
            self._cache[domain] = None
            return None

        result = self._db.lookup(ip)
        asn = result[0] if result else None
        if asn is None:
            self.stats["asn_fail"] += 1
        else:
            self.stats["resolved"] += 1
        self._cache[domain] = asn
        return asn
