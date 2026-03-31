import ipaddress
import logging
import pandas as pd
import random
import threading
from typing import Dict, Set, Optional, List

logger = logging.getLogger(__name__)


class VantagePoints:
    """Pool manager for per-country vantage-point IP addresses.

    The CSV file pointed at by ``ev_file`` is expected to have at least
    ``ipv4``/``ipv6`` and ``country`` columns (other columns are ignored).
    An optional ``blocklist_file`` may contain addresses that should never be
    used.  The maximum number of candidates remembered for each country is
    controlled with ``max_size``.

    Two internally managed sets are maintained for every country:

    * **active** — currently under evaluation by a measurement job.
    * **inactive** — candidates that may be drawn later.

    The class is thread-safe; all public methods acquire an internal lock to
    protect the data structures.
    """

    def __init__(self, ev_file: str, blocklist_file: Optional[str], max_size: int):
        self.all_vantages = pd.read_csv(ev_file)
        self.blocklist: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        if blocklist_file:
            with open(blocklist_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    self.blocklist.append(
                        ipaddress.ip_network(line, strict=False)
                    )

        self.ev_file = ev_file
        self.max_size = max_size
        # country -> {"active": set(), "inactive": set()}
        self.vantages: Dict[str, Dict[str, Set[str]]] = {}
        self._lock = threading.RLock()

        self.parse_vantages()

    def _is_blocked(self, ip: str) -> bool:
        addr = ipaddress.ip_address(ip)
        return any(addr in network for network in self.blocklist)

    # ------------------------------------------------------------------
    # initialization helpers
    # ------------------------------------------------------------------
    def parse_vantages(self) -> None:
        """Populate ``self.vantages`` from :pyattr:`all_vantages` DataFrame.

        Any entry whose address is in the blocklist is skipped.  We prefer
        IPv4 addresses over IPv6, but will accept either as long as the field
        is non-empty.  After collecting all candidates we reduce each country's
        pool to ``max_size`` addresses by randomly selecting a subset.
        """
        logger.info(f"Parsing vantage points from {self.ev_file}")
        with self._lock:
            self.vantages.clear()
            for row in self.all_vantages.itertuples(index=False):
                ip = None
                if hasattr(row, "ipv4") and row.ipv4 and not pd.isna(row.ipv4):
                    ip = row.ipv4
                elif hasattr(row, "ipv6") and row.ipv6 and not pd.isna(row.ipv6):
                    ip = row.ipv6
                if not ip:
                    continue
                if self._is_blocked(ip):
                    country_val = getattr(row, "country", "unknown")
                    # logger.info("Blocked IP %s (country=%s)", ip, country_val)
                    continue

                country = getattr(row, "country", None)
                if not country:
                    continue

                entry = self.vantages.setdefault(
                    country, {"active": set(), "inactive": set()}
                )
                entry["inactive"].add(ip)

            # trim pools to max_size
            for country, entry in self.vantages.items():
                inactive = entry["inactive"]
                if len(inactive) > self.max_size:
                    entry["inactive"] = set(random.sample(list(inactive), self.max_size))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_vantage(self, country: str) -> Optional[str]:
        """Return a randomly chosen inactive vantage point and mark it active.

        If there are no candidates remaining for ``country`` this returns
        ``None``.
        """
        with self._lock:
            entry = self.vantages.get(country)
            if not entry or not entry["inactive"]:
                return None
            vp = random.choice(tuple(entry["inactive"]))
            entry["inactive"].remove(vp)
            entry["active"].add(vp)
            return vp

    def evaluate(self, country: str, vp: str, ok: bool) -> None:
        """Inform the manager about the result of testing ``vp``.

        ``ok`` should be ``True`` if the vantage point responded correctly.
        In that case ``vp`` is returned to the inactive pool for future use.
        When ``ok`` is ``False`` the vantage point is removed from both sets
        and (if possible) replaced by another inactive candidate which is
        immediately promoted to *active*.
        """
        with self._lock:
            entry = self.vantages.get(country)
            if not entry:
                return
                
            entry["active"].discard(vp)
            if ok:
                entry["inactive"].add(vp)
                return

            # bad vantage point; delete completely
            entry["inactive"].discard(vp)
            if entry["inactive"]:
                replacement = random.choice(tuple(entry["inactive"]))
                entry["inactive"].remove(replacement)
                entry["active"].add(replacement)

    def get_n_vantages(self, country: str, n: int) -> List[str]:
        """Draw up to *n* inactive VPs, move them to active, return IPs."""
        with self._lock:
            entry = self.vantages.get(country)
            if not entry:
                return []
            drawn = []
            for _ in range(n):
                if not entry["inactive"]:
                    break
                vp = random.choice(tuple(entry["inactive"]))
                entry["inactive"].remove(vp)
                entry["active"].add(vp)
                drawn.append(vp)
            return drawn

    def confirm_active(self, country: str, vp: str) -> None:
        """VP passed evaluation — ensure it remains in the active set."""
        with self._lock:
            entry = self.vantages.get(country)
            if not entry:
                return
            entry["active"].add(vp)

    def reject_vp(self, country: str, vp: str) -> Optional[str]:
        """VP failed evaluation — remove entirely and draw a replacement
        into active.  Returns the replacement IP, or ``None``."""
        with self._lock:
            entry = self.vantages.get(country)
            if not entry:
                return None
            entry["active"].discard(vp)
            entry["inactive"].discard(vp)
            if entry["inactive"]:
                replacement = random.choice(tuple(entry["inactive"]))
                entry["inactive"].remove(replacement)
                entry["active"].add(replacement)
                return replacement
            return None

    # convenience accessors for inspection / testing
    def get_active(self, country: str) -> List[str]:
        with self._lock:
            entry = self.vantages.get(country)
            return list(entry["active"]) if entry else []

    def get_inactive(self, country: str) -> List[str]:
        with self._lock:
            entry = self.vantages.get(country)
            return list(entry["inactive"]) if entry else []

    def countries(self) -> List[str]:
        with self._lock:
            return list(self.vantages.keys())
