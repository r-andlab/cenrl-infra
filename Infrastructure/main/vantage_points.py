import ipaddress
import logging
import pandas as pd
import random
import threading
from typing import Dict, Set, Optional, List

logger = logging.getLogger(__name__)


class VantagePoints:
    """Pool manager for per-country vantage-point IP addresses.

    Can be initialized from either:

    * ``ev_file`` — a raw EV-certs CSV with ``ipv4``/``ipv6``, ``country``,
      and optional ``port`` columns.
    * ``vp_pool_file`` — a pre-filtered pool CSV (as produced by
      ``vantage_point_stats.py``) with ``country``, ``ip``, ``port`` columns.

    An optional ``blocklist_file`` may contain CIDR ranges that should never
    be used.  An optional ``max_countries`` limits the pool to the N countries
    with the most vantage points.

    Two internally managed sets are maintained for every country:

    * **active** — currently under evaluation by a measurement job.
    * **inactive** — candidates that may be drawn later.

    Port information is stored separately and can be looked up via
    :meth:`get_port`.

    The class is thread-safe; all public methods acquire an internal lock to
    protect the data structures.
    """

    def __init__(
        self,
        ev_file: Optional[str] = None,
        vp_pool_file: Optional[str] = None,
        blocklist_file: Optional[str] = None,
        max_countries: Optional[int] = None,
        blocked_countries: Optional[List[str]] = [],
    ):
        if not ev_file and not vp_pool_file:
            raise ValueError("Either ev_file or vp_pool_file must be given")

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
        self.vp_pool_file = vp_pool_file
        self.max_countries = max_countries
        # country -> {"active": set(), "inactive": set()}
        self.vantages: Dict[str, Dict[str, Set[str]]] = {}
        # ip -> port (e.g. "1.2.3.4" -> 443)
        self._ports: Dict[str, int] = {}
        self._lock = threading.RLock()
        self.blocked_countries = set(blocked_countries)

        if vp_pool_file:
            self._parse_pool_file()
        else:
            self._parse_ev_file()

        if self.max_countries is not None:
            self._trim_to_top_countries()

    def _is_blocked(self, ip: str) -> bool:
        addr = ipaddress.ip_address(ip)
        return any(addr in network for network in self.blocklist)

    # ------------------------------------------------------------------
    # initialization helpers
    # ------------------------------------------------------------------
    def _parse_ev_file(self) -> None:
        """Populate from a raw EV-certs CSV (ipv4/ipv6, country, port)."""
        logger.info("Parsing vantage points from %s", self.ev_file)
        df = pd.read_csv(self.ev_file)
        with self._lock:
            self.vantages.clear()
            self._ports.clear()
            for row in df.itertuples(index=False):
                ip = None
                if hasattr(row, "ipv4") and row.ipv4 and not pd.isna(row.ipv4):
                    ip = str(row.ipv4).strip()
                elif hasattr(row, "ipv6") and row.ipv6 and not pd.isna(row.ipv6):
                    ip = str(row.ipv6).strip()
                if not ip:
                    continue
                if self._is_blocked(ip):
                    continue

                country = getattr(row, "country", None)
                if not country or pd.isna(country):
                    continue
                country = str(country).strip()

                port = getattr(row, "port", None)
                if port is not None and not pd.isna(port):
                    self._ports.setdefault(ip, int(port))

                entry = self.vantages.setdefault(
                    country, {"active": set(), "inactive": set()}
                )
                entry["inactive"].add(ip)

    def _parse_pool_file(self) -> None:
        """Populate from a pre-filtered pool CSV (country, ip, port).

        The pool file is already blocklist-filtered so no blocklist check
        is performed.
        """
        logger.info("Parsing vantage points from pool file %s", self.vp_pool_file)
        df = pd.read_csv(self.vp_pool_file)
        with self._lock:
            self.vantages.clear()
            self._ports.clear()
            for row in df.itertuples(index=False):
                ip = str(row.ip).strip()
                if not ip:
                    continue

                country = str(row.country).strip()
                if not country:
                    continue

                if hasattr(row, "port") and not pd.isna(row.port):
                    self._ports.setdefault(ip, int(row.port))

                entry = self.vantages.setdefault(
                    country, {"active": set(), "inactive": set()}
                )
                entry["inactive"].add(ip)

    def _trim_to_top_countries(self) -> None:
        """Keep only the top ``max_countries`` countries ranked by total VP
        count (active + inactive).  Ports for removed IPs are also cleaned up."""
        with self._lock:
            ranked = sorted(
                self.vantages.keys(),
                key=lambda c: len(self.vantages[c]["active"]) + len(self.vantages[c]["inactive"]),
                reverse=True,
            )
            keep = set(ranked[: self.max_countries])
            logger.info(
                "Trimmed to top %d countries: %s",
                self.max_countries, sorted(keep),
            )
            keep -= self.blocked_countries
            for country in list(self.vantages.keys()):
                if country not in keep:
                    entry = self.vantages.pop(country)
                    removed_ips = entry["active"] | entry["inactive"]
                    for ip in removed_ips:
                        self._ports.pop(ip, None)
            logger.info(
                "Trimmed blocked countries: %s\n%d countries remain: %s",
                self.blocked_countries, len(keep), sorted(keep),
            )

    # Keep the old name as an alias so nothing breaks
    def parse_vantages(self) -> None:
        if self.vp_pool_file:
            self._parse_pool_file()
        else:
            self._parse_ev_file()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_port(self, ip: str) -> Optional[int]:
        """Return the port associated with *ip*, or ``None`` if unknown."""
        return self._ports.get(ip)

    def get_services(self, ip: str, base_services: List[str]) -> List[str]:
        """Return *base_services* with port suffixes for non-default ports.

        For example, if the VP uses port 909 and base_services is
        ``["https"]``, the result is ``["https:909"]``.  Default ports
        (443 for https, 80 for http) are left unchanged.
        """
        _DEFAULT_PORTS = {"https": 443, "http": 80}
        port = self._ports.get(ip)
        if port is None:
            return base_services
        return [
            f"{svc}:{port}" if _DEFAULT_PORTS.get(svc) != port else svc
            for svc in base_services
        ]

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
