"""
asn_aware_vantage_points.py

Drop-in extension of VantagePoints that adds ASN-diverse VP selection.

Instead of drawing VPs randomly from a flat per-country pool, this class
groups VPs by ASN within each country and round-robins across ASN groups
when drawing. This ensures selected VPs represent genuinely different
network paths rather than multiple IPs behind the same ISP.

ASN lookup priority:
  1. CAIDA pfx2as .dat file (if caida_dat path is provided) — most current
  2. 'asn' column already present in ev-certs.csv — from Censys/BigQuery
  3. 'unknown' if neither source has data for the IP

Usage (drop-in replacement for VantagePoints):

    from asn_aware_vantage_points import AsnAwareVantagePoints

    vp_pool = AsnAwareVantagePoints(
        ev_file="ev-certs.csv",
        caida_dat="ipasn_20260504.dat",  # optional but recommended
        max_countries=10,
    )

    # API is identical to VantagePoints — no other code changes needed.
    drawn = vp_pool.get_n_vantages("Australia", 3)

Generating the CAIDA .dat file from a raw pfx2as .gz:

    from convert_caida import convert
    convert("routeviews-rv2-20260504-1200_pfx2as.gz", "ipasn_20260504.dat")
"""

import logging
import random
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Set

import pandas as pd

from Infrastructure.main.as_org_resolver import AsOrgResolver

logger = logging.getLogger(__name__)

# pyasn is only needed if a caida_dat path is supplied
try:
    import pyasn as _pyasn
    _PYASN_AVAILABLE = True
except ImportError:
    _PYASN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Internal data structure
# ---------------------------------------------------------------------------
# Per country we keep:
#   _asn_pools[country][asn] = {"inactive": set(), "active": set()}
# The flat active/inactive sets from the parent design are preserved via
# _active and _inactive views that merge across ASNs — this keeps the
# reject_vp / confirm_active / get_active / get_inactive interface intact.

_UNKNOWN_ASN = "unknown"


class AsnAwareVantagePoints:
    """
    ASN-diverse vantage point pool manager.

    Public API is identical to the original VantagePoints class so it can
    be used as a drop-in replacement without changing orchestrator.py or
    any other caller.
    """

    def __init__(
        self,
        ev_file: Optional[str] = None,
        vp_pool_file: Optional[str] = None,
        blocklist_file: Optional[str] = None,
        max_countries: Optional[int] = None,
        blocked_countries: Optional[List[str]] = None,
        caida_dat: Optional[str] = None,
        guaranteed_countries: Optional[List[str]] = None,
        excluded_asns: Optional[List[str]] = None,
        as_org_path: Optional[str] = None,
        excluded_vps_file: Optional[str] = None,
    ):
        if not ev_file and not vp_pool_file:
            raise ValueError("Either ev_file or vp_pool_file must be given")

        self.blocked_countries: set = set(blocked_countries or [])
        # Countries kept regardless of max_countries volume ranking (still
        # subject to blocked_countries). Empty by default -- every existing
        # caller that doesn't pass this sees identical trimming behavior to
        # before this param existed.
        self.guaranteed_countries: set = set(guaranteed_countries or [])
        # ASNs excluded from the candidate pool entirely, at load time --
        # e.g. single-tenant corporate/cloud infrastructure (Apple, Oracle)
        # that can never be a representative national vantage point,
        # regardless of what country it's tagged with or what target it's
        # asked to test. Compared as strings since _resolve_asn() always
        # returns str. Empty by default -- no behavior change for any
        # existing caller.
        self.excluded_asns: set = set(str(a) for a in (excluded_asns or []))
        # Individual known-bad VP IPs, grown by hand over time as each one
        # gets confirmed (via WHOIS, live-probing, or
        # audit_vp_country_mismatches.py) rather than caught by a general
        # ASN- or country-level rule -- e.g. a VP whose ASN is correctly
        # registered in its claimed country but that's individually
        # unreliable, or a mismatched-country VP whose ASN is too small/
        # obscure to be worth an ASN-wide exclusion. See
        # local/excluded_vps.txt for the running list and each entry's
        # justification.
        self.excluded_vps: set = self._load_excluded_vps_file(excluded_vps_file)

        # Optional CAIDA ASN database
        self._caida_db = None
        if caida_dat:
            if not _PYASN_AVAILABLE:
                raise ImportError(
                    "pyasn is required when caida_dat is supplied. "
                    "Install it with: pip install pyasn"
                )
            logger.info("Loading CAIDA ASN database from %s", caida_dat)
            self._caida_db = _pyasn.pyasn(caida_dat)

        # Optional CAIDA AS-Organizations cross-check: excludes a VP whose
        # ASN is confirmed (not just suspected) to be registered to an
        # organization headquartered in a DIFFERENT country than the VP's
        # claimed one -- e.g. Apple Inc. (US) used as a "Hong Kong" VP.
        # Complementary to excluded_asns, not a replacement: confirmed this
        # session to catch about half of a set of known-bad VPs (the
        # mislabeled-country half); the other half (legitimately-registered
        # but low-quality hosting/VPS providers) isn't addressed by this
        # check. Fails open on unresolvable/ambiguous ASNs (see
        # AsOrgResolver.country_name_matches's docstring), same philosophy
        # as AsnResolver's DNS-failure handling elsewhere in this repo.
        self._as_org_resolver: Optional[AsOrgResolver] = None
        if as_org_path:
            self._as_org_resolver = AsOrgResolver(as_org_path)

        # country -> asn -> {"inactive": set, "active": set}
        self._asn_pools: Dict[str, Dict[str, Dict[str, Set[str]]]] = \
            defaultdict(lambda: defaultdict(lambda: {"inactive": set(), "active": set()}))

        # ip -> asn  (for quick reverse lookup)
        self._ip_asn: Dict[str, str] = {}
        # ip -> port
        self._ports: Dict[str, int] = {}

        self._lock = threading.RLock()
        self.max_countries = max_countries

        # Load data
        if vp_pool_file:
            self._parse_pool_file(vp_pool_file)
        else:
            self._parse_ev_file(ev_file)

        if max_countries is not None:
            self._trim_to_top_countries()

    # ------------------------------------------------------------------
    # ASN resolution
    # ------------------------------------------------------------------

    def _resolve_asn(self, ip: str, csv_asn: str) -> str:
        """
        Return the best available ASN for ip.

        Priority: CAIDA pfx2as > ev-certs CSV column > 'unknown'
        """
        if self._caida_db is not None:
            try:
                import ipaddress
                result = self._caida_db.lookup(ip)
                if result and result[0] is not None:
                    return str(result[0])
            except Exception:
                pass  # fall through to CSV column

        if csv_asn and csv_asn not in ("", "nan"):
            try:
                return str(int(float(csv_asn)))
            except (ValueError, TypeError):
                pass

        return _UNKNOWN_ASN

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_excluded_vps_file(path: Optional[str]) -> set:
        """One IP per line; '#' starts a trailing comment (the reason/
        evidence for that entry); blank lines and full-line comments
        ignored. Missing/None path -> empty set, no behavior change."""
        if not path:
            return set()
        ips = set()
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    ips.add(line)
        logger.info("Loaded %d excluded VP IP(s) from %s", len(ips), path)
        return ips

    def _parse_ev_file(self, ev_file: str) -> None:
        """Populate from a raw EV-certs CSV (ipv4/ipv6, country, port, asn)."""
        import ipaddress

        logger.info("Parsing vantage points from %s", ev_file)
        df = pd.read_csv(ev_file)

        # Load optional blocklist — kept simple, same logic as original
        excluded_count = 0
        hq_mismatch_count = 0
        excluded_vp_count = 0
        with self._lock:
            self._asn_pools.clear()
            self._ip_asn.clear()
            self._ports.clear()

            for row in df.itertuples(index=False):
                ip = None
                if hasattr(row, "ipv4") and row.ipv4 and not pd.isna(row.ipv4):
                    ip = str(row.ipv4).strip()
                elif hasattr(row, "ipv6") and row.ipv6 and not pd.isna(row.ipv6):
                    ip = str(row.ipv6).strip()
                if not ip:
                    continue
                if ip in self.excluded_vps:
                    excluded_vp_count += 1
                    continue

                country = getattr(row, "country", None)
                if not country or pd.isna(country):
                    continue
                country = str(country).strip()

                csv_asn = str(getattr(row, "asn", "")).strip() \
                    if hasattr(row, "asn") else ""
                asn = self._resolve_asn(ip, csv_asn)
                if asn in self.excluded_asns:
                    excluded_count += 1
                    continue
                if self._as_org_resolver is not None:
                    matches = self._as_org_resolver.country_name_matches(asn, country)
                    if matches is False:  # confirmed mismatch; None = unresolvable, fail open
                        hq_mismatch_count += 1
                        continue

                port = getattr(row, "port", None)
                if port is not None and not pd.isna(port):
                    self._ports.setdefault(ip, int(port))

                self._asn_pools[country][asn]["inactive"].add(ip)
                self._ip_asn[ip] = asn

        total = sum(
            len(pools["inactive"])
            for country_data in self._asn_pools.values()
            for pools in country_data.values()
        )
        logger.info(
            "Loaded %d VPs across %d countries (ASN source: %s), excluded %d VPs on %d excluded ASN(s), "
            "excluded %d VPs on AS-Org HQ-country mismatch, excluded %d individually-known-bad VP(s)",
            total,
            len(self._asn_pools),
            "CAIDA" if self._caida_db else "ev-certs CSV column",
            excluded_count, len(self.excluded_asns),
            hq_mismatch_count,
            excluded_vp_count,
        )

    def _parse_pool_file(self, vp_pool_file: str) -> None:
        """Populate from a pre-filtered pool CSV (country, ip, port)."""
        logger.info("Parsing vantage points from pool file %s", vp_pool_file)
        df = pd.read_csv(vp_pool_file)
        with self._lock:
            self._asn_pools.clear()
            self._ip_asn.clear()
            self._ports.clear()

            for row in df.itertuples(index=False):
                ip = str(row.ip).strip()
                if not ip:
                    continue
                if ip in self.excluded_vps:
                    continue
                country = str(row.country).strip()
                if not country:
                    continue

                csv_asn = str(getattr(row, "asn", "")).strip() \
                    if hasattr(row, "asn") else ""
                asn = self._resolve_asn(ip, csv_asn)
                if asn in self.excluded_asns:
                    continue
                if self._as_org_resolver is not None:
                    matches = self._as_org_resolver.country_name_matches(asn, country)
                    if matches is False:
                        continue

                if hasattr(row, "port") and not pd.isna(row.port):
                    self._ports.setdefault(ip, int(row.port))

                self._asn_pools[country][asn]["inactive"].add(ip)
                self._ip_asn[ip] = asn

    def _trim_to_top_countries(self) -> None:
        """Keep only the top max_countries countries by total VP count,
        plus any guaranteed_countries regardless of rank (e.g. a country
        this run's study depends on that a particular data source's volume
        ranking happens not to put in the top N)."""
        with self._lock:
            def _total(country):
                return sum(
                    len(p["inactive"]) + len(p["active"])
                    for p in self._asn_pools[country].values()
                )
            ranked = sorted(self._asn_pools.keys(), key=_total, reverse=True)
            keep = (set(ranked[: self.max_countries]) | self.guaranteed_countries) \
                - self.blocked_countries
            for country in list(self._asn_pools.keys()):
                if country not in keep:
                    del self._asn_pools[country]
            logger.info(
                "Trimmed to top %d countries (+ %d guaranteed): %s",
                self.max_countries, len(self.guaranteed_countries), sorted(keep),
            )

    # ------------------------------------------------------------------
    # Public API — identical signatures to original VantagePoints
    # ------------------------------------------------------------------

    def get_port(self, ip: str) -> Optional[int]:
        return self._ports.get(ip)

    def get_asn(self, ip: str) -> str:
        """Return the ASN for ip, or 'unknown'."""
        return self._ip_asn.get(ip, _UNKNOWN_ASN)

    def get_services(self, ip: str, base_services: List[str]) -> List[str]:
        _DEFAULT_PORTS = {"https": 443, "http": 80}
        port = self._ports.get(ip)
        if port is None:
            return base_services
        return [
            f"{svc}:{port}" if _DEFAULT_PORTS.get(svc) != port else svc
            for svc in base_services
        ]

    def countries(self) -> List[str]:
        with self._lock:
            return list(self._asn_pools.keys())

    # -- single VP draw --

    def get_vantage(self, country: str) -> Optional[str]:
        """
        Return one randomly chosen inactive VP, preferring under-represented
        ASNs. Moves the VP to active.
        """
        with self._lock:
            country_data = self._asn_pools.get(country)
            if not country_data:
                return None
            # Collect ASNs that still have inactive VPs
            available = {
                asn: pools
                for asn, pools in country_data.items()
                if pools["inactive"]
            }
            if not available:
                return None
            # Pick the ASN with fewest active VPs (most under-represented)
            asn = min(
                available,
                key=lambda a: len(country_data[a]["active"]),
            )
            vp = random.choice(tuple(available[asn]["inactive"]))
            country_data[asn]["inactive"].discard(vp)
            country_data[asn]["active"].add(vp)
            return vp

    # -- batch VP draw (main entry point) --

    def get_n_vantages(self, country: str, n: int) -> List[str]:
        """
        Draw up to n inactive VPs, round-robining across ASNs to maximise
        ISP diversity. Moves drawn VPs to active.
        """
        with self._lock:
            country_data = self._asn_pools.get(country)
            if not country_data:
                return []

            # Build per-ASN candidate lists (only ASNs with inactive VPs)
            asn_candidates: Dict[str, List[str]] = {
                asn: list(pools["inactive"])
                for asn, pools in country_data.items()
                if pools["inactive"]
            }
            if not asn_candidates:
                return []

            # Sort ASNs largest-first so we don't exhaust small pools early
            asns = sorted(asn_candidates, key=lambda a: -len(asn_candidates[a]))
            drawn: List[str] = []
            i = 0
            while len(drawn) < n and any(asn_candidates[a] for a in asns):
                asn = asns[i % len(asns)]
                if asn_candidates[asn]:
                    vp = asn_candidates[asn].pop(0)
                    country_data[asn]["inactive"].discard(vp)
                    country_data[asn]["active"].add(vp)
                    drawn.append(vp)
                i += 1

            asns_used = {self._ip_asn.get(vp, "?") for vp in drawn}
            logger.info(
                "Drew %d VPs for %s across %d ASN(s): %s",
                len(drawn), country, len(asns_used), drawn,
            )
            return drawn

    # -- lifecycle methods (unchanged logic, updated data structure) --

    def confirm_active(self, country: str, vp: str) -> None:
        """VP passed evaluation — ensure it remains in active."""
        with self._lock:
            asn = self._ip_asn.get(vp, _UNKNOWN_ASN)
            country_data = self._asn_pools.get(country)
            if not country_data:
                return
            country_data[asn]["active"].add(vp)

    def reject_vp(self, country: str, vp: str) -> Optional[str]:
        """
        VP failed — remove entirely and draw a replacement from a
        *different* ASN if possible, else any available ASN.
        """
        with self._lock:
            country_data = self._asn_pools.get(country)
            if not country_data:
                return None

            failed_asn = self._ip_asn.get(vp, _UNKNOWN_ASN)
            country_data[failed_asn]["active"].discard(vp)
            country_data[failed_asn]["inactive"].discard(vp)

            # Prefer a replacement from a different ASN
            other_asns = [
                asn for asn, pools in country_data.items()
                if asn != failed_asn and pools["inactive"]
            ]
            candidates = other_asns or [
                asn for asn, pools in country_data.items()
                if pools["inactive"]
            ]
            if not candidates:
                return None

            replacement_asn = random.choice(candidates)
            replacement = random.choice(
                tuple(country_data[replacement_asn]["inactive"])
            )
            country_data[replacement_asn]["inactive"].discard(replacement)
            country_data[replacement_asn]["active"].add(replacement)
            return replacement

    def restrict_to(self, country: str, keep) -> int:
        """
        Keep only the VPs in `keep` (an iterable of IPs) for `country`; every
        other VP in that country's pool is dropped. Used to pre-filter the
        candidate pool down to VPs that already passed Hyperquack's
        evaluation (see evaluate_all_vps.py). Returns how many VPs remain.
        """
        keep = set(keep)
        with self._lock:
            country_data = self._asn_pools.get(country, {})
            remaining = 0
            for pools in country_data.values():
                pools["inactive"] &= keep
                pools["active"] &= keep
                remaining += len(pools["inactive"]) + len(pools["active"])
            return remaining

    def discard_vp(self, country: str, vp: str) -> None:
        """
        VP failed -- remove it entirely and draw NO replacement. Used by
        all-VPs runs (Orchestrator(all_vps_mode=True)), where every eligible
        VP is already active, so "the pool" is fixed and a failed VP simply
        shrinks it (contrast reject_vp, which promotes a replacement).
        """
        with self._lock:
            country_data = self._asn_pools.get(country)
            if not country_data:
                return
            asn = self._ip_asn.get(vp, _UNKNOWN_ASN)
            if asn in country_data:
                country_data[asn]["active"].discard(vp)
                country_data[asn]["inactive"].discard(vp)

    def evaluate(self, country: str, vp: str, ok: bool) -> None:
        """Legacy method: ok=True returns VP to inactive, ok=False removes it."""
        with self._lock:
            asn = self._ip_asn.get(vp, _UNKNOWN_ASN)
            country_data = self._asn_pools.get(country)
            if not country_data:
                return
            country_data[asn]["active"].discard(vp)
            if ok:
                country_data[asn]["inactive"].add(vp)
            else:
                country_data[asn]["inactive"].discard(vp)

    # -- convenience accessors (same interface as original) --

    def get_active(self, country: str) -> List[str]:
        with self._lock:
            country_data = self._asn_pools.get(country)
            if not country_data:
                return []
            return [
                vp
                for pools in country_data.values()
                for vp in pools["active"]
            ]

    def get_inactive(self, country: str) -> List[str]:
        with self._lock:
            country_data = self._asn_pools.get(country)
            if not country_data:
                return []
            return [
                vp
                for pools in country_data.values()
                for vp in pools["inactive"]
            ]

    # -- reporting helper --

    def asn_summary(self, country: str) -> Dict[str, Dict[str, int]]:
        """
        Return ASN breakdown for a country.
        {asn: {"active": n, "inactive": m}, ...}
        """
        with self._lock:
            country_data = self._asn_pools.get(country, {})
            return {
                asn: {
                    "active": len(pools["active"]),
                    "inactive": len(pools["inactive"]),
                }
                for asn, pools in country_data.items()
            }