"""
vp_pool_adapter.py

The CdnCollisionFilter needs a vp_pool with two methods:
    get_asn(ip)                        -> str
    draw_replacement(country, exclude) -> ip or None

AsnAwareVantagePoints already provides get_asn. This adapter adds
draw_replacement on top of it, drawing an inactive VP whose ASN is not in
`exclude` and promoting it to active -- reusing the class's own per-ASN
_asn_pools structure so state stays consistent (verified directly against
Infrastructure/main/asn_aware_vantage_points.py: _asn_pools[country][asn] =
{"inactive": set, "active": set} is exactly the real internal layout).

Usage:

    from Infrastructure.main.asn_aware_vantage_points import AsnAwareVantagePoints
    from Infrastructure.main.vp_pool_adapter import VpPoolAdapter

    base = AsnAwareVantagePoints(ev_file="ev-certs.csv", caida_dat="ipasn.dat")
    pool = VpPoolAdapter(base)          # pass this to CdnCollisionFilter

Everything not overridden delegates to the wrapped instance via __getattr__,
so the adapter is transparent to the rest of the orchestrator -- it can be
used anywhere a plain VantagePoints/AsnAwareVantagePoints is expected.
"""

import logging
import random
from typing import Optional, Set

logger = logging.getLogger(__name__)


class VpPoolAdapter:
    def __init__(self, base):
        """base: an AsnAwareVantagePoints instance."""
        self._base = base

    # --- transparent delegation for everything else ---
    def __getattr__(self, name):
        return getattr(self._base, name)

    # --- the method the filter needs that the base class lacks ---
    def draw_replacement(self, country: str, exclude_asns: Set[str]) -> Optional[str]:
        """
        Draw one inactive VP for `country` whose ASN is not in exclude_asns,
        move it to active, and return its IP. Returns None if none available.

        Relies on the base class's internal per-ASN pools. We reach into the
        same structure get_n_vantages uses so active/inactive accounting stays
        correct.
        """
        # AsnAwareVantagePoints stores _asn_pools[country][asn] = {"inactive", "active"}
        country_data = getattr(self._base, "_asn_pools", {}).get(country)
        if not country_data:
            return None

        # candidate ASNs = have inactive VPs and are not excluded
        candidates = [
            asn for asn, pools in country_data.items()
            if pools["inactive"] and asn not in exclude_asns
        ]
        if not candidates:
            return None

        # prefer the ASN with the most inactive VPs (most headroom), tie broken randomly
        candidates.sort(key=lambda a: (-len(country_data[a]["inactive"]), random.random()))
        chosen_asn = candidates[0]

        vp = random.choice(tuple(country_data[chosen_asn]["inactive"]))
        country_data[chosen_asn]["inactive"].discard(vp)
        country_data[chosen_asn]["active"].add(vp)
        logger.info("draw_replacement: %s from ASN %s for %s", vp, chosen_asn, country)
        return vp
