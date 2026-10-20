"""
cdn_collision_filter.py

Given a target domain and a country's vantage points, decide which VPs may
safely measure it -- i.e. exclude any VP that shares an ASN with the target
(the Fastly/Akamai collision confirmed by analyze_cdn_collisions.py on the
outputs23 run), and top back up to a minimum count by drawing replacement
VPs from other ASNs when filtering leaves too few.

This is pure decision logic. It does not talk to HyperQuack or DNS directly;
it depends on two injected collaborators:
  - resolver : an AsnResolver (domain -> ASN)
  - vp_pool  : anything exposing get_asn(ip) and draw_replacement(country,
               exclude_asns) -> ip-or-None   (AsnAwareVantagePoints provides
               get_asn; draw_replacement is provided by VpPoolAdapter, a thin
               wrapper over its _asn_pools active/inactive tracking)

Returned decision object records what happened so runs can be audited.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FilterDecision:
    target: str
    target_asn: Optional[int]
    safe_vps: List[str] = field(default_factory=list)
    collided_vps: List[str] = field(default_factory=list)
    replacements: List[str] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""


class CdnCollisionFilter:
    def __init__(self, resolver, vp_pool, min_vps: int = 2):
        """
        resolver : AsnResolver
        vp_pool  : provides get_asn(ip) and draw_replacement(country, exclude_asns)
        min_vps  : minimum safe VPs required to measure a target (default 2)
        """
        self.resolver = resolver
        self.vp_pool = vp_pool
        self.min_vps = min_vps

    def filter_for_target(
        self, country: str, target: str, active_vps: List[str]
    ) -> FilterDecision:
        target_asn = self.resolver.asn_for(target)
        decision = FilterDecision(target=target, target_asn=target_asn)

        # Fail-open: if we can't determine the target's ASN, we can't detect a
        # collision, so measure normally with all active VPs.
        if target_asn is None:
            decision.safe_vps = list(active_vps)
            decision.reason = "target ASN unknown; measured with all VPs"
            return decision

        # Partition active VPs into safe vs colliding
        for vp in active_vps:
            if self.vp_pool.get_asn(vp) == str(target_asn):
                decision.collided_vps.append(vp)
            else:
                decision.safe_vps.append(vp)

        # Enough safe VPs already -- done.
        if len(decision.safe_vps) >= self.min_vps:
            decision.reason = "enough safe VPs after filtering"
            return decision

        # Otherwise, draw replacements from other ASNs until we reach min_vps
        # or run out. Exclude the target's ASN and any ASN already represented
        # among the safe VPs (so replacements add real diversity).
        exclude: Set[str] = {str(target_asn)}
        exclude.update(self.vp_pool.get_asn(vp) for vp in decision.safe_vps)

        while len(decision.safe_vps) < self.min_vps:
            repl = self.vp_pool.draw_replacement(country, exclude)
            if repl is None:
                break  # pool exhausted
            repl_asn = self.vp_pool.get_asn(repl)
            # draw_replacement already excludes these ASNs, but double-check
            if repl_asn == str(target_asn):
                continue
            decision.safe_vps.append(repl)
            decision.replacements.append(repl)
            exclude.add(repl_asn)

        if len(decision.safe_vps) >= self.min_vps:
            decision.reason = f"topped up with {len(decision.replacements)} replacement(s)"
        else:
            decision.skipped = True
            decision.reason = (
                f"only {len(decision.safe_vps)} safe VP(s) available "
                f"(< min {self.min_vps}); target skipped this round"
            )
        return decision
