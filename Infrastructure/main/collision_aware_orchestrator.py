"""
collision_aware_orchestrator.py

Extends Orchestrator to filter out VP<->target CDN collisions BEFORE scheduling,
turning the current "every target to every active VP" cross-product into
"every target to its own safe-VP subset".

This overrides Orchestrator._dispatch(country, active_vps, targets, now),
which orchestrator.py's tick() calls once per country each tick. This class
is actively wired in and used by run_all_vps.py (Experiments 5/6) as the
live orchestrator.

Why register_targets and schedule_measurements must always use the SAME VP set:
register_targets() snapshots the VP set the aggregator will wait for before
finalizing a target (Infrastructure/utils/aggregator.py). If a VP is registered
as "expected" but schedule_measurements() never actually sends it the work
(because the collision filter excluded it), the aggregator waits forever for a
vote that will never arrive -- the exact "stuck measurement" failure mode this
whole investigation started with (the Japan/106.73.19.129 incident), except by
construction, for every collision-filtered target. So every call below computes
one target's safe VP subset once and uses it for both calls, never the original
unfiltered active_vps.

Wiring (done wherever the orchestrator is constructed):

    from Infrastructure.main.asn_aware_vantage_points import AsnAwareVantagePoints
    from Infrastructure.main.vp_pool_adapter import VpPoolAdapter
    from Infrastructure.main.asn_resolver import AsnResolver
    from Infrastructure.main.cdn_collision_filter import CdnCollisionFilter
    from Infrastructure.main.collision_aware_orchestrator import CollisionAwareOrchestrator

    base_pool = AsnAwareVantagePoints(ev_file="local/ev-certs.csv", caida_dat="local/ipasn_20260720.dat")
    pool      = VpPoolAdapter(base_pool)
    resolver  = AsnResolver("local/ipasn_20260720.dat")
    filt      = CdnCollisionFilter(resolver, pool, min_vps=2)

    orch = CollisionAwareOrchestrator(
        params=..., vantage_points=pool, go_api_endpoint=..., services=[...],
        collision_filter=filt,
    )
"""

import logging
from typing import List

from Infrastructure.main.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class CollisionAwareOrchestrator(Orchestrator):
    def __init__(self, *args, collision_filter=None, **kwargs):
        super().__init__(*args, **kwargs)
        if collision_filter is None:
            raise ValueError("collision_filter is required")
        self._collision_filter = collision_filter
        # per-run audit of what got skipped: (country, target, reason)
        self._skipped_targets = []

    def _dispatch(
        self, country: str, active_vps: List[str], targets: List[str], now: float
    ) -> None:
        """
        Override of Orchestrator._dispatch. Instead of registering and
        scheduling the whole target batch against the full active_vps set in
        one shot, compute each target's safe (non-colliding) VP subset first,
        then register and schedule that exact same subset -- one target at a
        time, so the aggregator's expected-VP snapshot always matches what
        Hyperquack actually got asked to measure.
        """
        for target in targets:
            decision = self._collision_filter.filter_for_target(
                country, target, active_vps
            )

            if decision.skipped:
                self._skipped_targets.append((country, target, decision.reason))
                logger.warning(
                    "Collision skip: %s / %s -- %s",
                    country, target, decision.reason,
                )
                continue

            if decision.collided_vps:
                logger.info(
                    "Collision filter: %s / %s dropped %s, using %s (%s)",
                    country, target,
                    decision.collided_vps, decision.safe_vps, decision.reason,
                )

            # decision.replacements are VPs draw_replacement() just promoted
            # from inactive -> active in the pool's own bookkeeping, but that
            # promotion never tells Hyperquack the VP exists, and eval_store
            # has no entry for it either. Every other place in this codebase
            # that hands a VP real work (_process_eval_results,
            # _check_vp_health's replacement path) registers + updates it
            # with Hyperquack first -- do the same here, or Hyperquack drops
            # its eval result as "unknown VP" and the target gets stuck in
            # the DELAYED/resend loop forever.
            for replacement in decision.replacements:
                self.eval_store.register_vp(replacement, country)
                self.api.update_vps([replacement], self.services, tag=country)
                if self.api.debug:
                    self.api._inject_debug_eval_results([replacement])

            safe_vps = decision.safe_vps

            # Same safe_vps set feeds both calls below -- this is the fix for
            # the register/schedule mismatch described in the module docstring.
            vp_weights = self._compute_vp_weights(country, set(safe_vps))
            self.api.aggregator.set_vp_weights(country, vp_weights)
            self.api.aggregator.register_targets(
                country, [target], set(safe_vps),
                schedule_times={target: now},
            )
            self.api.schedule_measurements(
                vps=safe_vps,
                services=self.services,
                targets=[target],
                country=country,
            )
            self._inflight_times[country][target] = now

    def skipped_report(self):
        """Return the list of (country, target, reason) skipped this run."""
        return list(self._skipped_targets)
