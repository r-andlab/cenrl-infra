import logging
import threading
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from Infrastructure.utils.structures import MeasurementResponse

logger = logging.getLogger(__name__)


class MeasurementAggregator:
    """Per-target, per-VP aggregation buffer with majority-vote finalization.

    Buffers individual VP measurement results keyed by ``(country, target)``.
    Once every expected VP for a country has reported for a given target the
    entry is finalized: *blocked* if a strict majority of VPs report blocked.
    The aggregated ``MeasurementResponse`` is then placed on a per-country
    ready queue for the orchestrator to consume.

    The expected VP set is **snapshotted per-target at scheduling time** via
    ``register_targets()``.  This must be called before measurements are
    scheduled so that the aggregator knows exactly which VPs to wait for,
    preventing races when VPs are added or removed between scheduling and
    result arrival.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # (country, target) -> {vp_ip: blocked}
        self._pending: Dict[Tuple, Dict[str, bool]] = defaultdict(dict)
        # (country, target) -> frozenset of VPs this target must wait for
        self._target_expected: Dict[Tuple, frozenset] = {}
        # country -> current set of VP IPs (used as snapshot source)
        self._expected_vps: Dict[str, Set[str]] = {}
        # country -> list of completed MeasurementResponse
        self._ready: Dict[str, List[MeasurementResponse]] = defaultdict(list)
        # (country, target) -> monotonic time captured at register_targets() call.
        # Drained in _try_finalize so finalized MeasurementResponse can carry latency source. (Phase 3 D-04)
        self._schedule_times: Dict[Tuple, float] = {}

    def set_expected_vps(self, country: str, vps: Set[str]) -> None:
        """Update the set of VPs whose responses are required for *future*
        targets in *country*.  Already-pending targets keep their original
        snapshot."""
        with self._lock:
            self._expected_vps[country] = set(vps)

    def register_targets(
        self, country: str, targets: List[str], vps: Set[str],
        schedule_times: Dict[str, float] | None = None,
    ) -> None:
        """Snapshot expected VPs for each target at scheduling time (D-02).

        Must be called BEFORE schedule_measurements() with the same VP set.

        If ``schedule_times`` is provided (Phase 3 D-04), each target's monotonic
        schedule timestamp is recorded so the finalized ``MeasurementResponse``
        can carry it through to downstream consumers (latency_ms derivation).
        """
        with self._lock:
            snapshot = frozenset(vps)
            for target in targets:
                key = (country, target)
                self._target_expected[key] = snapshot
                if schedule_times is not None and target in schedule_times:
                    self._schedule_times[key] = schedule_times[target]
            # Keep _expected_vps updated for drop_vp() compatibility
            self._expected_vps[country] = set(vps)

    def record(self, country: str, vp: str, target: str, blocked: bool) -> None:
        """Record one VP's result.  If all expected VPs have now reported
        for this target, finalize via majority vote and enqueue."""
        with self._lock:
            key = (country, target)
            if key not in self._target_expected:
                logger.warning("record() for unregistered target %s/%s, dropping", country, target)
                return
            self._pending[key][vp] = blocked
            self._try_finalize(key)

    def get_ready(self, country: str) -> List[MeasurementResponse]:
        """Return and clear all finalized results for *country*."""
        with self._lock:
            return self._ready.pop(country, [])

    def drop_vp(self, country: str, vp: str) -> None:
        """Remove *vp* from pending entries and their snapshots for
        *country* — any that are now complete get finalized."""
        with self._lock:
            expected = self._expected_vps.get(country)
            if expected is not None:
                expected.discard(vp)

            # Iterate _target_expected (superset of _pending for registered targets)
            for key in list(self._target_expected):
                if key[0] != country:
                    continue
                if key in self._pending:
                    self._pending[key].pop(vp, None)
                # Shrink the per-target snapshot
                self._target_expected[key] = self._target_expected[key] - {vp}
                self._try_finalize(key)

    # ------------------------------------------------------------------
    # internals (caller must hold self._lock)
    # ------------------------------------------------------------------
    def _try_finalize(self, key: Tuple) -> None:
        expected = self._target_expected.get(key)
        if not expected:
            return
        vp_map = self._pending.get(key)
        if vp_map is None:
            return
        if not expected.issubset(vp_map.keys()):
            return

        votes = [vp_map[v] for v in expected]
        majority_blocked = sum(votes) > len(votes) / 2
        # Phase 3 D-04: pop the schedule timestamp so the response carries it
        # and the dict does not accumulate orphan entries.
        sched_mono = self._schedule_times.pop(key, None)
        self._ready[key[0]].append(
            MeasurementResponse(
                target=key[1],
                blocked=majority_blocked,
                scheduled_at_monotonic=sched_mono,
                vp_count=len(votes),
            )
        )
        del self._pending[key]
        del self._target_expected[key]
