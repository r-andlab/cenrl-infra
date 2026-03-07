import threading
from collections import defaultdict
from typing import Dict, List, Set

from Infrastructure.utils.structures import MeasurementResponse


class MeasurementAggregator:
    """Per-target, per-VP aggregation buffer with majority-vote finalization.

    Buffers individual VP measurement results keyed by ``(country, target)``.
    Once every expected VP for a country has reported for a given target the
    entry is finalized: *blocked* if a strict majority of VPs report blocked.
    The aggregated ``MeasurementResponse`` is then placed on a per-country
    ready queue for the orchestrator to consume.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # (country, target) -> {vp_ip: blocked}
        self._pending: Dict[tuple, Dict[str, bool]] = defaultdict(dict)
        # country -> set of VP IPs we expect responses from
        self._expected_vps: Dict[str, Set[str]] = {}
        # country -> list of completed MeasurementResponse
        self._ready: Dict[str, List[MeasurementResponse]] = defaultdict(list)

    def set_expected_vps(self, country: str, vps: Set[str]) -> None:
        """Update the set of VPs whose responses are required before a
        target can be finalized for *country*."""
        with self._lock:
            self._expected_vps[country] = set(vps)

    def record(self, country: str, vp: str, target: str, blocked: bool) -> None:
        """Record one VP's result.  If all expected VPs have now reported
        for this target, finalize via majority vote and enqueue."""
        with self._lock:
            key = (country, target)
            self._pending[key][vp] = blocked
            self._try_finalize(country, key)

    def get_ready(self, country: str) -> List[MeasurementResponse]:
        """Return and clear all finalized results for *country*."""
        with self._lock:
            return self._ready.pop(country, [])

    def drop_vp(self, country: str, vp: str) -> None:
        """Remove *vp* from the expected set and re-evaluate every pending
        entry for *country* — any that are now complete get finalized."""
        with self._lock:
            expected = self._expected_vps.get(country)
            if expected is None:
                return
            expected.discard(vp)

            for key in list(self._pending):
                if key[0] != country:
                    continue
                self._pending[key].pop(vp, None)
                self._try_finalize(country, key)

    # ------------------------------------------------------------------
    # internals (caller must hold self._lock)
    # ------------------------------------------------------------------
    def _try_finalize(self, country: str, key: tuple) -> None:
        expected = self._expected_vps.get(country)
        if not expected:
            return
        vp_map = self._pending.get(key)
        if vp_map is None:
            return
        if not expected.issubset(vp_map.keys()):
            return

        votes = [vp_map[v] for v in expected]
        majority_blocked = sum(votes) > len(votes) / 2
        self._ready[country].append(
            MeasurementResponse(target=key[1], blocked=majority_blocked)
        )
        del self._pending[key]
