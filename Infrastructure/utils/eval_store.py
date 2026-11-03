import threading
from typing import Dict, List, Optional

from Infrastructure.utils.structures import EvalPayload


class EvalStore:
    """Thread-safe buffer for VP evaluation results.

    The Go server's ``EvalPayload`` contains only the VP IP — no country.
    The orchestrator calls :meth:`register_vp` *before* sending a VP for
    evaluation so that incoming payloads can be resolved to a country.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._results: List[EvalPayload] = []
        self._vp_country_map: Dict[str, str] = {}

    def register_vp(self, vp: str, country: str) -> None:
        """Record which country *vp* belongs to."""
        with self._lock:
            self._vp_country_map[vp] = country

    def get_country(self, vp: str) -> Optional[str]:
        """Look up the country for *vp*, or ``None`` if unknown."""
        with self._lock:
            return self._vp_country_map.get(vp)

    def record(self, payload: EvalPayload) -> None:
        """Buffer an incoming evaluation result."""
        with self._lock:
            self._results.append(payload)

    def drain(self) -> List[EvalPayload]:
        """Return and clear all buffered evaluation results."""
        with self._lock:
            results = self._results[:]
            self._results.clear()
            return results
