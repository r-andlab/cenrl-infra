from collections import defaultdict
import threading
import queue
from typing import Any, Dict, List, Tuple, Optional
from utils.structures import RequestPayload


class MeasurementStore:
    """
    Stores measurement results keyed by country.
    - request_id == country name
    - record_result(request_id, result) is called when Go finishes a measurement
    - get_ready_batch() returns (country, [results]) for one country that has data
    """

    def __init__(self) -> None:
        # country -> list of results
        self._country_results: Dict[str, List[RequestPayload]] = defaultdict(list)
        # queue of countries that currently have *some* results ready
        self._ready_countries: "queue.Queue[str]" = queue.Queue()
        # to avoid enqueuing the same country multiple times
        self._in_queue: set[str] = set()
        self._lock = threading.Lock()

    # ------------- PUBLIC API -------------

    def make_request_id(self, country: str) -> str:
        """
        Use the country name as the request ID.
        You can call this when creating a request payload for Go.
        """
        return country

    def record_result(self, request_id: str, result: Any) -> None:
        """
        Called when a response comes back from the Go tool.
        request_id *is* the country name.
        """
        country = request_id

        with self._lock:
            self._country_results[country].append(result)
            # print(f"Stored for {country} :\n {result}\n")
            # only enqueue the country once even if multiple results arrive
            if country not in self._in_queue:
                self._ready_countries.put(country)
                self._in_queue.add(country)

    def get_ready_batch(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> Tuple[Optional[str], List[RequestPayload]]:
        """
        Returns (country, [results]) for a single country that has
        at least one completed measurement.

        If `block=True`, waits until some country has results (or timeout).
        If no results are ready and not blocking, returns (None, []).
        """
        try:
            country = self._ready_countries.get(block=block, timeout=timeout)
        except queue.Empty:
            return None, []

        with self._lock:
            results = self._country_results.pop(country, [])
            self._in_queue.discard(country)

        return country, results

    def get_country_batch(
        self, country: str, clear: bool = True
    ) -> List[RequestPayload]:
        """
        Return all currently stored results for a specific country.

        - If `clear=True` (default), the results are removed from the store,
          and the country is also removed from the "ready" set.
        - If `clear=False`, results are returned but kept in the store.

        This does *not* block: if there are no results yet for that country,
        it simply returns [].
        """
        with self._lock:
            results = self._country_results.get(country, [])
            if not results:
                return []

            if clear:
                # remove from internal structures
                results = self._country_results.pop(country, [])
                self._in_queue.discard(country)
                # we can't safely remove an arbitrary item from queue.Queue,
                # but get_ready_batch() will ignore stale entries when it pops.

            # if clear=False, we return a copy to avoid external mutation
            return list(results)
