import unittest

from Infrastructure.main.orchestrator import _infra_reason
from Infrastructure.utils.structures import TestPayload, TestResponseData

T0 = "2026-10-03T07:00:00+00:00"
T1 = "2026-10-03T07:00:10+00:00"


def kw(error=None, status=None):
    d = {"matches_template": False, "start_time": T0, "end_time": T1, "error": error}
    if status is not None:
        d["response"] = {"status_line": status}
    return d


def ctrl(status=None, error=None):
    d = {"matches_template": False, "start_time": T0, "end_time": T1,
         "control_url": "control-abc.com", "error": error}
    if status is not None:
        d["response"] = {"status_line": status}
    return d


def payload(responses, anomaly=True):
    return TestPayload(vp="1.2.3.4", service="https", test_url="x.com",
                       response=[TestResponseData(**r) for r in responses],
                       anomaly=anomaly, controls_failed=False, stateful_block=False)


class TestInfraReason(unittest.TestCase):
    def test_all_keyword_attempts_hung_is_hang(self):
        p = payload([kw("request canceled while waiting for connection")] * 3 + [ctrl(status="404 NF")] * 2)
        self.assertEqual(_infra_reason(p), "hang")

    def test_one_keyword_attempt_succeeded_is_not_hang(self):
        p = payload([kw("request canceled while waiting for connection"), kw(None, "200 OK")] + [ctrl(status="404 NF")])
        self.assertIsNone(_infra_reason(p))

    def test_empty_final_response_is_empty(self):
        p = payload([kw(None, None)] * 3 + [ctrl(status="404 NF")])
        self.assertEqual(_infra_reason(p), "empty")

    def test_backend_5xx_with_healthy_control_is_backend(self):
        p = payload([kw(None, "503 No healthy IP available")] * 3 + [ctrl(status="404 NF")])
        self.assertEqual(_infra_reason(p), "backend_5xx")

    def test_5xx_shared_by_control_is_not_infrastructure(self):
        p = payload([kw(None, "503 Service Unavailable")] * 3 + [ctrl(status="503 Service Unavailable")])
        self.assertIsNone(_infra_reason(p))

    def test_resets_keep_their_vote(self):
        p = payload([kw('Get "https://x": read: connection reset by peer')] * 3 + [ctrl(status="404 NF")])
        self.assertIsNone(_infra_reason(p))

    def test_4xx_mismatch_keeps_its_vote(self):
        p = payload([kw(None, "403 Forbidden")] * 3 + [ctrl(status="404 NF")])
        self.assertIsNone(_infra_reason(p))

    def test_3xx_redirect_keeps_its_vote(self):
        p = payload([kw(None, "301 Moved Permanently")] * 3 + [ctrl(status="404 NF")])
        self.assertIsNone(_infra_reason(p))


class TestResponseDataLiftsStatus(unittest.TestCase):
    def test_status_line_is_kept_and_response_body_is_dropped(self):
        d = TestResponseData(**kw(None, "200 OK"))
        self.assertEqual(d.status_line, "200 OK")
        self.assertFalse(hasattr(d, "response"))


if __name__ == "__main__":
    unittest.main()


from unittest.mock import MagicMock
from Infrastructure.main.orchestrator import Orchestrator, HANG_OUTAGE_STREAK


def _hang(vp, domain):
    return TestPayload(
        vp=vp, service="https", test_url=domain,
        response=[TestResponseData(**kw(
            'Get "https://x": net/http: request canceled while waiting for connection'))] * 3
        + [TestResponseData(**ctrl(status="404 NF"))],
        anomaly=True, controls_failed=False, stateful_block=False,
    )


def _ok(vp, domain):
    return TestPayload(
        vp=vp, service="https", test_url=domain,
        response=[TestResponseData(**kw(None, "200 OK"))],
        anomaly=False, controls_failed=False, stateful_block=False,
    )


def _orch():
    orch = Orchestrator.__new__(Orchestrator)
    orch._all_vps_mode = True
    return orch


class TestOutageRule(unittest.TestCase):
    def test_isolated_hang_keeps_its_vote(self):
        orch = _orch()
        self.assertIsNone(orch._infra_abstain_reason("C", _hang("1.1.1.1", "a.com")))

    def test_hang_shared_by_many_vps_keeps_its_vote(self):
        orch = _orch()
        for i in range(10):
            self.assertIsNone(orch._infra_abstain_reason("C", _hang(f"1.1.1.{i}", "onlineradiobox.com")))

    def test_outage_needs_full_streak_across_full_domain_count(self):
        orch = _orch()
        for i in range(HANG_OUTAGE_STREAK - 1):
            self.assertIsNone(orch._infra_abstain_reason("C", _hang("9.9.9.9", f"d{i}.com")))
        self.assertEqual(orch._infra_abstain_reason("C", _hang("9.9.9.9", "d49.com")), "hang")
        self.assertEqual(orch._infra_abstain_reason("C", _hang("9.9.9.9", "d50.com")), "hang")

    def test_repeated_domain_does_not_count_as_outage(self):
        orch = _orch()
        for _ in range(HANG_OUTAGE_STREAK + 5):
            self.assertIsNone(orch._infra_abstain_reason("C", _hang("9.9.9.9", "same.com")))

    def test_successful_result_ends_the_outage(self):
        orch = _orch()
        for i in range(HANG_OUTAGE_STREAK):
            orch._infra_abstain_reason("C", _hang("9.9.9.9", f"d{i}.com"))
        self.assertIsNone(orch._infra_abstain_reason("C", _ok("9.9.9.9", "fine.com")))
        self.assertIsNone(orch._infra_abstain_reason("C", _hang("9.9.9.9", "after.com")))

    def test_normal_mode_is_unchanged_by_streaks(self):
        orch = _orch()
        orch._all_vps_mode = False
        self.assertIsNone(orch._infra_abstain_reason("C", _hang("1.1.1.1", "a.com")))
