import unittest

from Infrastructure.utils.aggregator import MeasurementAggregator
from Infrastructure.utils.eval_store import EvalStore
from Infrastructure.utils.structures import EvalPayload


class TestMeasurementAggregator(unittest.TestCase):
    def test_majority_blocked(self):
        agg = MeasurementAggregator()
        agg.set_expected_vps("US", {"vp1", "vp2", "vp3"})

        agg.record("US", "vp1", "example.com", blocked=True)
        agg.record("US", "vp2", "example.com", blocked=True)
        # not all VPs yet
        self.assertEqual(agg.get_ready("US"), [])

        agg.record("US", "vp3", "example.com", blocked=False)
        results = agg.get_ready("US")
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].blocked)  # 2 of 3 blocked
        self.assertEqual(results[0].target, "example.com")

    def test_majority_not_blocked(self):
        agg = MeasurementAggregator()
        agg.set_expected_vps("DE", {"vp1", "vp2", "vp3"})

        agg.record("DE", "vp1", "google.com", blocked=False)
        agg.record("DE", "vp2", "google.com", blocked=True)
        agg.record("DE", "vp3", "google.com", blocked=False)

        results = agg.get_ready("DE")
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].blocked)  # 1 of 3 blocked

    def test_tie_resolves_not_blocked(self):
        agg = MeasurementAggregator()
        agg.set_expected_vps("FR", {"vp1", "vp2"})

        agg.record("FR", "vp1", "test.com", blocked=True)
        agg.record("FR", "vp2", "test.com", blocked=False)

        results = agg.get_ready("FR")
        self.assertEqual(len(results), 1)
        # exactly half => not blocked (strictly > 50%)
        self.assertFalse(results[0].blocked)

    def test_multiple_targets_finalize_independently(self):
        agg = MeasurementAggregator()
        agg.set_expected_vps("JP", {"vp1", "vp2"})

        # target A gets both VPs
        agg.record("JP", "vp1", "a.com", blocked=False)
        agg.record("JP", "vp2", "a.com", blocked=False)
        # target B only has vp1 so far
        agg.record("JP", "vp1", "b.com", blocked=True)

        results = agg.get_ready("JP")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].target, "a.com")

        # finish target B
        agg.record("JP", "vp2", "b.com", blocked=True)
        results = agg.get_ready("JP")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].target, "b.com")
        self.assertTrue(results[0].blocked)

    def test_drop_vp_finalizes_pending(self):
        agg = MeasurementAggregator()
        agg.set_expected_vps("BR", {"vp1", "vp2", "vp3"})

        agg.record("BR", "vp1", "site.com", blocked=True)
        agg.record("BR", "vp2", "site.com", blocked=False)
        # vp3 disappears before reporting
        self.assertEqual(agg.get_ready("BR"), [])

        agg.drop_vp("BR", "vp3")
        results = agg.get_ready("BR")
        self.assertEqual(len(results), 1)
        # Only vp1 and vp2 remain: 1 blocked, 1 not => tie => not blocked
        self.assertFalse(results[0].blocked)

    def test_get_ready_clears(self):
        agg = MeasurementAggregator()
        agg.set_expected_vps("CA", {"vp1"})
        agg.record("CA", "vp1", "test.ca", blocked=False)

        results = agg.get_ready("CA")
        self.assertEqual(len(results), 1)
        # calling again returns empty
        self.assertEqual(agg.get_ready("CA"), [])

    def test_no_expected_vps_nothing_finalizes(self):
        agg = MeasurementAggregator()
        agg.record("ZZ", "vp1", "x.com", blocked=True)
        self.assertEqual(agg.get_ready("ZZ"), [])


class TestEvalStore(unittest.TestCase):
    def test_register_and_lookup(self):
        store = EvalStore()
        store.register_vp("1.2.3.4", "Germany")
        self.assertEqual(store.get_country("1.2.3.4"), "Germany")
        self.assertIsNone(store.get_country("5.6.7.8"))

    def test_record_and_drain(self):
        store = EvalStore()
        p1 = EvalPayload(vp="1.1.1.1", service="https", response=[],
                          template={"ok": True})
        p2 = EvalPayload(vp="2.2.2.2", service="https", response=[],
                          issue="down", template=None)
        store.record(p1)
        store.record(p2)

        results = store.drain()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].vp, "1.1.1.1")
        self.assertEqual(results[1].vp, "2.2.2.2")

        # drain again is empty
        self.assertEqual(store.drain(), [])


if __name__ == "__main__":
    unittest.main()
