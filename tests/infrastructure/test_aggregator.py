"""Tests for MeasurementAggregator: register_targets() and fixed record()."""

import logging
import pytest
from Infrastructure.utils.aggregator import MeasurementAggregator


@pytest.fixture
def agg():
    return MeasurementAggregator()


class TestRegisterTargets:
    """Tests for register_targets() method."""

    def test_populates_target_expected(self, agg):
        """register_targets populates _target_expected for each target with frozenset of VPs."""
        vps = {"1.1.1.1", "2.2.2.2"}
        agg.register_targets("US", ["example.com", "test.com"], vps)

        assert ("US", "example.com") in agg._target_expected
        assert ("US", "test.com") in agg._target_expected
        assert agg._target_expected[("US", "example.com")] == frozenset(vps)
        assert agg._target_expected[("US", "test.com")] == frozenset(vps)

    def test_updates_expected_vps(self, agg):
        """register_targets updates _expected_vps[country] with the provided VP set."""
        vps = {"1.1.1.1", "2.2.2.2"}
        agg.register_targets("US", ["example.com"], vps)

        assert agg._expected_vps["US"] == vps

    def test_record_does_not_overwrite_target_expected(self, agg):
        """After register_targets, record() does NOT overwrite _target_expected."""
        vps = {"1.1.1.1", "2.2.2.2"}
        agg.register_targets("US", ["example.com"], vps)

        # Change _expected_vps to a different set (simulating VP changes)
        agg._expected_vps["US"] = {"3.3.3.3"}

        # Record should NOT overwrite the already-registered snapshot
        agg.record("US", "1.1.1.1", "example.com", True)
        assert agg._target_expected[("US", "example.com")] == frozenset(vps)


class TestRecordDefensivePath:
    """Tests for record() defensive early return on unregistered targets."""

    def test_unregistered_target_logs_warning_and_drops(self, agg, caplog):
        """record() for a target NOT in _target_expected logs a warning and returns."""
        with caplog.at_level(logging.WARNING):
            agg.record("US", "1.1.1.1", "unknown.com", True)

        assert "unregistered target" in caplog.text.lower()
        assert ("US", "unknown.com") not in agg._pending

    def test_registered_target_adds_to_pending_and_finalizes(self, agg):
        """record() for a registered target adds to _pending and triggers _try_finalize."""
        vps = {"1.1.1.1"}
        agg.register_targets("US", ["example.com"], vps)

        agg.record("US", "1.1.1.1", "example.com", True)

        # Single VP, should finalize immediately
        results = agg.get_ready("US")
        assert len(results) == 1
        assert results[0].target == "example.com"
        assert results[0].blocked is True

    def test_late_result_for_finalized_target_is_dropped(self, agg, caplog):
        """Late result for already-finalized target hits defensive path."""
        vps = {"1.1.1.1"}
        agg.register_targets("US", ["example.com"], vps)
        agg.record("US", "1.1.1.1", "example.com", True)

        # Consume the ready result
        agg.get_ready("US")

        # Late result arrives - target is no longer in _target_expected
        with caplog.at_level(logging.WARNING):
            agg.record("US", "1.1.1.1", "example.com", False)

        assert "unregistered target" in caplog.text.lower()


class TestDropVpWithRegisterTargets:
    """Tests for drop_vp() compatibility with register_targets()."""

    def test_drop_vp_shrinks_snapshot_and_finalizes(self, agg):
        """drop_vp() shrinks per-target snapshot and may finalize."""
        vps = {"1.1.1.1", "2.2.2.2"}
        agg.register_targets("US", ["example.com"], vps)

        # Record only from VP 1
        agg.record("US", "1.1.1.1", "example.com", True)

        # Drop VP 2 - should shrink snapshot and finalize
        agg.drop_vp("US", "2.2.2.2")

        results = agg.get_ready("US")
        assert len(results) == 1
        assert results[0].target == "example.com"
        assert results[0].blocked is True

    def test_drop_vp_without_results_shrinks_snapshot(self, agg):
        """drop_vp() on a registered target shrinks snapshot even if no results yet."""
        vps = {"1.1.1.1", "2.2.2.2"}
        agg.register_targets("US", ["example.com"], vps)

        agg.drop_vp("US", "2.2.2.2")

        # Snapshot should now only contain VP 1
        assert agg._target_expected[("US", "example.com")] == frozenset({"1.1.1.1"})


class TestVpCountPassthrough:
    """Phase 3 D-05: vp_count = len(votes) on finalized MeasurementResponse."""

    def test_vp_count_three_when_three_voters(self):
        agg = MeasurementAggregator()
        agg.register_targets("US", ["a.com"], {"vp1", "vp2", "vp3"})
        agg.record("US", "vp1", "a.com", blocked=True)
        agg.record("US", "vp2", "a.com", blocked=True)
        agg.record("US", "vp3", "a.com", blocked=False)
        results = agg.get_ready("US")
        assert len(results) == 1
        assert results[0].vp_count == 3

    def test_vp_count_two_after_drop_vp(self):
        agg = MeasurementAggregator()
        agg.register_targets("US", ["a.com"], {"vp1", "vp2", "vp3"})
        agg.record("US", "vp1", "a.com", blocked=True)
        agg.record("US", "vp2", "a.com", blocked=True)
        # vp3 disappears before reporting
        agg.drop_vp("US", "vp3")
        results = agg.get_ready("US")
        assert len(results) == 1
        assert results[0].vp_count == 2  # surviving voter count


class TestScheduledAtMonotonicPassthrough:
    """Phase 3 D-04: scheduled_at_monotonic flows from register_targets to MeasurementResponse."""

    def test_schedule_time_attached_when_provided(self):
        agg = MeasurementAggregator()
        agg.register_targets(
            "US", ["a.com"], {"vp1", "vp2"},
            schedule_times={"a.com": 100.5},
        )
        agg.record("US", "vp1", "a.com", blocked=False)
        agg.record("US", "vp2", "a.com", blocked=False)
        results = agg.get_ready("US")
        assert len(results) == 1
        assert results[0].scheduled_at_monotonic == 100.5

    def test_schedule_time_none_when_not_provided(self):
        agg = MeasurementAggregator()
        agg.register_targets("US", ["a.com"], {"vp1", "vp2"})
        agg.record("US", "vp1", "a.com", blocked=False)
        agg.record("US", "vp2", "a.com", blocked=False)
        results = agg.get_ready("US")
        assert len(results) == 1
        assert results[0].scheduled_at_monotonic is None

    def test_schedule_times_pop_after_finalize(self):
        agg = MeasurementAggregator()
        agg.register_targets(
            "US", ["a.com"], {"vp1"},
            schedule_times={"a.com": 99.0},
        )
        agg.record("US", "vp1", "a.com", blocked=False)
        agg.get_ready("US")
        assert ("US", "a.com") not in agg._schedule_times
