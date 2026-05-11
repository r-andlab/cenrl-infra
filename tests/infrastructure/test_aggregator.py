"""Tests for MeasurementAggregator: register_targets() and fixed record()."""

import logging
import pytest
from Infrastructure.utils.aggregator import MeasurementAggregator
from Infrastructure.utils.structures import AggregationMethod


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


class TestSetVpWeights:
    """Phase 4 D-02: set_vp_weights stores per-country weights for register_targets to snapshot."""

    def test_set_vp_weights_stores_country_dict(self):
        agg = MeasurementAggregator()
        agg.set_vp_weights("US", {"1.1.1.1": 0.7, "2.2.2.2": 0.4})
        assert agg._vp_weights["US"] == {"1.1.1.1": 0.7, "2.2.2.2": 0.4}

    def test_set_vp_weights_overwrites_previous(self):
        agg = MeasurementAggregator()
        agg.set_vp_weights("US", {"1.1.1.1": 0.7})
        agg.set_vp_weights("US", {"1.1.1.1": 0.9, "2.2.2.2": 0.4})
        assert agg._vp_weights["US"] == {"1.1.1.1": 0.9, "2.2.2.2": 0.4}

    def test_set_vp_weights_is_per_country(self):
        agg = MeasurementAggregator()
        agg.set_vp_weights("US", {"1.1.1.1": 0.7})
        agg.set_vp_weights("CN", {"3.3.3.3": 0.2})
        assert agg._vp_weights["US"] == {"1.1.1.1": 0.7}
        assert agg._vp_weights["CN"] == {"3.3.3.3": 0.2}

    def test_set_vp_weights_defensive_copy(self):
        """set_vp_weights stores a copy so external mutation does not leak."""
        agg = MeasurementAggregator()
        external = {"1.1.1.1": 0.7}
        agg.set_vp_weights("US", external)
        external["1.1.1.1"] = 0.0
        assert agg._vp_weights["US"]["1.1.1.1"] == 0.7


class TestPerTargetWeightSnapshot:
    """Phase 4 D-02: register_targets freezes _vp_weights[country] per target (WR-06 invariant)."""

    def test_register_targets_snapshots_current_weights(self):
        agg = MeasurementAggregator()
        agg.set_vp_weights("US", {"a": 0.8, "b": 0.2})
        agg.register_targets("US", ["t1"], {"a", "b"})
        assert agg._target_weights[("US", "t1")] == {"a": 0.8, "b": 0.2}

    def test_subsequent_weight_change_does_not_affect_registered_target(self):
        """A target registered before a weight push sees the OLD weights (WR-06 invariant)."""
        agg = MeasurementAggregator()
        agg.set_vp_weights("US", {"a": 0.8, "b": 0.2})
        agg.register_targets("US", ["t1"], {"a", "b"})
        # Now change the weights for future targets
        agg.set_vp_weights("US", {"a": 0.1, "b": 0.9})
        agg.register_targets("US", ["t2"], {"a", "b"})
        assert agg._target_weights[("US", "t1")] == {"a": 0.8, "b": 0.2}
        assert agg._target_weights[("US", "t2")] == {"a": 0.1, "b": 0.9}

    def test_target_weights_popped_after_finalize(self):
        agg = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        agg.set_vp_weights("US", {"a": 0.7})
        agg.register_targets("US", ["t1"], {"a"})
        agg.record("US", "a", "t1", blocked=True)
        # Drain ready queue to be sure finalize ran
        agg.get_ready("US")
        assert ("US", "t1") not in agg._target_weights

    def test_target_weights_popped_on_zero_expected_path(self):
        """WR-01 zero-expected path also pops _target_weights."""
        agg = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        agg.set_vp_weights("US", {"a": 0.7})
        agg.register_targets("US", ["t1"], {"a"})
        # Drop the only expected VP -- expected becomes empty, triggers WR-01 path
        agg.drop_vp("US", "a")
        assert ("US", "t1") not in agg._target_weights

    def test_register_targets_with_no_weights_pushed_yet(self):
        """Cold-start: register_targets before any set_vp_weights stores empty dict."""
        agg = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        agg.register_targets("US", ["t1"], {"a", "b"})
        assert agg._target_weights[("US", "t1")] == {}


class TestWeightedVoteBranch:
    """Phase 4 D-01: _try_finalize branches on aggregation_method."""

    def test_majority_default_bit_for_bit_2_of_3(self):
        """Default MAJORITY_VOTE: 2-of-3 blocked = True (unchanged from Plan 01)."""
        agg = MeasurementAggregator()
        agg.register_targets("US", ["t"], {"a", "b", "c"})
        agg.record("US", "a", "t", blocked=True)
        agg.record("US", "b", "t", blocked=True)
        agg.record("US", "c", "t", blocked=False)
        ready = agg.get_ready("US")
        assert len(ready) == 1
        assert ready[0].blocked is True

    def test_majority_default_1_of_3_is_false(self):
        """Default MAJORITY_VOTE: 1-of-3 blocked = False."""
        agg = MeasurementAggregator()
        agg.register_targets("US", ["t"], {"a", "b", "c"})
        agg.record("US", "a", "t", blocked=True)
        agg.record("US", "b", "t", blocked=False)
        agg.record("US", "c", "t", blocked=False)
        ready = agg.get_ready("US")
        assert ready[0].blocked is False

    def test_weighted_high_weight_dominates_low_weights(self):
        """WEIGHTED_VOTE: 1 high-weight blocked vote beats 2 low-weight not-blocked."""
        agg = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        agg.set_vp_weights("US", {"a": 0.95, "b": 0.05, "c": 0.05})
        agg.register_targets("US", ["t"], {"a", "b", "c"})
        agg.record("US", "a", "t", blocked=True)   # 0.95 * 1
        agg.record("US", "b", "t", blocked=False)  # 0.05 * 0
        agg.record("US", "c", "t", blocked=False)  # 0.05 * 0
        ready = agg.get_ready("US")
        # weighted_sum = 0.95; total = 1.05; 0.95 > 0.525 -> True
        assert ready[0].blocked is True

    def test_weighted_vs_majority_differential(self):
        """Same input + same VPs produces DIFFERENT blocked under weighted vs majority.

        This is the STRT-01 SC1 demonstrable behavior switch.
        """
        # MAJORITY: 1-of-3 = False
        agg_m = MeasurementAggregator(aggregation_method=AggregationMethod.MAJORITY_VOTE)
        agg_m.register_targets("US", ["t"], {"a", "b", "c"})
        agg_m.record("US", "a", "t", blocked=True)
        agg_m.record("US", "b", "t", blocked=False)
        agg_m.record("US", "c", "t", blocked=False)
        majority_ready = agg_m.get_ready("US")

        # WEIGHTED with skewed weights: 1-of-3 high-weight = True
        agg_w = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        agg_w.set_vp_weights("US", {"a": 0.95, "b": 0.05, "c": 0.05})
        agg_w.register_targets("US", ["t"], {"a", "b", "c"})
        agg_w.record("US", "a", "t", blocked=True)
        agg_w.record("US", "b", "t", blocked=False)
        agg_w.record("US", "c", "t", blocked=False)
        weighted_ready = agg_w.get_ready("US")

        assert majority_ready[0].blocked is False
        assert weighted_ready[0].blocked is True
        # Same vp_count regardless of branch
        assert majority_ready[0].vp_count == weighted_ready[0].vp_count == 3

    def test_weighted_strict_greater_tie_break_matches_majority(self):
        """Exact 50/50 weighted sum produces blocked=False (matches majority tie-break)."""
        agg = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        agg.set_vp_weights("US", {"a": 0.5, "b": 0.5})
        agg.register_targets("US", ["t"], {"a", "b"})
        agg.record("US", "a", "t", blocked=True)   # 0.5 * 1
        agg.record("US", "b", "t", blocked=False)  # 0.5 * 0
        # weighted_sum = 0.5; total = 1.0; 0.5 > 0.5 is False
        ready = agg.get_ready("US")
        assert ready[0].blocked is False

    def test_weighted_cold_start_missing_vp_defaults_to_half(self):
        """A VP not in the per-target snapshot defaults to 0.5 (Beta cold start)."""
        agg = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        # Only push weight for 'a'; 'b' and 'c' default to 0.5 at vote time
        agg.set_vp_weights("US", {"a": 0.5})
        agg.register_targets("US", ["t"], {"a", "b", "c"})
        agg.record("US", "a", "t", blocked=True)
        agg.record("US", "b", "t", blocked=True)
        agg.record("US", "c", "t", blocked=False)
        # weighted_sum = 0.5 + 0.5 + 0 = 1.0; total = 0.5+0.5+0.5 = 1.5; 1.0 > 0.75 -> True
        ready = agg.get_ready("US")
        assert ready[0].blocked is True

    def test_weighted_zero_weight_falls_back_to_majority(self):
        """All-zero weights -> warn-once + majority vote (D-01 Claude's Discretion)."""
        agg = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        agg.set_vp_weights("US", {"a": 0.0, "b": 0.0, "c": 0.0})
        agg.register_targets("US", ["t"], {"a", "b", "c"})
        agg.record("US", "a", "t", blocked=True)
        agg.record("US", "b", "t", blocked=True)
        agg.record("US", "c", "t", blocked=False)
        ready = agg.get_ready("US")
        # 2-of-3 majority = True
        assert ready[0].blocked is True
        assert ("US", "t") in agg._weighted_fallback_warned

    def test_weighted_zero_weight_warn_once_per_country_target(self):
        """Re-finalizing the same key must not re-warn (set-membership guard)."""
        agg = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        agg.set_vp_weights("US", {"a": 0.0})
        # First trigger
        agg.register_targets("US", ["t1"], {"a"})
        agg.record("US", "a", "t1", blocked=True)
        agg.get_ready("US")
        # Different target same country -> a new warn entry
        agg.register_targets("US", ["t2"], {"a"})
        agg.record("US", "a", "t2", blocked=True)
        agg.get_ready("US")
        assert ("US", "t1") in agg._weighted_fallback_warned
        assert ("US", "t2") in agg._weighted_fallback_warned

    def test_weighted_zero_weight_warning_logged(self, caplog):
        agg = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        agg.set_vp_weights("US", {"a": 0.0})
        agg.register_targets("US", ["t"], {"a"})
        with caplog.at_level(logging.WARNING):
            agg.record("US", "a", "t", blocked=True)
        assert "degenerate" in caplog.text.lower()

    def test_weighted_vote_vp_count_preserved(self):
        """WEIGHTED branch must preserve vp_count=len(votes) on MeasurementResponse."""
        agg = MeasurementAggregator(aggregation_method=AggregationMethod.WEIGHTED_VOTE)
        agg.set_vp_weights("US", {"a": 0.7, "b": 0.3, "c": 0.5})
        agg.register_targets("US", ["t"], {"a", "b", "c"})
        agg.record("US", "a", "t", blocked=True)
        agg.record("US", "b", "t", blocked=True)
        agg.record("US", "c", "t", blocked=False)
        ready = agg.get_ready("US")
        assert ready[0].vp_count == 3
