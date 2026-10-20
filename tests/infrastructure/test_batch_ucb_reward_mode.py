"""Ex 6: BatchUCB.absorb_measurement's reward_mode ("majority" vs "percentage").

Builds a bare BatchUCB instance (bypassing __init__, which needs a full
params/action-space/blocklist setup) and stubs the handful of methods
absorb_measurement calls besides observe()/reward selection, so these tests
stay focused on exactly the reward computation this change touches.
"""
from unittest.mock import MagicMock

from Infrastructure.models.BatchUCB import BatchUCB
from Infrastructure.utils.structures import Pending, PropagationMethod


def _ucb(reward_mode=None):
    ucb = BatchUCB.__new__(BatchUCB)
    ucb.country_name = "US"
    ucb.verbose = False
    ucb.logfile = None
    ucb._selected_targets = {}
    ucb.action_space = MagicMock()
    ucb.is_optimal_action = MagicMock(return_value=True)
    ucb.observe = MagicMock(side_effect=lambda arm, reward: reward)   # echoes reward as "q_value"
    ucb.update_optimal_value = MagicMock()
    ucb.update_blocklist_target_found = MagicMock()
    ucb.get_blocklist_coverage = MagicMock(return_value=0.0)
    ucb.prop_method = PropagationMethod.ON_RECEIPT
    ucb.propagate_rewards = MagicMock()
    ucb.disable_target = MagicMock()
    if reward_mode is not None:
        ucb.reward_mode = reward_mode
    return ucb


def _pending(target="a.com"):
    p = Pending(node_id="n1", arm_key="categories Pornography", arm_seq=["root", "categories Pornography"])
    return p


class TestRewardModeDefaultUnchanged:
    def test_reward_mode_unset_behaves_bit_for_bit_as_original(self):
        """No reward_mode attribute at all (every model_klass other than
        BatchUCB, or any BatchUCB built before this change) -- getattr(...,
        'majority') default keeps the original 0.0/1.0 behavior."""
        ucb = _ucb(reward_mode=None)
        ucb._selected_targets["a.com"] = _pending()
        out = ucb.absorb_measurement({"target": "a.com", "blocked": True, "vote_fraction": 0.37})
        assert out["rewards"] == 1.0
        ucb.observe.assert_called_once_with("categories Pornography", 1.0)

    def test_reward_mode_majority_explicit_same_as_unset(self):
        ucb = _ucb(reward_mode="majority")
        ucb._selected_targets["a.com"] = _pending()
        out = ucb.absorb_measurement({"target": "a.com", "blocked": False, "vote_fraction": 0.9})
        assert out["rewards"] == 0.0    # vote_fraction ignored entirely in majority mode


class TestPercentageRewardMode:
    def test_uses_vote_fraction_as_reward(self):
        ucb = _ucb(reward_mode="percentage")
        ucb._selected_targets["a.com"] = _pending()
        out = ucb.absorb_measurement({"target": "a.com", "blocked": False, "vote_fraction": 0.37})
        assert out["rewards"] == 0.37
        ucb.observe.assert_called_once_with("categories Pornography", 0.37)

    def test_is_blocked_and_downstream_fields_unaffected_by_reward_mode(self):
        """is_blocked (CSV column, update_blocklist_target_found) stays the
        boolean majority verdict regardless of reward_mode."""
        ucb = _ucb(reward_mode="percentage")
        ucb._selected_targets["a.com"] = _pending()
        out = ucb.absorb_measurement({"target": "a.com", "blocked": True, "vote_fraction": 0.12})
        assert out["is_blocked"] == 1               # majority said blocked...
        assert out["rewards"] == 0.12                # ...but reward is the fraction
        ucb.update_blocklist_target_found.assert_called_once_with("a.com")

    def test_missing_vote_fraction_falls_back_to_zero_or_one(self):
        """Old-shape input (no vote_fraction key, e.g. a stale checkpoint's
        replay) must not crash -- falls back to the 0.0/1.0 value."""
        ucb = _ucb(reward_mode="percentage")
        ucb._selected_targets["a.com"] = _pending()
        out = ucb.absorb_measurement({"target": "a.com", "blocked": True})
        assert out["rewards"] == 1.0

    def test_unanimous_and_zero_fractions_pass_through_exactly(self):
        for frac in (0.0, 1.0, 0.5):
            ucb = _ucb(reward_mode="percentage")
            ucb._selected_targets["a.com"] = _pending()
            out = ucb.absorb_measurement({"target": "a.com", "blocked": frac > 0.5, "vote_fraction": frac})
            assert out["rewards"] == frac


class TestAsnPercentageRewardMode:
    def test_uses_asn_vote_fraction_as_reward(self):
        ucb = _ucb(reward_mode="asn-percentage")
        ucb._selected_targets["a.com"] = _pending()
        out = ucb.absorb_measurement(
            {"target": "a.com", "blocked": False, "vote_fraction": 0.9, "asn_vote_fraction": 0.37}
        )
        # vote_fraction is ignored in this mode -- only asn_vote_fraction is read.
        assert out["rewards"] == 0.37
        ucb.observe.assert_called_once_with("categories Pornography", 0.37)

    def test_is_blocked_unaffected_by_reward_mode(self):
        ucb = _ucb(reward_mode="asn-percentage")
        ucb._selected_targets["a.com"] = _pending()
        out = ucb.absorb_measurement(
            {"target": "a.com", "blocked": True, "asn_vote_fraction": 0.12}
        )
        assert out["is_blocked"] == 1
        assert out["rewards"] == 0.12

    def test_missing_asn_vote_fraction_falls_back_to_zero_or_one(self):
        """Old-shape input (no asn_vote_fraction key) must not crash --
        falls back to the 0.0/1.0 value, same as the percentage mode does
        for a missing vote_fraction."""
        ucb = _ucb(reward_mode="asn-percentage")
        ucb._selected_targets["a.com"] = _pending()
        out = ucb.absorb_measurement({"target": "a.com", "blocked": True})
        assert out["rewards"] == 1.0
