"""Offline reward replay: reward definitions and the UCB replay on tiny synthetic data."""
import numpy as np

from Infrastructure.main.offline_reward_replay import (
    rewards_from_aggregates, ucb_replay, random_baseline, oracle_baseline)


def _agg():
    # counters: [n, anomaly, stateful, cf, first_mismatch, strict, vote]
    t = {"a.com": [100, 0, 0, 0, 0, 0, 0],        # nobody blocks
         "b.com": [100, 60, 0, 0, 60, 60, 60],    # majority blocks
         "c.com": [100, 15, 0, 0, 15, 15, 15]}    # 15% block: partial
    ta = {"a.com\t1": [50, 0, 0], "a.com\t2": [50, 0, 0],
          "b.com\t1": [50, 45, 45], "b.com\t2": [50, 15, 15],
          "c.com\t1": [50, 15, 15], "c.com\t2": [50, 0, 0]}
    return t, ta, ["a.com", "b.com", "c.com"]


def test_reward_definitions():
    t, ta, targets = _agg()
    r, t10, t50 = rewards_from_aggregates(t, ta, targets)
    assert list(r["majority"]) == [0.0, 1.0, 0.0]          # only b.com has a majority
    assert np.allclose(r["pct_vp"], [0.0, 0.6, 0.15])
    assert np.allclose(r["pct_asn"], [0.0, 0.5, 0.0])       # b.com: ASN1 blocks, ASN2 doesn't
    assert list(t10) == [False, True, True] and list(t50) == [False, True, False]


def _world(seed=0, arms=10, per_arm=20, blocked_arm=0, frac=0.2):
    """Arm `blocked_arm` has `frac` of its targets partially blocked; all others none."""
    rng = np.random.default_rng(seed)
    m = arms * per_arm
    arm_lists = [np.arange(a * per_arm, (a + 1) * per_arm) for a in range(arms)]
    truth = np.zeros(m, bool)
    truth[arm_lists[blocked_arm][:int(per_arm * frac)]] = True
    return arm_lists, truth


def test_pct_reward_finds_partial_blocking_that_majority_cannot():
    arm_lists, truth = _world()
    pct = np.where(truth, 0.2, 0.0)            # 20% of VPs block -> a small but real signal
    maj = np.zeros_like(pct)                    # majority never fires
    steps, seeds = 80, 40
    f = lambda rw, c: np.mean([ucb_replay(arm_lists, rw, [truth], steps, c, np.random.default_rng(s))[0, -1]
                               for s in range(seeds)])
    rand = np.mean([random_baseline(len(truth), [truth], steps, np.random.default_rng(s))[0, -1]
                    for s in range(seeds)])
    assert f(pct, 0.03) > 1.3 * rand            # signal -> the bandit concentrates on the blocked arm
    assert abs(f(maj, 0.03) - rand) < 0.35 * rand + 0.5   # no signal -> indistinguishable from random


def test_majority_reward_concentrates_when_blocking_is_universal():
    arm_lists, truth = _world(frac=1.0, per_arm=30)
    rw = truth.astype(float)
    got = np.mean([ucb_replay(arm_lists, rw, [truth], 60, 0.03, np.random.default_rng(s))[0, -1] for s in range(20)])
    assert got >= 25                            # nearly every pull after the exploration phase hits the blocked arm


def test_replay_is_deterministic_and_bounded():
    arm_lists, truth = _world()
    a = ucb_replay(arm_lists, truth.astype(float), [truth], 50, 0.03, np.random.default_rng(3))
    b = ucb_replay(arm_lists, truth.astype(float), [truth], 50, 0.03, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert np.all(np.diff(a[0]) >= 0)
    assert a[0, -1] <= min(50, truth.sum())
    assert np.all(oracle_baseline([truth], 50)[0] >= a[0])


def test_exhausting_the_pool_does_not_crash():
    arm_lists, truth = _world(arms=3, per_arm=5)
    out = ucb_replay(arm_lists, truth.astype(float), [truth], 100, 0.03, np.random.default_rng(0))
    assert out.shape == (1, 100) and out[0, -1] == truth.sum()      # saw everything, then stopped
