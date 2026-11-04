r"""
offline_reward_replay.py

Offline replay of Experiments 5-7 (majority vote / percentage-by-VP / percentage-by-ASN
rewards) on an all-VPs run that has ALREADY been measured.

Every one of a country's ~2,000 measured targets has the true outcome from EVERY VP, so
any reward function can be recomputed for free. This replays a UCB bandit (same update as
the live run: sample-average q, bonus c*sqrt(ln(t+1)/n_a), untried arm = +inf) over the
category arms, where pulling an arm hands back one not-yet-seen measured target from that
category, and scores how fast each reward finds the blocked targets.

  reward "majority" : 1 if > 50% of VPs voted blocked           (Ex 5, what the live run used)
  reward "pct_vp"   : fraction of VPs that voted blocked        (Ex 6)
  reward "pct_asn"  : fraction of ASNs whose own majority voted blocked   (Ex 7)
("voted blocked" = exactly the live orchestrator vote: stateful block, or first attempt did
not match the template. Ground truth used for SCORING is independent of that:
  T10 = >= 10% of VPs saw a strict block (anomaly or stateful), T50 = > 50%.)

LIMITS: only targets the original run's bandit happened to pick are available, so each arm
has a small fixed pool (selection bias); pulls are one target at a time (the live run
batched 5); budgets above ~50% of the pool mostly re-visit the same targets.

Usage (from /home/cenrl/cenrl):
    python3 -m Infrastructure.main.offline_reward_replay \
        --agg /home/cenrl/cenrl_outputs/experiment_five/analysis/per_target_aggregates.json \
        --run-dir /home/cenrl/cenrl_outputs/experiment_five/output5 \
        --out-dir /home/cenrl/cenrl_outputs/experiment_five/analysis/offline_replay
"""
import argparse
import collections
import csv
import json
import os
import time
from multiprocessing import Pool

import numpy as np

COUNTRIES = ["China", "Germany", "Iran", "Thailand", "Turkey", "United Arab Emirates", "Vietnam"]
REWARDS = ["majority", "pct_vp", "pct_asn"]
C_GRID = [0.01, 0.03, 0.1, 0.3]
CHECKPOINTS = [100, 250, 500, 1000]


def rewards_from_aggregates(t, ta, targets):
    """t: {target: [n, anomaly, stateful, cf, first_mismatch, strict, vote]};
    ta: {"target\\tasn": [n, strict, vote]}.  Returns dict of float arrays aligned to
    `targets` plus the two ground-truth boolean arrays."""
    asn_votes = collections.defaultdict(list)
    for k, (n, strict, vote) in ta.items():
        tgt = k.split("\t")[0]
        asn_votes[tgt].append(vote / n > 0.5)
    idx = {x: i for i, x in enumerate(targets)}
    m = len(targets)
    maj, pv, pa, t10, t50 = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m, bool), np.zeros(m, bool)
    for x, i in idx.items():
        n, _, _, _, _, strict, vote = t[x]
        maj[i] = 1.0 if vote / n > 0.5 else 0.0
        pv[i] = vote / n
        av = asn_votes.get(x)
        pa[i] = float(np.mean(av)) if av else 0.0
        t10[i] = strict / n >= 0.10
        t50[i] = strict / n > 0.50
    return {"majority": maj, "pct_vp": pv, "pct_asn": pa}, t10, t50


def ucb_replay(arm_lists, reward, truths, steps, c, rng):
    """One UCB run. arm_lists: list of int arrays (target indices per arm). Returns
    (cumulative-found array of shape (len(truths), steps))."""
    K = len(arm_lists)
    lists = [rng.permutation(a) for a in arm_lists]
    size = np.array([len(a) for a in lists])
    ptr = np.zeros(K, int)
    n = np.zeros(K)
    q = np.zeros(K)
    found = np.zeros((len(truths), steps))
    cum = np.zeros(len(truths))
    for t in range(steps):
        avail = ptr < size
        if not avail.any():
            found[:, t:] = cum[:, None]
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(n > 0, q + c * np.sqrt(np.log(t + 2) / np.maximum(n, 1)), np.inf)
        score = np.where(avail, score, -np.inf)
        cands = np.flatnonzero(score == score.max())
        a = cands[rng.integers(len(cands))]
        tgt = lists[a][ptr[a]]
        ptr[a] += 1
        n[a] += 1
        q[a] += (reward[tgt] - q[a]) / n[a]
        for j, tr in enumerate(truths):
            cum[j] += tr[tgt]
        found[:, t] = cum
    return found


def random_baseline(m, truths, steps, rng):
    perm = rng.permutation(m)
    return np.stack([np.cumsum(tr[perm][:steps]) for tr in truths])


def oracle_baseline(truths, steps):
    out = []
    for tr in truths:
        k = int(tr.sum())
        out.append(np.minimum(np.arange(1, steps + 1), k).astype(float))
    return np.stack(out)


def load_country(agg_c, run_dir, country):
    cn = country.replace(" ", "_")
    rows = list(csv.DictReader(open(f"{run_dir}/{cn}/{cn}_measurements.csv")))
    targets = [r["target"] for r in rows]                       # run order
    arm_name = {r["target"]: r["arm"].replace("categories ", "") for r in rows}
    run_flag = {r["target"]: r["blocked"] == "1" for r in rows}
    rewards, t10, t50 = rewards_from_aggregates(agg_c["t"], agg_c["ta"], targets)
    arms = sorted(set(arm_name.values()))
    ai = {a: i for i, a in enumerate(arms)}
    arm_lists = [[] for _ in arms]
    for i, x in enumerate(targets):
        arm_lists[ai[arm_name[x]]].append(i)
    arm_lists = [np.array(a, int) for a in arm_lists]
    return dict(targets=targets, arms=arms, arm_lists=arm_lists, rewards=rewards,
                t10=t10, t50=t50, run_flag=np.array([run_flag[x] for x in targets]))


_DATA = {}


def _task(args):
    country, reward_name, c, seeds, steps = args
    d = _DATA[country]
    curves = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 * s + 7)
        curves.append(ucb_replay(d["arm_lists"], d["rewards"][reward_name], [d["t10"], d["t50"]],
                                 steps, c, rng))
    return (country, reward_name, c), np.stack(curves)          # (seeds, 2, steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    agg = json.load(open(args.agg))
    for c in COUNTRIES:
        _DATA[c] = load_country(agg[c], args.run_dir, c)
    # sanity: does the recomputed majority reproduce the live run's blocked flag?
    print("majority-vote reproduction of the live run's flag:")
    for c in COUNTRIES:
        d = _DATA[c]; mj = d["rewards"]["majority"] > 0.5
        print(f"  {c:22s} agree on {np.mean(mj == d['run_flag']):.1%} of targets  "
              f"(live {int(d['run_flag'].sum())}, recomputed {int(mj.sum())})")
    t0 = time.time()
    tasks = [(c, r, cg, args.seeds, args.steps) for c in COUNTRIES for r in REWARDS for cg in C_GRID]
    res = {}
    with Pool(args.workers) as pl:
        for k, arr in pl.imap_unordered(_task, tasks):
            res[k] = arr
    print(f"replayed {len(tasks)} configs x {args.seeds} seeds in {time.time() - t0:.0f}s")
    summary = {}
    for c in COUNTRIES:
        d = _DATA[c]; m = len(d["targets"]); truths = [d["t10"], d["t50"]]
        rb = np.stack([random_baseline(m, truths, args.steps, np.random.default_rng(5000 + s))
                       for s in range(args.seeds)])
        ob = oracle_baseline(truths, args.steps)
        actual = np.stack([np.cumsum(tr) for tr in truths])[:, :args.steps]      # live run's own order
        summary[c] = {"n_targets": m, "n_T10": int(d["t10"].sum()), "n_T50": int(d["t50"].sum()),
                      "random": rb.mean(0).tolist(), "oracle": ob.tolist(), "actual_run": actual.tolist(),
                      "configs": {}}
        for r in REWARDS:
            for cg in C_GRID:
                arr = res[(c, r, cg)]
                summary[c]["configs"][f"{r}|{cg}"] = {"mean": arr.mean(0).tolist(), "sd": arr.std(0).tolist()}
    json.dump(summary, open(f"{args.out_dir}/replay_summary.json", "w"))
    print(f"wrote {args.out_dir}/replay_summary.json")


if __name__ == "__main__":
    main()
