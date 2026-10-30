r"""
aggregate_all_vps_results.py

One streaming pass over an all-VPs run's raw Hyperquack results
(<tests-dir>/<Country>_result.jsonl, one record per VP x target) that boils them
down to per-target / per-(target, ASN) / per-VP counters small enough to analyse
and replay offline (see offline_reward_replay.py).

Per target, counters are  [n, anomaly, stateful, controls_failed, first_mismatch,
strict, vote]  where
  strict = anomaly or stateful_block            (Hyperquack's own "blocked": the
                                                 keyword failed 10x but a control worked)
  vote   = stateful_block or first attempt did not match the template
                                                (exactly what Orchestrator._feed_aggregator
                                                 votes -- includes controls_failed)
Per (target, ASN): [n, strict, vote].  Per VP: [n, strict, controls_failed].

Usage (from /home/cenrl/cenrl):
    python3 -m Infrastructure.main.aggregate_all_vps_results \
        --tests-dir /home/cenrl/hyperquackv2/tests/54 \
        --out /home/cenrl/cenrl_outputs/experiment_five/analysis/per_target_aggregates.json
"""
import argparse
import collections
import json
import logging
import os
import time
from multiprocessing import Pool

DEFAULT_COUNTRIES = ["China", "Germany", "Iran", "Thailand", "Turkey",
                     "United Arab Emirates", "Vietnam"]
CHUNK = 256 * 1024 * 1024
_ASN = {}


def build_asn_map(countries):
    from Infrastructure.main.asn_aware_vantage_points import AsnAwareVantagePoints
    pool = AsnAwareVantagePoints(
        ev_file="local/ev-certs-monthly-000000000000.csv.gz", blocklist_file="local/blocklist.txt",
        max_countries=0, blocked_countries=[], caida_dat="local/ipasn_20260720.dat",
        guaranteed_countries=countries, excluded_asns=["714", "6185", "31898"],
        as_org_path=None, excluded_vps_file="local/excluded_vps.txt")
    asn = {}
    for c in countries:
        for p in pool._asn_pools.get(c, {}).values():
            for ip in p["inactive"] | p["active"]:
                asn[ip] = pool.get_asn(ip)
    return asn


def record_fields(r):
    """(target, vp, anomaly, stateful, controls_failed, first_mismatch, strict, vote)"""
    an = 1 if r.get("anomaly") else 0
    sb = 1 if r.get("stateful_block") else 0
    cf = 1 if r.get("controls_failed") else 0
    resp = r.get("response") or []
    fm = 1 if (resp and not resp[0].get("matches_template")) else 0
    return r["test_url"], r["vp"], an, sb, cf, fm, (1 if (an or sb) else 0), (1 if (sb or fm) else 0)


def work(job):
    country, path, start, end = job
    per_t = collections.defaultdict(lambda: [0] * 7)
    per_ta = collections.defaultdict(lambda: [0, 0, 0])
    per_v = collections.defaultdict(lambda: [0, 0, 0])
    bad = 0
    with open(path, "rb") as f:
        if start > 0:                       # start mid-file: skip to the next line boundary
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()
        else:
            f.seek(0)
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            try:
                t, vp, an, sb, cf, fm, strict, vote = record_fields(json.loads(line))
            except Exception:
                bad += 1
                continue
            x = per_t[t]
            for i, v in enumerate((1, an, sb, cf, fm, strict, vote)):
                x[i] += v
            y = per_ta[(t, _ASN.get(vp, "?"))]
            y[0] += 1; y[1] += strict; y[2] += vote
            z = per_v[vp]
            z[0] += 1; z[1] += strict; z[2] += cf
    return country, dict(per_t), {f"{k[0]}\t{k[1]}": v for k, v in per_ta.items()}, dict(per_v), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    _ASN.update(build_asn_map(args.countries))
    t0 = time.time()
    jobs = []
    for c in args.countries:
        p = f"{args.tests_dir}/{c}_result.jsonl"
        sz = os.path.getsize(p)
        jobs += [(c, p, s, min(s + CHUNK, sz)) for s in range(0, sz, CHUNK)]
    out = {c: {"t": collections.defaultdict(lambda: [0] * 7),
               "ta": collections.defaultdict(lambda: [0, 0, 0]),
               "v": collections.defaultdict(lambda: [0, 0, 0]), "bad": 0} for c in args.countries}
    with Pool(args.workers) as pl:                     # forked workers inherit _ASN
        for i, (c, pt, pta, pv, bad) in enumerate(pl.imap_unordered(work, jobs)):
            o = out[c]; o["bad"] += bad
            for k, v in pt.items():
                for j in range(7): o["t"][k][j] += v[j]
            for k, v in pta.items():
                for j in range(3): o["ta"][k][j] += v[j]
            for k, v in pv.items():
                for j in range(3): o["v"][k][j] += v[j]
            if i % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} chunks, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({c: {k: (dict(v) if isinstance(v, dict) else v) for k, v in o.items()} for c, o in out.items()},
              open(args.out, "w"))
    print(f"wrote {args.out} in {time.time() - t0:.0f}s; malformed lines: {sum(o['bad'] for o in out.values())}")


if __name__ == "__main__":
    main()
