"""
evaluate_all_vps.py

Evaluation-only census of the VP candidate pool: registers every candidate VP
with a (dedicated) Hyperquack instance so it runs only Hyperquack's evaluation
(8 control requests + template match), then counts how many pass per country
and per ASN. No measurements, no bandit.

Hyperquack must be started with a config where TestResultsOutputFile ==
EvaluationResultsOutputFile (so results are written locally instead of POSTed
to the Python receiver) and a large NumWorkers, e.g.:

    cd /home/cenrl/hyperquackv2
    go run cmd/server/worker_server.go -config config_evalall.json \\
        --logoutdir tests_evalall/all_vps

Usage (from the repo root):
    python3 -m Infrastructure.main.evaluate_all_vps \\
        --endpoint http://127.0.0.1:8889 \\
        --eval-file /home/cenrl/hyperquackv2/tests_evalall/all_vps/eval.json \\
        --out-dir /home/cenrl/cenrl_outputs/vp_evaluation_all
"""

import argparse
import collections
import csv
import json
import logging
import re
import time
from pathlib import Path

import requests

from Infrastructure.main.asn_aware_vantage_points import AsnAwareVantagePoints

DEFAULT_COUNTRIES = [
    "China", "Turkey", "Thailand", "Vietnam", "United Arab Emirates", "Iran", "Germany",
]


def build_pool(countries):
    return AsnAwareVantagePoints(
        ev_file="local/ev-certs-monthly-000000000000.csv.gz",
        blocklist_file="local/blocklist.txt",
        max_countries=0,
        blocked_countries=[],
        caida_dat="local/ipasn_20260720.dat",
        guaranteed_countries=countries,
        excluded_asns=["714", "6185", "31898"],
        as_org_path=None,
        excluded_vps_file="local/excluded_vps.txt",
    )


def candidates(pool, countries):
    out = {}
    for c in countries:
        for pools in pool._asn_pools.get(c, {}).values():
            for ip in pools["inactive"] | pools["active"]:
                out[ip] = c
    return out


def classify(issue: str) -> str:
    if "Bodies do not match" in issue:
        return "bodies_do_not_match"
    if "Client.Timeout" in issue or "i/o timeout" in issue or "deadline" in issue:
        return "timeout"
    if "reset" in issue:
        return "reset"
    if "refused" in issue:
        return "refused"
    if "no route" in issue or "unreachable" in issue:
        return "unreachable"
    if "EOF" in issue:
        return "eof"
    if re.search(r"(Cipher|Certificate|tls|TLS)", issue):
        return "tls"
    if "do not match" in issue:
        return "other_template_mismatch"
    return "other"


def read_eval(eval_file):
    """Return {ip: issue_or_None} using the first record per VP."""
    res = {}
    p = Path(eval_file)
    if not p.exists():
        return res
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ip = r.get("vp")
            if ip and ip not in res:
                res[ip] = r.get("issue") or None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8889")
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    ap.add_argument("--only", default=None,
                    help="usable_vps.json from an earlier pass: evaluate ONLY the VPs "
                         "listed there (to refresh a usable list quickly)")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--stall-secs", type=int, default=300)
    ap.add_argument("--max-secs", type=int, default=7200)
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = build_pool(args.countries)
    cand = candidates(pool, args.countries)
    if args.only:
        keep = {ip for ips in json.load(open(args.only)).values() for ip in ips}
        cand = {ip: c for ip, c in cand.items() if ip in keep}
    ips = list(cand)
    print(f"candidates: {len(ips)}  " + ", ".join(
        f"{c}={sum(1 for v in cand.values() if v == c)}" for c in args.countries), flush=True)

    invalid = []
    accepted = 0
    for i in range(0, len(ips), args.chunk):
        chunk = ips[i:i + args.chunk]
        body = {"vantage_points": [
            {"ip": ip, "services": pool.get_services(ip, ["https"])} for ip in chunk]}
        r = requests.post(args.endpoint + "/add-vantage-points", json=body, timeout=600)
        r.raise_for_status()
        resp = r.json()
        bad = resp.get("invalid_entries") or resp.get("invalid") or resp.get("Invalid") or []
        invalid.extend(b.get("ip") if isinstance(b, dict) else b for b in bad)
        accepted += len(chunk) - len(bad)
        print(f"  registered {min(i + args.chunk, len(ips))}/{len(ips)} (invalid so far {len(invalid)})", flush=True)

    expected = accepted
    print(f"accepted {accepted}, invalid {len(invalid)}; waiting for {expected} evaluations", flush=True)
    t0 = time.time()
    last_n, last_change = -1, time.time()
    while True:
        n = len(read_eval(args.eval_file))
        if n != last_n:
            last_n, last_change = n, time.time()
            print(f"  [{int(time.time() - t0):5d}s] evaluated {n}/{expected}", flush=True)
        if n >= expected:
            break
        if time.time() - last_change > args.stall_secs:
            print("  no progress for stall window; stopping wait", flush=True)
            break
        if time.time() - t0 > args.max_secs:
            print("  max wait reached", flush=True)
            break
        time.sleep(15)

    ev = read_eval(args.eval_file)
    per_country = collections.defaultdict(lambda: {"candidates": 0, "invalid": 0, "evaluated": 0, "passed": 0})
    per_asn = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))  # country->asn->[eval,pass]
    reasons = collections.defaultdict(collections.Counter)
    usable = collections.defaultdict(list)
    invalid_set = set(invalid)
    for ip, c in cand.items():
        pc = per_country[c]
        pc["candidates"] += 1
        if ip in invalid_set:
            pc["invalid"] += 1
            continue
        if ip not in ev:
            continue
        pc["evaluated"] += 1
        asn = pool.get_asn(ip)
        per_asn[c][asn][0] += 1
        if ev[ip] is None:
            pc["passed"] += 1
            per_asn[c][asn][1] += 1
            usable[c].append(ip)
        else:
            reasons[c][classify(ev[ip])] += 1

    print("\ncountry                 candidates  invalid  evaluated  passed   pass%   (of evaluated)")
    tot = collections.Counter()
    for c in args.countries:
        pc = per_country[c]
        rate = pc["passed"] / pc["evaluated"] if pc["evaluated"] else 0
        print(f"{c:22s}{pc['candidates']:11d}{pc['invalid']:9d}{pc['evaluated']:11d}{pc['passed']:8d}{rate:8.1%}")
        for k, v in pc.items():
            tot[k] += v
    rate = tot["passed"] / tot["evaluated"] if tot["evaluated"] else 0
    print(f"{'TOTAL':22s}{tot['candidates']:11d}{tot['invalid']:9d}{tot['evaluated']:11d}{tot['passed']:8d}{rate:8.1%}")
    print("\nusable ASNs per country:")
    for c in args.countries:
        n_asn = sum(1 for a, (e, p) in per_asn[c].items() if p > 0)
        print(f"  {c:22s}{n_asn:4d} ASNs with >=1 usable VP (of {len(per_asn[c])})")
    print("\nfailure reasons:")
    for c in args.countries:
        print(f"  {c:22s}{dict(reasons[c].most_common())}")

    (out_dir / "summary.json").write_text(json.dumps(
        {"per_country": per_country, "failure_reasons": {c: dict(v) for c, v in reasons.items()},
         "total": dict(tot)}, indent=2))
    (out_dir / "usable_vps.json").write_text(json.dumps(usable, indent=2))
    with (out_dir / "per_asn.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country", "asn", "evaluated", "passed"])
        for c in args.countries:
            for asn, (e, p) in sorted(per_asn[c].items(), key=lambda kv: -kv[1][0]):
                w.writerow([c, asn, e, p])
    print(f"\nwrote {out_dir}/summary.json, usable_vps.json, per_asn.csv")


if __name__ == "__main__":
    main()
