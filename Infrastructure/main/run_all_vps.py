r"""
run_all_vps.py

Experiment 5/6 entry point: the same bandit/aggregation stack as
run_experiment_one.py (UCB), but every target is measured from EVERY usable
VP in each country at the same time, instead of from 3 rotating VPs.
--reward-mode picks the reward: "majority" (Ex 5, default -- the
aggregation_method's 0.0/1.0 blocked verdict) or "percentage" (Ex 6 -- the
fraction of reporting VPs that voted blocked; the UCB policy itself is
unchanged, only what number counts as the reward). See MeasurementResponse
.vote_fraction (Infrastructure/utils/structures.py) and
BatchUCB.absorb_measurement for where this is actually read.

"Usable" = passed Hyperquack's own evaluation (evaluate_all_vps.py ->
usable_vps.json). The pool is restricted to exactly that list before the run
starts, and Orchestrator(all_vps_mode=True) then:
  - draws the whole pool at bootstrap (no vps_per_country),
  - never draws replacement VPs (nothing is left to draw),
  - has no 15-minute force-swap and never redirects a measurement to a
    different VP.
There is NO quorum: a target is finalized only once every VP still in its
expected set has reported (aggregator default), so the slowest VP sets a
country's pace. Waiting is logged at INFO ("taking longer than usual -- N min
so far, still waiting on K of M VP(s)"), not as a warning.

Which VPs get pulled out of circulation (see the ALL_VPS_* constants in
orchestrator.py):
  - VPs that fail evaluation at startup: dropped for good;
  - VPs Hyperquack no longer has / is removing: dropped for good (its own
    signal, not our impatience);
  - a VP that has NEVER returned a normal or blocked result and then fails its
    controls 10 times in a row (its results abstain): quarantined;
  - a VP that worked before but then fails its controls SLOWLY (>=60 s, i.e.
    timeouts) 5 times in a row (unreachable; its results abstain): quarantined.
Quarantine (ALL_VPS_QUARANTINE_MIN, 30 min) means excluded from new dispatches
and not dropped -- it stays registered with Hyperquack the whole time and is
retried after the window (re-probing output5's 137 permanent drops after that
run finished found roughly half would have recovered; see that constant's
comment). A VP that worked and fails its controls FAST (connection resets) is
NOT quarantined and keeps its "blocked" vote: in China that pattern is
residual censorship after a blocked domain, not a dead VP. Hyperquack's own
lifetime bad-evaluations drop is switched off (MaxBadEvaluations=1,000,000)
because it counts those too. A measurement is resent (to the same VP) only
after 30 minutes and only if the VP is neither queued for it nor running
anything.

Hyperquack must be started with the all-VPs config (NumWorkers=12000, 32 parallel
result senders, MaxBadEvaluations=1,000,000) and a fresh --logoutdir. In
/home/cenrl/hyperquackv2:
    go run cmd/server/worker_server.go -config config_allvps.json --logoutdir tests/54

Usage (from /home/cenrl/cenrl; add --reward-mode percentage for Ex 6):
    python3 -m Infrastructure.main.run_all_vps \
        --usable-vps /home/cenrl/cenrl_outputs/vp_evaluation_all/usable_vps.json \
        --vp-cap 2000 \
        -E 1 -m 2000 -v -f "categories" \
        -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv \
        -s 0.0 -c 0.03 -V 0.0 --reward-mode percentage \
        -o /home/cenrl/cenrl_outputs/experiment_six/output1

--vp-cap is optional (omit for no cap, the pre-existing behavior); without
it, a country with a much larger raw VP pool than others (e.g. China's
thousands of candidates vs. Iran's handful) finalizes every target far
slower than the rest, since all_vps_mode waits for every active VP to
report before a target finalizes -- so pool size directly sets the pace.
Ported from run_all_vps_linucb.py's --vp-cap/asn_round_robin_select.

--dry-run swaps Hyperquack for synthetic in-process results (no network) to
exercise the Python side at full VP count.
"""

import json

from Infrastructure.main.asn_aware_vantage_points import AsnAwareVantagePoints
from Infrastructure.main.orchestrator import Orchestrator, OrchestrationParser, STRATEGY_MAP


class AllVpsParser(OrchestrationParser):
    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument(
            "--usable-vps", required=True,
            help="usable_vps.json from evaluate_all_vps.py ({country: [ip, ...]})",
        )
        self.parser.add_argument(
            "--dry-run", action="store_true",
            help="Synthetic in-process results instead of Hyperquack (no network)",
        )
        self.parser.add_argument(
            "--ev-file", default="local/ev-certs-monthly-000000000000.csv.gz",
            help="Census the candidate pool is built from; must contain every VP in --usable-vps",
        )
        self.parser.add_argument(
            "--endpoint", default="http://127.0.0.1:8888",
            help="Hyperquack API endpoint",
        )
        self.parser.add_argument(
            "--vp-cap", type=int, default=None,
            help="Hard per-country VP pool limit, applied with the same "
                 "round-robin-across-ASN diversity get_n_vantages() uses internally. "
                 "Omit for no cap (whole usable-vps pool).",
        )

    def set_params(self, args):
        super().set_params(args)
        self.params["usable_vps"] = args.usable_vps
        self.params["dry_run"] = args.dry_run
        self.params["endpoint"] = args.endpoint
        self.params["ev_file"] = args.ev_file
        self.params["vp_cap"] = args.vp_cap


def asn_round_robin_select(pool: AsnAwareVantagePoints, country: str, cap: int) -> list:
    """Pick at most `cap` VPs for `country`, round-robining across ASN
    groups (largest-pool-first, one VP per visited ASN per pass) the same
    way AsnAwareVantagePoints.get_n_vantages() does internally -- but
    read-only: built purely from get_inactive()/get_asn(), never touches
    the pool's active/inactive state. Safe to call before Orchestrator
    exists, since nothing has been drawn yet at that point anyway.
    Ported verbatim from run_all_vps_linucb.py's asn_round_robin_select.
    """
    ips = pool.get_inactive(country)
    if len(ips) <= cap:
        return ips

    by_asn = {}
    for ip in ips:
        by_asn.setdefault(pool.get_asn(ip), []).append(ip)

    asns = sorted(by_asn, key=lambda a: -len(by_asn[a]))
    selected = []
    i = 0
    while len(selected) < cap and any(by_asn[a] for a in asns):
        asn = asns[i % len(asns)]
        if by_asn[asn]:
            selected.append(by_asn[asn].pop(0))
        i += 1
    return selected


if __name__ == "__main__":
    parser = AllVpsParser()
    params = parser.parse()

    agg_enum  = STRATEGY_MAP["aggregation"][params["aggregation"]]
    size_enum = STRATEGY_MAP["batch_size_method"][params["batch_size_method"]]
    sel_enum  = STRATEGY_MAP["target_selection"][params["target_selection"]]
    prop_enum = STRATEGY_MAP["propagation"][params["propagation"]]

    usable = json.load(open(params["usable_vps"]))
    countries = sorted(usable)

    vp_pool_config = {
        "ev_file":           params["ev_file"],
        "blocklist_file":    "local/blocklist.txt",
        "max_countries":     0,
        "blocked_countries": [],
        "caida_dat":         "local/ipasn_20260720.dat",
        "guaranteed_countries": countries,
        "excluded_asns":     ["714", "6185", "31898"],
        "as_org_path":       None,
        "excluded_vps_file": "local/excluded_vps.txt",
    }
    base_pool = AsnAwareVantagePoints(**vp_pool_config)
    for c in countries:
        n = base_pool.restrict_to(c, usable[c])
        if params["vp_cap"] is not None and n > params["vp_cap"]:
            capped = asn_round_robin_select(base_pool, c, params["vp_cap"])
            n = base_pool.restrict_to(c, capped)
        print(f"{c}: {n} usable VPs in pool (census listed {len(usable[c])})", flush=True)
    m = Orchestrator(
        params=params,
        vantage_points=base_pool,
        go_api_endpoint=params["endpoint"],
        services=["https"],
        countries=countries,
        vps_per_country=3,          # unused in all_vps_mode (whole pool is drawn)
        previous_values_folder=params["previous_values_folder"],
        vp_pool_config=vp_pool_config,
        aggregation_method=agg_enum,
        target_selection=sel_enum,
        batch_size_method=size_enum,
        qval_propagation_method=prop_enum,
        debug=params["dry_run"],
        all_vps_mode=True,
        reward_mode=params["reward_mode"],
    )
    m.run_forever()
