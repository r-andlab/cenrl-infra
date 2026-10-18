r"""
run_all_vps_linucb.py

LinUCB-family counterpart to run_all_vps.py (which drove output3-output12
with BatchUCB): same all-VPs pipeline -- every target measured from every
usable VP in each country at once, same VP health/quarantine/silent-drop
logic in orchestrator.py (none of that is UCB-specific, see that file) --
but driven by BatchLinUCB (--no-cluster), BatchClusterLinUCB (default), or
BatchLinUCBShared (--shared, one global A/b across every arm instead of one
per arm) instead of BatchUCB.

Deliberately does NOT reuse run_experiment_one_linucb.py's orchestrator
wiring: that file uses CollisionAwareOrchestrator, which was removed from
run_all_vps.py the day before output12 ran (commit 65ec578) -- copying it
here would make a LinUCB result not actually comparable to output12 on that
axis. This file's parser/orchestrator wiring mirrors run_all_vps.py itself,
with model_klass/action_space_klass swapped the same way
run_experiment_one_linucb.py already demonstrates for LinUCB specifically.

Adds one feature run_all_vps.py doesn't need: --vp-cap, a hard per-country
VP pool limit applied with the same round-robin-across-ASN diversity
AsnAwareVantagePoints.get_n_vantages() uses internally, computed read-only
(get_inactive() + get_asn(), no pool state mutation) and applied via a
second restrict_to() call before Orchestrator is constructed -- so
Orchestrator._bootstrap_vps()'s own later draw-and-register logic is
completely unaffected, it just sees a smaller, already-diverse pool.

Usage (from /home/cenrl/cenrl):
    python3 -m Infrastructure.main.run_all_vps_linucb \
        --usable-vps /home/cenrl/cenrl_outputs/vp_evaluation_oct_recheck/usable_vps_linucb.json \
        --vp-cap 2000 \
        -E 1 -m 2000 -v -f "categories" \
        -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv \
        -alpha 0.5 -V 0.0 --reward-mode percentage --no-cluster \
        -o /home/cenrl/cenrl_outputs/experiment_six/output_linucb1

Hyperquack must be started the same way run_all_vps.py's own docstring
describes (all-VPs config, NumWorkers=12000, MaxBadEvaluations=1,000,000):
    go run cmd/server/worker_server.go -config config_allvps.json --logoutdir tests/<n>

--dry-run swaps Hyperquack for synthetic in-process results (no network).
"""

import json

from Infrastructure.main.asn_aware_vantage_points import AsnAwareVantagePoints
from Infrastructure.main.orchestrator import Orchestrator, STRATEGY_MAP
from models.contextual_bandit.cluster_linucb import ClusterLinUCBParserOptions
from models.contextual_bandit.linucb import LinUCBActionSpace
from models.contextual_bandit.batch_linucb import BatchLinUCB, BatchClusterLinUCB, BatchLinUCBShared


class AllVpsLinUCBParser(ClusterLinUCBParserOptions):
    """Same orchestrator-level flags OrchestrationParser adds on top of
    UCBNaiveParserOptions (Infrastructure/main/orchestrator.py), layered on
    top of ClusterLinUCBParserOptions instead -- these two flag stacks can't
    be diamond-inherited directly (run_experiment_one_linucb.py's
    LinUCBOrchestrationParser re-declares them for the same reason), plus
    run_all_vps.py's own --usable-vps/--dry-run/--ev-file/--endpoint flags
    and --vp-cap, this file's one new addition."""

    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument(
            "--aggregation", choices=["majority", "weighted"], default="majority",
            help="VP-vote aggregation method",
        )
        self.parser.add_argument(
            "--batch-size", choices=["constant", "vary-on-success"], default="constant",
            help="Batch-size policy",
        )
        self.parser.add_argument(
            "--target-selection", choices=["top-k", "uniform-spread", "weighted-spread"],
            default="top-k", help="Target-selection method within an arm",
        )
        self.parser.add_argument(
            "--propagation", choices=["on-receipt", "in-order"], default="on-receipt",
            help="Reward-propagation policy",
        )
        self.parser.add_argument(
            "--reward-mode", choices=["majority", "percentage"], default="majority",
            help="Reward signal: 'majority' (the aggregation verdict's 0.0/1.0) or "
                 "'percentage' (Ex 6, fraction of voting VPs that voted blocked) -- "
                 "requires the reward_mode branch in BatchLinUCBMixin.absorb_measurement "
                 "(models/contextual_bandit/batch_linucb.py), ported there from "
                 "BatchUCB.py specifically so this flag isn't silently inert here.",
        )
        self.parser.add_argument(
            "--previous-values-folder", type=str, default=None,
            help="Per-country warm-start folder -- NOTE: existing prior files are "
                 "q_value/action_attempts CSVs built for BatchUCB's arm representation, "
                 "not compatible with LinUCB's A/b matrices, so this is untested for a "
                 "LinUCB-family run. Leave unset (cold start) until that's validated.",
        )
        self.parser.add_argument(
            "--no-cluster", action="store_true",
            help="Use plain LinUCB (BatchLinUCB) instead of ClusterLinUCB (BatchClusterLinUCB). "
                 "Ignored if --shared is also passed.",
        )
        self.parser.add_argument(
            "--shared", action="store_true",
            help="Use LinUCBShared (one global A/b matrix shared across every arm, "
                 "instead of one per arm) via BatchLinUCBShared. Takes priority over "
                 "--no-cluster if both are passed.",
        )
        self.parser.add_argument(
            "--usable-vps", required=True,
            help="{country: [ip, ...]} VP list, e.g. usable_vps_linucb.json",
        )
        self.parser.add_argument(
            "--vp-cap", type=int, default=None,
            help="Hard per-country VP pool limit, applied with the same "
                 "round-robin-across-ASN diversity get_n_vantages() uses internally. "
                 "Omit for no cap (whole usable-vps pool, matching run_all_vps.py).",
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
            "--receiver-port", type=int, default=9000,
            help="Port this process's own MeasurementReceiver listens on for "
                 "Hyperquack's callbacks (Infrastructure/apis/server.py). "
                 "Hardcoded to 9000 everywhere before this flag existed, which "
                 "meant no two CenRL orchestrator processes could ever run on "
                 "the same machine at once. Only needs changing to run a "
                 "second process alongside an existing one.",
        )

    def set_params(self, args):
        if args.outfile[-1] != "/":
            args.outfile += "/"
        super().set_params(args)
        self.params["aggregation"] = args.aggregation
        self.params["batch_size_method"] = getattr(args, "batch_size")
        self.params["target_selection"] = getattr(args, "target_selection")
        self.params["propagation"] = args.propagation
        self.params["reward_mode"] = getattr(args, "reward_mode")
        self.params["previous_values_folder"] = getattr(args, "previous_values_folder")
        self.params["no_cluster"] = args.no_cluster
        self.params["shared"] = args.shared
        self.params["usable_vps"] = args.usable_vps
        self.params["vp_cap"] = args.vp_cap
        self.params["dry_run"] = args.dry_run
        self.params["endpoint"] = args.endpoint
        self.params["receiver_port"] = args.receiver_port
        self.params["ev_file"] = args.ev_file


def asn_round_robin_select(pool: AsnAwareVantagePoints, country: str, cap: int) -> list:
    """Pick at most `cap` VPs for `country`, round-robining across ASN
    groups (largest-pool-first, one VP per visited ASN per pass) the same
    way AsnAwareVantagePoints.get_n_vantages() does internally -- but
    read-only: built purely from get_inactive()/get_asn(), never touches
    the pool's active/inactive state. Safe to call before Orchestrator
    exists, since nothing has been drawn yet at that point anyway.
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
    parser = AllVpsLinUCBParser()
    params = parser.parse()

    agg_enum  = STRATEGY_MAP["aggregation"][params["aggregation"]]
    size_enum = STRATEGY_MAP["batch_size_method"][params["batch_size_method"]]
    sel_enum  = STRATEGY_MAP["target_selection"][params["target_selection"]]
    prop_enum = STRATEGY_MAP["propagation"][params["propagation"]]

    if params["shared"]:
        model_klass = BatchLinUCBShared
    elif params["no_cluster"]:
        model_klass = BatchLinUCB
    else:
        model_klass = BatchClusterLinUCB

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
        model_klass=model_klass,
        action_space_klass=LinUCBActionSpace,
        receiver_port=params["receiver_port"],
    )
    m.run_forever()
