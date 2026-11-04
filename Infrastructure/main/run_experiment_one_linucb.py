"""
run_experiment_one_linucb.py

Same experiment_one 6-country pool as run_experiment_one.py (China, Turkey,
Thailand, Vietnam, United Arab Emirates, Iran), but driven by the
contextual-bandit algorithm (ClusterLinUCB by default, or plain LinUCB via
--no-cluster) instead of BatchUCB.

Requires models/contextual_bandit/batch_linucb.py (BatchLinUCB /
BatchClusterLinUCB), which adds the batching/async layer node.py's live
flow needs (queue_measurement/register_batch/can_step/absorb_measurement/
flush_held_batches) on top of the LinUCB-family choose_arm()/observe() --
see that file's module docstring for why this was needed and how it's
built (a mixin, so BatchUCB.py itself is never touched).

Usage (same flags as run_experiment_one.py, plus LinUCB's own --alpha /
--alpha_arm / --alpha_cluster -- see ClusterLinUCBParserOptions):
    python3 Infrastructure/main/run_experiment_one_linucb.py \\
        -E 1 -m 1000 -v -f "categories" \\
        -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv \\
        -alpha 0.5 -V 0.0 \\
        -o /home/cenrl/cenrl_outputs/experiment_one/output_linucb1

Add --no-cluster to run plain LinUCB (BatchLinUCB) instead of ClusterLinUCB.
"""

import argparse

from Infrastructure.main.asn_aware_vantage_points import AsnAwareVantagePoints
from Infrastructure.main.vp_pool_adapter import VpPoolAdapter
from Infrastructure.main.asn_resolver import AsnResolver
from Infrastructure.main.cdn_collision_filter import CdnCollisionFilter
from Infrastructure.main.collision_aware_orchestrator import CollisionAwareOrchestrator
from Infrastructure.main.orchestrator import STRATEGY_MAP
from Infrastructure.main.run_experiment_one import EXPERIMENT_ONE_COUNTRIES
from models.contextual_bandit.cluster_linucb import ClusterLinUCBParserOptions
from models.contextual_bandit.linucb import LinUCBActionSpace
from models.contextual_bandit.batch_linucb import BatchLinUCB, BatchClusterLinUCB


class LinUCBOrchestrationParser(ClusterLinUCBParserOptions):
    """Same orchestrator-level flags OrchestrationParser adds on top of
    UCBNaiveParserOptions (models/main/orchestrator.py), layered on top of
    ClusterLinUCBParserOptions instead of UCBNaiveParserOptions."""

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
            "--previous-values-folder", type=str, default=None,
            help="Per-country warm-start folder (see OrchestrationParser's own flag "
                 "for the full description) -- NOTE: existing prior files are "
                 "q_value/action_attempts CSVs built for BatchUCB's arm representation, "
                 "not compatible with LinUCB's A/b matrices, so this is untested for a "
                 "LinUCB-family run. Leave unset (cold start) until that's validated.",
        )
        self.parser.add_argument(
            "--no-cluster", action="store_true",
            help="Use plain LinUCB (BatchLinUCB) instead of ClusterLinUCB (BatchClusterLinUCB)",
        )

    def set_params(self, args):
        super().set_params(args)
        self.params["aggregation"] = args.aggregation
        self.params["batch_size_method"] = getattr(args, "batch_size")
        self.params["target_selection"] = getattr(args, "target_selection")
        self.params["propagation"] = args.propagation
        self.params["previous_values_folder"] = getattr(args, "previous_values_folder")
        self.params["no_cluster"] = args.no_cluster


if __name__ == "__main__":
    parser = LinUCBOrchestrationParser()
    params = parser.parse()

    agg_enum  = STRATEGY_MAP["aggregation"][params["aggregation"]]
    size_enum = STRATEGY_MAP["batch_size_method"][params["batch_size_method"]]
    sel_enum  = STRATEGY_MAP["target_selection"][params["target_selection"]]
    prop_enum = STRATEGY_MAP["propagation"][params["propagation"]]

    model_klass = BatchLinUCB if params["no_cluster"] else BatchClusterLinUCB

    excluded_asns = ["714", "6185", "31898"]
    as_org_path = None
    excluded_vps_file = "local/excluded_vps.txt"

    vp_pool_config = {
        "ev_file":           "local/ev-certs-monthly-000000000000.csv.gz",
        "blocklist_file":    "local/blocklist.txt",
        "max_countries":     0,
        "blocked_countries": [],
        "caida_dat":         "local/ipasn_20260720.dat",
        "guaranteed_countries": EXPERIMENT_ONE_COUNTRIES,
        "excluded_asns":     excluded_asns,
        "as_org_path":       as_org_path,
        "excluded_vps_file": excluded_vps_file,
    }
    vp_pool = VpPoolAdapter(AsnAwareVantagePoints(**vp_pool_config))
    collision_filter = CdnCollisionFilter(
        AsnResolver(vp_pool_config["caida_dat"]), vp_pool, min_vps=2,
    )

    m = CollisionAwareOrchestrator(
        params=params,
        vantage_points=vp_pool,
        go_api_endpoint="http://127.0.0.1:8888",
        services=["https"],
        vps_per_country=3,
        previous_values_folder=params["previous_values_folder"],
        vp_pool_config=vp_pool_config,
        aggregation_method=agg_enum,
        target_selection=sel_enum,
        batch_size_method=size_enum,
        qval_propagation_method=prop_enum,
        collision_filter=collision_filter,
        model_klass=model_klass,
        action_space_klass=LinUCBActionSpace,
    )
    m.run_forever()
