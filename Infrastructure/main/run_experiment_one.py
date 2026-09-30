"""
run_experiment_one.py

Standalone entry point for "experiment_one" -- a deliberately separate study
from the main 14-country pipeline, targeting 6 countries chosen for having
well-documented, internationally-recognized-as-distinct censorship
reputations (unlike most of the original 14, which have shown no confirmed
national censorship signal all session -- every non-zero result traced back
to a VP-representativeness artifact, not policy):

  - China                 the study's existing anchor/validated baseline
  - Turkey                administrative/court-order blocking, not DPI-driven.
                           output1/output2 both confirm this is real (porn,
                           gambling) but only visible when the VP draw
                           includes an ordinary consumer ISP -- bank/hosting
                           VPs (AS39095 Vakiflar Bankasi, AS207681 KKB) show
                           nothing.
  - Thailand              legally distinct motivation (lese-majeste law)
  - Vietnam               one-party state, often compared to China but a very
                           different (ISP-cooperation-driven) mechanism
  - United Arab Emirates  different motivation (VoIP/moral/religious), and a
                           major regional hosting hub. Confirmed real in
                           output2 (porn, gambling, grindr.com specifically)
                           -- same VP-type-dependence as Turkey: AS15802 (du,
                           a real UAE consumer ISP) shows it, AS8075
                           (Microsoft Azure) doesn't.
  - Iran                  added per explicit request -- the single most
                           internationally documented, systemic censorship
                           regime of any candidate, but the pool is thin:
                           only 83 candidate VPs after existing filtering
                           (vs. 1,999-8,123 for the other five). Real risk of
                           running out of replacement VPs mid-run given how
                           fast China alone churned through VPs this
                           experiment (7 force-rejections in <3 hours with a
                           pool nearly 100x this size) -- if Iran's region
                           dies mid-run ("no remaining vantage points"),
                           that's an expected, reportable outcome, not a bug.

Deliberately excludes Russia per explicit direction: its VP/hosting
landscape is too heterogeneous to be a clean, single comparison point.

This is a thin wrapper around the exact same orchestration stack
orchestrator.py's __main__ uses (AsnAwareVantagePoints, VpPoolAdapter,
CdnCollisionFilter, CollisionAwareOrchestrator) -- nothing about the
measurement/aggregation/model code differs. The ONLY thing that differs is
vp_pool_config: guaranteed_countries is set to exactly this list, and
max_countries=0 so nothing outside that guaranteed set survives
_trim_to_top_countries() (which otherwise deletes any country's VP pool
outright if it's neither in the top-N-by-volume nor guaranteed -- see that
method's docstring in asn_aware_vantage_points.py). Kept as a separate
script rather than editing orchestrator.py's own __main__, so the main
pipeline's default 14-country behavior is never at risk of being left in a
half-edited state.

Usage (same flags as orchestrator.py -- output dir is the one thing you
must point at experiment_one/outputN per run):
    python3 Infrastructure/main/run_experiment_one.py \\
        -E 1 -m 1000 -v -f "categories" \\
        -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv \\
        -s 0.0 -c 0.03 -V 0.0 \\
        -o /home/cenrl/cenrl_outputs/experiment_one/output1
"""

from Infrastructure.main.asn_aware_vantage_points import AsnAwareVantagePoints
from Infrastructure.main.vp_pool_adapter import VpPoolAdapter
from Infrastructure.main.asn_resolver import AsnResolver
from Infrastructure.main.cdn_collision_filter import CdnCollisionFilter
from Infrastructure.main.collision_aware_orchestrator import CollisionAwareOrchestrator
from Infrastructure.main.orchestrator import OrchestrationParser, STRATEGY_MAP

EXPERIMENT_ONE_COUNTRIES = ["China", "Turkey", "Thailand", "Vietnam", "United Arab Emirates", "Iran"]

if __name__ == "__main__":
    parser = OrchestrationParser()
    params = parser.parse()

    agg_enum  = STRATEGY_MAP["aggregation"][params["aggregation"]]
    size_enum = STRATEGY_MAP["batch_size_method"][params["batch_size_method"]]
    sel_enum  = STRATEGY_MAP["target_selection"][params["target_selection"]]
    prop_enum = STRATEGY_MAP["propagation"][params["propagation"]]

    # Same exclusion machinery as the main pipeline (Tier 1 ASN exclusion,
    # the hand-curated excluded_vps.txt list) -- no reason a "weird VP"
    # finding from the main study shouldn't also apply here.
    excluded_asns = ["714", "6185", "31898"]
    as_org_path = None
    excluded_vps_file = "local/excluded_vps.txt"

    vp_pool_config = {
        "ev_file":           "local/ev-certs-monthly-000000000000.csv.gz",
        "blocklist_file":    "local/blocklist.txt",
        # max_countries=0 + guaranteed_countries=exactly these 5 means
        # _trim_to_top_countries() keeps ONLY this list, regardless of
        # volume ranking -- see module docstring above.
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
    )
    m.run_forever()
