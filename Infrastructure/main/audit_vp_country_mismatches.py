"""
audit_vp_country_mismatches.py

Standalone add-on -- does not touch orchestrator.py's live VP-loading path
(as_org_path is deliberately left off there; see its comment for why an
auto-exclude filter turned out too noisy at scale). This script instead
automates the exact manual investigation done by hand earlier this
session: for every VP that actually cast a vote in a completed run, check
whether its ASN is registered to an organization headquartered in a
DIFFERENT country than the VP's claimed one (e.g. an Apple- or
Oracle-owned IP used as a "Hong Kong"/"South Korea" VP), and report the
mismatches for manual review.

Deliberately scoped to VPs that actually appeared in one run's raw
Hyperquack results (a few dozen, not the full multi-hundred-thousand
candidate pool) -- auditing the whole pool blind flags ~46% of everything
(mostly legitimate multinational hosting/telecom infrastructure, not
single-tenant self-collisions), which is far too noisy to read through by
hand. A completed run's actual VPs is exactly the small, concrete,
already-relevant set worth spending review time on.

Usage:
    python3 audit_vp_country_mismatches.py \\
        --hq-tests /home/cenrl/hyperquackv2/tests/26 \\
        --as-org-path local/as-org2info-20260801.jsonl.gz \\
        --caida-dat local/ipasn_20260720.dat
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import pyasn

from Infrastructure.main.as_org_resolver import AsOrgResolver


def find_country_result_files(hq_tests_dir: str) -> dict:
    """country_name -> path, for every <Country>_result.jsonl in the tests dir."""
    found = {}
    for path in glob.glob(os.path.join(hq_tests_dir, "*_result.jsonl")):
        country = Path(path).name[: -len("_result.jsonl")]
        found[country] = path
    return found


def unique_vps_per_country(hq_tests_dir: str) -> dict:
    """country -> set of every VP IP that appears anywhere in that country's
    raw results -- i.e. every VP that actually cast at least one vote in
    this run, VP rotation/replacements included, not just the initial
    bootstrap set."""
    result = {}
    for country, path in find_country_result_files(hq_tests_dir).items():
        vps = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vp = d.get("vp")
                if vp:
                    vps.add(vp)
        result[country] = vps
    return result


def audit_run(hq_tests_dir: str, as_org_path: str, caida_dat: str):
    resolver = AsOrgResolver(as_org_path)
    asn_db = pyasn.pyasn(caida_dat)

    country_vps = unique_vps_per_country(hq_tests_dir)
    if not country_vps:
        print(f"No *_result.jsonl files found under {hq_tests_dir}")
        return

    total_checked = 0
    total_mismatch = 0
    total_unresolvable = 0
    mismatches_by_country = defaultdict(list)

    for country, vps in sorted(country_vps.items()):
        for vp in sorted(vps):
            asn, _ = asn_db.lookup(vp)
            if asn is None:
                total_unresolvable += 1
                continue
            total_checked += 1
            matches = resolver.country_name_matches(str(asn), country)
            if matches is False:
                total_mismatch += 1
                resolved_iso2 = resolver.country_for_asn(str(asn))
                mismatches_by_country[country].append((vp, asn, resolved_iso2))

    print(f"Audited {total_checked} distinct VPs across {len(country_vps)} countries "
          f"({total_unresolvable} had no resolvable ASN, skipped)")
    print(f"{total_mismatch} confirmed country mismatch(es)\n")

    if not mismatches_by_country:
        print("No mismatches found -- every VP's ASN registration matches its claimed country.")
        return

    for country, entries in sorted(mismatches_by_country.items()):
        print(f"=== {country} ({len(entries)} mismatch(es)) ===")
        for vp, asn, resolved_iso2 in entries:
            print(f"  {vp:16s} ASN {str(asn):8s} -> registered in {resolved_iso2} (claims {country})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Flag VPs whose ASN is registered to an org headquartered in a "
                     "different country than they claim, for a completed run's actual VPs."
    )
    parser.add_argument("--hq-tests", required=True,
                        help="Hyperquack tests/N directory with the raw *_result.jsonl files")
    parser.add_argument("--as-org-path", required=True,
                        help="CAIDA AS-Organizations file, e.g. local/as-org2info-20260801.jsonl.gz")
    parser.add_argument("--caida-dat", required=True,
                        help="CAIDA pfx2as .dat file for ASN resolution, e.g. local/ipasn_20260720.dat")
    args = parser.parse_args()

    audit_run(args.hq_tests, args.as_org_path, args.caida_dat)


if __name__ == "__main__":
    main()
