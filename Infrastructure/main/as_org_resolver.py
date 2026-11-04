"""
as_org_resolver.py

Loads CAIDA's AS-Organizations dataset and exposes a simple ASN -> HQ
country lookup. Complementary to (not a replacement for) the CAIDA pfx2as
data AsnAwareVantagePoints already uses: pfx2as tells you WHICH ASN
originates an IP; this tells you WHERE THE ORGANIZATION THAT REGISTERED
THAT ASN IS HEADQUARTERED. The two can legitimately disagree -- Apple's
edge server can be correctly, physically located in Hong Kong (a GeoIP
database would agree) while Apple Inc. itself is headquartered in the US
(this file says US) -- and that mismatch is exactly the signal that caught
Apple/Oracle being used as mislabeled "Hong Kong"/"South Korea" VPs this
session, WITHOUT needing to hand-list specific ASN numbers.

Known limitation (confirmed empirically against real flaky VPs, not
theoretical): this only catches VPs whose claimed country doesn't match
their owning org's registered country. A VP that's a low-quality hosting/
VPS provider but IS genuinely, correctly registered in its claimed country
(e.g. a small local dedicated-server company) won't be caught by this --
that's a different problem (hosting/datacenter ASN vs residential/business
ISP), not solved here.

Download the dataset (free, no auth) from:
    https://publicdata.caida.org/datasets/as-organizations/
    (pick the latest YYYYMMDD.as-org2info.jsonl.gz)

Usage:
    resolver = AsOrgResolver("local/as-org2info-20260801.jsonl.gz")
    resolver.country_for_asn("6185")  # -> "US" (Apple Inc.)
"""

import gzip
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ISO 3166-1 alpha-2 -> the exact country name strings this codebase's VP
# pool / orchestrator / analyze_censorship.py use elsewhere. Only covers
# countries actually in play for this study -- extend as needed rather than
# pulling in a general country-code library for a handful of lookups.
ISO2_TO_COUNTRY_NAME = {
    "AU": "Australia",
    "BR": "Brazil",
    "CA": "Canada",
    "CN": "China",
    "FR": "France",
    "DE": "Germany",
    "HK": "Hong Kong",
    "IN": "India",
    "JP": "Japan",
    "NL": "Netherlands",
    "SG": "Singapore",
    "KR": "South Korea",
    "GB": "United Kingdom",
    "US": "United States",
}
# Reverse mapping, used to convert a *claimed* VP country name to its ISO2
# code so it can be compared directly against country_for_asn()'s ISO2
# output -- deliberately NOT done the other way around (resolved ISO2 ->
# name via ISO2_TO_COUNTRY_NAME) since the true registered country can be
# anywhere in the world (e.g. a VP claiming "United States" but actually
# registered in Honduras), not just one of the ~14 countries in this dict.
COUNTRY_NAME_TO_ISO2 = {name: iso2 for iso2, name in ISO2_TO_COUNTRY_NAME.items()}


class AsOrgResolver:
    def __init__(self, as_org_path: str):
        self._asn_to_country: dict = {}
        self._load(as_org_path)

    def _load(self, path: str) -> None:
        asn_to_org: dict = {}
        org_to_country: dict = {}
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rtype = d.get("type")
                if rtype == "ASN":
                    asn_to_org[d.get("asn")] = d.get("organizationId")
                elif rtype == "Organization":
                    org_to_country[d.get("organizationId")] = d.get("country")

        for asn, org_id in asn_to_org.items():
            country = org_to_country.get(org_id)
            if country:
                self._asn_to_country[asn] = country

        logger.info(
            "Loaded AS-Organizations data from %s: %d ASNs resolved to a country",
            path, len(self._asn_to_country),
        )

    def country_for_asn(self, asn: str) -> Optional[str]:
        """ISO 3166-1 alpha-2 code (e.g. 'US') for the org that registered
        this ASN, or None if unknown -- callers should fail open on None,
        not treat it as a mismatch."""
        return self._asn_to_country.get(str(asn))

    def country_name_matches(self, asn: str, claimed_country_name: str) -> Optional[bool]:
        """True if the ASN's registered org country matches claimed_country_name,
        False if it's a confirmed mismatch, None if we can't tell (ASN not in
        this dataset, or claimed_country_name isn't one of our known study
        countries) -- callers should fail open (treat None as "don't
        exclude") the same way AsnResolver.asn_for() fails open on DNS
        errors elsewhere in this repo.

        Compares by ISO2 code (claimed name -> ISO2, vs. the ASN's resolved
        ISO2 directly) rather than converting the resolved ISO2 back to a
        name -- the true registered country can be anywhere in the world
        (e.g. Honduras), not just one of our ~14 study countries, so only
        the claimed side needs to be one of ours.
        """
        resolved_iso2 = self.country_for_asn(asn)
        if resolved_iso2 is None:
            return None
        claimed_iso2 = COUNTRY_NAME_TO_ISO2.get(claimed_country_name)
        if claimed_iso2 is None:
            # claimed_country_name isn't one of our known study countries --
            # can't do a clean comparison, don't guess.
            return None
        return resolved_iso2 == claimed_iso2
