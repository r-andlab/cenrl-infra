Experiment mappings:

1. Naive UCB, partial reward with VPs. This is same configuration as experiment 5, so `data/exp1` is just a symlink. 
2. LinUCB, partial reward with VPs.
3. LinUCB Shared, partial reward with VPs.
4. ClusterUCB, partial reward with VPs.
5. Partial reward (with VPs) Naive UCB.
6. Partial reward (with ASNs) Naive UCB.
7. Majority vote Naive UCB.

Baseline is Tranco rank order sampling from CenRL.

Use the `<country_name>.csv` files. 

The baseline and `exp7` contain additional "fractional voting" files that record
the fraction of VPs that voted whether a domain was blocked. These experiments
contain additional `<country_name>_fractional.csv` files which report fractional
block levels under "rewards". All other experiments report "fractional voting"
in the "rewards" column of the data files.

`exp2_asns`, `exp5_asns`, and `exp6_asns` contains raw data from measurements to
help determine each ASN's vote.

`tranco_categories_subdomain_tld_entities_top10k.csv` is the list of domains we
used.
