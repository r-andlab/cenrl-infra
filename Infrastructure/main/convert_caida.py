"""
convert_caida.py

Converts a CAIDA RouteViews pfx2as .gz file into a pyasn .dat file
that AsnAwareVantagePoints (and pyasn generally) can query.

Usage from command line:
    python convert_caida.py routeviews-rv2-20260504-1200_pfx2as.gz ipasn_20260504.dat

Usage as a library:
    from convert_caida import convert
    convert("routeviews-rv2-20260504-1200_pfx2as.gz", "ipasn_20260504.dat")

Download the latest pfx2as files from:
    https://data.caida.org/datasets/routing/routeviews-prefix2as/
"""

import argparse
import logging
from bz2 import BZ2File
from gzip import GzipFile

logger = logging.getLogger(__name__)


def _open_archive(fpath: str):
    """Open a .gz or .bz2 archive and return a file-like object."""
    GZIP_MAGIC = b"\x1f\x8b"
    BZ2_MAGIC  = b"\x42\x5a\x68"
    with open(fpath, "rb") as fh:
        hdr = fh.read(max(len(BZ2_MAGIC), len(GZIP_MAGIC)))
    if hdr.startswith(BZ2_MAGIC):
        return BZ2File(fpath, "rb")
    elif hdr.startswith(GZIP_MAGIC):
        return GzipFile(fpath, "rb")
    else:
        raise TypeError(f"Cannot determine archive type for '{fpath}'")


def convert(gz_path: str, dat_path: str) -> int:
    """
    Convert a CAIDA pfx2as .gz file to a pyasn .dat file.

    Returns the number of prefixes written.
    """
    try:
        from pyasn import mrtx
    except ImportError:
        raise ImportError(
            "pyasn is required. Install it with: pip install pyasn"
        )

    logger.info("Converting %s -> %s", gz_path, dat_path)
    prefixes = {}

    with _open_archive(gz_path) as data:
        for raw_line in data:
            line = raw_line.decode().strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            ip, length, asn = parts
            prefixes[f"{ip}/{length}"] = asn

    mrtx.dump_prefixes_to_file(prefixes, dat_path, [gz_path])
    logger.info("Wrote %d prefixes to %s", len(prefixes), dat_path)
    return len(prefixes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Convert a CAIDA pfx2as .gz file to a pyasn .dat file"
    )
    parser.add_argument("gz_path",  help="Input .gz file from CAIDA")
    parser.add_argument("dat_path", help="Output .dat file for pyasn")
    args = parser.parse_args()
    n = convert(args.gz_path, args.dat_path)
    print(f"Done — {n:,} prefixes written to {args.dat_path}")