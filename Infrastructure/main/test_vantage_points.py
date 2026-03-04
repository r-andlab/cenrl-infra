import logging
import os
import tempfile
import unittest
import pandas as pd

from Infrastructure.main.vantage_points import VantagePoints

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EV_CSV = os.path.join(PROJECT_ROOT, "ev-certs.csv")
BLOCKLIST = os.path.join(PROJECT_ROOT, "blocklist.txt")


def make_ev_csv(rows):
    """Utility that returns path of a temporary CSV containing the given rows."""
    df = pd.DataFrame(rows)
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    df.to_csv(path, index=False)
    return path


def make_blocklist(ips):
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "w") as f:
        for ip in ips:
            f.write(ip + "\n")
    return path


class TestVantagePoints(unittest.TestCase):
    def test_parse_and_limit_size(self):
        # create three entries for same country but max_size=2 should trim
        rows = [
            {"ipv4": "1.1.1.1", "ipv6": "", "country": "A", "port": 443},
            {"ipv4": "2.2.2.2", "ipv6": "", "country": "A", "port": 443},
            {"ipv4": "3.3.3.3", "ipv6": "", "country": "A", "port": 443},
        ]
        csv = make_ev_csv(rows)
        vp = VantagePoints(ev_file=csv, blocklist_file=None, max_size=2)
        inactive = vp.get_inactive("A")
        self.assertEqual(len(inactive), 2)
        # all returned addresses should come from original set
        self.assertTrue(all(ip in {"1.1.1.1", "2.2.2.2", "3.3.3.3"} for ip in inactive))
        os.remove(csv)

    def test_blocklist_applied(self):
        rows = [
            {"ipv4": "10.0.0.1", "country": "X"},
            {"ipv4": "10.0.0.2", "country": "X"},
        ]
        csv = make_ev_csv(rows)
        blk = make_blocklist(["10.0.0.1/32"])
        vp = VantagePoints(ev_file=csv, blocklist_file=blk, max_size=10)
        self.assertEqual(vp.get_inactive("X"), ["10.0.0.2"])
        os.remove(csv)
        os.remove(blk)

    def test_get_and_evaluate_good(self):
        rows = [{"ipv4": "4.4.4.4", "country": "B"}]
        csv = make_ev_csv(rows)
        vp = VantagePoints(ev_file=csv, blocklist_file=None, max_size=5)

        chosen = vp.get_vantage("B")
        self.assertEqual(chosen, "4.4.4.4")
        self.assertEqual(vp.get_active("B"), ["4.4.4.4"])
        self.assertEqual(vp.get_inactive("B"), [])

        # tell it the point was OK - should move back to inactive
        vp.evaluate("B", "4.4.4.4", ok=True)
        self.assertEqual(vp.get_active("B"), [])
        self.assertEqual(vp.get_inactive("B"), ["4.4.4.4"])
        os.remove(csv)

    def test_get_and_evaluate_bad_replacement(self):
        rows = [
            {"ipv4": "5.5.5.5", "country": "C"},
            {"ipv4": "6.6.6.6", "country": "C"},
        ]
        csv = make_ev_csv(rows)
        vp = VantagePoints(ev_file=csv, blocklist_file=None, max_size=5)

        # grab first ip
        first = vp.get_vantage("C")
        self.assertIn(first, ["5.5.5.5", "6.6.6.6"])
        # there should be one remaining inactive
        self.assertEqual(len(vp.get_inactive("C")), 1)

        # mark first as bad, ensure it is removed and replacement active
        vp.evaluate("C", first, ok=False)
        self.assertNotIn(first, vp.get_active("C"))
        self.assertNotIn(first, vp.get_inactive("C"))
        # now exactly one active and zero inactive
        act = vp.get_active("C")
        self.assertEqual(len(act), 1)
        self.assertEqual(vp.get_inactive("C"), [])
        os.remove(csv)


    def test_real_data(self):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        vp = VantagePoints(ev_file=EV_CSV, blocklist_file=BLOCKLIST, max_size=50)

        countries = vp.countries()
        print(f"\n--- Real data: {len(countries)} countries loaded ---")

        total_inactive = 0
        for country in sorted(countries):
            inactive = vp.get_inactive(country)
            total_inactive += len(inactive)
            print(f"  {country}: {len(inactive)} vantage points")

        print(f"Total vantage points after blocklist filtering: {total_inactive}")
        self.assertGreater(len(countries), 0)


if __name__ == "__main__":
    unittest.main()
