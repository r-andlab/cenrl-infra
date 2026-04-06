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
    def test_parse_keeps_all_vps(self):
        # all VPs should be kept (no trimming)
        rows = [
            {"ipv4": "1.1.1.1", "ipv6": "", "country": "A", "port": 443},
            {"ipv4": "2.2.2.2", "ipv6": "", "country": "A", "port": 443},
            {"ipv4": "3.3.3.3", "ipv6": "", "country": "A", "port": 443},
        ]
        csv = make_ev_csv(rows)
        vp = VantagePoints(ev_file=csv, blocklist_file=None)
        inactive = vp.get_inactive("A")
        self.assertEqual(len(inactive), 3)
        self.assertEqual(set(inactive), {"1.1.1.1", "2.2.2.2", "3.3.3.3"})
        os.remove(csv)

    def test_max_countries(self):
        # max_countries=1 should keep only the country with the most VPs
        rows = [
            {"ipv4": "1.1.1.1", "country": "A"},
            {"ipv4": "2.2.2.2", "country": "B"},
            {"ipv4": "3.3.3.3", "country": "B"},
        ]
        csv = make_ev_csv(rows)
        vp = VantagePoints(ev_file=csv, blocklist_file=None, max_countries=1)
        self.assertEqual(vp.countries(), ["B"])
        self.assertEqual(len(vp.get_inactive("B")), 2)
        self.assertEqual(vp.get_inactive("A"), [])
        os.remove(csv)

    def test_blocklist_applied(self):
        rows = [
            {"ipv4": "10.0.0.1", "country": "X"},
            {"ipv4": "10.0.0.2", "country": "X"},
        ]
        csv = make_ev_csv(rows)
        blk = make_blocklist(["10.0.0.1/32"])
        vp = VantagePoints(ev_file=csv, blocklist_file=blk)
        self.assertEqual(vp.get_inactive("X"), ["10.0.0.2"])
        os.remove(csv)
        os.remove(blk)

    def test_get_and_evaluate_good(self):
        rows = [{"ipv4": "4.4.4.4", "country": "B"}]
        csv = make_ev_csv(rows)
        vp = VantagePoints(ev_file=csv, blocklist_file=None)

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
        vp = VantagePoints(ev_file=csv, blocklist_file=None)

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


    def test_get_n_vantages(self):
        rows = [
            {"ipv4": f"7.7.7.{i}", "country": "D"}
            for i in range(1, 6)
        ]
        csv = make_ev_csv(rows)
        vp = VantagePoints(ev_file=csv, blocklist_file=None)

        drawn = vp.get_n_vantages("D", 3)
        self.assertEqual(len(drawn), 3)
        self.assertEqual(len(vp.get_active("D")), 3)
        self.assertEqual(len(vp.get_inactive("D")), 2)

        # drawing more than remaining gives only what's left
        extra = vp.get_n_vantages("D", 5)
        self.assertEqual(len(extra), 2)
        self.assertEqual(len(vp.get_active("D")), 5)
        self.assertEqual(len(vp.get_inactive("D")), 0)
        os.remove(csv)

    def test_confirm_active(self):
        rows = [{"ipv4": "8.8.8.8", "country": "E"}]
        csv = make_ev_csv(rows)
        vp = VantagePoints(ev_file=csv, blocklist_file=None)

        drawn = vp.get_vantage("E")
        self.assertEqual(drawn, "8.8.8.8")
        # confirm_active should keep it in active (no-op)
        vp.confirm_active("E", "8.8.8.8")
        self.assertEqual(vp.get_active("E"), ["8.8.8.8"])
        self.assertEqual(vp.get_inactive("E"), [])
        os.remove(csv)

    def test_reject_vp_with_replacement(self):
        rows = [
            {"ipv4": "9.9.9.1", "country": "F"},
            {"ipv4": "9.9.9.2", "country": "F"},
        ]
        csv = make_ev_csv(rows)
        vp = VantagePoints(ev_file=csv, blocklist_file=None)

        first = vp.get_vantage("F")
        replacement = vp.reject_vp("F", first)
        self.assertIsNotNone(replacement)
        self.assertNotEqual(replacement, first)
        self.assertIn(replacement, vp.get_active("F"))
        self.assertNotIn(first, vp.get_active("F"))
        self.assertNotIn(first, vp.get_inactive("F"))
        os.remove(csv)

    def test_reject_vp_no_replacement(self):
        rows = [{"ipv4": "11.11.11.11", "country": "G"}]
        csv = make_ev_csv(rows)
        vp = VantagePoints(ev_file=csv, blocklist_file=None)

        first = vp.get_vantage("G")
        replacement = vp.reject_vp("G", first)
        self.assertIsNone(replacement)
        self.assertEqual(vp.get_active("G"), [])
        self.assertEqual(vp.get_inactive("G"), [])
        os.remove(csv)

    def test_real_data(self):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        vp = VantagePoints(ev_file=EV_CSV, blocklist_file=BLOCKLIST0)

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
