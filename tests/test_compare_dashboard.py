from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CompareDashboardTests(unittest.TestCase):
    def test_dashboard_covers_every_algorithm_in_suite(self) -> None:
        dashboard = (ROOT / "web" / "compare.html").read_text(encoding="utf-8")
        for algorithm in ("oasis", "ippo", "mappo", "qmix", "ps_dqn"):
            self.assertIn(f'id: "{algorithm}"', dashboard)


if __name__ == "__main__":
    unittest.main()
