from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TaskCatalogGeneratorTest(unittest.TestCase):
    def test_large_catalog_mode_skips_quadratic_audit_and_varies_coalitions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "targets"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "generate_task_catalogs.py"),
                "--output-dir",
                str(output),
                "--train-count",
                "96",
                "--test-count",
                "128",
                "--max-steps",
                "24",
                "--min-window-steps",
                "4",
                "--max-window-steps",
                "8",
                "--minimum-separation-deg",
                "0",
                "--skip-separation-audit",
                "--cooperative-observers-min",
                "2",
                "--cooperative-observers-max",
                "5",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(
                (output / "catalog_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["split_audit"]["separation_audit"],
                "skipped_for_large_catalog",
            )
            self.assertIsNone(
                manifest["split_audit"]["minimum_cross_split_separation_deg"]
            )
            with (output / "train_requests.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            coalition_sizes = {
                int(row["required_observers"])
                for row in rows
                if row["cooperation_mode"] != "single"
            }
            self.assertEqual(coalition_sizes, {2, 3, 4, 5})

    def test_train_and_test_catalogs_are_balanced_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "targets"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "generate_task_catalogs.py"),
                "--output-dir",
                str(output),
                "--train-count",
                "120",
                "--test-count",
                "120",
                "--train-seed",
                "11",
                "--test-seed",
                "12",
                "--minimum-separation-deg",
                "0.1",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(
                (output / "catalog_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["train"]["rows"], 120)
            self.assertEqual(manifest["test"]["rows"], 120)
            self.assertEqual(
                manifest["split_audit"]["exact_coordinate_overlap"], 0
            )
            self.assertGreaterEqual(
                manifest["split_audit"]["minimum_cross_split_separation_deg"],
                0.099,
            )
            self.assertEqual(
                manifest["train"]["mode_counts"],
                {"infrared": 12, "optical": 72, "sar": 36},
            )
            self.assertEqual(
                manifest["train"]["cooperation_counts"],
                {"sequential": 24, "simultaneous": 12, "single": 84},
            )
            with (output / "train_requests.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 120)
            self.assertTrue(all(int(row["deadline_step"]) < 240 for row in rows))
            self.assertTrue(
                all(
                    int(row["required_observers"]) == 1
                    for row in rows
                    if row["cooperation_mode"] == "single"
                )
            )
            first_hashes = manifest["files"]
            repeated = subprocess.run(
                command + ["--overwrite"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            repeated_manifest = json.loads(
                (output / "catalog_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(repeated_manifest["files"], first_hashes)

    def test_area_catalogs_have_balanced_sizes_and_auto_collaboration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "targets"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "generate_task_catalogs.py"),
                "--output-dir",
                str(output),
                "--train-count",
                "120",
                "--test-count",
                "120",
                "--train-seed",
                "21",
                "--test-seed",
                "22",
                "--minimum-separation-deg",
                "0.1",
                "--area-fraction",
                "0.75",
                "--file-prefix",
                "area_",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (output / "area_train_requests.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            area_rows = [row for row in rows if row["target_type"] == "area"]
            point_rows = [row for row in rows if row["target_type"] == "point"]
            self.assertEqual(len(area_rows), 90)
            self.assertEqual(len(point_rows), 30)
            self.assertTrue(
                all(row["cooperation_mode"] == "auto" for row in area_rows)
            )
            self.assertTrue(
                all(float(row["area_width_km"]) > 0.0 for row in area_rows)
            )
            manifest = json.loads(
                (output / "area_catalog_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["train"]["target_type_counts"],
                {"area": 90, "point": 30},
            )


if __name__ == "__main__":
    unittest.main()
