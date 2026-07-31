from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from metrics_to_tensorboard import discover_metrics, sync_metrics


class FakeWriter:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.text: list[tuple[str, str, int]] = []
        self.flushes = 0

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def add_text(self, tag: str, value: str, step: int) -> None:
        self.text.append((tag, value, step))

    def flush(self) -> None:
        self.flushes += 1


class MetricsToTensorBoardTest(unittest.TestCase):
    def test_existing_history_and_new_episode_are_mirrored_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rl64_seed701"
            run = root / "ippo"
            run.mkdir(parents=True)
            metrics = run / "metrics.json"
            payload = {
                "history": [
                    {
                        "episode": 1,
                        "mean_reward": 0.25,
                        "completed_tasks": 12,
                        "decision_opportunity_rate": 0.08,
                        "mean_loss": None,
                    },
                    {
                        "episode": 2,
                        "mean_reward": 0.50,
                        "completed_tasks": 18,
                        "mean_loss": 0.125,
                    },
                ]
            }
            metrics.write_text(json.dumps(payload), encoding="utf-8")

            output = Path(temporary) / "events"
            writers: dict[str, FakeWriter] = {}
            last_steps: dict[str, int] = {}
            factory = lambda log_dir: FakeWriter(log_dir)

            self.assertEqual(discover_metrics([root]), [(root, metrics)])
            self.assertEqual(
                sync_metrics(root, metrics, output, writers, last_steps, factory), 2
            )
            writer = next(iter(writers.values()))
            self.assertIn(("train/mean_reward", 0.5, 2), writer.scalars)
            self.assertIn(("tasks/completed", 18.0, 2), writer.scalars)
            self.assertNotIn(("loss/mean", 0.0, 1), writer.scalars)

            self.assertEqual(
                sync_metrics(root, metrics, output, writers, last_steps, factory), 0
            )
            payload["history"].append(
                {"episode": 3, "mean_reward": 0.75, "invalid_actions": 0}
            )
            metrics.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                sync_metrics(root, metrics, output, writers, last_steps, factory), 1
            )
            self.assertIn(("constraints/invalid_actions", 0.0, 3), writer.scalars)
            self.assertEqual(last_steps[str(metrics.resolve())], 3)


if __name__ == "__main__":
    unittest.main()
