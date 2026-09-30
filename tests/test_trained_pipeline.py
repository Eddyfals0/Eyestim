"""Integration checks for the locally installed research bundle.

Dataset/checkpoint tests skip cleanly in a fresh clone because the licensed,
large artifacts are intentionally ignored by Git.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import config
from diameter_estimator import LearnedDiameterEstimator
from reporter import SessionReporter


MANIFEST = ROOT / "data" / "external" / "eyedentify" / "manifest.csv"
DIAMETER_CHECKPOINT = ROOT / "src" / "models" / "eyestim_diameter.pth"
FINAL_SUMMARY = ROOT / "docs" / "training" / "final_validation_summary.json"


class TrainedPipelineTests(unittest.TestCase):
    def test_validated_threshold_is_active(self):
        self.assertEqual(config.PUPIL_THRESHOLD_OFFSET, 35)

    @unittest.skipUnless(MANIFEST.exists(), "EyeDentify is a local licensed dataset")
    def test_participant_splits_are_disjoint(self):
        with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        groups = {
            split: {
                int(row["participant_id"])
                for row in rows
                if row["split"] == split
            }
            for split in ("train", "validation", "test")
        }
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertEqual(len(set.union(*groups.values())), 51)

    @unittest.skipUnless(
        MANIFEST.exists() and DIAMETER_CHECKPOINT.exists(),
        "EyeDentify/checkpoint are local licensed artifacts",
    )
    def test_diameter_checkpoint_produces_physical_finite_output(self):
        with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        image = cv2.imread(str(MANIFEST.parent / row["image_path"]))
        self.assertIsNotNone(image)
        prediction = LearnedDiameterEstimator(DIAMETER_CHECKPOINT).predict(image)
        self.assertTrue(np.isfinite(prediction))
        self.assertGreaterEqual(prediction, 1.0)
        self.assertLessEqual(prediction, 9.0)

    def test_frozen_results_beat_naive_baselines(self):
        summary = json.loads(FINAL_SUMMARY.read_text(encoding="utf-8"))
        diameter = json.loads(
            (ROOT / "docs" / "training" / "training_summary.json").read_text(
                encoding="utf-8"
            )
        )
        center = json.loads(
            (ROOT / "docs" / "training" / "center_training_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(summary["eyedentify_integrity"]["passed"])
        self.assertLess(
            diameter["test"]["mae_mm"],
            diameter["constant_train_mean_baseline"]["mae_mm"],
        )
        self.assertLess(
            center["lpw_model"]["mean_error_px"],
            center["legacy_bioid_model"]["mean_error_px"],
        )
        self.assertGreaterEqual(
            summary["cambridge_end_to_end_held_out"]["detection_rate"], 0.99
        )

    def test_reporter_writes_markdown_and_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            reporter = SessionReporter(temporary)
            for index in range(20):
                reporter.log_frame(
                    10.0 + index / 100,
                    0.0,
                    0.0,
                    75.0,
                    "Centro",
                    diameter_mm=3.0,
                )
                reporter.timestamps[-1] = float(index) / 30.0
            report, graph = reporter.generate_report(10.0)
            self.assertTrue(Path(report).exists())
            self.assertTrue(Path(graph).exists())
            self.assertIn(
                "no una probabilidad",
                Path(report).read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
