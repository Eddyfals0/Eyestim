"""Reproduce EyeStim's culminating validation and visual evidence.

Run from the repository root with::

    python research/final_validation.py

The script is also invoked by ``notebooks/04_final_training_and_validation.ipynb``.
It does not train again; it audits the downloaded data and evaluates the frozen
checkpoints on held-out participants.
"""

from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from research.validation import benchmark_end_to_end_swirski, load_swirski_samples
from src.reporter import SessionReporter


TRAINING_DIR = PROJECT_ROOT / "docs" / "training"
MANIFEST_PATH = PROJECT_ROOT / "data" / "external" / "eyedentify" / "manifest.csv"
CENTER_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "eyestim_cnn.pth"
SWIRSKI_TEST_DIR = (
    PROJECT_ROOT / "data" / "external" / "swirski" / "p2-left"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_artifact(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def audit_eyedentify_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    """Check every manifest row, image, target, and participant split."""
    manifest = pd.read_csv(manifest_path)
    required = {
        "participant_id",
        "session_id",
        "left_pupil_mm",
        "image_path",
        "split",
    }
    missing_columns = sorted(required - set(manifest.columns))
    base_dir = manifest_path.parent
    missing_files = 0
    unreadable_files = 0
    shapes: set[tuple[int, ...]] = set()
    for relative_path in manifest["image_path"]:
        path = base_dir / str(relative_path)
        if not path.exists():
            missing_files += 1
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            unreadable_files += 1
        else:
            shapes.add(tuple(image.shape))

    participants = {
        split: set(group["participant_id"].astype(int))
        for split, group in manifest.groupby("split")
    }
    split_names = ("train", "validation", "test")
    overlaps = {
        f"{left}_{right}": sorted(participants.get(left, set()) & participants.get(right, set()))
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    target = pd.to_numeric(manifest["left_pupil_mm"], errors="coerce")
    split_summary = {}
    for split in split_names:
        subset = manifest[manifest["split"] == split]
        values = pd.to_numeric(subset["left_pupil_mm"], errors="coerce")
        split_summary[split] = {
            "frames": int(len(subset)),
            "participants": int(subset["participant_id"].nunique()),
            "target_mean_mm": float(values.mean()),
            "target_std_mm": float(values.std(ddof=0)),
        }

    audit = {
        "rows": int(len(manifest)),
        "participants": int(manifest["participant_id"].nunique()),
        "sessions": int(
            manifest[["participant_id", "session_id"]].drop_duplicates().shape[0]
        ),
        "missing_columns": missing_columns,
        "missing_images": missing_files,
        "unreadable_images": unreadable_files,
        "duplicate_image_paths": int(manifest["image_path"].duplicated().sum()),
        "non_finite_targets": int((~np.isfinite(target)).sum()),
        "target_outside_physiological_range_1_to_9_mm": int(
            ((target < 1.0) | (target > 9.0)).sum()
        ),
        "image_shapes": [list(shape) for shape in sorted(shapes)],
        "participant_overlaps": overlaps,
        "split_summary": split_summary,
    }
    audit["passed"] = not any(
        [
            missing_columns,
            missing_files,
            unreadable_files,
            audit["duplicate_image_paths"],
            audit["non_finite_targets"],
            audit["target_outside_physiological_range_1_to_9_mm"],
            any(overlaps.values()),
            sorted(shapes) != [(16, 32, 3)],
        ]
    )
    return audit


def paired_diameter_comparison(summary: dict) -> dict:
    predictions = pd.read_csv(TRAINING_DIR / "test_predictions.csv")
    baseline = float(summary["target_train_mean_mm"])
    predictions["model_absolute_error"] = (
        predictions["prediction_mm"] - predictions["reference_mm"]
    ).abs()
    predictions["baseline_absolute_error"] = (
        baseline - predictions["reference_mm"]
    ).abs()
    by_subject = predictions.groupby("participant_id")[[
        "model_absolute_error",
        "baseline_absolute_error",
    ]].mean()
    statistic = wilcoxon(
        by_subject["model_absolute_error"],
        by_subject["baseline_absolute_error"],
        alternative="two-sided",
    )
    return {
        "test_participants": int(len(by_subject)),
        "participants_where_model_beats_baseline": int(
            (by_subject["model_absolute_error"] < by_subject["baseline_absolute_error"]).sum()
        ),
        "paired_wilcoxon_two_sided_p": float(statistic.pvalue),
        "model_subject_mean_mae_mm": float(by_subject["model_absolute_error"].mean()),
        "baseline_subject_mean_mae_mm": float(by_subject["baseline_absolute_error"].mean()),
    }


def benchmark_reporter() -> dict:
    """Verify that a representative 20-second session report is generated."""
    example_dir = TRAINING_DIR / "session_example"
    reporter = SessionReporter(str(example_dir))
    rng = np.random.default_rng(42)
    initial = time.time()
    for frame in range(600):
        diameter_px = 11.5 + 0.55 * np.sin(frame / 45.0) + rng.normal(0, 0.08)
        diameter_mm = 3.1 + 0.12 * np.sin(frame / 45.0) + rng.normal(0, 0.025)
        ocular_index = float(np.clip(76 + 9 * np.sin(frame / 70.0), 0, 100))
        reporter.log_frame(
            diameter_px,
            float(0.10 * np.sin(frame / 33.0)),
            float(0.07 * np.cos(frame / 39.0)),
            ocular_index,
            "Centro" if frame % 10 else "Derecha",
            diameter_mm=diameter_mm,
        )
        reporter.timestamps[-1] = initial + frame / 30.0
    started = time.perf_counter()
    report_path, graph_path = reporter.generate_report(baseline_diameter=11.5)
    elapsed = time.perf_counter() - started
    return {
        "frames": 600,
        "generation_seconds": elapsed,
        "report_exists": Path(report_path).exists(),
        "graph_exists": Path(graph_path).exists(),
        "report_path": str(Path(report_path).relative_to(PROJECT_ROOT)),
        "graph_path": str(Path(graph_path).relative_to(PROJECT_ROOT)),
    }


def make_dashboard(
    audit: dict,
    diameter: dict,
    center: dict,
    end_to_end_results: pd.DataFrame,
    end_to_end: dict,
) -> Path:
    """Create one visual synopsis suitable for the app's ``t`` shortcut."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    splits = ["train", "validation", "test"]
    labels = ["Entrenamiento", "Validación", "Prueba"]
    frames = [audit["split_summary"][split]["frames"] for split in splits]
    subjects = [audit["split_summary"][split]["participants"] for split in splits]
    bars = axes[0, 0].bar(labels, frames, color=["#2563eb", "#7c3aed", "#f59e0b"])
    for bar, frame_count, subject_count in zip(bars, frames, subjects):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(frames) * 0.025,
            f"{frame_count:,} frames\n{subject_count} sujetos",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0, 0].set(
        title="EyeDentify: partición sin fuga entre participantes",
        ylabel="Imágenes",
        ylim=(0, max(frames) * 1.2),
    )

    metric_names = ["Media", "Mediana", "P95"]
    legacy = center["legacy_bioid_model"]
    lpw = center["lpw_model"]
    legacy_values = [legacy["mean_error_px"], legacy["median_error_px"], legacy["p95_error_px"]]
    lpw_values = [lpw["mean_error_px"], lpw["median_error_px"], lpw["p95_error_px"]]
    x = np.arange(len(metric_names))
    width = 0.36
    axes[0, 1].bar(x - width / 2, legacy_values, width, label="Modelo BioID legado", color="#dc2626")
    axes[0, 1].bar(x + width / 2, lpw_values, width, label="Modelo LPW nuevo", color="#059669")
    axes[0, 1].set(
        xticks=x,
        xticklabels=metric_names,
        ylabel="Error de centro (px)",
        title="Centro pupilar: prueba en sujeto LPW reservado",
    )
    axes[0, 1].legend()

    diameter_metrics = ["mae_mm", "rmse_mm", "p95_ae_mm"]
    diameter_labels = ["MAE", "RMSE", "P95 AE"]
    model_values = [diameter["test"][name] for name in diameter_metrics]
    baseline_values = [diameter["constant_train_mean_baseline"][name] for name in diameter_metrics]
    axes[1, 0].bar(x - width / 2, baseline_values, width, label="Media constante", color="#9ca3af")
    axes[1, 0].bar(x + width / 2, model_values, width, label="CNN EyeDentify", color="#2563eb")
    axes[1, 0].set(
        xticks=x,
        xticklabels=diameter_labels,
        ylabel="Error (mm)",
        title="Diámetro: 7 participantes EyeDentify no vistos",
    )
    axes[1, 0].legend()

    valid = end_to_end_results[end_to_end_results["detected"]]
    axes[1, 1].scatter(
        valid["true_diameter_px"],
        valid["estimated_diameter_px"],
        s=28,
        alpha=0.7,
        color="#7c3aed",
    )
    lower = min(valid["true_diameter_px"].min(), valid["estimated_diameter_px"].min())
    upper = max(valid["true_diameter_px"].max(), valid["estimated_diameter_px"].max())
    axes[1, 1].plot([lower, upper], [lower, upper], "--", color="#dc2626", label="Ideal")
    axes[1, 1].set(
        xlabel="Diámetro anotado (px)",
        ylabel="Diámetro estimado (px)",
        title=(
            "Pipeline completo en Cambridge reservado\n"
            f"MAE {end_to_end['diameter_mae_px']:.3f} px · "
            f"detección {end_to_end['detection_rate']:.0%} · "
            f"ρ {end_to_end['diameter_spearman']:.3f}"
        ),
    )
    axes[1, 1].legend()

    fig.suptitle(
        "EyeStim — evidencia culminatoria de datos, entrenamiento y prueba",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Resultado honesto: prototipo de pupilometría útil; el índice 0–100 no está validado como atención cognitiva.",
        ha="center",
        fontsize=11,
        color="#991b1b",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    output_path = TRAINING_DIR / "project_dashboard.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run() -> dict:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    diameter = load_json(TRAINING_DIR / "training_summary.json")
    center = load_json(TRAINING_DIR / "center_training_summary.json")
    audit = audit_eyedentify_manifest()
    paired = paired_diameter_comparison(diameter)

    samples = load_swirski_samples(SWIRSKI_TEST_DIR)
    end_to_end_results, end_to_end = benchmark_end_to_end_swirski(
        samples,
        CENTER_MODEL_PATH,
        crop_size=32,
        threshold_offset=35,
    )
    end_to_end_results.to_csv(TRAINING_DIR / "end_to_end_predictions.csv", index=False)
    reporter = benchmark_reporter()
    dashboard = make_dashboard(audit, diameter, center, end_to_end_results, end_to_end)

    result = {
        "eyedentify_integrity": audit,
        "diameter_participant_level_comparison": paired,
        "cambridge_end_to_end_held_out": end_to_end,
        "reporter_benchmark": reporter,
        "artifacts": {
            "dashboard": dashboard.relative_to(PROJECT_ROOT).as_posix(),
            "center_model": checkpoint_artifact(CENTER_MODEL_PATH),
            "center_training_checkpoint": checkpoint_artifact(
                PROJECT_ROOT / "src" / "models" / "eyestim_center_checkpoint.pth"
            ),
            "diameter_best_checkpoint": checkpoint_artifact(
                PROJECT_ROOT / "src" / "models" / "eyestim_diameter.pth"
            ),
            "diameter_last_checkpoint": checkpoint_artifact(
                PROJECT_ROOT / "src" / "models" / "eyestim_diameter_last.pth"
            ),
            "diameter_periodic_checkpoints": [
                checkpoint_artifact(path)
                for path in sorted(
                    (PROJECT_ROOT / "src" / "models" / "checkpoints").glob(
                        "diameter_epoch_*.pth"
                    )
                )
            ],
        },
    }
    (TRAINING_DIR / "final_validation_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
