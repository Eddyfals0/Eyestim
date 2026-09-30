"""Train and evaluate EyeStim's EyeDentify pupil-diameter model.

Outputs include best/last checkpoints, complete epoch history, held-out
predictions, per-participant metrics, and visual evidence under docs/training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import EyeDentifyDiameterDataset
from model import PupilDiameterCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_train_targets(manifest_path: str) -> np.ndarray:
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = [
        float(row["left_pupil_mm"])
        for row in rows
        if row.get("split") == "train" and row.get("left_pupil_mm")
    ]
    if not values:
        raise ValueError("El manifiesto no contiene targets de entrenamiento")
    return np.asarray(values, dtype=np.float32)


def evaluate(model, loader, device, target_mean, target_std):
    model.eval()
    predictions = []
    expected = []
    participants = []
    sessions = []
    paths = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device))
            predicted_mm = target_mean + target_std * output.cpu().numpy()
            predictions.extend(predicted_mm.tolist())
            expected.extend(batch["target_mm"].numpy().tolist())
            participants.extend(batch["participant_id"].numpy().tolist())
            sessions.extend(batch["session_id"].numpy().tolist())
            paths.extend(batch["image_path"])
    return {
        "predicted": np.asarray(predictions, dtype=np.float64),
        "expected": np.asarray(expected, dtype=np.float64),
        "participant": np.asarray(participants, dtype=np.int64),
        "session": np.asarray(sessions, dtype=np.int64),
        "path": paths,
    }


def regression_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - expected
    absolute = np.abs(error)
    denominator = np.sum((expected - expected.mean()) ** 2)
    r2 = 1.0 - np.sum(error**2) / denominator if denominator > 0 else float("nan")
    pearson = (
        float(np.corrcoef(expected, predicted)[0, 1])
        if len(expected) > 1 and expected.std() > 0 and predicted.std() > 0
        else float("nan")
    )
    expected_rank = np.argsort(np.argsort(expected))
    predicted_rank = np.argsort(np.argsort(predicted))
    spearman = (
        float(np.corrcoef(expected_rank, predicted_rank)[0, 1])
        if len(expected) > 1
        else float("nan")
    )
    return {
        "samples": int(len(expected)),
        "mae_mm": float(absolute.mean()),
        "median_ae_mm": float(np.median(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "p95_ae_mm": float(np.percentile(absolute, 95)),
        "bias_mm": float(error.mean()),
        "r2": float(r2),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "within_0_25_mm": float(np.mean(absolute <= 0.25)),
        "within_0_50_mm": float(np.mean(absolute <= 0.50)),
    }


def atomic_checkpoint(payload: dict, destination: str) -> None:
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    temporary = destination + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(history, test, per_subject, docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [row["train_mae_mm"] for row in history], label="Entrenamiento")
    ax.plot(epochs, [row["val_mae_mm"] for row in history], label="Validación")
    ax.set(xlabel="Época", ylabel="MAE (mm)", title="EyeDentify — curva de aprendizaje")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(docs_dir / "diameter_learning_curve.png", dpi=180)
    plt.close(fig)

    expected = test["expected"]
    predicted = test["predicted"]
    lo = min(float(expected.min()), float(predicted.min()))
    hi = max(float(expected.max()), float(predicted.max()))
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        expected,
        predicted,
        c=test["participant"],
        cmap="viridis",
        s=20,
        alpha=0.72,
    )
    ax.plot([lo, hi], [lo, hi], "--", color="#ef4444", label="Predicción perfecta")
    ax.set(
        xlabel="Tobii / referencia (mm)",
        ylabel="EyeStim CNN (mm)",
        title="Predicciones en participantes nunca vistos",
    )
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Participante")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(docs_dir / "diameter_test_predictions.png", dpi=180)
    plt.close(fig)

    errors = predicted - expected
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors, bins=30, color="#2563eb", alpha=0.8)
    ax.axvline(0, color="#111827", linestyle="--")
    ax.axvline(errors.mean(), color="#ef4444", label=f"Sesgo {errors.mean():.3f} mm")
    ax.set(xlabel="Error predicción − referencia (mm)", ylabel="Muestras", title="Distribución del error en prueba")
    ax.legend()
    fig.tight_layout()
    fig.savefig(docs_dir / "diameter_error_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in history], label="Validación")
    axes[0, 0].set(title="Pérdida normalizada", xlabel="Época", ylabel="MSE")
    axes[0, 0].legend()
    axes[0, 1].scatter(expected, predicted, s=15, alpha=0.6, color="#0f766e")
    axes[0, 1].plot([lo, hi], [lo, hi], "--", color="#ef4444")
    axes[0, 1].set(title="Referencia frente a predicción", xlabel="Real (mm)", ylabel="Predicho (mm)")
    axes[1, 0].hist(np.abs(errors), bins=25, color="#7c3aed", alpha=0.8)
    axes[1, 0].set(title="Error absoluto", xlabel="Error (mm)", ylabel="Muestras")
    axes[1, 1].bar(
        [str(row["participant_id"]) for row in per_subject],
        [row["mae_mm"] for row in per_subject],
        color="#f59e0b",
    )
    axes[1, 1].set(title="MAE por participante de prueba", xlabel="Participante", ylabel="MAE (mm)")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle("Panel culminatorio de entrenamiento EyeStim / EyeDentify", fontsize=16)
    fig.tight_layout()
    fig.savefig(docs_dir / "training_dashboard.png", dpi=180)
    plt.close(fig)

    count = min(12, len(expected))
    indices = np.linspace(0, len(expected) - 1, count, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 7))
    for ax, index in zip(axes.flat, indices):
        image = cv2.imread(test["path"][index], cv2.IMREAD_GRAYSCALE)
        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"P{test['participant'][index]} · real {expected[index]:.2f}\n"
            f"pred {predicted[index]:.2f} mm",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes.flat[count:]:
        ax.axis("off")
    fig.suptitle("Capturas de prueba: diámetro Tobii frente a EyeStim")
    fig.tight_layout()
    fig.savefig(docs_dir / "sample_test_predictions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=config.EYEDENTIFY_MANIFEST_PATH)
    parser.add_argument("--epochs", type=int, default=config.DIAMETER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DIAMETER_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=config.DIAMETER_EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Activar aumento fotométrico (desactivado por defecto, como la configuración oficial)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = torch.device(args.device)
    docs_dir = Path(config.TRAINING_DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)

    targets = read_train_targets(args.manifest)
    target_mean = float(targets.mean())
    target_std = float(targets.std())
    datasets = {
        "train": EyeDentifyDiameterDataset(
            args.manifest, "train", args.augment, target_mean, target_std
        ),
        "validation": EyeDentifyDiameterDataset(
            args.manifest, "validation", False, target_mean, target_std
        ),
        "test": EyeDentifyDiameterDataset(
            args.manifest, "test", False, target_mean, target_std
        ),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, dataset in datasets.items()
    }

    model = PupilDiameterCNN().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.DIAMETER_LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    history = []
    best_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.time()

    split_participants = {
        name: sorted({int(row["participant_id"]) for row in dataset.rows})
        for name, dataset in datasets.items()
    }
    print("=== Entrenamiento EyeStim / EyeDentify ===")
    print(f"Dispositivo: {device}")
    print(f"Media/std target train: {target_mean:.4f} / {target_std:.4f} mm")
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset)} frames | sujetos {split_participants[name]}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_abs_errors = []
        for batch in loaders["train"]:
            images = batch["image"].to(device)
            targets_norm = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, targets_norm)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()) * len(images))
            predicted_mm = target_mean + target_std * output.detach().cpu().numpy()
            train_abs_errors.extend(
                np.abs(predicted_mm - batch["target_mm"].numpy()).tolist()
            )

        validation = evaluate(
            model, loaders["validation"], device, target_mean, target_std
        )
        val_metrics = regression_metrics(validation["expected"], validation["predicted"])
        val_loss = float(
            np.mean(
                np.abs(validation["predicted"] - validation["expected"])
                / target_std
            )
        )
        scheduler.step(val_metrics["mae_mm"])
        row = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / len(datasets["train"]),
            "val_loss": val_loss,
            "train_mae_mm": float(np.mean(train_abs_errors)),
            "val_mae_mm": val_metrics["mae_mm"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Época {epoch:03d} | train MAE {row['train_mae_mm']:.4f} mm | "
            f"val MAE {row['val_mae_mm']:.4f} mm | lr {row['learning_rate']:.2e}"
        )

        metadata = {
            "dataset": "EyeDentify webcam crops with Tobii reference",
            "license": "CC BY-NC 4.0",
            "epoch": epoch,
            "participant_disjoint_splits": split_participants,
            "input_size": [config.DIAMETER_INPUT_HEIGHT, config.DIAMETER_INPUT_WIDTH],
            "input_channels": "BGR",
            "augmentation": bool(args.augment),
        }
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "metadata": metadata,
        }
        if val_metrics["mae_mm"] < best_mae - 1e-5:
            best_mae = val_metrics["mae_mm"]
            best_epoch = epoch
            stale_epochs = 0
            atomic_checkpoint(payload, config.DIAMETER_MODEL_SAVE_PATH)
        else:
            stale_epochs += 1
        if epoch % 10 == 0:
            atomic_checkpoint(
                payload,
                os.path.join(config.MODELS_DIR, "checkpoints", f"diameter_epoch_{epoch:03d}.pth"),
            )
        if stale_epochs >= args.patience:
            print(f"Early stopping en época {epoch}; mejor época {best_epoch}")
            break

    payload["metadata"]["stopped_epoch"] = history[-1]["epoch"]
    atomic_checkpoint(payload, config.DIAMETER_LAST_CHECKPOINT_PATH)
    best = torch.load(
        config.DIAMETER_MODEL_SAVE_PATH, map_location=device, weights_only=True
    )
    model.load_state_dict(best["model_state_dict"])
    test = evaluate(model, loaders["test"], device, target_mean, target_std)
    test_metrics = regression_metrics(test["expected"], test["predicted"])

    per_subject = []
    for participant in sorted(set(test["participant"].tolist())):
        mask = test["participant"] == participant
        metrics = regression_metrics(test["expected"][mask], test["predicted"][mask])
        per_subject.append({"participant_id": participant, **metrics})

    sample = next(iter(loaders["test"]))["image"][:1].to(device)
    with torch.no_grad():
        for _ in range(10):
            model(sample)
        infer_started = time.perf_counter()
        repetitions = 250
        for _ in range(repetitions):
            model(sample)
        inference_ms = (time.perf_counter() - infer_started) * 1000 / repetitions

    predictions_rows = [
        {
            "participant_id": int(participant),
            "session_id": int(session),
            "reference_mm": f"{expected:.6f}",
            "prediction_mm": f"{predicted:.6f}",
            "error_mm": f"{predicted - expected:.6f}",
            "image_path": path,
        }
        for participant, session, expected, predicted, path in zip(
            test["participant"],
            test["session"],
            test["expected"],
            test["predicted"],
            test["path"],
        )
    ]
    save_rows(docs_dir / "history.csv", history)
    save_rows(docs_dir / "test_predictions.csv", predictions_rows)
    save_rows(docs_dir / "per_subject_metrics.csv", per_subject)
    make_figures(history, test, per_subject, docs_dir)

    baseline_prediction = np.full_like(test["expected"], target_mean)
    baseline_metrics = regression_metrics(test["expected"], baseline_prediction)
    summary = {
        "dataset": "EyeDentify",
        "license": "CC BY-NC 4.0",
        "split_participants": split_participants,
        "split_samples": {name: len(dataset) for name, dataset in datasets.items()},
        "target_train_mean_mm": target_mean,
        "target_train_std_mm": target_std,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "training_seconds": time.time() - started,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "cpu_inference_ms_per_eye": inference_ms,
        "cpu_inference_fps_single_eye": 1000.0 / inference_ms,
        "test": test_metrics,
        "constant_train_mean_baseline": baseline_metrics,
    }
    (docs_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    shutil.copy2(args.manifest, docs_dir / "dataset_manifest_snapshot.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
