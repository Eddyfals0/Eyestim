"""Retrain EyeStim pupil-centre localization on subject-disjoint LPW data."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import LPWCenterDataset
from model import EyePupilCNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def evaluate(model, loader, device):
    model.eval()
    predicted = []
    expected = []
    participants = []
    frames = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device)).cpu().numpy()
            predicted.append(output)
            expected.append(batch["target"].numpy())
            participants.extend(batch["participant_id"].numpy().tolist())
            frames.extend(batch["frame"].numpy().tolist())
    predicted = np.concatenate(predicted)
    expected = np.concatenate(expected)
    error_px = np.linalg.norm(
        (predicted - expected) * (config.ROI_SIZE / 2.0), axis=1
    )
    return {
        "predicted": predicted,
        "expected": expected,
        "participant": np.asarray(participants),
        "frame": np.asarray(frames),
        "error_px": error_px,
    }


def metrics(result):
    error = result["error_px"]
    return {
        "samples": int(len(error)),
        "mean_error_px": float(error.mean()),
        "median_error_px": float(np.median(error)),
        "p95_error_px": float(np.percentile(error, 95)),
        "within_3px": float(np.mean(error <= 3.0)),
        "within_5px": float(np.mean(error <= 5.0)),
        "prediction_std_x": float(result["predicted"][:, 0].std()),
        "prediction_std_y": float(result["predicted"][:, 1].std()),
    }


def save_visuals(history, result, dataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [row["train_error_px"] for row in history], label="Train")
    ax.plot(epochs, [row["val_error_px"] for row in history], label="Validación")
    ax.set(
        xlabel="Época",
        ylabel="Error euclidiano (px en ROI 64×64)",
        title="LPW — aprendizaje del centro pupilar",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "center_learning_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(result["error_px"], bins=35, color="#0891b2", alpha=0.85)
    ax.axvline(3.0, color="#f59e0b", linestyle="--", label="3 px")
    ax.axvline(5.0, color="#ef4444", linestyle="--", label="5 px")
    ax.set(xlabel="Error de centro (px)", ylabel="Frames", title="Error en participante LPW reservado")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "center_error_distribution.png", dpi=180)
    plt.close(fig)

    indices = np.linspace(0, len(dataset) - 1, 12, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, index in zip(axes.flat, indices):
        image = dataset.images[index]
        expected = result["expected"][index]
        predicted = result["predicted"][index]
        true_x, true_y = (expected + 1.0) * config.ROI_SIZE / 2.0
        pred_x, pred_y = (predicted + 1.0) * config.ROI_SIZE / 2.0
        ax.imshow(image, cmap="gray")
        ax.scatter([true_x], [true_y], c="#22c55e", marker="+", s=80, label="Real")
        ax.scatter([pred_x], [pred_y], c="#ef4444", marker="x", s=55, label="CNN")
        ax.set_title(f"Frame {dataset.frames[index]} · {result['error_px'][index]:.1f} px")
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Capturas de prueba LPW: centro real frente a EyeStim")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output_dir / "center_sample_predictions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    seed_everything(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.TRAINING_DOCS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)

    datasets = {
        "train": LPWCenterDataset((1, 2, 3, 4, 5, 6), training=True),
        "validation": LPWCenterDataset((7,), training=False),
        "test": LPWCenterDataset((8,), training=False),
    }
    loaders = {
        name: DataLoader(dataset, batch_size=64, shuffle=name == "train", num_workers=0)
        for name, dataset in datasets.items()
    }

    legacy_metrics = None
    legacy_path = os.path.join(config.MODELS_DIR, "eyestim_cnn_bioid_legacy.pth")
    if os.path.exists(config.MODEL_SAVE_PATH):
        if not os.path.exists(legacy_path):
            shutil.copy2(config.MODEL_SAVE_PATH, legacy_path)
        legacy = EyePupilCNN().to(device)
        legacy.load_state_dict(torch.load(legacy_path, map_location=device, weights_only=True))
        legacy_metrics = metrics(evaluate(legacy, loaders["test"], device))

    model = EyePupilCNN().to(device)
    criterion = nn.SmoothL1Loss(beta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    best_error = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()

    print("=== Entrenamiento de centro pupilar LPW ===")
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset)} frames")
    for epoch in range(1, 41):
        model.train()
        train_error = []
        train_loss = 0.0
        for batch in loaders["train"]:
            images = batch["image"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(images)
            train_error.extend(
                torch.linalg.vector_norm(
                    (output.detach() - target) * (config.ROI_SIZE / 2.0), dim=1
                ).cpu().numpy().tolist()
            )
        validation = evaluate(model, loaders["validation"], device)
        validation_metrics = metrics(validation)
        scheduler.step(validation_metrics["mean_error_px"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(datasets["train"]),
            "train_error_px": float(np.mean(train_error)),
            "val_error_px": validation_metrics["mean_error_px"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Época {epoch:02d} | train {row['train_error_px']:.3f} px | "
            f"val {row['val_error_px']:.3f} px"
        )
        if row["val_error_px"] < best_error - 1e-4:
            best_error = row["val_error_px"]
            best_epoch = epoch
            stale = 0
            temporary = config.MODEL_SAVE_PATH + ".tmp"
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, config.MODEL_SAVE_PATH)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_error_px": best_error,
                    "dataset": "LPW participants 1-8, subject-disjoint",
                },
                config.CENTER_CHECKPOINT_PATH,
            )
        else:
            stale += 1
        if stale >= 10:
            print(f"Early stopping; mejor época {best_epoch}")
            break

    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True)
    )
    test = evaluate(model, loaders["test"], device)
    test_metrics = metrics(test)
    save_visuals(history, test, datasets["test"], output_dir)

    with torch.no_grad():
        sample = next(iter(loaders["test"]))["image"][:1].to(device)
        for _ in range(10):
            model(sample)
        inference_started = time.perf_counter()
        for _ in range(250):
            model(sample)
        inference_ms = (time.perf_counter() - inference_started) * 4.0

    rows = [
        {
            "participant_id": int(participant),
            "frame": int(frame),
            "true_x_norm": float(expected[0]),
            "true_y_norm": float(expected[1]),
            "pred_x_norm": float(predicted[0]),
            "pred_y_norm": float(predicted[1]),
            "error_px": float(error),
        }
        for participant, frame, expected, predicted, error in zip(
            test["participant"],
            test["frame"],
            test["expected"],
            test["predicted"],
            test["error_px"],
        )
    ]
    with (output_dir / "center_test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "center_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "dataset": "Labelled Pupils in the Wild",
        "license": "CC BY-NC-SA 4.0",
        "splits": {
            "train_participants": [1, 2, 3, 4, 5, 6],
            "validation_participants": [7],
            "test_participants": [8],
            "samples": {name: len(dataset) for name, dataset in datasets.items()},
        },
        "best_epoch": best_epoch,
        "training_seconds": time.time() - started,
        "cpu_inference_ms_per_eye": inference_ms,
        "legacy_bioid_model": legacy_metrics,
        "lpw_model": test_metrics,
    }
    (output_dir / "center_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
