"""Generate publication figures directly from frozen EyeStim result files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "training"
FIGURES = ROOT / "tmp" / "pdfs" / "eyestim_academic_paper"

BLUE = "#1d4ed8"
NAVY = "#172554"
TEAL = "#0f766e"
GREEN = "#15803d"
ORANGE = "#c2410c"
RED = "#b91c1c"
GRAY = "#64748b"
LIGHT = "#eff6ff"


def _clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def pipeline_figure() -> Path:
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.25, 1.75, 1.65, 1.25, "Webcam", "imagen RGB", LIGHT, NAVY),
        (2.25, 1.75, 1.8, 1.25, "Haar", "rostro + ojos", "#f8fafc", GRAY),
        (4.4, 1.75, 2.0, 1.25, "CNN LPW", "centro (x, y)", "#ecfdf5", GREEN),
        (7.0, 3.1, 2.15, 1.25, "OpenCV", "contorno + elipse", "#fff7ed", ORANGE),
        (7.0, 0.4, 2.15, 1.25, "CNN EyeDentify", "diámetro (mm)", "#eef2ff", BLUE),
        (9.8, 3.1, 2.55, 1.25, "Pupilometría", "centro refinado + px", "#f0fdf4", TEAL),
        (9.8, 0.4, 2.55, 1.25, "Salida experimental", "mm + incertidumbre", "#fefce8", ORANGE),
    ]
    for x, y, width, height, title, subtitle, fill, edge in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.08,rounding_size=0.10",
            linewidth=1.8, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + 0.78, title, ha="center", va="center", fontsize=12, fontweight="bold", color=edge)
        ax.text(x + width / 2, y + 0.36, subtitle, ha="center", va="center", fontsize=9.5, color="#334155")

    arrows = [
        ((1.92, 2.38), (2.23, 2.38)),
        ((4.07, 2.38), (4.38, 2.38)),
        ((6.42, 2.38), (6.97, 3.65)),
        ((6.42, 2.38), (6.97, 1.02)),
        ((9.17, 3.72), (9.77, 3.72)),
        ((9.17, 1.02), (9.77, 1.02)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": NAVY})

    ax.text(6.5, 4.65, "Dos estimadores complementarios, no un único modelo", ha="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(6.5, 0.02, "El índice ocular no forma parte de la prueba geométrica y no se interpreta como una medida de atención.", ha="center", fontsize=9.5, color=RED)
    return _save(fig, "figure_1_pipeline.png")


def dataset_figure() -> Path:
    summary = json.loads((RESULTS / "final_validation_summary.json").read_text(encoding="utf-8"))
    audit = summary["eyedentify_integrity"]["split_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))

    labels = ["Entrenamiento", "Validación", "Prueba"]
    keys = ["train", "validation", "test"]
    frames = [audit[key]["frames"] for key in keys]
    people = [audit[key]["participants"] for key in keys]
    colors = [BLUE, "#7c3aed", "#d97706"]
    bars = axes[0].bar(labels, frames, color=colors, width=0.66)
    for bar, count, subjects in zip(bars, frames, people):
        axes[0].text(bar.get_x() + bar.get_width() / 2, count + 120, f"{count:,}\n({subjects} participantes)", ha="center", fontsize=9)
    axes[0].set_title("EyeDentify: separación por participante", fontweight="bold")
    axes[0].set_ylabel("Imágenes seleccionadas")
    axes[0].set_ylim(0, 7600)
    axes[0].grid(axis="y", alpha=0.25)

    dataset_names = ["LPW\ncentro", "EyeDentify\ndiámetro", "Cambridge\nelipse", "NEMAR\natención"]
    sample_counts = [16000, 9425, 300, 51]
    bar_colors = [GREEN, BLUE, ORANGE, RED]
    bars = axes[1].bar(dataset_names, sample_counts, color=bar_colors, width=0.7)
    labels_above = ["16,000 imágenes", "9,425 imágenes", "300 elipses", "51 grabaciones"]
    for bar, label in zip(bars, labels_above):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 350, label, ha="center", fontsize=8.5)
    axes[1].set_title("Una fuente para cada pregunta observable", fontweight="bold")
    axes[1].set_ylabel("Muestras usadas localmente")
    axes[1].set_ylim(0, 18500)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Separación correcta de los datos", fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return _save(fig, "figure_2_datasets.png")


def _read_bioid_eye_points(path: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    values = path.read_text(encoding="ascii").splitlines()[1].split()
    lx, ly, rx, ry = map(float, values)
    return (lx, ly), (rx, ry)


def _read_cambridge_sample(participant: int, sample_index: int) -> tuple[np.ndarray, tuple[float, ...]]:
    base = ROOT / "data" / "external" / "swirski" / f"p{participant}-left"
    lines = (base / "pupil-ellipses.txt").read_text(encoding="ascii").splitlines()
    parts = lines[sample_index].replace("|", " ").split()
    frame = int(parts[0])
    ellipse = tuple(map(float, parts[1:6]))
    image = plt.imread(base / "frames" / f"{frame}-eye.png")
    return image, ellipse


def sample_data_figure() -> Path:
    """Show what a participant sample looks like in each visual dataset."""
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.2))
    fig.patch.set_facecolor("white")

    bioid_ids = [0, 900]
    for row, image_id in enumerate(bioid_ids):
        image_path = ROOT / "data" / "bioid" / f"BioID_{image_id:04d}.pgm"
        image = plt.imread(image_path)
        axes[row, 0].imshow(image, cmap="gray")
        for x, y in _read_bioid_eye_points(image_path.with_suffix(".eye")):
            axes[row, 0].plot(x, y, marker="+", color="#22c55e", markersize=9, markeredgewidth=2)
        axes[row, 0].set_title(f"BioID {image_id:04d}\nrostro y puntos generales", fontsize=9.5)
        _clean_axis(axes[row, 0])

    lpw_specs = [(1, "9.roi64.npz", 600), (8, "9.roi64.npz", 1300)]
    for row, (participant, filename, frame) in enumerate(lpw_specs):
        data = np.load(ROOT / "data" / "external" / "lpw_center" / str(participant) / filename)
        image = data["images"][frame]
        target = data["targets"][frame]
        x, y = (target + 1.0) * 32.0
        axes[row, 1].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].plot(x, y, marker="+", color="#22c55e", markersize=10, markeredgewidth=2)
        axes[row, 1].set_title(f"LPW P{participant}\nojo infrarrojo y centro", fontsize=9.5)
        _clean_axis(axes[row, 1])

    eye_specs = [
        (11, ROOT / "data" / "external" / "eyedentify" / "left_eyes" / "11" / "1" / "frame_15.png"),
        (51, ROOT / "data" / "external" / "eyedentify" / "left_eyes" / "51" / "1" / "frame_15.png"),
    ]
    for row, (participant, image_path) in enumerate(eye_specs):
        axes[row, 2].imshow(plt.imread(image_path), cmap="gray", interpolation="nearest")
        axes[row, 2].set_title(f"EyeDentify P{participant}\nrecorte de webcam", fontsize=9.5)
        _clean_axis(axes[row, 2])

    cambridge_specs = [(1, 12), (2, 18)]
    for row, (participant, index) in enumerate(cambridge_specs):
        image, ellipse = _read_cambridge_sample(participant, index)
        cx, cy, width, height, angle = ellipse
        margin_x, margin_y = 85, 55
        x0, x1 = max(0, int(cx - margin_x)), min(image.shape[1], int(cx + margin_x))
        y0, y1 = max(0, int(cy - margin_y)), min(image.shape[0], int(cy + margin_y))
        axes[row, 3].imshow(image[y0:y1, x0:x1], cmap="gray")
        patch = Ellipse(
            (cx - x0, cy - y0), width, height,
            angle=np.degrees(angle), fill=False, edgecolor="#22c55e", linewidth=2,
        )
        axes[row, 3].add_patch(patch)
        axes[row, 3].set_title(f"Cambridge P{participant}\nelipse marcada", fontsize=9.5)
        _clean_axis(axes[row, 3])

    fig.suptitle("Muestras reales y anónimas de los conjuntos de imágenes", fontsize=14, fontweight="bold", color=NAVY)
    fig.text(
        0.5, 0.015,
        "La marca verde indica la referencia disponible. BioID contiene rostros, pero no una medida precisa de la pupila.",
        ha="center", fontsize=9.5, color=RED,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    return _save(fig, "figure_3_sample_data.png")


def sample_predictions_figure() -> Path:
    """Show representative model outputs without exposing participant names."""
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.7))

    center_rows = pd.read_csv(RESULTS / "center_test_predictions.csv").set_index("frame")
    lpw = np.load(ROOT / "data" / "external" / "lpw_center" / "8" / "9.roi64.npz")
    center_frames = [0, 363, 1090, 1999]
    for col, frame in enumerate(center_frames):
        row = center_rows.loc[frame]
        true_x, true_y = (np.array([row["true_x_norm"], row["true_y_norm"]]) + 1.0) * 32.0
        pred_x, pred_y = (np.array([row["pred_x_norm"], row["pred_y_norm"]]) + 1.0) * 32.0
        axes[0, col].imshow(lpw["images"][frame], cmap="gray", vmin=0, vmax=255)
        axes[0, col].plot(true_x, true_y, marker="+", color="#22c55e", markersize=10, markeredgewidth=2)
        axes[0, col].plot(pred_x, pred_y, marker="x", color="#ef4444", markersize=8, markeredgewidth=2)
        axes[0, col].set_title(f"LPW imagen {frame}\nerror {row['error_px']:.1f} px", fontsize=9.5)
        _clean_axis(axes[0, col])

    diameter_rows = pd.read_csv(RESULTS / "test_predictions.csv")
    diameter_participants = [11, 15, 41, 51]
    for col, participant in enumerate(diameter_participants):
        row = diameter_rows[diameter_rows["participant_id"] == participant].iloc[0]
        axes[1, col].imshow(plt.imread(Path(row["image_path"])), cmap="gray", interpolation="nearest")
        axes[1, col].set_title(
            f"EyeDentify P{participant}\nreal {row['reference_mm']:.2f} | modelo {row['prediction_mm']:.2f} mm",
            fontsize=9.5,
        )
        _clean_axis(axes[1, col])

    fig.suptitle("Ejemplos de las predicciones realizadas por EyeStim", fontsize=14, fontweight="bold", color=NAVY)
    fig.text(
        0.5, 0.015,
        "Fila superior: centro real (+ verde) y centro calculado (x roja). Fila inferior: diámetro Tobii y diámetro calculado.",
        ha="center", fontsize=9.5, color="#334155",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    return _save(fig, "figure_6_sample_predictions.png")


def learning_figure() -> Path:
    center = pd.read_csv(RESULTS / "center_history.csv")
    diameter = pd.read_csv(RESULTS / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    axes[0].plot(center["epoch"], center["train_error_px"], color=BLUE, lw=2, label="Entrenamiento")
    axes[0].plot(center["epoch"], center["val_error_px"], color=ORANGE, lw=2, label="Validación")
    axes[0].axvline(12, color=GREEN, ls="--", label="Mejor época: 12")
    axes[0].set(title="CNN de centro - LPW", xlabel="Época", ylabel="Error euclidiano (px)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(diameter["epoch"], diameter["train_mae_mm"], color=BLUE, lw=2, label="Entrenamiento")
    axes[1].plot(diameter["epoch"], diameter["val_mae_mm"], color=ORANGE, lw=2, label="Validación")
    axes[1].axvline(9, color=GREEN, ls="--", label="Mejor época: 9")
    axes[1].set(title="CNN de diámetro - EyeDentify", xlabel="Época", ylabel="MAE (mm)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Curvas para seleccionar los mejores modelos", fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return _save(fig, "figure_3_learning.png")


def results_figure() -> Path:
    center = json.loads((RESULTS / "center_training_summary.json").read_text(encoding="utf-8"))
    diameter = json.loads((RESULTS / "training_summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    x = np.arange(3)
    width = 0.36

    old = center["legacy_bioid_model"]
    new = center["lpw_model"]
    old_values = [old["mean_error_px"], old["median_error_px"], old["p95_error_px"]]
    new_values = [new["mean_error_px"], new["median_error_px"], new["p95_error_px"]]
    axes[0].bar(x - width / 2, old_values, width, color=RED, label="BioID legado")
    axes[0].bar(x + width / 2, new_values, width, color=GREEN, label="LPW nuevo")
    axes[0].set(xticks=x, xticklabels=["Media", "Mediana", "P95"], ylabel="Error de centro (px)", title="Centro: participante LPW no visto")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    baseline = diameter["constant_train_mean_baseline"]
    model = diameter["test"]
    base_values = [baseline["mae_mm"], baseline["rmse_mm"], baseline["p95_ae_mm"]]
    model_values = [model["mae_mm"], model["rmse_mm"], model["p95_ae_mm"]]
    axes[1].bar(x - width / 2, base_values, width, color="#94a3b8", label="Media constante")
    axes[1].bar(x + width / 2, model_values, width, color=BLUE, label="CNN EyeDentify")
    axes[1].set(xticks=x, xticklabels=["MAE", "RMSE", "P95 AE"], ylabel="Error de diámetro (mm)", title="Diámetro: 7 participantes no vistos")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Comparación contra soluciones sencillas", fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return _save(fig, "figure_4_results.png")


def end_to_end_figure() -> Path:
    values = pd.read_csv(RESULTS / "end_to_end_predictions.csv")
    valid = values[values["detected"].astype(str).str.lower().isin(["true", "1"])]
    errors = valid["absolute_diameter_error_px"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))

    axes[0].scatter(valid["true_diameter_px"], valid["estimated_diameter_px"], color="#7c3aed", alpha=0.72, s=24)
    lower = min(valid["true_diameter_px"].min(), valid["estimated_diameter_px"].min())
    upper = max(valid["true_diameter_px"].max(), valid["estimated_diameter_px"].max())
    axes[0].plot([lower, upper], [lower, upper], "--", color=RED, label="Predicción ideal")
    axes[0].set(title="Referencia frente a estimación", xlabel="Diámetro anotado (px)", ylabel="Diámetro estimado (px)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].hist(errors, bins=18, color=TEAL, alpha=0.85)
    axes[1].axvline(errors.mean(), color=RED, ls="--", lw=2, label=f"MAE = {errors.mean():.3f} px")
    axes[1].axvline(errors.quantile(0.95), color=ORANGE, ls=":", lw=2, label=f"P95 = {errors.quantile(0.95):.3f} px")
    axes[1].set(title="Distribución del error absoluto", xlabel="Error absoluto (px)", ylabel="Imágenes")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Prueba completa en Cambridge P2 reservado (n=150)", fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return _save(fig, "figure_5_end_to_end.png")


def main() -> None:
    outputs = [
        pipeline_figure(),
        dataset_figure(),
        sample_data_figure(),
        learning_figure(),
        results_figure(),
        sample_predictions_figure(),
        end_to_end_figure(),
    ]
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
