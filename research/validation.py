"""Dataset fitness and empirical validation helpers for EyeStim.

These functions keep the notebooks short while leaving every transformation and
metric in ordinary, testable Python code.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def benchmark_lpw_cnn(
    video_path: Path,
    labels_path: Path,
    model_path: Path,
    roi_size: int = 64,
) -> dict[str, float]:
    """Evaluate the saved EyeStim CNN against LPW pupil-centre annotations."""
    import torch

    from src.model import EyePupilCNN

    labels = np.loadtxt(labels_path, dtype=np.float32)
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    targets: list[tuple[float, float]] = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    index = 0
    while index < len(labels):
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        resized = cv2.resize(gray, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
        frames.append(clahe.apply(resized).astype(np.float32) / 255.0)
        x, y = labels[index]
        targets.append((2.0 * x / width - 1.0, 2.0 * y / height - 1.0))
        index += 1
    capture.release()

    model = EyePupilCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    inputs = torch.from_numpy(np.stack(frames)[:, None, :, :])
    predictions = []
    with torch.no_grad():
        for batch in inputs.split(128):
            predictions.append(model(batch))
    predicted = torch.cat(predictions).numpy()
    expected = np.asarray(targets, dtype=np.float32)
    errors = np.linalg.norm((predicted - expected) * (roi_size / 2.0), axis=1)
    centre_baseline = np.linalg.norm(expected * (roi_size / 2.0), axis=1)
    return {
        "frames": float(len(errors)),
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "constant_centre_baseline_px": float(centre_baseline.mean()),
        "prediction_std_x": float(predicted[:, 0].std()),
        "prediction_std_y": float(predicted[:, 1].std()),
    }


def audit_bioid(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check BioID file integrity and map its labels to EyeStim objectives."""
    images = sorted(data_dir.glob("*.pgm"))
    labels = sorted(data_dir.glob("*.eye"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    shapes: set[tuple[int, int]] = set()
    out_of_bounds = 0
    separations: list[float] = []
    malformed = 0

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            malformed += 1
            continue
        height, width = image.shape
        shapes.add((height, width))
        eye_path = data_dir / f"{image_path.stem}.eye"
        if not eye_path.exists():
            continue
        rows = [
            row.strip()
            for row in eye_path.read_text(encoding="utf-8").splitlines()
            if row.strip() and not row.startswith("#")
        ]
        try:
            lx, ly, rx, ry = map(float, rows[0].split()[:4])
        except (IndexError, ValueError):
            malformed += 1
            continue
        if not (0 <= lx < width and 0 <= rx < width and 0 <= ly < height and 0 <= ry < height):
            out_of_bounds += 1
        separations.append(float(np.hypot(lx - rx, ly - ry)))

    integrity = pd.DataFrame(
        {
            "metric": [
                "images",
                "annotation_files",
                "paired_samples",
                "missing_annotations",
                "missing_images",
                "malformed_or_unreadable",
                "out_of_bounds_annotations",
                "unique_image_shapes",
                "median_inter_eye_distance_px",
            ],
            "value": [
                len(images),
                len(labels),
                len(image_stems & label_stems),
                len(image_stems - label_stems),
                len(label_stems - image_stems),
                malformed,
                out_of_bounds,
                sorted(shapes),
                float(np.median(separations)) if separations else np.nan,
            ],
        }
    )

    fitness = pd.DataFrame(
        [
            ("Detección/localización facial", "Sí", "Rostros y centros aproximados de ambos ojos"),
            ("Centro pupilar preciso", "No", "La etiqueta representa posición del ojo, no centro pupilar validado"),
            ("Elipse/diámetro pupilar", "No", "No hay máscara, ejes ni diámetro de pupila"),
            ("Dirección o punto de mirada", "No", "No hay objetivo de pantalla ni vector de mirada"),
            ("Atención/carga cognitiva", "No", "No hay tarea, condición, conducta ni etiqueta cognitiva"),
            ("Partición por participante", "No con el loader actual", "Los nombres usados no codifican el sujeto"),
        ],
        columns=["objetivo", "cubierto", "evidencia"],
    )
    return integrity, fitness


def load_swirski_samples(dataset_dir: Path, roi_size: int = 64) -> list[dict]:
    """Load annotated frames and transform the ellipse into the 64x64 space."""
    samples: list[dict] = []
    labels_path = dataset_dir / "pupil-ellipses.txt"
    for row in labels_path.read_text(encoding="utf-8").splitlines():
        frame_text, ellipse_text = row.split("|", maxsplit=1)
        frame_id = int(frame_text.strip())
        x, y, radius_a, radius_b, angle_rad = map(float, ellipse_text.split())
        image_path = dataset_dir / "frames" / f"{frame_id}-eye.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape
        resized = cv2.resize(image, (roi_size, roi_size), interpolation=cv2.INTER_AREA)

        mask = np.zeros_like(image)
        cv2.ellipse(
            mask,
            (round(x), round(y)),
            (round(radius_a), round(radius_b)),
            np.degrees(angle_rad),
            0,
            360,
            255,
            -1,
        )
        resized_mask = cv2.resize(mask, (roi_size, roi_size), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(
            resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        ellipse = cv2.fitEllipse(max(contours, key=cv2.contourArea))
        (true_x, true_y), (axis_a, axis_b), _ = ellipse
        true_diameter = (axis_a + axis_b) / 2.0

        samples.append(
            {
                "frame": frame_id,
                "image": resized,
                "center_norm": (2.0 * x / width - 1.0, 2.0 * y / height - 1.0),
                "true_center": (true_x, true_y),
                "true_diameter": true_diameter,
            }
        )
    return samples


def _estimate_pupil(
    eye_roi: np.ndarray,
    center_norm: tuple[float, float],
    crop_size: int,
    threshold_offset: int,
) -> tuple[bool, float, tuple[float, float]]:
    """Parameterised form of the classical detector currently in pupilometry.py."""
    roi_size = eye_roi.shape[0]
    nx, ny = center_norm
    center_x = int((nx + 1.0) * roi_size / 2.0)
    center_y = int((ny + 1.0) * roi_size / 2.0)
    x1 = max(0, center_x - crop_size // 2)
    y1 = max(0, center_y - crop_size // 2)
    x2 = min(roi_size, center_x + crop_size // 2)
    y2 = min(roi_size, center_y + crop_size // 2)
    crop = eye_roi[y1:y2, x1:x2]
    if min(crop.shape, default=0) < 5:
        return False, np.nan, (np.nan, np.nan)

    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    minimum = cv2.minMaxLoc(blurred)[0]
    threshold = int(np.clip(minimum + threshold_offset, 0, 255))
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, np.nan, (np.nan, np.nan)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 8 or len(contour) < 5:
        return False, np.nan, (np.nan, np.nan)
    try:
        (local_x, local_y), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
    except cv2.error:
        return False, np.nan, (np.nan, np.nan)
    diameter = (axis_a + axis_b) / 2.0
    if not 2.0 <= diameter <= 25.0:
        return False, np.nan, (np.nan, np.nan)
    return True, diameter, (x1 + local_x, y1 + local_y)


def benchmark_swirski(
    samples: list[dict],
    crop_size: int = 32,
    threshold_offset: int = 15,
    brightness_shift: int = 0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate detection, pupil diameter, and centre error on labelled ellipses."""
    rows: list[dict] = []
    for sample in samples:
        image = np.clip(
            sample["image"].astype(np.int16) + brightness_shift, 0, 255
        ).astype(np.uint8)
        detected, diameter, center = _estimate_pupil(
            image, sample["center_norm"], crop_size, threshold_offset
        )
        center_error = (
            float(np.hypot(center[0] - sample["true_center"][0], center[1] - sample["true_center"][1]))
            if detected
            else np.nan
        )
        rows.append(
            {
                "frame": sample["frame"],
                "detected": detected,
                "true_diameter": sample["true_diameter"],
                "estimated_diameter": diameter,
                "diameter_error": diameter - sample["true_diameter"] if detected else np.nan,
                "absolute_diameter_error": abs(diameter - sample["true_diameter"]) if detected else np.nan,
                "center_error": center_error,
            }
        )
    results = pd.DataFrame(rows)
    valid = results[results["detected"]]
    correlation = (
        float(spearmanr(valid["true_diameter"], valid["estimated_diameter"]).statistic)
        if len(valid) >= 3
        else np.nan
    )
    summary = {
        "n": float(len(results)),
        "detection_rate": float(results["detected"].mean()),
        "diameter_mae_px": float(valid["absolute_diameter_error"].mean()),
        "diameter_bias_px": float(valid["diameter_error"].mean()),
        "diameter_spearman": correlation,
        "center_mae_px": float(valid["center_error"].mean()),
    }
    return results, summary


def benchmark_end_to_end_swirski(
    samples: list[dict],
    model_path: Path,
    crop_size: int = 32,
    threshold_offset: int = 35,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run the deployed centre CNN followed by the classical ellipse detector.

    Unlike :func:`benchmark_swirski`, this test never supplies the annotated
    centre to the detector. It therefore measures the complete pupil pipeline
    on a participant that was absent from LPW training and parameter tuning.
    """
    import torch

    from src.model import EyePupilCNN

    model = EyePupilCNN()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = np.stack(
        [clahe.apply(sample["image"]) for sample in samples]
    ).astype(np.float32) / 255.0
    with torch.no_grad():
        predictions = torch.cat(
            [model(batch) for batch in torch.from_numpy(processed[:, None]).split(128)]
        ).numpy()

    rows: list[dict] = []
    for sample, predicted_norm in zip(samples, predictions):
        detected, diameter, ellipse_center = _estimate_pupil(
            sample["image"], tuple(predicted_norm), crop_size, threshold_offset
        )
        predicted_center = tuple(
            (predicted_norm + 1.0) * sample["image"].shape[0] / 2.0
        )
        cnn_center_error = float(
            np.hypot(
                predicted_center[0] - sample["true_center"][0],
                predicted_center[1] - sample["true_center"][1],
            )
        )
        ellipse_center_error = (
            float(
                np.hypot(
                    ellipse_center[0] - sample["true_center"][0],
                    ellipse_center[1] - sample["true_center"][1],
                )
            )
            if detected
            else np.nan
        )
        rows.append(
            {
                "frame": sample["frame"],
                "detected": detected,
                "true_center_x": sample["true_center"][0],
                "true_center_y": sample["true_center"][1],
                "cnn_center_x": predicted_center[0],
                "cnn_center_y": predicted_center[1],
                "cnn_center_error_px": cnn_center_error,
                "ellipse_center_error_px": ellipse_center_error,
                "true_diameter_px": sample["true_diameter"],
                "estimated_diameter_px": diameter,
                "diameter_error_px": (
                    diameter - sample["true_diameter"] if detected else np.nan
                ),
                "absolute_diameter_error_px": (
                    abs(diameter - sample["true_diameter"])
                    if detected
                    else np.nan
                ),
            }
        )

    results = pd.DataFrame(rows)
    valid = results[results["detected"]]
    summary = {
        "n": int(len(results)),
        "detection_rate": float(results["detected"].mean()),
        "cnn_center_mae_px": float(results["cnn_center_error_px"].mean()),
        "ellipse_center_mae_px": float(valid["ellipse_center_error_px"].mean()),
        "diameter_mae_px": float(valid["absolute_diameter_error_px"].mean()),
        "diameter_median_ae_px": float(
            valid["absolute_diameter_error_px"].median()
        ),
        "diameter_p95_ae_px": float(
            valid["absolute_diameter_error_px"].quantile(0.95)
        ),
        "diameter_bias_px": float(valid["diameter_error_px"].mean()),
        "diameter_spearman": float(
            spearmanr(
                valid["true_diameter_px"], valid["estimated_diameter_px"]
            ).statistic
        ),
        "crop_size": int(crop_size),
        "threshold_offset": int(threshold_offset),
    }
    return results, summary


def sweep_swirski(
    samples: list[dict],
    crop_sizes: tuple[int, ...] = (24, 32, 40, 48, 64),
    threshold_offsets: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35),
) -> pd.DataFrame:
    rows = []
    for crop_size in crop_sizes:
        for offset in threshold_offsets:
            _, summary = benchmark_swirski(samples, crop_size, offset)
            rows.append({"crop_size": crop_size, "threshold_offset": offset, **summary})
    return pd.DataFrame(rows)


def load_nemar_recordings(dataset_dir: Path) -> pd.DataFrame:
    """Summarise paired attentive/distracted recordings from NEMAR nm000150."""
    pattern = re.compile(r"sub-(\d+)_ses-(\d+)_task-stim01_desc-pupil")
    rows: list[dict] = []
    pupil_files = sorted(dataset_dir.glob("derivatives/**/*desc-pupil_eyetrack.tsv.gz"))
    for pupil_path in pupil_files:
        match = pattern.search(pupil_path.name)
        if not match:
            continue
        subject, session = match.groups()
        gaze_path = pupil_path.with_name(
            pupil_path.name.replace("desc-pupil_eyetrack", "desc-gaze_visualangle_eyetrack")
        )
        if not gaze_path.exists():
            continue
        pupil_area = pd.read_csv(pupil_path, sep="\t", compression="gzip", header=None).iloc[:, 0]
        gaze = pd.read_csv(gaze_path, sep="\t", compression="gzip", header=None)
        length = min(len(pupil_area), len(gaze))
        pupil_area = pd.to_numeric(pupil_area.iloc[:length], errors="coerce").interpolate(limit_direction="both")
        # EyeLink reports pupil area in arbitrary units. EyeStim thresholds a
        # diameter ratio, so sqrt(area) is the dimensionally compatible proxy.
        pupil = np.sqrt(pupil_area.clip(lower=0))
        gaze = gaze.iloc[:length].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")
        angle_x = gaze.iloc[:, -2]
        angle_y = gaze.iloc[:, -1]

        baseline = float(pupil.iloc[:100].median())
        ratio = pupil / baseline
        pupil_factor = np.ones(length)
        pupil_factor[(ratio >= 1.02) & (ratio <= 1.20)] = 1.15
        pupil_factor[(ratio < 0.85) | (ratio > 1.30)] = 0.80

        # A fixed +/-20 degree field maps gaze angles to EyeStim's nominal [-1, 1].
        nx = angle_x / 20.0
        ny = angle_y / 20.0
        rolling_x = nx.rolling(30, min_periods=5).std(ddof=0)
        rolling_y = ny.rolling(30, min_periods=5).std(ddof=0)
        rolling_total = np.sqrt(rolling_x**2 + rolling_y**2)
        stability = np.clip(1.0 - rolling_total / 0.25, 0.0, 1.0).fillna(0.5)
        score = np.clip(stability.to_numpy() * pupil_factor * 100.0, 0.0, 100.0)

        rows.append(
            {
                "subject": f"sub-{subject}",
                "session": f"ses-{session}",
                "condition": "attentive" if session == "01" else "distracted",
                "samples": length,
                "pupil_baseline": baseline,
                "pupil_median": float(pupil.median()),
                "pupil_change_pct": float((pupil.median() / baseline - 1.0) * 100.0),
                "pupil_cv_pct": float(pupil.std(ddof=0) / pupil.mean() * 100.0),
                "gaze_dispersion_deg": float(np.hypot(angle_x.std(ddof=0), angle_y.std(ddof=0))),
                "gaze_step_deg": float(np.hypot(angle_x.diff(), angle_y.diff()).median()),
                "eyestim_score_mean": float(np.mean(score)),
            }
        )
    return pd.DataFrame(rows)


def paired_condition_tests(recordings: pd.DataFrame) -> pd.DataFrame:
    """Perform within-subject comparisons to avoid between-person scale bias."""
    metrics = [
        "pupil_change_pct",
        "pupil_cv_pct",
        "gaze_dispersion_deg",
        "gaze_step_deg",
        "eyestim_score_mean",
    ]
    rows = []
    for metric in metrics:
        pivot = recordings.pivot(index="subject", columns="condition", values=metric).dropna()
        difference = pivot["attentive"] - pivot["distracted"]
        result = wilcoxon(difference) if len(difference) else None
        rows.append(
            {
                "metric": metric,
                "paired_subjects": len(difference),
                "attentive_mean": float(pivot["attentive"].mean()),
                "distracted_mean": float(pivot["distracted"].mean()),
                "attentive_minus_distracted": float(difference.mean()),
                "paired_effect_dz": float(difference.mean() / difference.std(ddof=1)),
                "wilcoxon_p": float(result.pvalue) if result is not None else np.nan,
                "expected_direction_rate": float((difference > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def subject_disjoint_classifier(recordings: pd.DataFrame) -> dict[str, float]:
    """Evaluate whether summary signals distinguish condition on unseen subjects."""
    feature_names = [
        "pupil_change_pct",
        "pupil_cv_pct",
        "gaze_dispersion_deg",
        "gaze_step_deg",
    ]
    clean = recordings.dropna(subset=feature_names).copy()
    features = clean[feature_names].to_numpy()
    labels = (clean["condition"] == "attentive").astype(int).to_numpy()
    groups = clean["subject"].to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    splitter = LeaveOneGroupOut()
    probabilities = cross_val_predict(
        model,
        features,
        labels,
        groups=groups,
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "recordings": float(len(clean)),
        "subjects": float(clean["subject"].nunique()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }
