# 👁️ EyeStim — Pupilometría experimental (híbrido CNN + OpenCV)

This repository contains an experimental real-time pupilometry module. It combines a pupil-centre CNN trained on LPW, a pupil-diameter CNN trained on EyeDentify/Tobii references, and classical local ellipse fitting with OpenCV. The displayed 0–100 value is an **experimental ocular-stability index**, not a validated probability of cognitive attention.

The project is designed under strict principles of modularity and efficiency to run in real-time on a conventional CPU using low-cost webcams.

---

## 🏗️ Repository Structure

The project architecture is organized in a modular structure:

```text
Eyestim/
│
├── data/                    # Local Dataset (train/val/testing) - [Git Ignored]
├── docs/                    # Scientific documentation, training evidence and session reports
│   ├── cnn_architecture.md  # Mermaid representation of the CNN architecture
│   ├── guia_sencilla_eyestim.md # Simplified non-technical guide
│   ├── session_report.md    # Statistical report of the last session
│   └── attention_evolution.png # Temporal chart of attention and pupil diameter
│
├── output/                  # Official deliverables
│   └── pdf/
│       └── eyestim_paper_academico_2026.pdf # Official academic research paper (BUAP Social Service)
│
├── models/                  # Weights and trained models (.pth) - [Git Ignored]
│
├── src/                     # Unified source code
│   ├── config.py            # System constants and configuration thresholds
│   ├── utils.py             # Helper routines and OpenCV validation check
│   ├── dataset.py           # LPW/EyeDentify loaders plus legacy BioID loader
│   ├── model.py             # Centre and diameter CNN architectures
│   ├── train_center.py      # Subject-disjoint LPW training
│   ├── train_diameter.py    # Subject-disjoint EyeDentify training
│   ├── predict.py           # Static inference and visual comparison (CNN)
│   ├── pupilometry.py       # Classical local processing of pupil diameter
│   ├── attention.py         # Spatial mapper and experimental ocular index
│   ├── reporter.py          # Statistical reporter generator in Markdown and PNG charts
│   └── show_eyes.py         # Real-time HUD interactive viewer using webcam
│
├── requirements.txt         # Project dependencies
└── .gitignore               # Git exclusions
```

---

## ⚙️ Hybrid pupilometry pipeline

The system integrates a processing flow across four parallel phases:
1. **Facial and Eye Detection**: Uses fast Haar Cascades classifiers to delimit the eye region and crop the eye ROI into a $64 \times 64$ pixels window.
2. **Deep Iris Localization (CNN)**: The lightweight convolutional network `EyePupilCNN` performs Cartesian regression on the ROI to locate the exact geometric center of the iris.
3. **Classical Precision Pupilometry**: Extracts a local crop centered on the CNN estimation, applying local adaptive thresholding and least-squares ellipse fitting (`cv2.fitEllipse`) to estimate the physical pupil diameter in pixels.
4. **Experimental Ocular Index**: Logs diameter fluctuations and spatial stability over five heuristic regions. Research validation found that the current formula must not be interpreted as cognitive attention.

## 📚 Research datasets and trained checkpoints

- **LPW**: 16,000 annotated frames from 8 participants, split by participant, for pupil-centre localization.
- **EyeDentify**: a 9,425-image representative subset covering all 51 participants and lighting sessions, with participant-disjoint splits and Tobii pupil-diameter references.
- **Cambridge/Świrski**: an independent participant used for end-to-end ellipse and diameter testing.
- **NEMAR BBBD**: paired attentive/distracted recordings used only to test the validity of the ocular index.

The datasets remain under `data/external/` and are ignored by Git according to their licences. The active local checkpoints are `src/models/eyestim_cnn.pth` and `src/models/eyestim_diameter.pth`. Reproduce the download, training, audit, plots, and executed notebooks with the commands in `notebooks/README.md`. The final scientific decision is in `docs/research/project_culmination_report.md`.

---

## 🚀 Installation and Setup

### 1. Clone the repository and set up the environment
Setting up a local virtual environment is highly recommended:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

---

## 🧪 Validation and testing

The module includes assertions for the pupil detector and the mathematical behaviour of the experimental ocular index. These are software tests, not cognitive validation:

```bash
# Run classical pupilometry unit tests
python src/pupilometry.py

# Run attention tracking unit tests
python src/attention.py

# Audit frozen checkpoints and regenerate the final visual dashboard
python research/final_validation.py
```

*Both commands will output success (`All unit tests passed!`) if the mathematical calculations of calibration, quadrant mapping, and ellipse fitting match the expected values.*

---

## 💻 Interactive Real-Time Demo

Start the webcam in real-time with the **HUD interactive viewer**:
```bash
python src/show_eyes.py
```

### Viewer Controls:
*   **[ Ocular Index Bar ]**: Experimental stability indicator; it is not a cognitive-attention probability.
*   **[ Gaze Vector ]**: Yellow vector pointing from the iris center to the direction of your screen focus.
*   **[ Pupil Outline ]**: Green ellipse contouring your pupil in real-time along with its diameter in pixels.
*   **[ Key 'q' ]**: Safely exits the interactive demo and triggers the reporter module to generate the session report.
*   **[ Key 't' ]**: Opens the final training and validation dashboard.

---

## 📊 Session Statistical Report

Upon pressing `q` to exit the webcam view, `src/reporter.py` automatically compiles a complete report in `docs/`:

*   **`docs/session_report.md`**: session-level pupil measurements, experimental ocular index, and heuristic region distribution.
*   **`docs/attention_evolution.png`**: temporal pupil and ocular-index signals.

---

## 🎓 Academic Research Paper (PDF)

The official **academic paper (12 pages, BUAP Social Service report)** documents the project background, technical problem, dataset provenance, model configurations, CPU resources, training protocol, results, statistical validation, viability, limitations, deliverables, and references. It also distinguishes the proposed adaptive-stimulus scope from the experimentally validated pupilometry work.

The deliverable is available at **`output/pdf/eyestim_paper_academico_2026.pdf`**. Its reproducible builders are `research/paper_figures.py` and `scripts/build_academic_paper.py`.
