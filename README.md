# 👁️ EyeStim — Visual Attention Analyzer (Deep-Classic Hybrid)

This repository contains the **Real-Time Pupilometry and Attention Tracking Module (EyeStim)**. Its purpose is to capture, model, and record the user's visual behavior in a non-invasive manner through a hybrid pipeline: deep gaze estimation by convolutional neural networks (CNN) and classical local pupillometric measurement using OpenCV.

The project is designed under strict principles of modularity and efficiency to run in real-time on a conventional CPU using low-cost webcams.

---

## 🏗️ Repository Structure

The project architecture is organized in a modular structure:

```text
Eyestim/
│
├── data/                    # Local Dataset (train/val/testing) - [Git Ignored]
├── docs/                    # Scientific documentation and session reports
│   ├── paper_eyestim.pdf    # Academic paper of the project in two-columns (PDF)
│   ├── paper_eyestim.tex    # LaTeX source code of the academic paper
│   ├── cnn_architecture.md  # Mermaid representation of the CNN architecture
│   ├── session_report.md    # Statistical report of the last session
│   └── attention_evolution.png # Temporal chart of attention and pupil diameter
│
├── models/                  # Weights and trained models (.pth) - [Git Ignored]
│
├── src/                     # Unified source code
│   ├── config.py            # System constants and configuration thresholds
│   ├── utils.py             # Helper routines and OpenCV validation check
│   ├── dataset.py           # PyTorch dataset loader and preprocessor for BioID
│   ├── model.py             # Lightweight CNN architecture (EyePupilCNN)
│   ├── train.py             # CNN training and convergence script
│   ├── predict.py           # Static inference and visual comparison (CNN)
│   ├── pupilometry.py       # Classical local processing of pupil diameter
│   ├── attention.py         # Spatial mapper and cognitive attention estimator
│   ├── reporter.py          # Statistical reporter generator in Markdown and PNG charts
│   └── show_eyes.py         # Real-time HUD interactive viewer using webcam
│
├── requirements.txt         # Project dependencies
└── .gitignore               # Git exclusions
```

---

## ⚙️ Hybrid Vision and Attention Pipeline

The system integrates a processing flow across four parallel phases:
1. **Facial and Eye Detection**: Uses fast Haar Cascades classifiers to delimit the eye region and crop the eye ROI into a $64 \times 64$ pixels window.
2. **Deep Iris Localization (CNN)**: The lightweight convolutional network `EyePupilCNN` performs Cartesian regression on the ROI to locate the exact geometric center of the iris.
3. **Classical Precision Pupilometry**: Extracts a local crop centered on the CNN estimation, applying local adaptive thresholding and least-squares ellipse fitting (`cv2.fitEllipse`) to estimate the physical pupil diameter in pixels.
4. **Tracking and Attention Score**: Logs diameter fluctuations relative to an initially calibrated baseline, evaluating spatial gaze stability over 5 screen quadrants (Center, Up, Down, Left, Right) to compute a cognitive attention score (0%-100%).

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

## 🧪 Validation and Unit Testing

The module includes unit assertion tests to verify that both classical pupilometry and cognitive attention tracking function accurately before interactive deployment:

```bash
# Run classical pupilometry unit tests
python src/pupilometry.py

# Run attention tracking unit tests
python src/attention.py
```

*Both commands will output success (`All unit tests passed!`) if the mathematical calculations of calibration, quadrant mapping, and ellipse fitting match the expected values.*

---

## 💻 Interactive Real-Time Demo

Start the webcam in real-time with the **HUD interactive viewer**:
```bash
python src/show_eyes.py
```

### Viewer Controls:
*   **[ Attention Progress Bar ]**: Dynamic progress bar at the top of the screen that changes colors (cyan/green/yellow/red) based on your attention score.
*   **[ Gaze Vector ]**: Yellow vector pointing from the iris center to the direction of your screen focus.
*   **[ Pupil Outline ]**: Green ellipse contouring your pupil in real-time along with its diameter in pixels.
*   **[ Key 'q' ]**: Safely exits the interactive demo and triggers the reporter module to generate the session report.

---

## 📊 Session Statistical Report

Upon pressing `q` to exit the webcam view, `src/reporter.py` automatically compiles a complete report in `docs/`:

*   **[docs/session_report.md](file:///c:/Users/Eduar/AREA_PROGRAMCION/01_PROYECTOS/Eyestim/docs/session_report.md)**: Detailed Markdown report with average attention, average pupil diameter, and screen quadrant time distribution.
*   **[docs/attention_evolution.png](file:///c:/Users/Eduar/AREA_PROGRAMCION/01_PROYECTOS/Eyestim/docs/attention_evolution.png)**: Dual-panel plot displaying temporal pupil diameter changes against the baseline (top) and attention score dynamics (bottom).

---

## 🎓 Academic Research Paper (LaTeX / PDF)

For scientific documentation and social service validation, a full **5-page** research paper was written and formatted in two columns under the standard academic layout.

The pre-compiled PDF featuring native TikZ vector diagrams of the CNN blocks can be accessed here:
👉 **[paper_eyestim.pdf](file:///c:/Users/Eduar/AREA_PROGRAMCION/01_PROYECTOS/Eyestim/docs/paper_eyestim.pdf)**
