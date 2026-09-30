"""Inference wrapper for the EyeDentify pupil-diameter regressor."""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

import config
from dataset import EyeDentifyDiameterDataset
from model import PupilDiameterCNN


class LearnedDiameterEstimator:
    """Load a training checkpoint and predict left-pupil diameter in mm."""

    def __init__(self, checkpoint_path=config.DIAMETER_MODEL_SAVE_PATH, device="cpu"):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        self.device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model = PupilDiameterCNN().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.target_mean = float(checkpoint["target_mean"])
        self.target_std = float(checkpoint["target_std"])
        self.metadata = checkpoint.get("metadata", {})

    def predict(self, eye_roi: np.ndarray) -> float:
        image = EyeDentifyDiameterDataset.preprocess(eye_roi)
        tensor = torch.from_numpy(
            np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)).copy()
        )
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            normalized = float(self.model(tensor).item())
        diameter_mm = self.target_mean + self.target_std * normalized
        return float(np.clip(diameter_mm, 1.0, 9.0))
