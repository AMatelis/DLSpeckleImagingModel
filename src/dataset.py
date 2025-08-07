import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple


class SpeckleDataset(Dataset):
    def __init__(self, data_folder: str, transform: bool = True):
        """
        Args:
            data_folder (str): Path to folder containing .avi videos.
            transform (bool): Whether to normalize video frames.
        """
        self.data_folder = data_folder
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, float]]:
        """Collect all video file paths and their associated labels (from filename)."""
        samples = []
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith('.avi'):
                try:
                    label = self._extract_label_from_filename(file_name)
                    file_path = os.path.join(self.data_folder, file_name)
                    samples.append((file_path, label))
                except ValueError:
                    print(f"[WARNING] Skipping file with invalid label format: {file_name}")
        return samples

    def _extract_label_from_filename(self, filename: str) -> float:
        """Extract numeric label from filename, e.g. '5ulpermin.avi' -> 5.0."""
        base = os.path.splitext(filename)[0]
        digits = ''.join(c if c.isdigit() or c == '.' else '' for c in base)
        if not digits:
            raise ValueError(f"No numeric label found in filename: {filename}")
        return float(digits)

    def _load_video(self, filepath: str) -> torch.Tensor:
        """Loads video file and converts it into a (C, T, H, W) tensor."""
        cap = cv2.VideoCapture(filepath)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            frame = cv2.resize(frame, (64, 64))               # Resize for uniformity
            frames.append(frame)

        cap.release()

        video_np = np.stack(frames, axis=0)  # Shape: (T, H, W)
        video_np = np.expand_dims(video_np, axis=0)  # Shape: (C=1, T, H, W)

        video_tensor = torch.from_numpy(video_np).float() / 255.0  # Normalize to [0, 1]

        if self.transform:
            mean = video_tensor.mean()
            std = video_tensor.std()
            video_tensor = (video_tensor - mean) / (std + 1e-6)

        return video_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[idx]
        video_tensor = self._load_video(video_path)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return video_tensor, label_tensor
