import os
import re
import logging
from typing import List, Tuple, Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_flowrate_from_filename(filename: str) -> Optional[float]:
    """
    Extracts the numeric flowrate value from the filename.
    Assumes the first number in the filename corresponds to the flowrate.
    """
    matches = re.findall(r"(\d+\.?\d*)", filename)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            logging.warning(f"Could not convert extracted flowrate to float from filename: {filename}")
            return None
    return None


def load_video_frames(video_path: str) -> List[np.ndarray]:
    """
    Loads video frames as grayscale numpy arrays from the given video file.
    Raises errors if file not found or no frames are extracted.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    return frames


def normalize_frames(frames: List[np.ndarray], mode: str = "scale") -> np.ndarray:
    """
    Normalize frames with either scaling [0,1] or z-score normalization.
    Returns numpy array of shape (T, H, W), dtype float32.
    """
    stack = np.stack(frames).astype(np.float32)
    if mode == "scale":
        return stack / 255.0
    elif mode == "zscore":
        mean = stack.mean()
        std = stack.std()
        if std > 1e-6:
            return (stack - mean) / std
        else:
            logging.warning("Std deviation too low for z-score normalization, returning unnormalized frames.")
            return stack
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


class SpeckleDataset(Dataset):
    """
    PyTorch Dataset for loading speckle video frame sequences and their flowrate labels.

    Args:
        data_dir (str): Path to directory containing .avi videos.
        sequence_len (int): Number of consecutive frames per sample.
        stride (int): Step size between sequences.
        normalize_mode (str): "scale" or "zscore" normalization.
        transform (Callable, optional): Optional transform applied on tensor.
        cache_frames (bool): If True, cache entire video frames in memory (uses more RAM).
    """

    def __init__(
        self,
        data_dir: str,
        sequence_len: int,
        stride: int = 1,
        normalize_mode: str = "scale",
        transform: Optional[Callable] = None,
        cache_frames: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.sequence_len = sequence_len
        self.stride = stride
        self.normalize_mode = normalize_mode
        self.transform = transform
        self.cache_frames = cache_frames

        self.samples: List[Tuple[str, int]] = []
        self.flowrates: List[float] = []
        self._video_cache: dict[str, List[np.ndarray]] = {}

        self._prepare_index()

    def _prepare_index(self) -> None:
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        video_files = sorted(f for f in os.listdir(self.data_dir) if f.lower().endswith(".avi"))
        if not video_files:
            logging.warning(f"No .avi videos found in {self.data_dir}")

        for vf in video_files:
            flowrate = extract_flowrate_from_filename(vf)
            if flowrate is None:
                logging.warning(f"Skipping {vf}: Could not extract flowrate.")
                continue

            video_path = os.path.join(self.data_dir, vf)
            try:
                if self.cache_frames:
                    if video_path not in self._video_cache:
                        self._video_cache[video_path] = load_video_frames(video_path)
                    frames = self._video_cache[video_path]
                else:
                    frames = load_video_frames(video_path)

                if len(frames) < self.sequence_len:
                    logging.warning(f"Skipping {vf}: video length {len(frames)} < sequence length {self.sequence_len}")
                    continue

                for start_idx in range(0, len(frames) - self.sequence_len + 1, self.stride):
                    self.samples.append((video_path, start_idx))
                    self.flowrates.append(flowrate)

            except (FileNotFoundError, RuntimeError) as e:
                logging.warning(f"Skipping {vf} due to error: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, start_idx = self.samples[idx]

        if self.cache_frames and video_path in self._video_cache:
            frames = self._video_cache[video_path][start_idx : start_idx + self.sequence_len]
        else:
            all_frames = load_video_frames(video_path)
            frames = all_frames[start_idx : start_idx + self.sequence_len]

        norm_frames = normalize_frames(frames, mode=self.normalize_mode)
        # Add channel dimension (C=1), result shape: (1, T, H, W)
        tensor = torch.from_numpy(norm_frames).unsqueeze(0).float()

        if self.transform:
            tensor = self.transform(tensor)

        label = torch.tensor(self.flowrates[idx], dtype=torch.float32)

        return tensor, label


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python dataset.py <path_to_video_folder>")
        sys.exit(1)

    dataset = SpeckleDataset(
        data_dir=sys.argv[1],
        sequence_len=5,
        stride=1,
        normalize_mode="scale",
        cache_frames=False,
    )

    print(f"Dataset initialized with {len(dataset)} samples.")

    sample_tensor, sample_label = dataset[0]
    print(f"Example sample tensor shape: {sample_tensor.shape}")
    print(f"Example label: {sample_label.item()}")