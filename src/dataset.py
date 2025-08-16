import os
import re
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_flow_from_filename(fname: str) -> Optional[float]:
    """Extract numeric flow rate from filename like '5ul.avi', 'flow_5.0.avi'."""
    m = re.search(r"([0-9]+\.?[0-9]*)", fname)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def read_video_segment(video_path: str, start: int, length: int) -> List[np.ndarray]:
    """Read a segment of frames from a video in grayscale starting at `start`."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Could not open video: {video_path}")

    # Jump directly to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    for _ in range(length):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.float32))

    cap.release()
    if len(frames) < length:
        raise RuntimeError(f"[ERROR] Not enough frames in segment from: {video_path}")
    return frames


class SpeckleDataset(Dataset):
    """
    Memory-efficient speckle dataset that loads only the frames needed for each sample.

    Args:
        folder (str): Path to folder containing .avi videos.
        sequence_len (int): Frames per sequence.
        stride (int): Step size for sliding window extraction.
        normalize (str): 'scale' or 'zscore' normalization.
        cache_videos (bool): If True, caches full videos in memory (RAM heavy).
    """

    def __init__(
        self,
        folder: str,
        sequence_len: int = 5,
        stride: int = 1,
        normalize: str = 'scale',
        cache_videos: bool = False
    ):
        self.folder = folder
        self.sequence_len = sequence_len
        self.stride = stride
        self.normalize = normalize
        self.cache_videos = cache_videos

        self.samples: List[Tuple[str, int, float]] = []
        self.cache = {}

        if not os.path.isdir(folder):
            raise NotADirectoryError(f"[ERROR] Dataset folder not found: {folder}")

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.avi')])
        if not files:
            raise RuntimeError(f"[ERROR] No .avi files found in {folder}")

        for fn in files:
            flow = extract_flow_from_filename(fn)
            if flow is None:
                print(f"[WARN] Could not parse flow rate from filename: {fn}")
                continue

            path = os.path.join(folder, fn)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"[WARN] Skipping {fn}: cannot open")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count < sequence_len:
                print(f"[WARN] Skipping {fn}: only {frame_count} frames (< {sequence_len})")
                continue

            if cache_videos:
                try:
                    self.cache[path] = read_video_segment(path, 0, frame_count)
                except Exception as e:
                    print(f"[WARN] Skipping {fn} due to cache error: {e}")
                    continue

            # Build sample index list
            for start in range(0, frame_count - sequence_len + 1, stride):
                self.samples.append((path, start, float(flow)))

        if not self.samples:
            raise RuntimeError(f"[ERROR] No valid sequences found in {folder}")

        print(f"[INFO] Prepared {len(self.samples)} sequences from {len(files)} videos.")

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_clip(self, arr: np.ndarray) -> np.ndarray:
        """Apply selected normalization mode."""
        if self.normalize == 'scale':
            arr = arr / 255.0
        elif self.normalize == 'zscore':
            mu, sd = arr.mean(), arr.std()
            arr = (arr - mu) / (sd + 1e-8)
        return arr.astype(np.float32)

    def _get_clip(self, path: str, start: int) -> torch.Tensor:
        """Retrieve a clip (1, T, H, W) from cache or disk."""
        if path in self.cache:
            frames = self.cache[path][start:start + self.sequence_len]
        else:
            frames = read_video_segment(path, start, self.sequence_len)

        arr = np.stack(frames, axis=0)  # (T, H, W)
        arr = self._normalize_clip(arr)
        arr = arr[None, ...]  # (1, T, H, W)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        path, start, flow = self.samples[idx]
        x = self._get_clip(path, start)
        y = torch.tensor(flow, dtype=torch.float32)
        return x, y
