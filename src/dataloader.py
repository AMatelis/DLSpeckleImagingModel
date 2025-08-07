import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List

from src.dataset import SpeckleDataset  # Your dataset class that accepts stacks and labels


def extract_flowrate_from_filename(filename: str) -> Optional[float]:
    """
    Extract flow rate from filename like '5ulpermin.avi' or '0.2mlpermin.avi'.
    Converts mL to µL.
    """
    fname = filename.lower()
    match_ul = re.search(r"([\d\.]+)ulpermin", fname)
    if match_ul:
        return float(match_ul.group(1))
    match_ml = re.search(r"([\d\.]+)mlpermin", fname)
    if match_ml:
        return float(match_ml.group(1)) * 1000.0
    return None


def load_video_frames(video_path: str) -> List[np.ndarray]:
    """
    Load grayscale frames from video file.
    Raises errors on failure.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return frames


def prepare_dataset(
    data_folder: str,
    sequence_len: int = 5,
    stride: int = 1,
    cache_file: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert videos in folder to stacked numpy arrays and labels.
    Optionally cache processed data.
    Returns:
      stacks: (N, 1, T, H, W) float32 normalized clips
      labels: (N,) float32 flow rates
    """
    if cache_file and os.path.exists(cache_file):
        print(f"[INFO] Loading cached dataset from: {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        return cached["stacks"], cached["labels"]

    print(f"[INFO] Preparing dataset from folder: {data_folder}")
    all_clips, all_labels = [], []

    for filename in sorted(os.listdir(data_folder)):
        if not filename.lower().endswith(".avi"):
            continue
        flowrate = extract_flowrate_from_filename(filename)
        if flowrate is None or flowrate < 0:
            print(f"[WARN] Skipping {filename}: invalid flowrate")
            continue

        video_path = os.path.join(data_folder, filename)
        try:
            frames = load_video_frames(video_path)
        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            continue

        if len(frames) < sequence_len:
            print(f"[WARN] Skipping short video ({len(frames)} frames): {filename}")
            continue

        for start in range(0, len(frames) - sequence_len + 1, stride):
            clip = frames[start:start + sequence_len]
            clip = np.stack(clip).astype(np.float32) / 255.0  # Normalize [0,1]
            clip = clip[:, np.newaxis, :, :]  # (T, 1, H, W)
            all_clips.append(clip)
            all_labels.append(flowrate)

    if not all_clips:
        raise RuntimeError("[ERROR] No valid clips found in dataset.")

    # Convert to (N, 1, T, H, W)
    stacks = np.transpose(np.array(all_clips), (0, 2, 1, 3, 4))
    labels = np.array(all_labels, dtype=np.float32)

    if cache_file:
        print(f"[INFO] Saving dataset cache to: {cache_file}")
        np.savez_compressed(cache_file, stacks=stacks, labels=labels)

    print(f"[INFO] Dataset prepared with {len(stacks)} samples.")
    return stacks, labels


def create_dataloaders(
    data_folder: str,
    batch_size: int = 8,
    test_split: float = 0.2,
    sequence_len: int = 5,
    stride: int = 1,
    use_subset: bool = False,
    max_samples: int = 300,
    val_sample_size: Optional[int] = None,
    normalize_mode: str = "scale",
    num_workers: int = 0,
    seed: int = 42,
    cache_file: Optional[str] = None,
    augment: bool = False,
    debug: bool = False
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for train and validation datasets.
    Runs only on CPU.
    """
    stacks, labels = prepare_dataset(
        data_folder=data_folder,
        sequence_len=sequence_len,
        stride=stride,
        cache_file=cache_file,
    )

    gen = torch.Generator()
    gen.manual_seed(seed)
    indices = torch.randperm(len(stacks), generator=gen).tolist()

    if use_subset and len(indices) > max_samples:
        indices = indices[:max_samples]
    stacks = stacks[indices]
    labels = labels[indices]

    if len(stacks) < 2:
        raise ValueError("[ERROR] Not enough data for splitting")

    split_idx = int(len(stacks) * (1 - test_split))
    X_train, X_val = stacks[:split_idx], stacks[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]

    if val_sample_size and len(X_val) > val_sample_size:
        val_indices = torch.randperm(len(X_val), generator=gen)[:val_sample_size].tolist()
        X_val, y_val = X_val[val_indices], y_val[val_indices]

    train_dataset = SpeckleDataset(
        stacks=X_train,
        labels=y_train,
        normalize_mode=normalize_mode,
        augment=augment,
        train=True,
        seed=seed,
        debug=debug,
    )
    val_dataset = SpeckleDataset(
        stacks=X_val,
        labels=y_val,
        normalize_mode=normalize_mode,
        augment=False,
        train=False,
        seed=seed,
        debug=debug,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, None


if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python dataloader.py <path_to_video_folder>")
        sys.exit(1)
    train_loader, val_loader, _ = create_dataloaders(
        data_folder=sys.argv[1],
        batch_size=4,
        test_split=0.2,
        sequence_len=5,
        stride=1,
        use_subset=False,
        num_workers=0,
        seed=42,
        cache_file=None,
        augment=False,
        debug=True,
    )

    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    batch = next(iter(train_loader))
    print(f"Sample batch data shape: {batch[0].shape}, labels shape: {batch[1].shape}")
