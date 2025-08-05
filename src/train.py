import os
import random
import logging
import argparse
import csv
from typing import Optional, List, Tuple, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ------------------------------
# Utils: Reproducibility & Logging
# ------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )
    return logging.getLogger()


def save_csv_predictions(preds: List[float], targets: List[float], filepath: str) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "TrueFlowRate", "PredictedFlowRate", "AbsError", "SquaredError", "RelError(%)", "FlowClass"])
        for i, (t, p) in enumerate(zip(targets, preds)):
            abs_err = abs(t - p)
            sq_err = abs_err ** 2
            rel_err = (abs_err / t * 100) if t != 0 else 0.0
            flow_class = classify_flowrate(t)
            writer.writerow([i, round(t, 2), round(p, 2), round(abs_err, 2), round(sq_err, 2), round(rel_err, 2), flow_class])


def classify_flowrate(value: float) -> str:
    if value < 20:
        return "Low"
    elif value < 200:
        return "Medium"
    else:
        return "High"


def plot_scatter(true_vals: List[float], pred_vals: List[float], filepath: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.7, edgecolors='k')
    min_val, max_val = min(true_vals), max(true_vals)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("True Flow Rate (µL/min)")
    plt.ylabel("Predicted Flow Rate (µL/min)")
    plt.title("Predicted vs True Flow Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_loss_curve(losses: List[float], filepath: str) -> None:
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# ------------------------------
# Dataset with Lazy Loading and Augmentation Hooks
# ------------------------------

def extract_flowrate_from_filename(filename: str) -> Optional[float]:
    import re
    matches = re.findall(r"(\d+\.?\d*)", filename)
    if matches:
        return float(matches[0])
    return None


def load_video_frames(video_path: str) -> List[np.ndarray]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video missing: {video_path}")
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
        raise RuntimeError(f"No frames extracted: {video_path}")
    return frames


def normalize_frames(frames: List[np.ndarray], mode: str = "scale") -> np.ndarray:
    stack = np.stack(frames).astype(np.float32)
    if mode == "scale":
        return stack / 255.0
    elif mode == "zscore":
        mean = stack.mean()
        std = stack.std()
        return (stack - mean) / std if std > 1e-6 else stack
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


class SpeckleDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        sequence_len: int,
        stride: int,
        normalize_mode: str = "scale",
        transform: Optional[Callable] = None,
        cache_frames: bool = False
    ):
        self.data_dir = data_dir
        self.sequence_len = sequence_len
        self.stride = stride
        self.normalize_mode = normalize_mode
        self.transform = transform
        self.cache_frames = cache_frames

        self.samples: List[Tuple[str, int]] = []
        self.flowrates: List[float] = []
        self.video_frames_cache = {}

        self._prepare_index()

    def _prepare_index(self) -> None:
        video_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".avi")]
        video_files.sort()
        for vf in video_files:
            flowrate = extract_flowrate_from_filename(vf)
            if flowrate is None:
                logging.warning(f"Cannot extract flowrate from filename: {vf}, skipping.")
                continue
            video_path = os.path.join(self.data_dir, vf)
            try:
                if self.cache_frames and video_path not in self.video_frames_cache:
                    frames = load_video_frames(video_path)
                    self.video_frames_cache[video_path] = frames
                else:
                    frames = load_video_frames(video_path)
            except Exception as e:
                logging.warning(f"Error loading video {vf}: {e}, skipping.")
                continue
            if len(frames) < self.sequence_len:
                logging.warning(f"Video {vf} too short ({len(frames)} frames), skipping.")
                continue
            for start_idx in range(0, len(frames) - self.sequence_len + 1, self.stride):
                self.samples.append((video_path, start_idx))
                self.flowrates.append(flowrate)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, start_idx = self.samples[idx]
        if self.cache_frames and video_path in self.video_frames_cache:
            frames = self.video_frames_cache[video_path][start_idx:start_idx + self.sequence_len]
        else:
            frames_all = load_video_frames(video_path)
            frames = frames_all[start_idx:start_idx + self.sequence_len]

        seq_norm = normalize_frames(frames, mode=self.normalize_mode)
        seq_norm = seq_norm[:, np.newaxis, :, :]
        tensor = torch.from_numpy(seq_norm).float().permute(1, 0, 2, 3)

        if self.transform is not None:
            tensor = self.transform(tensor)

        label = torch.tensor(self.flowrates[idx], dtype=torch.float32)
        return tensor, label


# ------------------------------
# Model: 3D CNN with Positive Output Activation
# ------------------------------

class SpeckleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)


# ------------------------------
# Training & Validation Loops
# ------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: StandardScaler,
    device: torch.device,
    grad_clip: float,
    scaler_amp: torch.cuda.amp.GradScaler
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        targets_np = targets.cpu().numpy().reshape(-1, 1)
        targets_scaled = torch.tensor(scaler.transform(targets_np), dtype=torch.float32, device=device).flatten()

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cpu'):
            outputs = model(inputs).flatten()
            loss = criterion(outputs, targets_scaled)

        scaler_amp.scale(loss).backward()
        if grad_clip > 0:
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    scaler: StandardScaler,
    device: torch.device
) -> Tuple[float, List[float], List[float]]:
    model.eval()
    preds, targets_all = [], []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs).flatten()
                targets_np = targets.cpu().numpy().reshape(-1, 1)
                targets_scaled = scaler.transform(targets_np).flatten()
                loss = criterion(outputs, torch.tensor(targets_scaled, dtype=torch.float32, device=device))
            total_loss += loss.item() * inputs.size(0)

            preds_batch = scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()
            preds.extend(preds_batch)
            targets_all.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, preds, targets_all


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int,
    checkpoint_dir: str,
    output_dir: str,
    patience: int,
    val_samples: int,
    grad_clip: float,
):
    logger = logging.getLogger()
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info("Fitting target scaler on training labels...")
    train_targets = []
    for _, y in train_loader:
        train_targets.extend(y.numpy())
    scaler = StandardScaler().fit(np.array(train_targets).reshape(-1, 1))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    logger.info("Scaler saved.")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    scaler_amp = torch.cuda.amp.GradScaler()  # Mixed precision
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    writer = SummaryWriter(log_dir=output_dir)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_clip, scaler_amp)
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, scaler, device)
        train_losses.append(train_loss)

        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        epoch_pred_path = os.path.join(output_dir, f"predictions_epoch_{epoch}.csv")
        save_csv_predictions(val_preds[:val_samples], val_targets[:val_samples], epoch_pred_path)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model at epoch {epoch}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch}")

        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered.")
            break

    writer.close()

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    _, final_preds, final_targets = validate(model, val_loader, criterion, scaler, device)
    mse = mean_squared_error(final_targets, final_preds)
    mae = mean_absolute_error(final_targets, final_preds)
    r2 = r2_score(final_targets, final_preds)

    logger.info("Final validation metrics:")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R2:  {r2:.4f}")

    final_pred_path = os.path.join(output_dir, "final_predictions.csv")
    save_csv_predictions(final_preds[:val_samples], final_targets[:val_samples], final_pred_path)

    plot_scatter(final_targets, final_preds, os.path.join(output_dir, "pred_vs_true.png"))
    plot_loss_curve(train_losses, os.path.join(output_dir, "training_loss.png"))

    logger.info("Training complete.")


# ------------------------------
# DataLoader creation
# ------------------------------

def create_data_loaders(
    data_dir: str,
    batch_size: int,
    sequence_len: int,
    stride: int,
    normalize_mode: str,
    test_split: float,
    num_workers: int,
    transform: Optional[Callable] = None,
    cache_frames: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    dataset = SpeckleDataset(
        data_dir=data_dir,
        sequence_len=sequence_len,
        stride=stride,
        normalize_mode=normalize_mode,
        transform=transform,
        cache_frames=cache_frames,
    )

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(np.floor(test_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ------------------------------
# Main & Argparse
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Speckle Flow Estimation Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to speckle videos dataset folder")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output folder for checkpoints, logs, and plots")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sequence_len", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--normalize_mode", type=str, choices=["scale", "zscore"], default="scale")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_samples", type=int, default=50)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max norm for gradient clipping. Set 0 to disable.")
    parser.add_argument("--cache_frames", action='store_true', help="Cache video frames in memory (uses more RAM).")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()

    logger = setup_logger(args.output_dir)
    logger.info("Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_len=args.sequence_len,
        stride=args.stride,
        normalize_mode=args.normalize_mode,
        test_split=args.test_split,
        num_workers=args.num_workers,
        cache_frames=args.cache_frames
    )

    model = SpeckleRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        output_dir=args.output_dir,
        patience=args.patience,
        val_samples=args.val_samples,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    main()