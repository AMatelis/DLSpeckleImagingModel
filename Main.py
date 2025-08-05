import argparse
import os
import sys
from typing import Optional

import torch
from torch import nn, optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adjust path for modular imports (modify as needed)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.train import train_and_evaluate
from src.dataloader import create_dataloaders
from models.bloodflow_cnn import BloodFlowCNN


def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    sequence_len: int = 5,
    batch_size: int = 8,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    device = torch.device(device if device else "cpu")
    print(f"[INFO] Evaluating on device: {device}")

    _, val_loader, _ = create_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        test_split=0.2,
        sequence_len=sequence_len,
        use_subset=False,
        num_workers=4,
    )
    if val_loader is None or len(val_loader) == 0:
        raise RuntimeError("Validation data not found or improperly formatted.")

    model = BloodFlowCNN().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    scaler_path = os.path.join(ROOT_DIR, "outputs", "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Missing scaler.pkl. Train the model first.")
    scaler = joblib.load(scaler_path)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.cpu().numpy()
            outputs = scaler.inverse_transform(outputs.reshape(-1, 1)).flatten()
            all_preds.extend(outputs)
            all_targets.extend(targets.numpy().flatten())

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print("\n[RESULTS]")
    print(f"MSE     : {mse:.4f}")
    print(f"MAE     : {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({"Target": all_targets, "Prediction": all_preds})
        df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

        plt.figure(figsize=(10, 6))
        plt.scatter(df["Target"], df["Prediction"], alpha=0.6, edgecolors='k')
        min_val = min(df["Target"].min(), df["Prediction"].min())
        max_val = max(df["Target"].max(), df["Prediction"].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
        plt.title("Flow Rate Prediction")
        plt.xlabel("True Flow Rate")
        plt.ylabel("Predicted Flow Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "flowrate_scatter.png"))
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood Flow Estimation")
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True, help="Run mode: train or evaluate")
    parser.add_argument("--checkpoint", type=str, help="Path to model .pt file (for evaluation)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for dataset")
    parser.add_argument("--sequence_len", type=int, default=5, help="Length of frame sequence")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu or cuda)")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Running on: {device}")

    try:
        if args.mode == "train":
            train_loader, val_loader, _ = create_dataloaders(
                data_folder=args.data_dir,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                test_split=0.2,
                use_subset=False,
                num_workers=args.num_workers,
            )
            if train_loader is None or len(train_loader) == 0:
                raise ValueError("No training data found.")

            model = BloodFlowCNN().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

            output_dir = os.path.join(ROOT_DIR, "outputs")
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)

            train_and_evaluate(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=args.num_epochs,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                patience=10,
                val_samples=120,
                grad_clip=1.0,
            )

        elif args.mode == "evaluate":
            if not args.checkpoint or not os.path.isfile(args.checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
            evaluate_model(
                checkpoint_path=args.checkpoint,
                data_dir=args.data_dir,
                sequence_len=args.sequence_len,
                batch_size=args.batch_size,
                device=args.device,
                output_dir=os.path.join(ROOT_DIR, "outputs"),
            )

    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()