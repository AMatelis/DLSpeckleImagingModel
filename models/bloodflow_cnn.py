import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for adaptive channel weighting.
    Improves representational power by re-scaling channels based on global context.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        b, c, t, h, w = x.size()
        squeeze = x.view(b, c, -1).mean(dim=2)  # Global avg over T,H,W
        excitation = F.relu(self.fc1(squeeze), inplace=True)
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = excitation.view(b, c, 1, 1, 1)
        return x * excitation


class BloodFlowCNN(nn.Module):
    """
    Deep 3D CNN for speckle pattern blood flow estimation.
    Input: (B, 1, T, H, W)
    Output: (B,) predicted flow rates.
    """
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,
        reduction: int = 16,
        dropout_rate: float = 0.4,
        input_norm: bool = True,
    ):
        super().__init__()

        self.input_norm = nn.BatchNorm3d(input_channels) if input_norm else nn.Identity()

        # 3D CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            SEBlock(base_channels * 4, reduction=reduction),

            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        # Fully connected regression head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(base_channels * 8, base_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(base_channels * 4, 1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input_shape(x)
        x = self.input_norm(x)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    def _initialize_weights(self):
        """Kaiming-normal initialization for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _validate_input_shape(x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, T, H, W), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.size(1)} channels.")

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


def get_model(**kwargs) -> BloodFlowCNN:
    """Factory function to create BloodFlowCNN."""
    return BloodFlowCNN(**kwargs)


if __name__ == "__main__":
    # Quick sanity test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    dummy_input = torch.randn(2, 1, 16, 128, 128).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape:", output.shape)
