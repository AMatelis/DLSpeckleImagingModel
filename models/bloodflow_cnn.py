import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for adaptive channel weighting."""
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.size()
        squeeze = x.view(b, c, -1).mean(dim=2)  # Global Average Pooling (spatial+temporal)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = excitation.view(b, c, 1, 1, 1)
        return x * excitation


class BloodFlowCNN(nn.Module):
    """
    Deep 3D CNN model for speckle pattern blood flow estimation.
    Designed for research/medical-grade regression tasks on video input.
    """
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,
        reduction: int = 16,
        dropout_rate: float = 0.4,
        input_norm: bool = True,
    ):
        super(BloodFlowCNN, self).__init__()

        self.input_norm = nn.BatchNorm3d(input_channels) if input_norm else nn.Identity()

        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            # SEBlock inserted here to leverage spatial features better
            SEBlock(base_channels * 4, reduction=reduction),

            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # output: [B, C, 1, 1, 1]
        )

        # We'll infer the flattened feature size dynamically after a dummy forward
        self._feature_dim: Optional[int] = None

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(base_channels * 8, base_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(base_channels * 4, 1)  # Regression output

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input shape
        self._validate_input_shape(x)

        x = self.input_norm(x)
        x = self.encoder(x)
        x = torch.flatten(x, 1)  # Flatten all except batch dim

        # Lazy init feature dim if not set (allows dynamic input size)
        if self._feature_dim is None:
            self._feature_dim = x.size(1)
            # Re-initialize fully connected layers with correct input dim
            self.fc1 = nn.Linear(self._feature_dim, self.fc1.out_features).to(x.device)
            self.fc2 = nn.Linear(self.fc1.out_features, 1).to(x.device)
            self._initialize_weights()

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        """Initialize weights with Kaiming normal."""
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
            raise ValueError(f"Expected 5D input (B,C,T,H,W), got {x.ndim}D")
        if x.size(1) != 1:
            raise ValueError(f"Expected input with 1 channel, got {x.size(1)}")

    def freeze_encoder(self):
        """Freeze encoder layers for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder layers."""
        for param in self.encoder.parameters():
            param.requires_grad = True


def test_model():
    """
    Sanity check: Runs a forward pass with dummy input on available device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BloodFlowCNN().to(device)
    model.eval()
    dummy_input = torch.randn(2, 1, 16, 128, 128).to(device)  # [B, C, T, H, W]
    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_model()