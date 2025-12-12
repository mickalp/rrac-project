import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Basic residual block with two 3x3 conv layers and a skip connection.

    in_channels == out_channels, no downsampling, as in the RRAC paper.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out


class RRACNet(nn.Module):
    """
    RRAC-like artifact correction network.

    Based on the description in:
    "Deep-learning-based ring artifact correction for tomographic reconstruction"
    (J. Synchrotron Rad. 2023, 30, 620–626).

    - 14 conv layers total:
      * 2 initial conv layers
      * 6 residual blocks (each has 2 conv layers)
    - Input: wavelet coefficients of sinogram (4 subbands) -> 4 channels
    - Output: predicted artifact in wavelet coefficients (also 4 channels)
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_blocks: int = 6,
        out_channels: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.out_channels = out_channels

        # First two conv layers (input -> feature space)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(base_channels) for _ in range(num_blocks)]
        )

        # Final conv to map back to wavelet artifact space
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (batch, 4, H, W) wavelet coefficients (4 subbands as channels)

        returns: (batch, 4, H, W) predicted artifact coefficients
        """
        # Initial feature extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Final prediction
        out = self.conv_out(x)
        return out


if __name__ == "__main__":
    # Small self-test when running this file directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = RRACNet().to(device)
    dummy_input = torch.randn(2, 4, 256, 256, device=device)  # batch=2, 4-channel wavelet
    dummy_output = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", dummy_output.shape)
