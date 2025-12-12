import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyNet(nn.Module):
    """
    Very small CNN just to test GPU forward + backward.
    Input: (batch, 1, 64, 64)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("CUDA device name:", torch.cuda.get_device_name(0))

    # Create model and move to GPU (if available)
    model = TinyNet().to(device)

    # Dummy input batch: 8 images, 1 channel, 64x64
    x = torch.randn(8, 1, 64, 64, device=device)

    # Dummy target: we just try to predict all zeros
    target = torch.zeros_like(x)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # One tiny training loop (3 iterations)
    for step in range(3):
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x)

        # Compute loss
        loss = criterion(y_pred, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        print(f"Step {step+1}: loss = {loss.item():.6f}")

    print("Done. If this ran without errors and 'Using device: cuda',")
    print("your GPU training setup is working.")


if __name__ == "__main__":
    main()
