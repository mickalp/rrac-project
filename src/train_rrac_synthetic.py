import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rrac_model import RRACNet
from dataset_synthetic import SyntheticStripeDataset


def main():
    # --- Paths ---
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_root = os.path.join(base_dir, "data", "synthetic")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))

    # --- Dataset & DataLoader ---
    dataset = SyntheticStripeDataset(root_dir=data_root, wavelet="haar", normalize=False)
    print("Total samples:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # keep 0 on Windows to avoid issues
        pin_memory=True if device.type == "cuda" else False,
    )

    # --- Model, loss, optimizer ---
    model = RRACNet(in_channels=4, base_channels=64, num_blocks=6, out_channels=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- Training loop (small sanity test) ---
    num_epochs = 2           # keep it small for now
    print_every = 5          # print every N steps
    max_steps_per_epoch = 20 # cap steps per epoch to keep this fast

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, (x, y) in enumerate(dataloader):
            # Optionally stop early to keep first test fast
            if step >= max_steps_per_epoch:
                break

            # Move data to device
            x = x.to(device)  # striped wavelet
            y = y.to(device)  # artifact wavelet

            # Forward
            pred = model(x)   # predicted artifact

            # Loss between predicted artifact and true artifact
            loss = criterion(pred, y)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % print_every == 0:
                avg_loss = running_loss / print_every
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step+1}] "
                      f"Loss: {avg_loss:.6f}")
                running_loss = 0.0

    print("Training sanity test finished.")


if __name__ == "__main__":
    main()
