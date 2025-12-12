import os
import argparse
import time  # for timing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rrac_model import RRACNet
from dataset_synthetic import SyntheticStripeDataset
from dataset_experimental import ExperimentalStripeDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RRACNet on synthetic or experimental sinograms."
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "experimental"],
        default="synthetic",
        help="Which dataset to use (default: synthetic).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="rrac_checkpoint.pth",
        help="Path to save/load model checkpoint (default: rrac_checkpoint.pth).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, resume training from the checkpoint if it exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_root_synth = os.path.join(base_dir, "data", "synthetic")
    data_root_exp = os.path.join(base_dir, "data", "experimental")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))

    # --- Dataset choice ---
    if args.dataset == "synthetic":
        dataset = SyntheticStripeDataset(
            root_dir=data_root_synth,
            wavelet="haar",
            normalize=False,
        )
    else:
        dataset = ExperimentalStripeDataset(
            root_dir=data_root_exp,
            wavelet="haar",
            normalize=True,   # normalize real sinograms
            patch_size=1024,  # central 1024x1024 patch; adjust if needed
        )

    print(f"Using dataset: {args.dataset}, total samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows-safe
        pin_memory=True if device.type == "cuda" else False,
    )

    # --- Model, loss, optimizer ---
    model = RRACNet(in_channels=4, base_channels=64, num_blocks=6, out_channels=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Optionally resume from checkpoint ---
    ckpt_path = os.path.join(base_dir, args.checkpoint)
    if args.resume and os.path.isfile(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0)
    else:
        start_epoch = 0

    # --- Training loop ---
    num_epochs = args.epochs
    print_every = 10

    model.train()

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        time_accum = 0.0

        # Wrap dataloader with tqdm for this epoch
        dataloader_epoch = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True,
        )

        for step, (x, y) in enumerate(dataloader_epoch):
            step_start = time.time()

            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_time = time.time() - step_start
            time_accum += step_time
            running_loss += loss.item()

            # Update tqdm bar with current loss and step time
            dataloader_epoch.set_postfix(
                loss=float(loss.item()),
                step_time=f"{step_time:.3f} s",
            )

            if (step + 1) % print_every == 0:
                avg_loss = running_loss / print_every
                avg_time = time_accum / print_every
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{step+1}/{len(dataloader)}] "
                    f"Loss: {avg_loss:.6f} "
                    f"(avg step time: {avg_time:.3f} s)"
                )
                running_loss = 0.0
                time_accum = 0.0

        # --- Save checkpoint at end of each epoch ---
        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
