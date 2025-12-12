import os
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from algotom.io import loadersaver as losa
from rrac_model import RRACNet
from wavelet_utils import sino_to_wavelet_4ch, wavelet_4ch_to_sino


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean experimental striped sinograms using a trained RRACNet model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="rrac_checkpoint.pth",
        help="Path to trained model checkpoint (default: rrac_checkpoint.pth in project root).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Folder with striped experimental sinograms. "
            "Default: data/experimental/sinograms_striped in project root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Folder to save corrected sinograms. "
            "Default: data/experimental/sinograms_corrected in project root."
        ),
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="haar",
        help="Wavelet name to use (must match training, default: haar).",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".tif",
        help="File extension to process (default: .tif).",
    )
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> RRACNet:
    print(f"Loading model checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = RRACNet(
        in_channels=4,
        base_channels=64,
        num_blocks=6,
        out_channels=4,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def normalize_sino(sino: np.ndarray):
    """Simple per-sinogram normalization: (x - mean) / std."""
    eps = 1e-6
    mean = float(np.mean(sino))
    std = float(np.std(sino)) + eps
    sino_norm = (sino - mean) / std
    return sino_norm, mean, std


def denormalize_sino(sino_norm: np.ndarray, mean: float, std: float):
    return sino_norm * std + mean


def clean_single_sino(
    sino_striped: np.ndarray,
    model: RRACNet,
    device: torch.device,
    wavelet: str = "haar",
) -> np.ndarray:
    """
    Apply the trained model to a single striped sinogram (full-size).

    Steps:
        - normalize sinogram
        - 2D DWT -> 4-channel wavelet (striped)
        - model predicts artifact in wavelet space
        - subtract artifact
        - IDWT -> clean normalized sinogram
        - denormalize back
    """
    # Ensure float32
    sino_striped = sino_striped.astype(np.float32)

    # Normalize (same style as training on experimental data)
    sino_norm, mean, std = normalize_sino(sino_striped)

    # Forward wavelet transform: (4, H/2, W/2)
    w_striped = sino_to_wavelet_4ch(sino_norm, wavelet=wavelet)

    # Prepare tensor: (1, 4, H, W)
    x = torch.from_numpy(w_striped).unsqueeze(0).to(device)

    with torch.no_grad():
        w_art_pred = model(x)[0].cpu().numpy()  # remove batch dim -> (4, H, W)

    # Clean wavelet (normalized)
    w_clean = w_striped - w_art_pred

    # Inverse wavelet -> clean sinogram (normalized)
    sino_clean_norm = wavelet_4ch_to_sino(w_clean, wavelet=wavelet)

    # Denormalize back to original scale
    sino_clean = denormalize_sino(sino_clean_norm, mean, std)

    return sino_clean.astype(np.float32)


def main():
    args = parse_args()

    # --- Resolve paths ---
    base_dir = os.path.dirname(os.path.dirname(__file__))

    if args.input_dir is None:
        input_dir = os.path.join(base_dir, "data", "experimental", "sinograms_striped")
    else:
        input_dir = args.input_dir

    if args.output_dir is None:
        output_dir = os.path.join(base_dir, "data", "experimental", "sinograms_corrected")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(base_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))

    # --- Load model ---
    model = load_model(ckpt_path, device)

    # --- List files to process ---
    all_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(args.ext.lower())
    )

    if not all_files:
        print(f"No files with extension {args.ext} found in {input_dir}")
        return

    print(f"Found {len(all_files)} striped sinograms to clean.")
    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Wavelet      : {args.wavelet}")

    # --- Process each sinogram ---
    for fname in tqdm(all_files, desc="Cleaning sinograms", leave=True):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        # Load striped sinogram
        sino_striped = losa.load_image(in_path).astype(np.float32)

        t0 = time.time()
        sino_clean = clean_single_sino(
            sino_striped,
            model=model,
            device=device,
            wavelet=args.wavelet,
        )
        dt = time.time() - t0

        # Save corrected sinogram
        losa.save_image(out_path, sino_clean)

        tqdm.write(f"{fname}: cleaned in {dt:.2f} s -> {out_path}")

    print("All sinograms cleaned and saved.")


if __name__ == "__main__":
    main()
