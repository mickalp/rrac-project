import os
import argparse

import numpy as np
import torch  # just to check for GPU if you ever want to switch to gpu=True
from algotom.io import loadersaver as losa
from algotom.rec import reconstruction as rec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct images from experimental sinograms using Algotom FBP."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Folder with sinograms to reconstruct. "
            "Default: data/experimental/sinograms_corrected in project root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Folder to save reconstructed images. "
            "Default: data/experimental/reconstructed_pictures in project root."
        ),
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".tif",
        help="File extension to process (default: .tif).",
    )
    parser.add_argument(
        "--center",
        type=float,
        default=None,
        help=(
            "Center of rotation in pixels (default: detector_width/2). "
            "If not set, the script will use (num_detectors - 1) / 2."
        ),
    )
    parser.add_argument(
        "--angles-start",
        type=float,
        default=0.0,
        help="Start angle in degrees (default: 0).",
    )
    parser.add_argument(
        "--angles-end",
        type=float,
        default=180.0,
        help="End angle in degrees (default: 180 for parallel-beam CT).",
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help=(
            "Set this if your sinograms are stored as (detectors, projections) instead of "
            "(projections, detectors). The script will transpose them before reconstruction."
        ),
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help=(
            "Do NOT apply log to the sinogram. Use this if data are already log-transformed "
            "or you don't want log(I0/I) preprocessing."
        ),
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="hann",
        help="FBP filter name (default: hann).",
    )
    return parser.parse_args()


def reconstruct_single_sino(
    sino: np.ndarray,
    center: float,
    angles_start: float,
    angles_end: float,
    apply_log: bool,
    filter_name: str = "hann",
) -> np.ndarray:
    """
    Reconstruct a single 2D slice from a 2D sinogram using Algotom's FBP.

    sino: (num_projections, num_detectors)
    """
    num_proj, num_det = sino.shape

    # Angles in radians
    angles_deg = np.linspace(angles_start, angles_end, num_proj, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)

    # FBP reconstruction
    recon = rec.fbp_reconstruction(
        sino,
        center=center,
        angles=angles_rad,
        apply_log=apply_log,
        filter_name=filter_name,
        gpu=False,  # set True if you want to use GPU and have it configured
    )

    return recon.astype(np.float32)


def main():
    args = parse_args()

    # --- Resolve base paths ---
    base_dir = os.path.dirname(os.path.dirname(__file__))

    if args.input_dir is None:
        input_dir = os.path.join(base_dir, "data", "experimental", "sinograms_corrected")
    else:
        input_dir = args.input_dir

    if args.output_dir is None:
        output_dir = os.path.join(base_dir, "data", "experimental", "reconstructed_pictures")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("Input directory :", input_dir)
    print("Output directory:", output_dir)

    # --- List sinograms ---
    all_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(args.ext.lower())
    )

    if not all_files:
        print(f"No files with extension {args.ext} found in {input_dir}")
        return

    print(f"Found {len(all_files)} sinograms to reconstruct.")

    # --- Process each sinogram ---
    for fname in all_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        print(f"Reconstructing {fname} ...", end="", flush=True)

        # Load sinogram
        sino = losa.load_image(in_path).astype(np.float32)

        # Ensure 2D
        if sino.ndim != 2:
            raise ValueError(f"Expected 2D sinogram for {fname}, got shape {sino.shape}")

        # Optional transpose
        if args.transpose:
            sino = sino.T  # (detectors, projections) -> (projections, detectors)

        num_proj, num_det = sino.shape

        # Center of rotation
        if args.center is None:
            center = (num_det - 1) / 2.0
        else:
            center = float(args.center)

        # Apply FBP
        recon = reconstruct_single_sino(
            sino,
            center=center,
            angles_start=args.angles_start,
            angles_end=args.angles_end,
            apply_log=not args.no_log,
            filter_name=args.filter,
        )

        # Save reconstructed image
        losa.save_image(out_path, recon)
        print(" done.")

    print("All reconstructions finished.")


if __name__ == "__main__":
    main()
