import os
import argparse

import numpy as np
from algotom.io import loadersaver as losa
import algotom.prep.calculation as calc
from algotom.rec import reconstruction as rec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct images from synthetic sinograms using Algotom FBP."
    )
    parser.add_argument(
        "--variant",
        choices=["clean", "striped", "corrected"],
        default="striped",
        help=(
            "Which synthetic sinograms to reconstruct: "
            "'clean' (sinograms_clean), "
            "'striped' (sinograms_striped), "
            "'corrected' (sinograms_corrected). "
            "Default: striped."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Override input folder with sinograms to reconstruct. "
            "If not set, it is derived from --variant under data/synthetic/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override output folder to save reconstructed images. "
            "If not set, it is derived from --variant under data/synthetic/."
        ),
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".tif",
        help="File extension to process (default: .tif).",
    )
    # parser.add_argument(
        # "--center",
        # default=calc.find_center_vo,
        # help=(
            # "Center of rotation in pixels (default: detector_width/2). "
            # "If not set, the script will use (num_detectors - 1) / 2."
        # ),
    # )
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
    center,
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
        center=calc.find_center_vo(sino),
        angles=angles_rad,
        apply_log=apply_log,
        filter_name=filter_name,
        gpu=False,  # keep CPU, can switch to True if GPU recon is configured
    )

    return recon.astype(np.float32)


def main():
    args = parse_args()

    # --- Resolve base paths ---
    base_dir = os.path.dirname(os.path.dirname(__file__))
    synth_root = os.path.join(base_dir, "data", "synthetic")

    # Derive default input/output from variant if not explicitly given
    if args.input_dir is None:
        if args.variant == "clean":
            input_dir = os.path.join(synth_root, "sinograms_clean")
        elif args.variant == "striped":
            input_dir = os.path.join(synth_root, "sinograms_striped")
        else:  # "corrected"
            input_dir = os.path.join(synth_root, "sinograms_corrected")
    else:
        input_dir = args.input_dir

    if args.output_dir is None:
        if args.variant == "clean":
            output_dir = os.path.join(synth_root, "reconstructed_clean")
        elif args.variant == "striped":
            output_dir = os.path.join(synth_root, "reconstructed_striped")
        else:  # "corrected"
            output_dir = os.path.join(synth_root, "reconstructed_corrected")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("Variant         :", args.variant)
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
        # if args.center is None:
            # center = (num_det - 1) / 2.0
        # else:
            # center = float(args.center)

        # Apply FBP
        recon = reconstruct_single_sino(
            sino,
            center=calc.find_center_vo(sino),
            angles_start=args.angles_start,
            angles_end=args.angles_end,
            apply_log=not args.no_log,
            filter_name=args.filter,
        )

        # Save reconstructed image
        losa.save_image(out_path, recon)
        print(" done.")

    print("All synthetic reconstructions finished.")


if __name__ == "__main__":
    main()
