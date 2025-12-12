import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from algotom.io import loadersaver as losa
from rrac_model import RRACNet
from wavelet_utils import sino_to_wavelet_4ch, wavelet_4ch_to_sino


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_root = os.path.join(base_dir, "data", "synthetic")
    clean_dir = os.path.join(data_root, "sinograms_clean")
    striped_dir = os.path.join(data_root, "sinograms_striped")

    # Pick one sample (first file)
    fname = sorted(os.listdir(striped_dir))[112]
    path_clean = os.path.join(clean_dir, fname)
    path_striped = os.path.join(striped_dir, fname)

    print("Using file:", fname)

    # Load sinograms
    sino_clean = losa.load_image(path_clean).astype(np.float32)
    sino_striped = losa.load_image(path_striped).astype(np.float32)

    # Wavelet transform
    w_clean = sino_to_wavelet_4ch(sino_clean, wavelet="haar")
    w_striped = sino_to_wavelet_4ch(sino_striped, wavelet="haar")

    # Prepare input tensor for model: (1, 4, H, W)
    x = torch.from_numpy(w_striped).unsqueeze(0)  # add batch dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = RRACNet().to(device)
    model.eval()

    # For now, we use the model as it is in memory (no checkpoint),
    # so results correspond to the model from your last training run.
    x = x.to(device)

    with torch.no_grad():
        w_art_pred = model(x)[0].cpu().numpy()  # remove batch dim

    # Clean wavelet estimate
    w_clean_est = w_striped - w_art_pred

    # Inverse wavelet
    sino_clean_est = wavelet_4ch_to_sino(w_clean_est, wavelet="haar")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(sino_striped, cmap="gray")
    axes[0].set_title("Striped sinogram")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sino_clean_est, cmap="gray")
    axes[1].set_title("Corrected (network)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(sino_clean, cmap="gray")
    axes[2].set_title("Ground truth clean")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
