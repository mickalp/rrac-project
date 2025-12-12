import numpy as np
import pywt


def sino_to_wavelet_4ch(sino: np.ndarray, wavelet: str = "haar") -> np.ndarray:
    """
    Compute a single-level 2D DWT of a sinogram and pack the 4 subbands
    (LL, LH, HL, HH) into a 4-channel array.

    Parameters
    ----------
    sino : np.ndarray, shape (H, W)
        2D sinogram (float or int). Will be converted to float32.
    wavelet : str
        Wavelet name, default is "haar" (as in the RRAC paper).

    Returns
    -------
    w4 : np.ndarray, shape (4, H//2, W//2)
        Channels order: [LL, LH, HL, HH].
        This matches the idea in the paper where 4 subbands are used as input.
    """
    if sino.ndim != 2:
        raise ValueError("sino_to_wavelet_4ch expects a 2D array (H, W).")

    # Ensure float32
    sino = np.asarray(sino, dtype=np.float32)

    # 2D DWT
    cA, (cH, cV, cD) = pywt.dwt2(sino, wavelet)

    # Stack into 4 channels
    w4 = np.stack([cA, cH, cV, cD], axis=0).astype(np.float32)

    return w4


def wavelet_4ch_to_sino(w4: np.ndarray, wavelet: str = "haar") -> np.ndarray:
    """
    Inverse of sino_to_wavelet_4ch:
    Take 4-channel wavelet coefficients and reconstruct the 2D sinogram.

    Parameters
    ----------
    w4 : np.ndarray, shape (4, H, W)
        Channels: [LL, LH, HL, HH].
    wavelet : str
        Wavelet name, default "haar".

    Returns
    -------
    sino_rec : np.ndarray, shape (H*2, W*2)
        Reconstructed sinogram (float32).
    """
    w4 = np.asarray(w4, dtype=np.float32)

    if w4.ndim != 3 or w4.shape[0] != 4:
        raise ValueError(
            "wavelet_4ch_to_sino expects shape (4, H, W) with channels [LL, LH, HL, HH]."
        )

    cA = w4[0]
    cH = w4[1]
    cV = w4[2]
    cD = w4[3]

    # Inverse 2D DWT
    sino_rec = pywt.idwt2((cA, (cH, cV, cD)), wavelet).astype(np.float32)

    return sino_rec


# --- Optional: small self-test when run as a script --- #
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create a dummy sinogram-like image (gradient + noise)
    H, W = 512, 512
    y = np.linspace(0, 1, H, dtype=np.float32)
    x = np.linspace(0, 1, W, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    sino = yy + 0.1 * np.random.randn(H, W).astype(np.float32)

    # Forward wavelet
    w4 = sino_to_wavelet_4ch(sino, wavelet="haar")
    print("Wavelet 4-ch shape:", w4.shape)  # (4, H//2, W//2)

    # Inverse wavelet
    sino_rec = wavelet_4ch_to_sino(w4, wavelet="haar")
    print("Reconstructed sino shape:", sino_rec.shape)

    # Check reconstruction quality (should be very close)
    diff = np.abs(sino - sino_rec)
    print("Max abs diff:", diff.max())
    print("Mean abs diff:", diff.mean())

    # Plot original and reconstructed for sanity check
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(sino, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(sino_rec, cmap="gray")
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
