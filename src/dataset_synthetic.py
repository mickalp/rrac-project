import os
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset

from algotom.io import loadersaver as losa
from wavelet_utils import sino_to_wavelet_4ch


class SyntheticStripeDataset(Dataset):
    """
    PyTorch Dataset for synthetic sinograms with and without stripe artifacts.

    Folder structure (root_dir):
        root_dir/
            sinograms_clean/
                synth_0000.tif, ...
            sinograms_striped/
                synth_0000.tif, ...

    For each file name present in both, we:
        - load clean sinogram S_clean
        - load striped sinogram S_striped
        - compute 2D wavelet (4 channels) for each
        - define:
            input  = W_striped   (4, H, W)
            target = W_striped - W_clean  (artifact in wavelet space)

    Returned tensors:
        x: (4, H, W), float32
        y: (4, H, W), float32
    """

    def __init__(
        self,
        root_dir: str,
        wavelet: str = "haar",
        file_ext: str = ".tif",
        normalize: bool = False,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.clean_dir = os.path.join(root_dir, "sinograms_clean")
        self.striped_dir = os.path.join(root_dir, "sinograms_striped")
        self.wavelet = wavelet
        self.file_ext = file_ext
        self.normalize = normalize

        # Collect matching filenames in both clean and striped folders
        clean_files = sorted(
            f for f in os.listdir(self.clean_dir) if f.endswith(self.file_ext)
        )
        striped_files = sorted(
            f for f in os.listdir(self.striped_dir) if f.endswith(self.file_ext)
        )

        # Use intersection of basenames
        clean_set = set(clean_files)
        striped_set = set(striped_files)
        common = sorted(clean_set.intersection(striped_set))

        if not common:
            raise RuntimeError(
                f"No matching files with extension {self.file_ext} found in:\n"
                f"  {self.clean_dir}\n  {self.striped_dir}"
            )

        self.filenames: List[str] = common
        print(f"SyntheticStripeDataset found {len(self.filenames)} file pairs.")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]

        path_clean = os.path.join(self.clean_dir, fname)
        path_striped = os.path.join(self.striped_dir, fname)

        # Load sinograms as numpy arrays
        sino_clean = losa.load_image(path_clean).astype(np.float32)
        sino_striped = losa.load_image(path_striped).astype(np.float32)

        if sino_clean.shape != sino_striped.shape:
            raise ValueError(
                f"Shape mismatch between clean and striped for {fname}: "
                f"{sino_clean.shape} vs {sino_striped.shape}"
            )

        # Optional simple per-sinogram normalization (you can tweak later)
        if self.normalize:
            # Avoid division by zero
            eps = 1e-6
            mean_c = np.mean(sino_clean)
            std_c = np.std(sino_clean) + eps
            sino_clean = (sino_clean - mean_c) / std_c
            sino_striped = (sino_striped - mean_c) / std_c

        # Wavelet transform -> (4, H/2, W/2)
        w_clean = sino_to_wavelet_4ch(sino_clean, wavelet=self.wavelet)
        w_striped = sino_to_wavelet_4ch(sino_striped, wavelet=self.wavelet)

        # Artifact in wavelet space
        w_artifact = w_striped - w_clean

        # Convert to torch tensors, shape (4, H, W)
        x = torch.from_numpy(w_striped)   # input
        y = torch.from_numpy(w_artifact)  # target

        return x, y


if __name__ == "__main__":
    # Small self-test
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")
    dataset = SyntheticStripeDataset(root_dir=base_dir, wavelet="haar", normalize=False)

    print("Dataset length:", len(dataset))
    x0, y0 = dataset[0]
    print("x0 shape:", x0.shape)
    print("y0 shape:", y0.shape)
    print("x0 dtype:", x0.dtype, "y0 dtype:", y0.dtype)
