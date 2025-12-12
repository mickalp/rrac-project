import os
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from algotom.io import loadersaver as losa
from wavelet_utils import sino_to_wavelet_4ch


class ExperimentalStripeDataset(Dataset):
    """
    Dataset for real experimental sinograms with and without stripe artifacts.

    Folder structure (root_dir):
        root_dir/
            sinograms_clean/
                sample001.tif, ...
            sinograms_striped/
                sample001.tif, ...

    For each common filename:
        - load clean sinogram S_clean
        - load striped sinogram S_striped
        - optionally center-crop to a fixed patch_size x patch_size
        - compute 2D wavelet (4 channels) for each
        - define:
            input  = W_striped
            target = W_striped - W_clean  (artifact in wavelet space)

    Returned:
        x: (4, H, W) float32
        y: (4, H, W) float32
    """

    def __init__(
        self,
        root_dir: str,
        wavelet: str = "haar",
        file_ext: str = ".tif",
        normalize: bool = False,
        patch_size: Optional[int] = 1024,  # central crop size; None = use full image
    ):
        super().__init__()

        self.root_dir = root_dir
        self.clean_dir = os.path.join(root_dir, "sinograms_clean")
        self.striped_dir = os.path.join(root_dir, "sinograms_striped")
        self.wavelet = wavelet
        self.file_ext = file_ext
        self.normalize = normalize
        self.patch_size = patch_size

        clean_files = sorted(
            f for f in os.listdir(self.clean_dir) if f.endswith(self.file_ext)
        )
        striped_files = sorted(
            f for f in os.listdir(self.striped_dir) if f.endswith(self.file_ext)
        )

        clean_set = set(clean_files)
        striped_set = set(striped_files)
        common = sorted(clean_set.intersection(striped_set))

        if not common:
            raise RuntimeError(
                f"No matching files with extension {self.file_ext} found in:\n"
                f"  {self.clean_dir}\n  {self.striped_dir}"
            )

        self.filenames: List[str] = common
        print(f"ExperimentalStripeDataset found {len(self.filenames)} file pairs.")

    def __len__(self) -> int:
        return len(self.filenames)

    def _center_crop(self, sino_clean: np.ndarray, sino_striped: np.ndarray):
        """
        Center-crop both clean and striped sinograms to patch_size x patch_size.
        Assumes both have the same original shape.
        """
        if self.patch_size is None:
            return sino_clean, sino_striped

        H, W = sino_clean.shape
        ps = self.patch_size

        if H < ps or W < ps:
            raise ValueError(
                f"patch_size={ps} is larger than sinogram shape {H}x{W}. "
                f"Use a smaller patch_size."
            )

        y0 = (H - ps) // 2
        x0 = (W - ps) // 2
        y1 = y0 + ps
        x1 = x0 + ps

        return (
            sino_clean[y0:y1, x0:x1],
            sino_striped[y0:y1, x0:x1],
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]
        path_clean = os.path.join(self.clean_dir, fname)
        path_striped = os.path.join(self.striped_dir, fname)

        sino_clean = losa.load_image(path_clean).astype(np.float32)
        sino_striped = losa.load_image(path_striped).astype(np.float32)

        if sino_clean.shape != sino_striped.shape:
            raise ValueError(
                f"Shape mismatch between clean and striped for {fname}: "
                f"{sino_clean.shape} vs {sino_striped.shape}"
            )

        # Center-crop to patch_size x patch_size if requested
        sino_clean, sino_striped = self._center_crop(sino_clean, sino_striped)

        if self.normalize:
            eps = 1e-6
            mean_c = np.mean(sino_clean)
            std_c = np.std(sino_clean) + eps
            sino_clean = (sino_clean - mean_c) / std_c
            sino_striped = (sino_striped - mean_c) / std_c

        w_clean = sino_to_wavelet_4ch(sino_clean, wavelet=self.wavelet)
        w_striped = sino_to_wavelet_4ch(sino_striped, wavelet=self.wavelet)

        w_artifact = w_striped - w_clean

        x = torch.from_numpy(w_striped)
        y = torch.from_numpy(w_artifact)

        return x, y


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "experimental")
    dataset = ExperimentalStripeDataset(root_dir=base_dir, wavelet="haar", normalize=False)
    print("Dataset length:", len(dataset))
    x0, y0 = dataset[0]
    print("x0 shape:", x0.shape)
    print("y0 shape:", y0.shape)
