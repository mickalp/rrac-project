import os
import numpy as np
from skimage.transform import radon
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from algotom.io import loadersaver as losa


def make_output_dirs(base_dir):
    d_clean = os.path.join(base_dir, "sinograms_clean")
    d_striped = os.path.join(base_dir, "sinograms_striped")
    os.makedirs(d_clean, exist_ok=True)
    os.makedirs(d_striped, exist_ok=True)
    return d_clean, d_striped


def generate_phantom_image(size=512):
    """
    Generate a simple Shepp-Logan-type phantom image of given size.
    """
    # skimage's phantom is 400x400 float, resize to (size, size)
    ph = shepp_logan_phantom()
    ph_resized = resize(ph, (size, size), mode="reflect", anti_aliasing=True).astype(
        np.float32
    )
    return ph_resized


def make_sinogram(img, num_angles=360):
    """
    Compute a simple parallel-beam sinogram using skimage.radon.
    Shape: (num_detectors, num_projections) or (H, num_angles), depending on implementation.
    We'll keep it as (num_angles, num_detectors).
    """
    # skimage.radon expects angles in degrees
    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
    sino = radon(img, theta=theta, circle=False).astype(np.float32)
    # skimage.radon returns shape (num_detectors, num_angles), i.e. (H, num_angles)
    # We want shape (num_angles, num_detectors)
    sino = sino.T  # now (num_angles, num_detectors)
    return sino, theta


def add_stripes(sino, num_stripes=10, max_intensity=0.2, rng=None):
    """
    Add synthetic vertical stripe artifacts to a sinogram.
    We add stripes along the detector axis, i.e. constants in the projection dimension.

    sino: (num_projections, num_detectors)
    """
    if rng is None:
        rng = np.random.default_rng()

    sino = sino.copy()
    num_proj, num_det = sino.shape

    # Randomly choose stripe positions along detector axis (columns)
    cols = rng.choice(num_det, size=num_stripes, replace=False)

    for c in cols:
        # Random stripe amplitude
        amp = (rng.random() * 2 - 1) * max_intensity  # in range [-max_intensity, max_intensity]
        # Add constant offset along projection dimension
        sino[:, c] += amp

    return sino


def main():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")
    out_clean_dir, out_striped_dir = make_output_dirs(base_dir)

    rng = np.random.default_rng(1234)

    num_samples = 20  # start small; you can increase later
    size = 512
    num_angles = 360

    print(f"Saving synthetic data to:\n  Clean:   {out_clean_dir}\n  Striped: {out_striped_dir}")
    print(f"Generating {num_samples} samples...")

    for i in range(num_samples):
        # 1) Create phantom image
        img = generate_phantom_image(size=size)

        # 2) Make clean sinogram
        sino_clean, theta = make_sinogram(img, num_angles=num_angles)

        # 3) Add synthetic stripes
        sino_striped = add_stripes(sino_clean, num_stripes=10, max_intensity=0.2, rng=rng)

        # 4) Save as TIFF (using Algotom loadersaver)
        fname = f"synth_{i:04d}.tif"
        path_clean = os.path.join(out_clean_dir, fname)
        path_striped = os.path.join(out_striped_dir, fname)

        losa.save_image(path_clean, sino_clean)
        losa.save_image(path_striped, sino_striped)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{num_samples} done")

    print("Done generating synthetic sinograms.")


if __name__ == "__main__":
    main()
