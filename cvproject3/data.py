import os
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from skimage.transform import resize

# Define the base path to the local zip dataset
ZIP_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--isp-uv-es--SEN2NAIP/snapshots/79e93461ad93911ebcd5aa0f342376e4c41e8743/cross-sensor/cross-sensor.zip",
)


def load_image(roi: str) -> np.ndarray:
    """Load high-resolution image from specified ROI within a zip file.

    Returns: (RGB (h,w,3), channel4 (h,w))
    """
    with zipfile.ZipFile(ZIP_PATH) as z:
        # Construct the path of the image inside the zip file
        image_path = f"cross-sensor/{roi}/hr.tif"
        with z.open(image_path) as fh:
            hr_data = rioxarray.open_rasterio(fh)

    img_n = hr_data.to_numpy()
    img = img_n.transpose(1, 2, 0) / 255.0
    return img.astype(np.float32)


def show_image(title: str, img: np.ndarray):
    # Ensure img has 3 or 4 channels (RGB or RGBA)
    if img.ndim not in {3, 4} or img.shape[2] not in {3, 4}:
        raise ValueError("Input image must have shape (H, W, 3) or (H, W, 4).")

    # Setup the figure: rows=3, columns=4
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle(title, fontsize=16)

    # Define the channels
    channels = (
        ["Red", "Green", "Blue", "NIR"]
        if img.shape[2] == 4
        else ["Red", "Green", "Blue"]
    )

    # Display original image channels
    for i, channel in enumerate(channels):
        axes[0, i].imshow(img[..., i], cmap="gray")
        axes[0, i].set_title(f"Original {channel}")
        axes[0, i].axis("off")

    # Downscale the image by a factor of 2
    downscaled_img = resize(
        img,
        (img.shape[0] // 2, img.shape[1] // 2),
        anti_aliasing=True,
    )

    # Display downscaled image channels
    for i, channel in enumerate(channels):
        axes[1, i].imshow(downscaled_img[..., i], cmap="gray")
        axes[1, i].set_title(f"Downscaled {channel}")
        axes[1, i].axis("off")

    # Upscale the downscaled image using bicubic interpolation
    bicubic_upscaled_img = resize(downscaled_img, img.shape[:2], order=3)

    # Calculate the difference
    difference = np.clip(bicubic_upscaled_img - img, -1, 1)

    # Display difference channels
    for i, channel in enumerate(channels):
        axes[2, i].imshow(difference[..., i], cmap="coolwarm", vmin=-0.1, vmax=0.1)
        axes[2, i].set_title(f"Difference {channel}")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust top to give space for title
    plt.show()


N_IMAGES = 2851


def image_names():
    return [f"ROI_{img:04d}" for img in range(N_IMAGES)]
