from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .data import image_names, load_image

# Directory structure:
# data
#   train|val|test
#     XX.hr.npy
#     XX.lr.npy


def _write_dir(images, dir_: str):
    (Path("data") / dir_).mkdir(parents=True, exist_ok=True)

    for img_roi in tqdm(images, desc=f"Writing {dir_}/"):
        img = np.clip(load_image(img_roi) * 255, 0, 255).astype(np.uint8)
        img_downscaled = cv2.resize(
            img,
            (img.shape[1] // 2, img.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )

        np.savez(Path("data") / dir_ / f"{img_roi}.hr.npz", img)
        np.savez(Path("data") / dir_ / f"{img_roi}.lr.npz", img_downscaled)


def generate_dataset():
    imgs = image_names()

    imgs_train, imgs_test = train_test_split(imgs, test_size=0.2)
    imgs_train, imgs_val = train_test_split(imgs_train, test_size=0.1)

    _write_dir(imgs_train, "train")
    _write_dir(imgs_val, "val")
    _write_dir(imgs_test, "test")


if __name__ == "__main__":
    generate_dataset()
