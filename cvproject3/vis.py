import argparse
import matplotlib.pyplot as plt
import torch
import typesum as ts
from torch.utils.data import DataLoader

import torchviz

from .models import Model
from .train import Dataset, cvimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    *,
    show_lr: bool = True,
    show_model: bool = True,
    show_bicubic: bool = True,
    data_dir: str = "data/train",
):
    model = Model().to(device)
    model.load_state_dict(torch.load("models/best.pt"))

    dataset = Dataset(data_dir, augment=False)

    for dd in dataset:
        lr, hr = dd

        lr_up = model(lr.to(device).unsqueeze(0)).squeeze(0)

        lr_up_bic = (
            torch.nn.Upsample(mode="bicubic", scale_factor=2)(
                lr.unsqueeze(0),
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .detach()
            .numpy()
        )
        lr_up = lr_up.cpu().permute(1, 2, 0).detach().numpy()

        # convert tensors to numpy for consistent plotting
        hr_np = hr.permute(1, 2, 0).detach().numpy()
        lr_np = lr.permute(1, 2, 0).detach().numpy()

        # upsample LR by 2x so it matches HR / model sizes (keeps shared axes aligned)
        lr_np_up = (
            torch.nn.Upsample(mode="nearest", scale_factor=2)(lr.unsqueeze(0))
            .squeeze(0)
            .permute(1, 2, 0)
            .detach()
            .numpy()
        )

        # prepare list of images to show
        imgs = [("HR", hr_np)]
        if show_lr:
            imgs.append(("LR", lr_np_up))
        if show_model:
            imgs.append(("Model", lr_up))
        if show_bicubic:
            imgs.append(("Bicubic", lr_up_bic))

        n = len(imgs)

        # create subplots with shared axes so zoom/pan is synchronized
        fig, axs = plt.subplots(1, n, sharex=True, sharey=True, figsize=(4 * n, 4))
        if n == 1:
            axs = [axs]

        for ax, (title, arr) in zip(axs, imgs):
            ax.imshow(arr)
            # ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize HR/LR and upsampled images")
    parser.add_argument("--data-dir", default="data/train", help="Dataset directory")
    parser.add_argument(
        "--no-lr", action="store_true", help="Hide low-resolution image"
    )
    parser.add_argument(
        "--no-model", action="store_true", help="Hide model and bicubic output"
    )
    args = parser.parse_args()

    main(
        show_lr=not args.no_lr,
        show_model=not args.no_model,
        show_bicubic=not args.no_model,
        data_dir=args.data_dir,
    )
