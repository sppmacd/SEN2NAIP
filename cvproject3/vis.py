import matplotlib.pyplot as plt
import torch
import typesum as ts
from torch.utils.data import DataLoader

import torchviz

from .models import Model
from .train import Dataset, cvimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model().to(device)
model.load_state_dict(torch.load("models/best.pt"))

dataset = Dataset("data/train")

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

    plt.subplot(1, 4, 1)
    plt.imshow(hr.permute(1, 2, 0))

    plt.subplot(1, 4, 2)
    plt.imshow(lr.permute(1, 2, 0))

    plt.subplot(1, 4, 3)
    plt.imshow(lr_up)

    plt.subplot(1, 4, 4)
    plt.imshow(lr_up_bic)

    plt.show()
