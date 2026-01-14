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

dataset = Dataset("data/test")

for dd in dataset:
    lr, hr = dd

    lr_up = model(lr.to(device).unsqueeze(0)).squeeze(0)
    torchviz.make_dot(lr_up).render("model.png")

    lr_up = lr_up.cpu().permute(1, 2, 0).detach().numpy()

    plt.subplot(1, 3, 1)
    plt.imshow(hr.permute(1, 2, 0))

    plt.subplot(1, 3, 2)
    plt.imshow(lr.permute(1, 2, 0))

    plt.subplot(1, 3, 3)
    plt.imshow(lr_up)

    plt.show()
