import numpy as np
import torch
import typesum as ts
from ignite.contrib.handlers import (
    ProgressBar,
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import PSNR, SSIM, Loss
from torch.utils.data import DataLoader

import dvclive

from .models import Model
from .train import Dataset, cvimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model().to(device)
# model.load_state_dict(torch.load("models/best.pt"))


test_metrics = {
    "loss": Loss(torch.nn.MSELoss()),
    "ssim": SSIM(255),
    "psnr": PSNR(255),
}

test_evaluator = create_supervised_evaluator(
    model,
    metrics=test_metrics,
    device=device,
)
ProgressBar().attach(test_evaluator)

dataset = Dataset("data/test")
test_loader = DataLoader(dataset, batch_size=4)

test_evaluator.run(test_loader)

metrics = test_evaluator.state.metrics


with dvclive.Live(dir="dvclive/eval") as live:
    print(metrics)
    live.log_metric("test_loss", metrics["loss"])
    live.log_metric("test_ssim", metrics["ssim"])
    live.log_metric("test_psnr", metrics["psnr"])

    # Show some random prediction
    lr, hr = dataset[len(dataset) // 2]
    print(lr.max(), hr.max())
    with torch.no_grad():
        upscaled = model(lr.unsqueeze(0).to(device)).squeeze(0).cpu()
    ts.print(lr, hr, upscaled)
    live.log_image("low_res.png", cvimage(lr.numpy()))
    live.log_image("high_res.png", cvimage(hr.numpy()))
    live.log_image("upscaled.png", cvimage(upscaled.numpy()))
