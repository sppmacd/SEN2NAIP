import torch
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
from .train import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model().to(device)
model.load_state_dict(torch.load("models/best.pt"))


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

test_loader = DataLoader(Dataset("data/test"), batch_size=4)

test_evaluator.run(test_loader)

metrics = test_evaluator.state.metrics

with dvclive.Live(dir="dvclive/eval") as live:
    print(metrics)
    live.log_metric("test_loss", metrics["loss"])
    live.log_metric("test_ssim", metrics["ssim"])
    live.log_metric("test_psnr", metrics["psnr"])
