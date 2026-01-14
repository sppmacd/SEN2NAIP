import os
from pathlib import Path

import numpy as np
import torch
import yaml
from box import ConfigBox
from ignite.contrib.handlers import (
    ProgressBar,
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

import dvclive

from .model_impls.content_loss import resnet18_content_loss
from .models import Model


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.hr_images = list(Path(path).glob("*.hr.npz"))

    @staticmethod
    def _transform():
        return Compose([ToTensor()])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx: int):
        hr_image_path = self.hr_images[idx]
        lr_image_path = str(hr_image_path).replace("hr.npz", "lr.npz")

        hr_image_file = np.load(hr_image_path)
        hr_image = hr_image_file["arr_0"]
        hr_image_file.close()
        lr_image_file = np.load(lr_image_path)
        lr_image = lr_image_file["arr_0"]
        lr_image_file.close()

        t = self._transform()
        return t(lr_image), t(hr_image)


import torch
import torch.nn as nn


# 1) Weight Signal-to-Noise Ratio (SNR)
# Measures mean(|w|) / (std(w) + eps) aggregated across learnable params.
# Returns a single scalar (log-SNR) so values are more stable and comparable.
def weight_snr(model: nn.Module, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute a log signal-to-noise ratio across model parameters.
    Higher values indicate larger mean absolute weights relative to std (more 'coherent' signal).
    """
    means = []
    stds = []
    for p in model.parameters():
        if p.numel() == 0:
            continue
        w = p.detach()
        # ignore parameters that are effectively scalar bias with no variance? still include
        means.append(w.abs().mean())
        stds.append(w.std(unbiased=False))
    if not means:
        return torch.tensor(float("nan"))
    means = torch.stack(means)
    stds = torch.stack(stds)
    snr = means / (stds + eps)  # per-parameter tensor
    # aggregate by taking robust mean: median of per-param SNRs, then log1p for stability
    median_snr = snr.median()
    return torch.log1p(median_snr)


# 2) Gradient Norm Sparsity
# Fraction of gradient elements whose absolute value is below a small threshold.
# Useful to detect collapse / dead parameters or extremely small updates.
def grad_norm_sparsity(model: nn.Module, threshold: float = 1e-6) -> torch.Tensor:
    """
    Compute fraction of parameter gradient elements with abs < threshold.
    Requires gradients to be present (after backward). If no grad for any param, returns nan.
    """
    total = 0
    near_zero = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        n = g.numel()
        if n == 0:
            continue
        total += n
        near_zero += int((g.abs() < threshold).sum().item())
    if total == 0:
        return torch.tensor(float("nan"))
    return torch.tensor(near_zero / total, dtype=torch.float32)


# 3) Empirical Fisher Trace Estimate (single-batch)
# Approximates trace of empirical Fisher Information matrix by summing squared gradients
# normalized per-parameter dimension. This captures effective curvature magnitude.
def fisher_trace_estimate(model: nn.Module, normalize: bool = True) -> torch.Tensor:
    """
    Compute an empirical Fisher trace proxy using squared gradients:
    trace_est = sum( (g^2).mean() ) over parameters  (optionally normalized by number of params)
    Requires gradients to be present (after backward). Returns scalar tensor.
    """
    sq_means = []
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.numel() == 0:
            continue
        # use mean of squared grads for stability across sizes
        sq_means.append((g * g).mean())
    if not sq_means:
        return torch.tensor(float("nan"))
    total = torch.stack(sq_means).sum()
    if normalize:
        # normalize by number of parameter tensors to avoid scale differences between architectures
        total = total / float(len(sq_means))
    # return log1p to keep metric on reasonable scale
    return torch.log1p(total)


def grad_L2(model: nn.Module, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute global L2 norm of gradients across all parameters:
    ||g||_2 = sqrt(sum_i sum_j g_ij^2)
    Returns NaN if no parameter has a gradient.
    """
    sq_sum = None
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.numel() == 0:
            continue
        # accumulate in float32 for stability
        val = float((g.float() ** 2).sum().cpu().item())
        sq_sum = val if sq_sum is None else sq_sum + val
    if sq_sum is None:
        return torch.tensor(float("nan"))
    return torch.tensor((sq_sum + eps) ** 0.5)


def cvimage(img):
    return np.transpose(img[:3] * 255, (1, 2, 0)).astype(np.uint8)


def run(config: ConfigBox, live: dvclive.Live):
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)

    batch_size = config.train.batch_size

    # Show some random prediction
    train_dataset = Dataset("data/train")
    lr, hr = train_dataset[len(train_dataset) // 2]
    live.log_image("low_res.png", cvimage(lr.numpy()))
    live.log_image("high_res.png", cvimage(hr.numpy()))
    live.log_image(
        "prediction_random.png",
        cvimage(model(lr.to(device).unsqueeze(0)).squeeze(0).cpu().detach().numpy()),
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(Dataset("data/val"), shuffle=True, batch_size=batch_size)

    match config.train.optimizer:
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        case "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    match config.train.loss:
        case "mse":
            criterion = nn.MSELoss()
        case "resnet18":
            criterion = resnet18_content_loss

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        device,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    val_metrics = {
        "loss": Loss(criterion),
    }

    val_evaluator = create_supervised_evaluator(
        model,
        metrics=val_metrics,
        device=device,
    )

    log_interval = 10

    @trainer.on(Events.ITERATION_COMPLETED(every=5))
    def tb_log_dynamics(engine):
        tb_logger.add_scalar(
            "weight_snr",
            weight_snr(model),
            engine.state.iteration,
        )
        tb_logger.add_scalar(
            "grad_norm_sparsity",
            grad_norm_sparsity(model),
            engine.state.iteration,
        )
        tb_logger.add_scalar(
            "fisher_trace_estimate",
            fisher_trace_estimate(model),
            engine.state.iteration,
        )
        tb_logger.add_scalar(
            "grad_L2",
            grad_L2(model),
            engine.state.iteration,
        )

    @trainer.on(Events.ITERATION_COMPLETED)
    def dvclive_log_step(engine):
        live.next_step()

    @trainer.on(
        Events.ITERATION_COMPLETED(every=config.train.gradient_accumulation_steps),
    )
    def dvclive_log_loss(engine):
        live.log_metric("train_loss", engine.state.output)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.5f}"
        )
        live.log_metric("val_loss", metrics["loss"])

    @trainer.on(Events.COMPLETED)
    def log_training_time(trainer):
        print(f"Total time: {trainer.state.times['COMPLETED']}")
        live.log_metric("train_time", trainer.state.times["COMPLETED"])

    # Score function to return current value of any metric we defined above in val_metrics
    def score_function(engine):
        return engine.state.metrics["loss"]

    # Checkpoint to store n_saved best models wrt score function
    model_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=2,
        filename_prefix="best",
        score_function=score_function,
        score_name="loss",
        global_step_transform=global_step_from_engine(
            trainer,
        ),  # helps fetch the trainer's state
        require_empty=False,
    )

    # Save the model after every epoch of val_evaluator is completed
    val_evaluator.add_event_handler(
        Events.COMPLETED,
        model_checkpoint,
        {"model": model},
    )

    # Define a Tensorboard logger
    tb_logger = TensorboardLogger(log_dir="tb-logger")

    # Attach handler to plot trainer's loss every 100 iterations
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    # Attach handler for plotting both evaluators' metrics after every epoch completes
    for tag, evaluator in [
        ("validation", val_evaluator),
    ]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    ProgressBar().attach(trainer, output_transform=lambda x: {"batch loss": x})
    ProgressBar().attach(val_evaluator)

    trainer.run(train_loader, max_epochs=100)
    torch.save(model.state_dict(), "models/best.pt")

    # after training, rename the best checkpoint to models/best.pt
    # best_path = model_checkpoint.last_checkpoint  # full path to the chosen best file
    # os.makedirs("models", exist_ok=True)
    # if best_path:
    #     target = os.path.join("models", "best.pt")
    #     # overwrite if exists
    #     if os.path.abspath(best_path) != os.path.abspath(target):
    #         os.replace(best_path, target)


if __name__ == "__main__":
    with open("params.yaml", "r") as f, dvclive.Live(dir="dvclive/train") as live:
        run(ConfigBox(yaml.safe_load(f), box_dots=True), live)
