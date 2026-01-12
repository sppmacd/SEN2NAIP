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


def cvimage(img):
    print(img.min(), img.max(), img)
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

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(Dataset("data/val"), shuffle=True, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    trainer = create_supervised_trainer(model, optimizer, criterion, device)

    val_metrics = {
        "loss": Loss(criterion),
    }

    train_evaluator = create_supervised_evaluator(
        model,
        metrics=val_metrics,
        device=device,
    )
    val_evaluator = create_supervised_evaluator(
        model,
        metrics=val_metrics,
        device=device,
    )

    log_interval = 10

    @trainer.on(Events.ITERATION_COMPLETED)
    def dvclive_log(engine):
        live.log_metric("train_loss", engine.state.output)
        live.next_step()

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(
            f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(
            f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.5f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.5f}"
        )
        live.log_metric("val_loss", metrics["loss"])

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
        ("training", train_evaluator),
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
    ProgressBar().attach(train_evaluator)
    ProgressBar().attach(val_evaluator)

    trainer.run(train_loader, max_epochs=1)

    # after training, rename the best checkpoint to models/best.pt
    best_path = model_checkpoint.last_checkpoint  # full path to the chosen best file
    os.makedirs("models", exist_ok=True)
    if best_path:
        target = os.path.join("models", "best.pt")
        # overwrite if exists
        if os.path.abspath(best_path) != os.path.abspath(target):
            os.replace(best_path, target)


if __name__ == "__main__":
    with open("params.yaml", "r") as f, dvclive.Live(dir="dvclive/train") as live:
        run(ConfigBox(yaml.safe_load(f), box_dots=True), live)
