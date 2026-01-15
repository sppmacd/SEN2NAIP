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

from .model_impls.content_loss import VGGPerceptualLoss
from .model_impls.gan import ResNet18Discriminator, create_gan_trainer
from .models import Model


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, *, overfitting: bool = False):
        if overfitting:
            self.hr_images = list(Path(path).glob("*.hr.npz"))[:1]
        else:
            self.hr_images = list(Path(path).glob("*.hr.npz"))[:256]

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
        return t(lr_image)[:3], t(hr_image)[:3]


def cvimage(img):
    import typesum as ts

    ts.print(img)
    return np.transpose(np.clip(img, 0, 1) * 255, (1, 2, 0)).astype(np.uint8)


def run(config: ConfigBox, live: dvclive.Live):
    print(config)

    torch.autograd.set_detect_anomaly(True)
    torch.cuda.memory._record_memory_history()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    generator_model = Model().to(device)
    # discriminator_model = ResNet18Discriminator().to(device)

    batch_size = config.train.batch_size

    overfitting = config.train.mode == "overfitting"

    train_dataset = Dataset("data/train", overfitting=overfitting)

    # Show some random prediction
    lr, hr = train_dataset[len(train_dataset) // 2]
    live.log_image("low_res.png", cvimage(lr.numpy()))
    live.log_image("high_res.png", cvimage(hr.numpy()))
    live.log_image(
        "prediction_random.png",
        cvimage(
            generator_model(lr.to(device).unsqueeze(0))
            .squeeze(0)
            .cpu()
            .detach()
            .numpy(),
        ),
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(Dataset("data/val"), shuffle=True, batch_size=batch_size)

    match config.train.optimizer:
        case "adam":
            optimizer = torch.optim.Adam(generator_model.parameters(), lr=1e-3)
        case "rmsprop":
            optimizer = torch.optim.RMSprop(generator_model.parameters(), lr=1e-3)

    match config.train.loss:
        case "mse":

            def msedebug(y1, y2):
                print(f"y1 range: {y1.min()}..{y1.max()}")
                print(f"y2 range: {y2.min()}..{y2.max()}")
                return nn.MSELoss()(y1, y2)

            criterion = nn.MSELoss()
        case "vgg16":
            criterion = VGGPerceptualLoss(resize=False).to(device)

    trainer = create_supervised_trainer(
        generator_model,
        optimizer,
        criterion,
        device,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    # trainer = create_gan_trainer(
    #     generator=generator_model,
    #     discriminator=discriminator_model,
    #     optimizer_class=torch.optim.Adam,
    #     device=device,
    #     gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    # )

    val_metrics = {
        "loss": Loss(criterion),
    }

    val_evaluator = create_supervised_evaluator(
        generator_model,
        metrics=val_metrics,
        device=device,
    )

    log_interval = 10

    @trainer.on(Events.ITERATION_COMPLETED)
    def dvclive_log_step(engine):
        live.next_step()

    @trainer.on(
        Events.ITERATION_COMPLETED(every=config.train.gradient_accumulation_steps),
    )
    def dvclive_log_loss(engine):
        live.log_metric("train_loss", engine.state.output)
        # print(engine.state.output)
        # for k, v in engine.state.output.items():
        #     live.log_metric(k, v)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        print(f"Epoch {trainer.state.epoch} finished")
        if not overfitting:
            val_evaluator.run(val_loader)
            metrics = val_evaluator.state.metrics
            print(
                f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.5f}"
            )
            live.log_metric("val_loss", metrics["loss"])

    @trainer.on(Events.EPOCH_COMPLETED(every=20))
    def log_images(trainer):
        live.log_image(
            f"e{trainer.state.epoch}_state.png",
            cvimage(
                generator_model(lr.to(device).unsqueeze(0))
                .squeeze(0)
                .cpu()
                .detach()
                .numpy(),
            ),
        )

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
        {"model": generator_model},
    )

    # Define a Tensorboard logger
    tb_logger = TensorboardLogger()

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

    trainer.run(train_loader, max_epochs=config.train.max_epochs)
    torch.save(generator_model.state_dict(), "models/best.pt")

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
