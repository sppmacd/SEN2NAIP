import gc

import torch
from ignite.engine import Engine
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision import models

from .dynamics import grad_L2, weights_L2


class ResNet18Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Map 4->3 channels (hacky but allows us to reuse pretrained resnet)
        self.input_conv = nn.Conv2d(4, 3, kernel_size=1)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            nn.LeakyReLU(inplace=True),
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
        # self._init_weights()

    def input_conv_params(self):
        return self.input_conv.parameters()

    def backbone_params(self):
        return self.backbone.parameters()

    def classifier_params(self):
        return self.classifier.parameters()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_lr, x_hr):
        # x_lr: (B,4,H//2, W//2) low-resolution image (generator/upscaler input)
        # x_hr: (B,4,H,W) high-resolution image (real or SR output)
        x_lr_up = nn.Upsample(scale_factor=2, mode="bicubic")(x_lr)
        # Discriminator learns on the difference between bicubic scaling and generator output.
        x = self.input_conv(x_hr - x_lr_up)  # (B,3,H,W)
        feat = self.backbone(x)  # (B,512,1,1)
        logits = self.classifier(feat)  # (B,1)
        return logits.squeeze(1)


def print_local_tensor_info(local_vars):
    gc.collect()
    tensors = []
    for name, val in local_vars.items():
        try:
            if isinstance(val, torch.Tensor):
                tensors.append(
                    (name, val.shape, val.dtype, val.device, val.requires_grad)
                )
            elif isinstance(val, (list, tuple)):
                # count contained tensors without recursing deeply
                cnt = sum(1 for x in val if isinstance(x, torch.Tensor))
                if cnt:
                    tensors.append(
                        (name, f"{type(val).__name__} len={len(val)} tensors={cnt}")
                    )
            elif isinstance(val, dict):
                cnt = sum(1 for x in val.values() if isinstance(x, torch.Tensor))
                if cnt:
                    tensors.append((name, f"dict len={len(val)} tensors={cnt}"))
            elif "numpy" in str(type(val)) and hasattr(val, "shape"):
                tensors.append((name, getattr(val, "shape", None)))
        except Exception:
            pass

    if not tensors:
        print("No tensors found in locals")
        return

    print("Local tensors:")
    for info in tensors:
        print(info)


def create_gan_trainer(
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_class,
    device,
    gradient_accumulation_steps: int,
):
    opt_generator = optimizer_class(generator.parameters(), lr=8e-4)
    opt_discriminator = optimizer_class(discriminator.parameters(), lr=1e-3)

    def training_step(engine, data):
        # https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
        generator.train()
        discriminator.train()

        lr, hr = data
        real_lr = lr.to(device)
        real = hr.to(device)

        it = engine.state.iteration

        if (
            it // (8 * gradient_accumulation_steps) % 4 != 3
        ):  # 0,1,2 - train discriminator, 3 - train generator
            generator.requires_grad_(False)

            if it % gradient_accumulation_steps == 0:
                discriminator.zero_grad()

            output_real = discriminator(real_lr, real).view(-1)
            fake = generator(lr.to(device))
            output_fake = discriminator(real_lr, fake.detach()).view(-1)

            torch.cuda.empty_cache()

            # Earth mover's distance (Maximize)
            err_d = (output_fake - output_real).mean()
            err_d.backward()

            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            opt_discriminator.step()

            # Clip weights
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            torch.cuda.empty_cache()

            return {
                "EMD": -err_d.item(),
                "D_x": output_real.mean().item(),
                "D_G_z1": output_fake.mean().item(),
                "discr_grad_L2": grad_L2(discriminator).item(),
                "discr_weights_L2": weights_L2(discriminator).item(),
            }
        else:
            # Train Generator
            generator.requires_grad_(True)

            if it % gradient_accumulation_steps == 0:
                generator.zero_grad()

            output_real = discriminator(real_lr, real).view(-1)

            fake = generator(lr.to(device))
            output_fake = discriminator(real_lr, fake).view(-1)

            torch.cuda.empty_cache()

            # Earth mover's distance (Minimize)
            err_g_emd = (output_real - output_fake).mean()
            output_fake_mean = output_fake.mean().item()
            del output_fake
            output_real_mean = output_real.mean().item()
            del output_real

            torch.cuda.empty_cache()

            # Penalty for fake outputs being out of -1..1
            err_norm = fake.mean() ** 4

            err_g = err_g_emd + err_norm
            err_g.backward()

            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            opt_generator.step()

            # Clip weights
            for p in generator.parameters():
                p.data.clamp_(-0.1, 0.1)

            gc.collect()

            return {
                "EMD_pure": err_g_emd.item(),
                "err_norm": err_norm.item(),
                "EMD": err_g.item(),
                "D_x": output_real_mean,
                "D_G_z2": output_fake_mean,
                "MSE": nn.MSELoss()(fake, real).item(),
                "gen_grad_L2": grad_L2(generator).item(),
                "gen_weights_L2": weights_L2(generator).item(),
                "gen_weights_L2_exit_": weights_L2(generator.carn.exit).item(),
            }

    def training_step_profile(engine, data):
        try:
            t = training_step(engine, data)
        except torch.OutOfMemoryError:
            torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
            raise
        return t

    return Engine(training_step_profile)
