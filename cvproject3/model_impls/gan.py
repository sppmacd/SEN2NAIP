import gc

import torch
from ignite.engine import Engine
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision import models


class ResNet18Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Map 4->3 channels (hacky but allows us to reuse pretrained resnet)
        self.input_conv = nn.Conv2d(4, 3, kernel_size=1)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.backbone.requires_grad_(False)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

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

    def forward(self, x_hr):
        # x_hr: (B,4,H,W) high-resolution image (real or SR output)
        x = self.input_conv(x_hr)  # (B,3,H,W)
        feat = self.backbone(x)  # (B,512,1,1)
        probs = self.classifier(feat)  # (B,1)
        return probs.squeeze(1)


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
    opt_generator = optimizer_class(generator.parameters(), lr=5e-3)
    opt_discriminator = optimizer_class(discriminator.parameters(), lr=5e-4)

    criterion = nn.BCELoss()

    labels_real = torch.ones(32, device=device)
    labels_fake = torch.zeros(32, device=device)

    def training_step(engine, data):
        # https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
        generator.train()
        discriminator.train()

        lr, hr = data

        it = engine.state.iteration

        b_size = hr.size(0)

        if it // 50 % 2 == 0:
            # Train Discriminator (real)
            discriminator.zero_grad()
            real = hr.to(device)
            label = labels_real[:b_size]
            output1 = discriminator(real).view(-1)
            err_d_real = criterion(output1, label)
            err_d_real.backward()

            gc.collect()

            # Train Discriminator (fake)
            fake = generator(lr.to(device))
            label = labels_fake[:b_size]
            output2 = discriminator(fake.detach()).view(-1)
            err_d_fake = criterion(output2, label)
            err_d_fake.backward()

            gc.collect()

            err_d = err_d_real + err_d_fake
            opt_discriminator.step()

            gc.collect()

            return {
                "Loss_D": err_d.item(),
                "D_x": output1.mean().item(),
                "D_G_z1": output2.mean().item(),
            }
        else:
            # Train Generator
            generator.zero_grad()
            label = labels_real[:b_size]
            output3 = discriminator(generator(lr.to(device)))
            err_g = criterion(output3, label)
            err_g.backward()
            opt_generator.step()

            gc.collect()

            return {
                "Loss_G": err_g.item(),
                "D_G_z2": output3.mean().item(),
            }

    def training_step_profile(engine, data):
        with profile(
            activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True
        ) as prof:
            t = training_step(engine, data)

        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        return t

    return Engine(training_step)
