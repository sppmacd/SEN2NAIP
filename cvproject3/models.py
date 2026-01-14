import torch

from .model_impls.unet import UNet
from .model_impls.carn.model.carn_m import Net as CARN_m

# All models take (B, C, W, H) downscaled image
# and return (B, C, W*2, H*2) upscaled image


class BaselineModel(torch.nn.Module):
    def forward(self, x):
        return torch.nn.Upsample(scale_factor=2, mode="nearest")(x)


class SimpleModel(torch.nn.Module):
    """A simple model with just one conv layer.

    Just to see if the dvc+ignite setup works.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, (3, 3), padding=1)

    def forward(self, x):
        xup = torch.nn.Upsample(scale_factor=2, mode="bicubic")(x)
        return self.conv(xup)


class UNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(4, 4)

    def forward(self, x):
        xup = torch.nn.Upsample(scale_factor=2, mode="bicubic")(x)
        # Note: U-Net learns the difference here.
        return xup + self.unet(xup)


class CARNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.carn = CARN_m(scale=2)

    def forward(self, x):
        return self.carn(x, 2)


Model = CARNModel
