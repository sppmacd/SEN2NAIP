import torch

# Assuming notation:
#  I -> original image,
#  I' -> downscaled image,
#  I^ -> bicubic upscaled image (BicubicUpscale(I'))
# We perform downscaling: I' = Downscale(I)
# Then train model to predict the difference: Model(I') ~= I - BicubicUpscale(I')
# So loss is e.g L = MSE(Model(I'), I - BicubicUpscale(I'))

# All models take (B, C, W, H) original image I
# and return (B, C, W*2, H*2)


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
