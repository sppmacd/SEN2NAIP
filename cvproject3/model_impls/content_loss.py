import torch
import torch.nn as nn
import torchvision.models as models


# feature extractor using pretrained torchvision resnet18 (first 3 layers: conv1..layer3)
class _ResNet18First3(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            base = models.resnet18(weights=weights)
        except Exception:
            base = models.resnet18(pretrained=True)
        children = list(base.children())
        kept = children[:7]  # conv1, bn1, relu, maxpool, layer1, layer2, layer3
        self.features = nn.Sequential(*kept)
        for p in self.features.parameters():
            p.requires_grad = False
        if device is not None:
            self.to(device)

    def forward(self, x):
        return self.features(x)


# ImageNet normalization stats
imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])

_resnet_extractor = None
_resnet_device = None


def _get_extractor(device):
    global _resnet_extractor, _resnet_device
    if _resnet_extractor is None or _resnet_device != device:
        _resnet_extractor = _ResNet18First3(device=device)
        _resnet_device = device
    return _resnet_extractor


def _preprocess_for_resnet(x):
    # expect float tensor; convert to [0,1] if in [0,255]
    if x.dtype != torch.float32:
        x = x.float()
    if x.max() > 2.0:
        x = x / 255.0
    # if single-channel, replicate to 3
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    # if 3+ channels, keep first 3 for resnet
    if x.shape[1] >= 3:
        x3 = x[:, :3, :, :]
    else:
        x3 = x
    device = x.device
    mean = imagenet_mean.to(device).view(1, -1, 1, 1)
    std = imagenet_std.to(device).view(1, -1, 1, 1)
    return (x3 - mean) / std


_mse = nn.MSELoss(reduction="mean")


def resnet18_content_loss(y_pred, y_real, alpha=1.0, beta=1.0):
    """
    Combined loss:
      - RGB pixel-wise MSE (beta weight)
      - ResNet18(first3) feature MSE
      - 4th-channel pixel-wise MSE (alpha weight)

    Args:
      y_pred, y_real: tensors shaped (B,4,H,W). ALWAYS include a 4th channel.
      alpha: weight for 4th-channel MSE (default 1.0).
      beta: weight for RGB pixel-wise MSE (default 1.0).

    Returns:
      scalar torch.Tensor: combined loss = feat_loss + beta * rgb_mse + alpha * ch4_mse
    """
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"y_pred and y_real must have same shape, got {y_pred.shape} vs {y_real.shape}"
        )
    if y_pred.dim() != 4 or y_pred.shape[1] < 4:
        raise ValueError("Inputs must be 4D tensors with at least 4 channels (B,4,H,W)")

    device = y_pred.device
    extractor = _get_extractor(device)

    # Separate RGB and 4th channel
    rgb_pred = y_pred[:, :3, :, :].contiguous()
    rgb_real = y_real[:, :3, :, :].contiguous()
    ch4_pred = y_pred[:, 3:4, :, :].contiguous()
    ch4_real = y_real[:, 3:4, :, :].contiguous()

    # Preprocess RGB for ResNet (normalization, scaling)
    x_pred = _preprocess_for_resnet(rgb_pred)
    x_real = _preprocess_for_resnet(rgb_real)

    # Extract features: target with no_grad to save memory; predicted keeps grad
    with torch.no_grad():
        feat_real = extractor(x_real).detach()
    feat_pred = extractor(x_pred)

    feat_loss = torch.mean((feat_pred - feat_real) ** 2)

    # RGB pixel-wise MSE: ensure scaling to [0,1]
    rp = rgb_pred.float()
    rr = rgb_real.float()
    if rp.max() > 2.0:
        rp = rp / 255.0
        rr = rr / 255.0
    rgb_mse = _mse(rp, rr)

    # 4th-channel MSE: scale to [0,1] if needed, ensure float
    c4p = ch4_pred.float()
    c4r = ch4_real.float()
    if c4p.max() > 2.0:
        c4p = c4p / 255.0
        c4r = c4r / 255.0
    ch4_mse = _mse(c4p, c4r)

    return feat_loss + beta * rgb_mse + alpha * ch4_mse
