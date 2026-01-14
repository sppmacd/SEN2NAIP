import torch
import torch.nn as nn

import math
import torch
import matplotlib.pyplot as plt

from . import ops


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.b1 = ops.EResidualBlock(64, 64, group=group)
        self.c1 = ops.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


def chshow(t: torch.Tensor, cmap="viridis", vmin=None, vmax=None, figsize=None):
    """
    Plot channels of a CUDA tensor with shape (1, C, W, H) on a square grid, each channel in its own axis.

    Args:
      t (torch.Tensor): CUDA tensor of shape (1, C, W, H) or (1, C, H, W).
      cmap (str): Matplotlib colormap for each channel.
      vmin, vmax: Optional intensity bounds passed to imshow.
      figsize (tuple): Optional (width, height) for the entire figure. If None, computed from grid size.

    Notes:
      - The function moves the tensor to CPU and detaches it.
      - If tensor has shape (1, C, H, W) that's also handled.
    """
    # Validate shape
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor")
    if t.dim() != 4 or t.size(0) != 1:
        raise ValueError("t must have shape (1, C, W, H) or (1, C, H, W)")
    # ensure CPU float
    tt = t.detach().cpu().float()
    _, C, d1, d2 = tt.shape

    # If channels-first spatial dims are (H, W) or (W, H) doesn't matter for display.
    # Build square-ish grid
    cols = math.ceil(math.sqrt(C))
    rows = math.ceil(C / cols)

    # Figure size default: 3in per subplot
    if figsize is None:
        figsize = (cols * 3, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # normalize axes to 2D array
    if isinstance(axes, plt.Axes):
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    for i in range(rows * cols):
        ax = axes_flat[i]
        ax.axis("off")
        if i < C:
            channel = tt[0, i]
            im = ax.imshow(
                channel.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal"
            )
            ax.set_title(f"Ch {i}")
            ax.axis("on")  # show ticks if desired; remove if not
            ax.set_xticks([])
            ax.set_yticks([])

            # add colorbar for this image
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.6)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.show()


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()

        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040, 0.5000), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040, 0.5000), sub=False)

        self.entry = nn.Conv2d(4, 64, 3, 1, 1)

        self.b1 = Block(64, 64, group=group)
        self.b2 = Block(64, 64, group=group)
        self.b3 = Block(64, 64, group=group)
        self.c1 = ops.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64 * 4, 64, 1, 1, 0)

        self.upsample = ops.UpsampleBlock(
            64, scale=scale, multi_scale=multi_scale, group=group
        )
        self.exit = nn.Conv2d(64, 4, 3, 1, 1)

    def forward(self, x, scale):
        # chshow(x)
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        # chshow(o3)

        out = self.upsample(o3, scale=scale)
        # chshow(out)

        out = self.exit(out)
        # chshow(out)
        out = self.add_mean(out)
        # chshow(out)

        return out
