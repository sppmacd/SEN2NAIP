import torch
from torch import nn


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

    l2 = torch.tensor((sq_sum + eps) ** 0.5)
    print(f"grad L2 = {l2}")
    return l2


def weights_L2(model: nn.Module, eps: float = 1e-12) -> torch.Tensor:
    """L2 norm of weights."""
    sq_sum = None
    for p in model.parameters():
        if p.numel() == 0:
            continue
        # accumulate in float32 for stability
        val = float((p.float() ** 2).sum().cpu().item())
        sq_sum = val if sq_sum is None else sq_sum + val
    if sq_sum is None:
        return torch.tensor(float("nan"))

    l2 = torch.tensor((sq_sum + eps) ** 0.5)
    print(f"weights L2 = {l2}")
    return l2
