from __future__ import annotations

import math

import torch


def mdn_nll(
    y: torch.Tensor,
    logit_pi: torch.Tensor,
    mu: torch.Tensor,
    log_sx: torch.Tensor,
    log_sz: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood for a mixture of 2D Gaussians with correlation.

    Shapes:
      y:        [B, 2]
      logit_pi: [B, K]
      mu:       [B, K, 2]
      log_sx:   [B, K]
      log_sz:   [B, K]
      rho:      [B, K]  (in (-1, 1))
    """

    if y.ndim != 2 or y.size(-1) != 2:
        raise ValueError("y must be [B,2]")

    # Clamp rho to avoid numerical blow-ups near |rho|=1
    rho = rho.clamp(min=-0.95, max=0.95)

    sx = torch.exp(log_sx).clamp(min=1e-3)
    sz = torch.exp(log_sz).clamp(min=1e-3)

    x = y[:, None, 0]  # [B,1]
    z = y[:, None, 1]
    mx = mu[:, :, 0]
    mz = mu[:, :, 1]

    # Standardized residuals
    dx = (x - mx) / sx
    dz = (z - mz) / sz

    one_minus_rho2 = (1.0 - rho**2).clamp(min=1e-3)

    # log N(x,z | mu, Sigma) for each component
    # See bivariate normal pdf; compute in log-space.
    log_norm = -math.log(2.0 * math.pi) - torch.log(sx) - torch.log(sz) - 0.5 * torch.log(one_minus_rho2)
    z_term = (dx**2 + dz**2 - 2.0 * rho * dx * dz) / one_minus_rho2
    log_prob = log_norm - 0.5 * z_term  # [B,K]

    log_pi = torch.log_softmax(logit_pi, dim=-1)
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # [B]
    return -log_mix.mean()


def mdn_mean(logit_pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """
    Mixture mean: E[y] = sum_k pi_k * mu_k
    Shapes:
      logit_pi: [B,K]
      mu: [B,K,2]
    """

    pi = torch.softmax(logit_pi, dim=-1)
    return torch.einsum("bk,bkd->bd", pi, mu)
