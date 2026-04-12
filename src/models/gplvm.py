from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


class GPLVM(nn.Module):
    """
    Deterministic GPLVM with Gaussian likelihood and shared RBF kernel.

    Observed data:
        Y in R^{N x D}

    Latent variables:
        X in R^{N x Q}

    Model:
        For each observed dimension d=1,...,D:
            y_d ~ GP(0, K(X, X) + sigma^2 I)

    Notes:
    - This is a practical, deterministic GPLVM for latent analysis.
    - It is intended for small/moderate subsets because training is O(N^3).
    """

    def __init__(
        self,
        Y: torch.Tensor,
        latent_dim: int = 2,
        X_init: torch.Tensor | None = None,
        ard: bool = True,
        jitter: float = 1e-5,
    ) -> None:
        super().__init__()

        if Y.ndim != 2:
            raise ValueError("Y must be a 2D tensor of shape [N, D].")

        self.register_buffer("Y", Y)
        self.n, self.data_dim = Y.shape
        self.latent_dim = latent_dim
        self.ard = ard
        self.jitter = jitter

        if X_init is None:
            X_init = 0.01 * torch.randn(self.n, latent_dim, dtype=Y.dtype, device=Y.device)

        if X_init.shape != (self.n, latent_dim):
            raise ValueError(
                f"X_init must have shape {(self.n, latent_dim)}, got {tuple(X_init.shape)}"
            )

        self.X = nn.Parameter(X_init.clone())

        if ard:
            self.log_lengthscale = nn.Parameter(torch.zeros(latent_dim, dtype=Y.dtype, device=Y.device))
        else:
            self.log_lengthscale = nn.Parameter(torch.zeros(1, dtype=Y.dtype, device=Y.device))

        self.log_outputscale = nn.Parameter(torch.tensor(0.0, dtype=Y.dtype, device=Y.device))
        self.log_noise = nn.Parameter(torch.tensor(-2.0, dtype=Y.dtype, device=Y.device))

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale)

    @property
    def outputscale(self) -> torch.Tensor:
        return torch.exp(self.log_outputscale)

    @property
    def noise(self) -> torch.Tensor:
        return torch.exp(self.log_noise)

    def rbf_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        ARD RBF kernel:
            k(x, x') = sigma_f^2 exp(-0.5 ||(x-x') / ell||^2)
        """
        if self.ard:
            ls = self.lengthscale.view(1, 1, -1)
        else:
            ls = self.lengthscale.view(1, 1, 1)

        X1_scaled = X1.unsqueeze(1) / ls
        X2_scaled = X2.unsqueeze(0) / ls
        sqdist = ((X1_scaled - X2_scaled) ** 2).sum(dim=-1)

        return self.outputscale * torch.exp(-0.5 * sqdist)

    def kernel_matrix(self) -> torch.Tensor:
        K = self.rbf_kernel(self.X, self.X)
        eye = torch.eye(self.n, dtype=self.Y.dtype, device=self.Y.device)
        return K + (self.noise + self.jitter) * eye

    def negative_log_likelihood(self) -> torch.Tensor:
        """
        Negative log marginal likelihood:
            0.5 * tr(Y^T K^{-1} Y) + D/2 * log|K| + ND/2 * log(2pi)
        """
        K = self.kernel_matrix()
        L = torch.linalg.cholesky(K)

        alpha = torch.cholesky_solve(self.Y, L)  # [N, D]
        data_fit = 0.5 * torch.sum(self.Y * alpha)

        logdet = self.data_dim * torch.sum(torch.log(torch.diagonal(L)))
        constant = 0.5 * self.n * self.data_dim * math.log(2.0 * math.pi)

        return data_fit + logdet + constant

    def latent_regularizer(self, weight: float = 1e-3) -> torch.Tensor:
        """
        Small regularizer to keep latent positions bounded.
        """
        return weight * torch.mean(self.X ** 2)

    def loss(self, latent_reg_weight: float = 1e-3) -> torch.Tensor:
        return self.negative_log_likelihood() + self.latent_regularizer(latent_reg_weight)

    @torch.no_grad()
    def get_latent_positions(self) -> torch.Tensor:
        return self.X.detach().clone()

    @torch.no_grad()
    def get_hyperparameters(self) -> dict[str, list[float] | float]:
        return {
            "lengthscale": self.lengthscale.detach().cpu().numpy().tolist(),
            "outputscale": float(self.outputscale.detach().cpu().item()),
            "noise": float(self.noise.detach().cpu().item()),
        }