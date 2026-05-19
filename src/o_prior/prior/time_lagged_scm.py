from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .utils import XSampler


class TimeLaggedSCM(nn.Module):
    """Generate synthetic time-lagged data with autoregressive causal structure.

    Each feature at time t depends on a weighted combination of all features
    from the previous `lag_order` steps, plus noise.

    Parameters
    ----------
    seq_len : int, default=1024
        Sequence length to generate.
    num_features : int, default=100
        Number of features in X.
    num_outputs : int, default=1
        Number of outputs in y.
    lag_order : int, default=3
        Number of past time steps used for each update.
    weight_sparsity : float, default=0.7
        Fraction of weights set to zero (sparsity).
    activation : callable, default=nn.Tanh
        Nonlinearity applied to the autoregressive update.
    noise_std : float, default=0.01
        Std of Gaussian noise for X updates.
    output_noise_std : float, default=0.01
        Std of Gaussian noise for y.
    sampling : str, default="normal"
        Sampling strategy for initial history via XSampler.
    device : str, default="cpu"
        Torch device.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 100,
        num_outputs: int = 1,
        lag_order: int = 3,
        weight_sparsity: float = 0.7,
        activation: Any = nn.Tanh,
        noise_std: float = 0.01,
        output_noise_std: float = 0.01,
        sampling: str = "normal",
        device: str = "cpu",
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.lag_order = max(1, lag_order)
        self.weight_sparsity = min(max(weight_sparsity, 0.0), 0.99)
        self.activation = activation()
        self.noise_std = noise_std
        self.output_noise_std = output_noise_std
        self.sampling = sampling
        self.register_buffer("_dev", torch.empty(0, device=device))

        # Initialize autoregressive weights (target, source, lag)
        w = torch.randn(
            self.num_features,
            self.num_features,
            self.lag_order,
            device=self._dev.device,
        ) / math.sqrt(max(self.num_features, 1))
        if self.weight_sparsity > 0:
            mask = torch.rand_like(w) > self.weight_sparsity
            w = w * mask
        self.register_buffer("weights", w)

        # Output projection
        self.register_buffer(
            "out_weights",
            torch.randn(self.num_features, self.num_outputs, device=self._dev.device)
            / math.sqrt(max(self.num_features, 1)),
        )

        # Initial history sampler
        self.xsampler = XSampler(
            self.lag_order,
            self.num_features,
            sampling=self.sampling,
            device=self._dev.device,
        )

        self.eval()

    @torch.no_grad()
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Initialize X with history
        X = torch.zeros(self.seq_len, self.num_features, device=self._dev.device)
        X[: self.lag_order] = self.xsampler.sample()

        # Autoregressive generation
        for t in range(self.lag_order, self.seq_len):
            past = X[t - self.lag_order : t]  # (lag, features)
            update = torch.einsum("ijl,lj->i", self.weights, past)
            update = self.activation(update)
            update = update + self.noise_std * torch.randn_like(update)
            X[t] = update

        # Outputs from X
        y = X @ self.out_weights
        if self.output_noise_std > 0:
            y = y + self.output_noise_std * torch.randn_like(y)

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y
