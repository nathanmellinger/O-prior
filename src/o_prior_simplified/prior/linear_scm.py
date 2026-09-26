from __future__ import annotations

from typing import Dict, Any

import torch
from torch import nn


from .utils import GaussianNoise, XSampler


class LinearSCM(nn.Module):
    """Simple linear regression prior for curriculum learning Stage 1.
    
    Generates datasets where y = X @ w + b + noise.
    Crucially, 'w' and 'b' are sampled randomly *per batch*, allowing the
    downstream model to learn the general algorithm of linear regression
    rather than memorizing a single function.
    
    Parameters
    ----------
    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.
    
    num_features : int, default=100
        The number of features.
    
    num_outputs : int, default=1
        The number of outputs (targets).
    
    noise_std : float, default=0.01
        Standard deviation of observation noise. Set to 0.0 for noise-free tasks.
    
    weight_std : float, default=1.0
        Standard deviation for sampling regression weights.
    
    bias_std : float, default=1.0
        Standard deviation for sampling bias term.

    sampling : str, default='normal'
        Feature sampling strategy ('normal', 'mixed', 'uniform', 'beta').
    
    use_joint_covariance_sampling : bool, default=False
        If True, use random node selection: sample from Multivariate
        Normal with random covariance, then randomly select X and y (strictly
        disjoint). If False, use deterministic y = X @ w + b.
    
    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.
    
    **kwargs : dict
        Unused hyperparameters passed from parent configurations.
    """
    
    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 100,
        num_outputs: int = 1,
        noise_std: float = 0.01,
        weight_std: float = 1.0,
        bias_std: float = 1.0,
        sampling: str = "normal",
        use_joint_covariance_sampling: bool = False,
        device: str = "cpu",
        **kwargs: Dict[str, Any],
    ):
        super(LinearSCM, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.noise_std = noise_std
        self.weight_std = weight_std
        self.bias_std = bias_std
        self.sampling = sampling
        self.use_joint_covariance_sampling = use_joint_covariance_sampling
        self.device = device

        self.xsampler = XSampler(
            self.seq_len,
            self.num_features,
            sampling=self.sampling,
            device=self.device,
        )
    
    def _generate_multivariate_normal(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates data by sampling from a random Joint Covariance Matrix."""
        total_dims = self.num_features + self.num_outputs
        
        # 1. Generate random positive semi-definite Covariance Matrix
        #    Cov = L @ L.T
        L_rand = torch.randn(total_dims, total_dims, device=self.device) * 0.5
        Cov = L_rand @ L_rand.T
        
        # Add diagonal loading for stability
        Cov.diagonal().add_(0.1)
        
        # 2. Sample from Multivariate Normal: N(0, Cov)
        #    Optimization: Use Cholesky(Cov) directly instead of full eigendecomp
        try:
            L_chol = torch.linalg.cholesky(Cov)
        except RuntimeError:
            # Robust Fallback: SVD for singular matrices
            U, S, _ = torch.linalg.svd(Cov)
            L_chol = U @ torch.diag(torch.sqrt(S.clamp(min=1e-6)))
            
        Z = torch.randn(self.seq_len, total_dims, device=self.device)
        data = Z @ L_chol.T  # (seq_len, total_dims)
        
        # 3. Randomly split dimensions into X and y
        perm = torch.randperm(total_dims, device=self.device)
        y = data[:, perm[:self.num_outputs]]
        X = data[:, perm[self.num_outputs:self.num_outputs + self.num_features]]
        
        return X, y

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate (X, y) dataset for a NEW random linear task.
        
        Returns
        -------
        tuple
            (X, y) where:
            - X: Features tensor of shape (seq_len, num_features)
            - y: Targets tensor of shape (seq_len, num_outputs)
        """
        if self.use_joint_covariance_sampling:
            X, y = self._generate_multivariate_normal()
        else:
            # Deterministic Linear: y = Xw + b
            # 1. Generate features using xsampler
            X = self.xsampler.sample()
            
            # 2. Sample NEW weights/bias for this specific batch
            #    (Do not store as self.weights, or the task never changes!)
            weights = torch.randn(self.num_features, self.num_outputs, device=self.device) * self.weight_std
            bias = torch.randn(self.num_outputs, device=self.device) * self.bias_std
            
            y = X @ weights + bias

        # Add Observation Noise
        if self.noise_std > 0:
            X = X + torch.randn_like(X) * self.noise_std
            y = y + torch.randn_like(y) * self.noise_std

        # Handle Single Output Squeeze
        if self.num_outputs == 1:
            y = y.squeeze(-1)

        # Sanity Check
        if torch.isnan(X).any() or torch.isnan(y).any():
            raise RuntimeError("LinearSCM generated NaNs. Check noise/std parameters.")
        if torch.isinf(X).any() or torch.isinf(y).any():
            raise RuntimeError("LinearSCM generated Infs. Check noise/std parameters.")

        return X, y

