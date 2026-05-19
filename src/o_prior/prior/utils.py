from __future__ import annotations

import random
import numpy as np

import torch
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, X):
        std = self.std

        if torch.is_tensor(std):
            # Align per-channel std with common tensor shapes
            if X.dim() == 3:
                # Conv1d output: (batch, channels, length)
                if std.dim() == 1 and std.shape[0] == X.shape[1]:
                    std = std.view(1, X.shape[1], 1)
                elif std.dim() == 2 and std.shape[1] == X.shape[1]:
                    # (1, channels) or (batch, channels) -> add spatial dim
                    std = std.view(std.shape[0], X.shape[1], 1)
            elif X.dim() == 2:
                # MLP output: (batch, features)
                if std.dim() == 1 and std.shape[0] == X.shape[1]:
                    std = std.view(1, X.shape[1])
        
        return X + torch.normal(torch.zeros_like(X), std)


class XSampler:
    """Input sampler for generating features for prior datasets.

    Supports multiple feature distribution types:
    - Normal: Standard normal distribution
    - Multinomial: Categorical features with random number of categories
    - Zipf: Power law distributed features
    - Mixed: Random combination of the above

    Parameters
    ----------
    seq_len : int
        Length of sequence to generate

    num_features : int
        Number of features to generate

    pre_stats : bool
        Whether to pre-generate statistics for the input features

    sampling : str, default='mixed'
        Feature sampling strategy ('normal', 'mixed', 'uniform')

    device : str, default='cpu'
        Device to store tensors on
    """

    def __init__(self, seq_len, num_features, pre_stats=False, sampling="mixed", device="cpu"):
        self.seq_len = seq_len
        self.num_features = num_features
        self.pre_stats = pre_stats
        self.sampling = sampling
        self.device = device

        if pre_stats:
            self._pre_stats()

    def _pre_stats(self):
        means = np.random.normal(0, 1, self.num_features)
        stds = np.abs(np.random.normal(0, 1, self.num_features) * means)
        self.means = torch.tensor(means, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.seq_len, 1)
        self.stds = torch.tensor(stds, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.seq_len, 1)

    def sample(self, return_numpy=False):
        """Generate features according to the specified sampling strategy.

        Returns
        -------
        X : torch.Tensor
            Generated features of shape (seq_len, num_features)
        """
        samplers = {"normal": self.sample_normal_all, "mixed": self.sample_mixed, "uniform": self.sample_uniform, "beta": self.sample_beta}
        if self.sampling not in samplers:
            raise ValueError(f"Invalid sampling method: {self.sampling}")
        X = samplers[self.sampling]()

        return X.cpu().numpy() if return_numpy else X

    def sample_normal_all(self):
        if self.pre_stats:
            X = torch.normal(self.means, self.stds.abs()).float()
        else:
            X = torch.normal(0.0, 1.0, (self.seq_len, self.num_features), device=self.device).float()
        return X

    def sample_uniform(self):
        """Generate uniformly distributed features."""
        return torch.rand((self.seq_len, self.num_features), device=self.device)

    def sample_normal(self, n=None):
        """Generate normally distributed features.

        Parameters
        ----------
        n : int
            Index of the feature to generate
        """

        if self.pre_stats:
            return torch.normal(self.means[:, n], self.stds[:, n].abs()).float()
        else:
            return torch.normal(0.0, 1.0, (self.seq_len,), device=self.device).float()

    def sample_multinomial(self):
        """Generate categorical features."""
        n_categories = random.randint(2, 20)
        probs = torch.rand(n_categories, device=self.device)
        x = torch.multinomial(probs, self.seq_len, replacement=True)
        x = x.float()
        return (x - x.mean()) / x.std()

    def sample_zipf(self):
        """Generate Zipf-distributed features."""
        x = np.random.zipf(2.0 + random.random() * 2, (self.seq_len,))
        x = torch.tensor(x, device=self.device).clamp(max=10)
        x = x.float()
        return x - x.mean()

    def sample_mixed(self):
        """Generate features with mixed distributions."""
        X = []
        zipf_p, multi_p, normal_p = random.random() * 0.66, random.random() * 0.66, random.random() * 0.66
        for n in range(self.num_features):
            if random.random() > normal_p:
                x = self.sample_normal(n)
            elif random.random() > multi_p:
                x = self.sample_multinomial()
            elif random.random() > zipf_p:
                x = self.sample_zipf()
            else:
                x = torch.rand((self.seq_len,), device=self.device)
            X.append(x)
        return torch.stack(X, -1)

    def sample_beta(self):
        """Generate Beta-distributed features (power-law style)."""
        # Randomize alpha/beta for each feature to create diverse distributions
        # Range [0.1, 5.0] covers U-shaped (<1), uniform(1), and bell-shaped (>1)
        alpha = torch.rand(self.num_features, device=self.device) * 4.9 + 0.1
        beta = torch.rand(self.num_features, device=self.device) * 4.9 + 0.1
        
        dist = torch.distributions.Beta(alpha, beta)
        x = dist.sample((self.seq_len,)) # (seq_len, num_features)
        
        # Normalize to mean=0, std=1 for consistency with other samplers
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False).clip(min=1e-6)
        return (x - mean) / std

