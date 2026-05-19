"""Optional warping functions for data generation (realistic/complex distributions)."""

import torch
from torch import Tensor
import numpy as np


def kumaraswamy_warp(x: Tensor, a: float = 2.0, b: float = 5.0) -> Tensor:
    """
    Apply Kumaraswamy distribution warping to input tensor.
    
    Optional Kumaraswamy warping for additional "realism/challenge"
    in post-processing. This transforms the input distribution using the Kumaraswamy
    cumulative distribution function (CDF).
    
    The Kumaraswamy distribution is a bounded continuous probability distribution
    defined on [0, 1]. The CDF is: F(x) = 1 - (1 - x^a)^b
    
    Parameters
    ----------
    x : Tensor
        Input tensor to warp. Should be normalized to [0, 1] range.
    a : float, default=2.0
        First shape parameter of Kumaraswamy distribution
    b : float, default=5.0
        Second shape parameter of Kumaraswamy distribution
        
    Returns
    -------
    Tensor
        Warped tensor with same shape as input
    """
    # Ensure input is in [0, 1] range (clip if needed)
    x_clipped = torch.clamp(x, min=0.0, max=1.0)
    
    # Kumaraswamy CDF: F(x) = 1 - (1 - x^a)^b
    x_powered = torch.pow(x_clipped, a)
    one_minus_x_powered = 1.0 - x_powered
    one_minus_powered = torch.pow(one_minus_x_powered, b)
    warped = 1.0 - one_minus_powered
    
    return warped


def apply_kumaraswamy_warping(
    X: Tensor,
    a: float = 2.0,
    b: float = 5.0,
    normalize_first: bool = True
) -> Tensor:
    """
    Apply Kumaraswamy warping to feature tensor.
    
    This function normalizes features to [0, 1] range, applies Kumaraswamy warping,
    and optionally re-normalizes the result.
    
    Parameters
    ----------
    X : Tensor
        Feature tensor of shape (seq_len, num_features)
    a : float, default=2.0
        First shape parameter of Kumaraswamy distribution
    b : float, default=5.0
        Second shape parameter of Kumaraswamy distribution
    normalize_first : bool, default=True
        If True, normalize to [0, 1] before warping
        
    Returns
    -------
    Tensor
        Warped feature tensor with same shape as input
    """
    if normalize_first:
        # Normalize to [0, 1] using min-max per feature
        X_min = X.min(dim=0, keepdim=True)[0]
        X_max = X.max(dim=0, keepdim=True)[0]
        X_range = (X_max - X_min).clip(min=1e-6)
        X_norm = (X - X_min) / X_range
    else:
        X_norm = X
    
    # Apply Kumaraswamy warping per feature
    X_warped = kumaraswamy_warp(X_norm, a=a, b=b)
    
    return X_warped

