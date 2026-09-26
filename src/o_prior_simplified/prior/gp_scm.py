from __future__ import annotations

import math
import warnings
from typing import Dict, Any

import torch
from torch import nn

# Hard cap on seq_len for GP-SCM: GP sampling scales O(N²) memory and O(N³) time
MAX_GP_SEQ_LEN = 2048


class GPSCM(nn.Module):
    """Generates synthetic tabular datasets using Gaussian Process (GP) priors with RBF kernels.
    
    This generates smooth functions using GP sampling, which is beneficial for learning
    smooth real-world functions. Each feature is sampled from an independent GP, and
    the target is a combination of these GP features.
    
    Parameters
    ----------
    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.
    
    num_features : int, default=100
        The number of features.
    
    num_outputs : int, default=1
        The number of outputs (targets).
    
    length_scale : float, default=1.0
        Length scale parameter for RBF kernel. Controls smoothness of the GP.
        Smaller values = more wiggly, larger values = smoother.
    
    signal_variance : float, default=1.0
        Signal variance parameter for RBF kernel. Controls amplitude of the GP.
    
    noise_variance : float, default=0.01
        Observation noise variance. Added to the GP output.
    
    gp_combination : str, default="linear"
        How to combine GP features into target:
        - "linear": y = X @ w + b
        - "quadratic": y = X @ w + (X**2) @ w2 + b
        - "mlp": Pass through small MLP
    
    use_joint_covariance_sampling : bool, default=False
        If True, use random node selection: generate latent GPs,
        mix them to create correlations, then randomly select X and y from the
        correlated set (strictly disjoint). If False, use deterministic y = f(X).
    
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
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 0.01,
        gp_combination: str = "linear",
        use_joint_covariance_sampling: bool = False,
        device: str = "cpu",
        **kwargs: Dict[str, Any],
    ):
        super(GPSCM, self).__init__()
        
        # Clamp seq_len to prevent OOM crashes (GP scales O(N²) memory and O(N³) time)
        if seq_len > MAX_GP_SEQ_LEN:
            warnings.warn(
                f"GP-SCM seq_len ({seq_len}) exceeds maximum ({MAX_GP_SEQ_LEN}). "
                f"Clamping to {MAX_GP_SEQ_LEN} to prevent OOM. "
                f"GP sampling scales O(N²) memory and O(N³) time. "
                f"Consider using approximate GP methods for longer sequences.",
                UserWarning,
                stacklevel=2
            )
            seq_len = MAX_GP_SEQ_LEN
        
        # Clamp length_scale to prevent nearly-constant GP functions
        # For coordinates in [0, 1], length_scale > 3.0 produces features with very little variation
        # This ensures data quality even if old cached config samples extreme values
        MAX_SAFE_LENGTH_SCALE = 3.6
        if length_scale > MAX_SAFE_LENGTH_SCALE:
            # Silently clamp to safe maximum
            length_scale = MAX_SAFE_LENGTH_SCALE
        
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.gp_combination = gp_combination
        self.use_joint_covariance_sampling = use_joint_covariance_sampling
        self.device = device
        
        # Initialize combination layers
        self._init_combination_layers()
    
    def _init_combination_layers(self):
        """Initialize the mixing mechanism (Linear, Quadratic, or MLP)."""
        if self.gp_combination == "linear":
            self.combination_weights = nn.Parameter(
                torch.randn(self.num_features, self.num_outputs, device=self.device) * 0.1
            )
            self.combination_bias = nn.Parameter(
                torch.zeros(self.num_outputs, device=self.device)
            )
        elif self.gp_combination == "quadratic":
            self.combination_weights = nn.Parameter(
                torch.randn(self.num_features, self.num_outputs, device=self.device) * 0.1
            )
            self.combination_weights2 = nn.Parameter(
                torch.randn(self.num_features, self.num_outputs, device=self.device) * 0.05
            )
            self.combination_bias = nn.Parameter(
                torch.zeros(self.num_outputs, device=self.device)
            )
        elif self.gp_combination == "mlp":
            hidden_dim = max(32, self.num_features // 2)
            self.combination_mlp = nn.Sequential(
                nn.Linear(self.num_features, hidden_dim, device=self.device),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim, device=self.device),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.num_outputs, device=self.device),
            )
        else:
            raise ValueError(f"Unknown gp_combination: {self.gp_combination}")
    
    def rbf_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF (Radial Basis Function) kernel matrix.
        
        K(x1, x2) = σ² * exp(-0.5 * ||x1 - x2||² / l²)
        
        Uses torch.cdist for optimized distance calculation.
        
        Parameters
        ----------
        X1 : torch.Tensor
            First set of points, shape (n1, d)
        X2 : torch.Tensor
            Second set of points, shape (n2, d)
        
        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (n1, n2)
        """
        # torch.cdist is efficient and handles the broadcasting in optimized C++ backends
        dist_matrix = torch.cdist(X1, X2, p=2)  # (n1, n2)
        squared_dist = dist_matrix ** 2
        
        # RBF kernel: σ² * exp(-0.5 * ||x1 - x2||² / l²)
        K = self.signal_variance * torch.exp(-0.5 * squared_dist / (self.length_scale ** 2))
        
        return K
    
    def _validate_hyperparameters(self):
        """Validate that hyperparameters are finite and within valid ranges."""
        for name in ["length_scale", "signal_variance", "noise_variance"]:
            v = getattr(self, name)
            if not (isinstance(v, (float, int)) and math.isfinite(v)):
                raise ValueError(
                    f"GP-SCM: {name} must be a finite float, got {v!r}. "
                    f"This likely indicates a bug in hyperparameter sampling."
                )
        
        if self.length_scale <= 0:
            raise ValueError(f"GP-SCM: length_scale must be > 0, got {self.length_scale}")
        if self.signal_variance <= 0:
            raise ValueError(f"GP-SCM: signal_variance must be > 0, got {self.signal_variance}")
        if self.noise_variance < 0:
            raise ValueError(f"GP-SCM: noise_variance must be >= 0, got {self.noise_variance}")
    
    def _get_stable_cholesky(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix and Cholesky decomposition with robust error handling.
        
        This implements production-grade numerical stability:
        - Validates hyperparameters are finite and positive
        - Uses float64 precision for decomposition (even if model uses fp16/bf16)
        - Preserves PSD property when handling NaNs/Infs
        - Adaptive jitter with cholesky_ex for better error detection
        - Falls back to eigendecomposition (eigh) instead of SVD for symmetric matrices
        
        Parameters
        ----------
        coords : torch.Tensor
            Input coordinates, shape (seq_len, coord_dim)
        
        Returns
        -------
        torch.Tensor
            Cholesky factor L, shape (seq_len, seq_len) where K = L @ L^T
        """
        # Validate hyperparameters first
        self._validate_hyperparameters()
        
        seq_len = coords.shape[0]
        device = coords.device
        original_dtype = coords.dtype
        
        # Always do kernel math in float64 for numerical stability
        # This prevents issues with fp16/bf16 causing NaNs in Cholesky
        work_dtype = torch.float64
        coords_w = coords.to(dtype=work_dtype)
        
        # Warn about extreme length_scale values that can cause numerical issues
        coord_range = (coords_w.max() - coords_w.min()).item()
        if coord_range > 0 and self.length_scale / coord_range > 3.0:
            warnings.warn(
                f"GP-SCM length_scale ({self.length_scale:.2f}) is very large relative to "
                f"coordinate range ({coord_range:.2f}). This may cause numerical instability. "
                f"Consider using length_scale < {coord_range * 3.0:.2f}",
                UserWarning,
                stacklevel=3
            )
        
        # Compute RBF kernel more stably (avoid cdist's sqrt + square)
        # For 1D coords: sqdist = (x_i - x_j)^2
        x = coords_w  # (seq_len, 1)
        sqdist = (x - x.T) ** 2  # (seq_len, seq_len)
        
        # Guard against tiny length_scale producing numerical issues
        ls = max(float(self.length_scale), 1e-6)
        sv = max(float(self.signal_variance), 1e-12)
        
        # RBF kernel: σ² * exp(-0.5 * ||x1 - x2||² / l²)
        K = sv * torch.exp(-0.5 * sqdist / (ls * ls))
        
        # Symmetrize to eliminate tiny numerical asymmetry
        K = 0.5 * (K + K.T)
        
        # Replace non-finite entries WITHOUT touching valid ones (preserves PSD)
        # This is critical: torch.clamp would break positive semi-definiteness!
        if not torch.isfinite(K).all():
            K = K.masked_fill(~torch.isfinite(K), 0.0)
        
        eye = torch.eye(seq_len, device=device, dtype=work_dtype)
        
        # Adaptive jitter loop with cholesky_ex for better error detection
        jitter = 1e-6 * sv  # Start with small jitter proportional to signal variance
        L = None
        
        for attempt in range(8):  # Up to 8 attempts with increasing jitter
            K_j = K + jitter * eye
            L, info = torch.linalg.cholesky_ex(K_j)
            
            # Check if Cholesky succeeded and result is finite
            if info.item() == 0 and torch.isfinite(L).all():
                break
            
            # Increase jitter by 10x for next attempt
            jitter *= 10.0
            L = None
        
        # If Cholesky failed, fall back to eigendecomposition
        # eigh is better than SVD for symmetric matrices
        if L is None:
            evals, evecs = torch.linalg.eigh(K + jitter * eye)
            # Clamp eigenvalues to ensure positive semi-definite
            evals = torch.clamp(evals, min=1e-10)
            # Reconstruct L from eigendecomposition: L = Q * sqrt(Λ)
            L = evecs @ torch.diag(torch.sqrt(evals))
        
        # Final safety check
        if not torch.isfinite(L).all():
            raise RuntimeError(
                f"GP-SCM Cholesky/eigh produced NaNs/Infs after all fallbacks. "
                f"Hyperparameters: length_scale={self.length_scale:.3f}, "
                f"signal_variance={self.signal_variance:.3f}, "
                f"noise_variance={self.noise_variance:.3f}, "
                f"coord_range={coord_range:.3f}, "
                f"seq_len={seq_len}, "
                f"input_dtype={original_dtype}, device={device}. "
                f"This indicates a fundamental numerical issue - please report this bug."
            )
        
        # Return in float32 (good compromise between precision and memory)
        # Cast back to original device (in case it changed)
        return L.to(dtype=torch.float32, device=device)
    
    def sample_gp_function(
        self,
        input_coords: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample smooth function from GP using RBF kernel.
        
        Uses Cholesky decomposition for efficient sampling:
        f ~ GP(0, K) => f = L @ ε where L = cholesky(K), ε ~ N(0, I)
        
        Parameters
        ----------
        input_coords : torch.Tensor
            Input coordinates, shape (seq_len, coord_dim)
        num_samples : int, default=1
            Number of independent GP samples to generate
        
        Returns
        -------
        torch.Tensor
            GP samples, shape (num_samples, seq_len)
        """
        seq_len = input_coords.shape[0]
        
        # Use centralized stable Cholesky computation
        L = self._get_stable_cholesky(input_coords)
        
        # Sample: f = L @ ε where ε ~ N(0, I)
        # Match dtype to L for numerical consistency
        epsilon = torch.randn(num_samples, seq_len, device=self.device, dtype=L.dtype)
        gp_samples = (L @ epsilon.T).T  # (num_samples, seq_len)
        
        return gp_samples
    
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate (X, y) dataset using GP priors.
        
        Returns
        -------
        tuple
            (X, y) where:
            - X: Features tensor of shape (seq_len, num_features)
            - y: Targets tensor of shape (seq_len, num_outputs)
        """
        # Random 1D coordinates avoid imposing smooth trends along row order.
        coords = torch.rand(self.seq_len, 1, device=self.device)
        
        # OPTIMIZATION: Vectorized GP sampling - compute Cholesky once for all features
        # Since all features use the same coords, the kernel matrix K is identical
        # Compute K and L once, then sample all features simultaneously
        
        if self.use_joint_covariance_sampling:
            # Latent Mixing + Random Selection
            # 1. Generate Latents (independent GPs)
            # Use fewer latents than total dimensions to create mixing
            num_latents = max(1, min(self.num_features, self.num_features // 2))
            latents = self.sample_gp_function(coords, num_samples=num_latents)  # (num_latents, seq_len)
            latents = latents.T  # (seq_len, num_latents)
            
            # 2. Mix them to create correlations (Data = Latents @ MixingMatrix)
            total_dims = self.num_features + self.num_outputs
            # Matrix shape: (num_latents, total_dims)
            mixing_weights = torch.randn(num_latents, total_dims, device=self.device) * 0.1
            data_correlated = latents @ mixing_weights  # (seq_len, total_dims)
            
            # Center to reduce bias
            data_correlated = data_correlated - data_correlated.mean(dim=0, keepdim=True)
            
            # Add observation noise
            if self.noise_variance > 0:
                noise = torch.randn_like(data_correlated) * math.sqrt(self.noise_variance)
                data_correlated = data_correlated + noise
            
            # 3. Random Selection (Strictly Disjoint)
            perm = torch.randperm(total_dims, device=self.device)
            idx_y = perm[:self.num_outputs]
            idx_X = perm[self.num_outputs:self.num_outputs + self.num_features]
            
            y = data_correlated[:, idx_y]
            X = data_correlated[:, idx_X]
            
            # Squeeze if single output
            if self.num_outputs == 1:
                y = y.squeeze(-1)
        else:
            # Original deterministic behavior: y = f(X)
            # Compute Cholesky ONCE for all features (centralized stability logic)
            L = self._get_stable_cholesky(coords)
            
            # Vectorized sampling: sample all features at once
            # epsilon shape: (num_features, seq_len)
            epsilon = torch.randn(self.num_features, self.seq_len, device=self.device)
            
            # X = L @ epsilon.T -> (seq_len, num_features)
            # L is (seq_len, seq_len), epsilon.T is (seq_len, num_features)
            # Result: (seq_len, num_features)
            X = L @ epsilon.T
            
            # Check for NaNs in X after sampling
            if torch.any(torch.isnan(X)) or torch.any(torch.isinf(X)):
                raise RuntimeError("GP-SCM generated NaNs/Infs in features. This may indicate numerical instability.")
            
            # Center features to reduce bias and improve train/test overlap
            X = X - X.mean(dim=0, keepdim=True)
            
            # Add observation noise to features
            if self.noise_variance > 0:
                noise = torch.randn_like(X) * math.sqrt(self.noise_variance)
                X = X + noise
            
            # Generate target from GP features
            if self.gp_combination == "linear":
                y = X @ self.combination_weights + self.combination_bias
            elif self.gp_combination == "quadratic":
                y = (X @ self.combination_weights + 
                     (X ** 2) @ self.combination_weights2 + 
                     self.combination_bias)
            elif self.gp_combination == "mlp":
                y = self.combination_mlp(X)
            else:
                raise ValueError(f"Unknown gp_combination: {self.gp_combination}")
            
            # Add noise to target
            if self.noise_variance > 0:
                target_noise = torch.randn_like(y) * math.sqrt(self.noise_variance)
                y = y + target_noise
            
            # Squeeze if single output
            if self.num_outputs == 1:
                y = y.squeeze(-1)
        
        # Final check for NaNs/Infs in output
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)) or torch.any(torch.isinf(X)) or torch.any(torch.isinf(y)):
            raise RuntimeError("GP-SCM generated NaNs/Infs in output. This may indicate numerical instability with current hyperparameters.")
        
        return X, y
