from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import nn

from .utils import GaussianNoise, XSampler


class MLPSCM(nn.Module):
    """Generates synthetic tabular datasets using a Multi-Layer Perceptron (MLP) based Structural Causal Model (SCM).
    
    Copied from : https://github.com/soda-inria/tabicl/blob/main/src/tabicl/prior/mlp_scm.

    Parameters
    ----------
    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.

    num_features : int, default=100
        The number of features.

    num_outputs : int, default=1
        The number of outputs.

    is_causal : bool, default=True
        - If `True`, simulates a causal graph: `X` and `y` are sampled from the
          intermediate hidden states of the MLP transformation applied to initial causes.
          The `num_causes` parameter controls the number of initial root variables.
        - If `False`, simulates a direct predictive mapping: Initial causes are used
          directly as `X`, and the final output of the MLP becomes `y`. `num_causes`
          is effectively ignored and set equal to `num_features`.

    num_causes : int, default=10
        The number of initial root 'cause' variables sampled by `XSampler`.
        Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
        set to `num_features`.

    y_is_effect : bool, default=True
        Specifies how the target `y` is selected when `is_causal=True`.
        - If `True`, `y` is sampled from the outputs of the final MLP layer(s),
          representing terminal effects in the causal chain.
        - If `False`, `y` is sampled from the earlier intermediate outputs (after
          permutation), representing variables closer to the initial causes.

    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled from the flattened
        intermediate MLP outputs when `is_causal=True`.
        - If `True`, `X` and `y` are selected from a contiguous block of the
          intermediate outputs, potentially creating denser dependencies among them.
        - If `False`, `X` and `y` indices are chosen randomly and independently
          from all available intermediate outputs.

    sort_features : bool, default=True
        Determines whether to sort the features based on their original indices from
        the intermediate MLP outputs. Only relevant when `is_causal=True`.

    num_layers : int, default=10
        The total number of layers in the MLP transformation network. Must be >= 2.
        Includes the initial linear layer and subsequent blocks of
        (Activation -> Linear -> Noise).

    hidden_dim : int, default=20
        The dimensionality of the hidden representations within the MLP layers.
        If `is_causal=True`, this is automatically increased if it's smaller than
        `num_outputs + 2 * num_features` to ensure enough intermediate variables
        are generated for sampling `X` and `y`.

    mlp_activations : default=nn.Tanh
        The activation function to be used after each linear transformation
        in the MLP layers (except the first).

    init_std : float, default=1.0
        The standard deviation of the normal distribution used for initializing
        the weights of the MLP's linear layers.

    block_wise_dropout : bool, default=True
        Specifies the weight initialization strategy.
        - If `True`, uses a 'block-wise dropout' initialization where only random
          blocks within the weight matrix are initialized with values drawn from
          a normal distribution (scaled by `init_std` and potentially dropout),
          while the rest are zero. This encourages sparsity.
        - If `False`, uses standard normal initialization for all weights, followed
          by applying dropout mask based on `mlp_dropout_prob`.

    mlp_dropout_prob : float, default=0.1
        The dropout probability applied to weights during *standard* initialization
        (i.e., when `block_wise_dropout=False`). Ignored if
        `block_wise_dropout=True`. The probability is clamped between 0 and 0.99.

    scale_init_std_by_dropout : bool, default=True
        Whether to scale the `init_std` during weight initialization to compensate
        for the variance reduction caused by dropout. If `True`, `init_std` is
        divided by `sqrt(1 - dropout_prob)` or `sqrt(keep_prob)` depending on the
        initialization method.

    sampling : str, default="normal"
        The method used by `XSampler` to generate the initial 'cause' variables.
        Options:
        - "normal": Standard normal distribution (potentially with pre-sampled stats).
        - "uniform": Uniform distribution between 0 and 1.
        - "mixed": A random combination of normal, multinomial (categorical),
          Zipf (power-law), and uniform distributions across different cause variables.

    pre_sample_cause_stats : bool, default=False
        If `True` and `sampling="normal"`, the mean and standard deviation for
        each initial cause variable are pre-sampled. Passed to `XSampler`.

    noise_std : float, default=0.01
        The base standard deviation for the Gaussian noise added after each MLP
        layer's linear transformation (except the first layer).

    pre_sample_noise_std : bool, default=False
        Controls how the standard deviation for the `GaussianNoise` layers is determined.

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
        is_causal: bool = True,
        num_causes: int = 10,
        y_is_effect: bool = True,
        in_clique: bool = False,
        sort_features: bool = True,
        num_layers: int = 10,
        hidden_dim: int = 20,
        mlp_activations: Any = nn.Tanh,
        init_std: float = 1.0,
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        use_joint_covariance_sampling: bool = False,
        device: str = "cpu",
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.use_joint_covariance_sampling = use_joint_covariance_sampling

        if num_layers < 2:
            warnings.warn(
                f"num_layers ({num_layers}) < 2, adjusting to 2. "
                f"This ensures minimum network depth for proper feature extraction.",
                UserWarning,
                stacklevel=2
            )
            num_layers = 2
        self.num_layers = num_layers

        self.hidden_dim = hidden_dim
        self.mlp_activations = mlp_activations
        
        # SAFETY FIX: Clamp init_std to prevent explosion regardless of config
        # Values > 0.5 can cause NaNs with deep networks and unbounded activations (Exp, Square)
        # Even with this clamp, dropout scaling can increase effective std, so we also clamp in init methods
        self.init_std = min(init_std, 0.5)
        
        self.block_wise_dropout = block_wise_dropout
        self.mlp_dropout_prob = mlp_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        # Register buffer to track device - this ensures device follows module when moved with .to()
        self.register_buffer("_dev", torch.empty(0, device=device))

        if self.is_causal:
            # Ensure enough intermediate variables for sampling X and y
            # Total available nodes = hidden_dim * (num_layers - 1), so we can use smaller hidden_dim
            # with more layers while still having enough nodes to sample from
            blocks = max(1, self.num_layers - 1)  # Number of intermediate blocks
            required_nodes = self.num_outputs + 2 * self.num_features
            min_hidden_dim = math.ceil(required_nodes / blocks)
            self.hidden_dim = max(self.hidden_dim, min_hidden_dim)
        else:
            # In non-causal mode, features are the causes
            self.num_causes = self.num_features

        # Define the input sampler
        self.xsampler = XSampler(
            self.seq_len,
            self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self._dev.device,
        )

        # Build layers
        layers = [nn.Linear(self.num_causes, self.hidden_dim)]
        for _ in range(self.num_layers - 1):
            layers.append(self._make_layer_block())
        if not self.is_causal:
            layers.append(self._make_layer_block(is_output=True))
        self.layers = nn.Sequential(*layers).to(device)

        # Initialize layers
        self._init_parameters()
        
        # Set to eval mode for faster generation (no dropout, batch norm updates, etc.)
        self.eval()

    def _make_layer_block(self, is_output: bool = False) -> nn.Sequential:
        """Create activation -> linear -> noise block."""
        out_dim = self.num_outputs if is_output else self.hidden_dim
        # Always call mlp_activations as a factory to get a new instance per layer
        # This matches the original design and ensures each layer gets its own activation module
        activation = self.mlp_activations()
        linear_layer = nn.Linear(self.hidden_dim, out_dim)

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(torch.zeros(size=(1, out_dim), device=self._dev.device), float(self.noise_std))
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)

        return nn.Sequential(activation, linear_layer, noise_layer)

    def _init_parameters(self) -> None:
        """Initialize MLP parameters."""
        for i, (_, param) in enumerate(self.layers.named_parameters()):
            if self.block_wise_dropout and param.dim() == 2:
                self._init_block_dropout(param, i)
            else:
                self._init_normal(param, i)

    def _init_block_dropout(self, param: torch.Tensor, index: int) -> None:
        """Block-wise sparse initialization with safety checks."""
        nn.init.zeros_(param)
        # Safety: ensure n_blocks doesn't exceed dimensions
        max_blocks = min(param.shape[0], param.shape[1])
        if max_blocks < 1:
            max_blocks = 1
        n_blocks = torch.randint(1, min(math.ceil(math.sqrt(max_blocks)) + 1, max_blocks + 1), (1,), device=self._dev.device).item()
        block_size = [max(1, dim // n_blocks) for dim in param.shape]  # Ensure block_size >= 1
        keep_prob = (n_blocks * block_size[0] * block_size[1]) / max(param.numel(), 1)  # Guard against division by zero
        # Clamp keep_prob to reasonable range to prevent extreme std values
        keep_prob = max(min(keep_prob, 1.0), 1e-6)
        for block in range(n_blocks):
            block_slice = tuple(slice(dim * block, min(dim * (block + 1), param.shape[i])) for i, dim in enumerate(block_size))
            # Clamp std to prevent explosion when scaling by dropout compensation
            std = self.init_std / max(keep_prob**0.5 if self.scale_init_std_by_dropout else 1, 1e-6)
            # Clamp std to reasonable range: too small causes vanishing gradients, too large causes explosion
            std = max(min(std, 0.75), 1e-6)  # Clamp between 1e-6 and 0.75
            nn.init.normal_(param[block_slice], std=std)

    def _init_normal(self, param: torch.Tensor, index: int) -> None:
        """Standard normal initialization with He scaling and dropout."""
        if param.dim() == 2:  # Applies only to weights, not biases
            fan_in = param.shape[1]
            # He initialization: scale by 1/sqrt(fan_in) to keep variance stable across layers
            # This prevents variance explosion in deep networks
            # Guard against division by zero or very small fan_in
            if fan_in <= 0:
                warnings.warn(
                    f"Invalid fan_in={fan_in} for parameter shape {param.shape}. "
                    f"Using default fan_in=1 to prevent division by zero.",
                    UserWarning,
                    stacklevel=2
                )
                fan_in = 1
            std = self.init_std / math.sqrt(max(fan_in, 1))
            # Then apply dropout compensation if needed
            dropout_prob = self.mlp_dropout_prob if index > 0 else 0  # No dropout for the first layer's weights
            dropout_prob = min(dropout_prob, 0.99)
            if self.scale_init_std_by_dropout:
                std = std / max((1 - dropout_prob) ** 0.5, 1e-6)  # Guard against division by zero
            # Clamp std to reasonable range: too small causes vanishing gradients, too large causes explosion
            std = max(min(std, 0.75), 1e-6)  # Clamp between 1e-6 and 0.75
            nn.init.normal_(param, std=std)
            # Apply dropout mask: use no_grad() to avoid in-place operation on leaf parameter with grad
            # Use shape-based RNG (torch.rand_like doesn't support generator parameter)
            if dropout_prob > 0:
                with torch.no_grad():
                    mask = (torch.rand(param.shape, device=param.device, dtype=param.dtype) < (1 - dropout_prob)).to(param.dtype)
                    param.mul_(mask)
        elif param.dim() == 1:  # Bias initialization
            # Initialize biases to small values to prevent instability
            # Use a conservative std that's smaller than weight initialization
            bias_std = min(self.init_std * 0.1, 0.1)  # Much smaller than weights
            nn.init.normal_(param, std=bias_std)

    @torch.no_grad()
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic (X, y) data."""
        causes = self.xsampler.sample()  # (seq_len, num_causes)

        # OPTIMIZATION: Avoid list accumulation for better performance
        # Process first layer separately (linear only, no activation)
        x = self.layers[0](causes)
        
        # Collect intermediate outputs only if needed for causal mode
        if self.is_causal:
            outputs = []
            # Process remaining layers and collect outputs
            for layer in self.layers[1:]:
                x = layer(x)
                outputs.append(x)
        else:
            # Non-causal: only need last output, no need to store intermediates
            for layer in self.layers[1:]:
                x = layer(x)
            outputs = [x]  # Only last output needed

        # Handle outputs based on causality
        X, y = self._extract_xy(causes, outputs)

        # Check for NaNs and Infs and raise error to trigger retry mechanism
        # This is cleaner than returning zeros, which creates invalid datasets that get filtered later
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)) or torch.any(torch.isinf(X)) or torch.any(torch.isinf(y)):
            raise RuntimeError("MLPSCM generated NaNs/Infs due to unstable initialization")

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y

    def _extract_xy(
        self, causes: torch.Tensor, outputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features X and targets y from MLP outputs."""
        if self.is_causal:
            # Concatenate all intermediate outputs along feature dimension
            # torch.cat is well-optimized in PyTorch for this use case
            outputs_flat = torch.cat(outputs, dim=-1)
            
            # Joint Selection Strategy (Disjoint Non-Causal Selection)
            # Allow y to be selected from "earlier" nodes (non-causal), but strictly enforce disjoint sets
            if self.use_joint_covariance_sampling or (not self.y_is_effect):
                # Flatten all intermediate outputs
                total_nodes = outputs_flat.shape[1]
                
                # Ensure we have enough nodes
                min_required = self.num_outputs + self.num_features
                if total_nodes < min_required:
                    warnings.warn(
                        f"Insufficient output dimensions: got {total_nodes}, need at least {min_required}. "
                        f"Reducing num_features to fit available dimensions.",
                        UserWarning,
                        stacklevel=2
                    )
                    # Reduce num_features to fit available dimensions
                    # We can't regenerate outputs here, so we adjust num_features
                    reduction = min_required - total_nodes
                    self.num_features = max(1, self.num_features - reduction)
                    warnings.warn(
                        f"Reduced num_features to {self.num_features} to fit available dimensions.",
                        UserWarning,
                        stacklevel=2
                    )
                    # Recalculate min_required with new num_features
                    min_required = self.num_outputs + self.num_features
                
                # Randomly select indices without replacement (Ensures DISJOINT sets)
                available_indices = torch.randperm(total_nodes, device=self._dev.device)
                
                indices_y = available_indices[:self.num_outputs]
                indices_X = available_indices[self.num_outputs : self.num_outputs + self.num_features]
                
                # X and y are disjoint, but y might be 'upstream' of X (Reverse Causality)
                if self.sort_features:
                    indices_X, _ = torch.sort(indices_X)
                    X = outputs_flat[:, indices_X]
                else:
                    X = outputs_flat[:, indices_X]
                y = outputs_flat[:, indices_y]
            else:
                # Original behavior: y from later nodes, no overlap
                # Safety check: ensure enough space for X and y without overlap
                min_required = 2 * self.num_outputs + self.num_features
                if outputs_flat.shape[-1] < min_required:
                    warnings.warn(
                        f"Insufficient output dimensions: got {outputs_flat.shape[-1]}, "
                        f"need at least {min_required} (num_outputs={self.num_outputs}, "
                        f"num_features={self.num_features}, y_is_effect={self.y_is_effect}). "
                        f"Reducing num_features to fit available dimensions.",
                        UserWarning,
                        stacklevel=2
                    )
                    # Reduce num_features to fit available dimensions
                    reduction = min_required - outputs_flat.shape[-1]
                    self.num_features = max(1, self.num_features - reduction)
                    warnings.warn(
                        f"Reduced num_features to {self.num_features} to fit available dimensions.",
                        UserWarning,
                        stacklevel=2
                    )
                    # Recalculate min_required with new num_features
                    min_required = 2 * self.num_outputs + self.num_features
                
                if self.in_clique:
                    # When in_clique=True, features and targets are sampled as a block, ensuring that
                    # selected variables may share dense dependencies.
                    # CRITICAL FIX: When y_is_effect=True, exclude last num_outputs columns to prevent leakage
                    # Ensure clique block cannot overlap with last num_outputs columns used for y
                    # Clique block is [start, start + num_outputs + num_features)
                    # We need: start + num_outputs + num_features <= outputs_flat.shape[-1] - num_outputs
                    max_start = outputs_flat.shape[-1] - self.num_outputs - self.num_features - self.num_outputs
                    max_start = max(0, max_start)  # Ensure non-negative
                    start = torch.randint(0, max_start + 1, (1,), device=self._dev.device).item()
                    
                    random_perm = start + torch.randperm(self.num_outputs + self.num_features, device=self._dev.device)
                else:
                    # Exclude last num_outputs elements (reserved for y when y_is_effect=True)
                    random_perm = torch.randperm(outputs_flat.shape[-1] - self.num_outputs, device=self._dev.device)

                indices_X = random_perm[self.num_outputs : self.num_outputs + self.num_features]
                # If targets are effects, take last output dims from outputs_flat (not from random_perm)
                indices_y = list(range(-self.num_outputs, 0))

                if self.sort_features:
                    indices_X, _ = torch.sort(indices_X)

                # Select input features and targets from outputs
                X = outputs_flat[:, indices_X]
                y = outputs_flat[:, indices_y]
        else:
            # In non-causal mode, use original causes and last layer output
            X = causes
            y = outputs[-1]

        return X, y
