from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .utils import GaussianNoise, XSampler


class ConvSCM(nn.Module):
    """Generates synthetic tabular datasets using a Convolutional Neural Network (CNN) based Structural Causal Model (SCM).
    
    Similar to MLPSCM but uses 1D convolutional layers instead of fully connected layers.
    This creates different dependency patterns where features are transformed through
    localized receptive fields rather than global connections.

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
          intermediate hidden states of the CNN transformation applied to initial causes.
          The `num_causes` parameter controls the number of initial root variables.
        - If `False`, simulates a direct predictive mapping: Initial causes are used
          directly as `X`, and the final output of the CNN becomes `y`. `num_causes`
          is effectively ignored and set equal to `num_features`.

    num_causes : int, default=10
        The number of initial root 'cause' variables sampled by `XSampler`.
        Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
        set to `num_features`.

    y_is_effect : bool, default=True
        Specifies how the target `y` is selected when `is_causal=True`.
        - If `True`, `y` is sampled from the outputs of the final CNN layer(s),
          representing terminal effects in the causal chain.
        - If `False`, `y` is sampled from the earlier intermediate outputs (after
          permutation), representing variables closer to the initial causes.

    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled from the flattened
        intermediate CNN outputs when `is_causal=True`.
        - If `True`, `X` and `y` are selected from a contiguous block of the
          intermediate outputs, potentially creating denser dependencies among them.
        - If `False`, `X` and `y` indices are chosen randomly and independently
          from all available intermediate outputs.

    sort_features : bool, default=True
        Determines whether to sort the features based on their original indices from
        the intermediate CNN outputs. Only relevant when `is_causal=True`.

    num_layers : int, default=10
        The total number of convolutional layers in the network. Must be >= 2.
        Includes the initial conv layer and subsequent blocks of
        (Activation -> Conv1d -> Noise).

    hidden_channels : int, default=20
        The number of output channels for convolutional layers (analogous to hidden_dim in MLP).
        If `is_causal=True`, this is automatically increased if needed to ensure enough
        intermediate variables are generated for sampling `X` and `y`.

    kernel_size : int, default=3
        The size of the convolutional kernel. Determines the receptive field for each layer.

    stride : int, default=1
        The stride of the convolution. Use values > 1 to reduce sequence length.

    padding : int or str, default="same"
        Padding added to the input. Use "same" to maintain sequence length, or an integer.

    conv_activations : default=nn.Tanh
        The activation function to be used after each convolutional transformation
        (except the first).

    init_std : float, default=1.0
        The standard deviation of the normal distribution used for initializing
        the weights of the convolutional layers.

    block_wise_dropout : bool, default=True
        Specifies the weight initialization strategy.
        - If `True`, uses a 'block-wise dropout' initialization where only random
          blocks within the weight tensor are initialized with values drawn from
          a normal distribution, while the rest are zero. This encourages sparsity.
        - If `False`, uses standard normal initialization for all weights, followed
          by applying dropout mask based on `conv_dropout_prob`.

    conv_dropout_prob : float, default=0.1
        The dropout probability applied to weights during *standard* initialization
        (i.e., when `block_wise_dropout=False`). Ignored if
        `block_wise_dropout=True`. The probability is clamped between 0 and 0.99.

    scale_init_std_by_dropout : bool, default=True
        Whether to scale the `init_std` during weight initialization to compensate
        for the variance reduction caused by dropout.

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
        The base standard deviation for the Gaussian noise added after each
        convolutional layer's transformation (except the first layer).

    pre_sample_noise_std : bool, default=False
        Controls how the standard deviation for the `GaussianNoise` layers is determined.

    use_joint_covariance_sampling : bool, default=False
        If `True`, allows joint selection of X and y from all intermediate outputs,
        potentially creating non-causal relationships.

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
        hidden_channels: int = 20,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | str = "same",
        conv_activations: Any = nn.Tanh,
        init_std: float = 1.0,
        block_wise_dropout: bool = True,
        conv_dropout_prob: float = 0.1,
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

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_activations = conv_activations
        
        # SAFETY FIX: Clamp init_std to prevent explosion
        self.init_std = min(init_std, 0.5)
        
        self.block_wise_dropout = block_wise_dropout
        self.conv_dropout_prob = conv_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        
        # Register buffer to track device
        self.register_buffer("_dev", torch.empty(0, device=device))

        if self.is_causal:
            # Ensure enough intermediate variables for sampling X and y
            # Each layer produces hidden_channels outputs per sample
            blocks = max(1, self.num_layers - 1)
            required_nodes = self.num_outputs + 2 * self.num_features
            min_hidden_channels = math.ceil(required_nodes / blocks)
            self.hidden_channels = max(self.hidden_channels, min_hidden_channels)
            
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
        # First conv layer: in_channels=1, out_channels=hidden_channels
        # Input shape: (batch, 1, num_causes) treating each sample as 1D sequence
        # Use integer padding to avoid 'same' warning with even kernel sizes
        first_kernel = min(self.kernel_size, self.num_causes)
        first_padding = self._resolve_padding(first_kernel, self.stride)
        layers = [nn.Conv1d(1, self.hidden_channels, kernel_size=first_kernel, 
                           stride=self.stride, padding=first_padding)]
        
        for _ in range(self.num_layers - 1):
            layers.append(self._make_layer_block())
        
        if not self.is_causal:
            layers.append(self._make_layer_block(is_output=True))

        self.layers = nn.Sequential(*layers).to(device)
        # Initialize layers
        self._init_parameters()

        # Set to eval mode for faster generation
        self.eval()

    def _make_layer_block(self, is_output: bool = False) -> nn.Sequential:
        """Create activation -> conv1d -> noise block."""
        #print('LOLOLOL')
        out_channels = self.num_outputs if is_output else self.hidden_channels
        
        # Always call conv_activations as a factory to get a new instance per layer
        activation = self.conv_activations()
        # Use configured stride and resolve padding (avoid 'same' warning with even kernels)
        padding = self._resolve_padding(self.kernel_size, self.stride)
        conv_layer = nn.Conv1d(self.hidden_channels, out_channels, 
                               kernel_size=self.kernel_size, 
                               stride=self.stride,
                               padding=padding)

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(torch.zeros(size=(1, out_channels), device=self._dev.device), float(self.noise_std))
            )
        else:
            noise_std = self.noise_std

        noise_layer = GaussianNoise(noise_std)

        return nn.Sequential(activation, conv_layer, noise_layer)

    def _resolve_padding(self, kernel_size: int, stride: int) -> int:
        """Resolve padding for Conv1d layers.

        Note: For stride > 1, true 'same' padding depends on input length,
        so we fall back to symmetric padding.
        """
        if self.padding == "same":
            return kernel_size // 2
        if isinstance(self.padding, int):
            return self.padding
        raise ValueError(f"Invalid padding: {self.padding}. Use 'same' or int.")

    def _init_parameters(self) -> None:
        """Initialize CNN parameters."""
        for i, (_, param) in enumerate(self.layers.named_parameters()):
            # Keep the fixed parameters sampled by activation modules.
            if not param.requires_grad:
                continue
            if self.block_wise_dropout and param.dim() >= 2:
                self._init_block_dropout(param, i)
            else:
                self._init_normal(param, i)

    def _init_block_dropout(self, param: torch.Tensor, index: int) -> None:
        """Block-wise sparse initialization with safety checks."""
        nn.init.zeros_(param)
        
        # For conv weights, shape is (out_channels, in_channels, kernel_size)
        # We'll apply block dropout across the channel dimensions
        if param.dim() >= 2:
            max_blocks = min(param.shape[0], param.shape[1])
            if max_blocks < 1:
                max_blocks = 1
            n_blocks = torch.randint(1, min(math.ceil(math.sqrt(max_blocks)) + 1, max_blocks + 1), 
                                    (1,), device=self._dev.device).item()
            block_size = [max(1, param.shape[i] // n_blocks) for i in range(2)]
            last_block_size = [param.shape[i] - (n_blocks - 1) * block_size[i] for i in range(2)]
            kept_pairs = (n_blocks - 1) * block_size[0] * block_size[1] + last_block_size[0] * last_block_size[1]
            keep_prob = kept_pairs / max(param.shape[0] * param.shape[1], 1)
            keep_prob = max(min(keep_prob, 1.0), 1e-6)
            
            for block in range(n_blocks):
                # Slice channels only; the last block includes any remainder.
                start_0 = min(block_size[0] * block, param.shape[0])
                end_0 = param.shape[0] if block == n_blocks - 1 else block_size[0] * (block + 1)
                start_1 = min(block_size[1] * block, param.shape[1])
                end_1 = param.shape[1] if block == n_blocks - 1 else block_size[1] * (block + 1)
                
                std = self.init_std / max(keep_prob**0.5 if self.scale_init_std_by_dropout else 1, 1e-6)
                std = max(min(std, 0.75), 1e-6)
                
                if param.dim() == 2:
                    nn.init.normal_(param[start_0:end_0, start_1:end_1], std=std)
                elif param.dim() == 3:
                    nn.init.normal_(param[start_0:end_0, start_1:end_1, :], std=std)

    def _init_normal(self, param: torch.Tensor, index: int) -> None:
        """Standard normal initialization with He scaling and dropout."""
        #print(param.shape)
        if param.dim() >= 2:  # Applies to weights, not biases
            # For conv layers, fan_in = in_channels * kernel_size
            if param.dim() == 3:  # Conv1d weights: (out_channels, in_channels, kernel_size)
                fan_in = param.shape[1] * param.shape[2]
            else:  # 2D weights
                fan_in = param.shape[1]
            
            if fan_in <= 0:
                warnings.warn(
                    f"Invalid fan_in={fan_in} for parameter shape {param.shape}. "
                    f"Using default fan_in=1 to prevent division by zero.",
                    UserWarning,
                    stacklevel=2
                )
                fan_in = 1
            
            std = self.init_std / math.sqrt(max(fan_in, 1))
            
            # Apply dropout compensation
            dropout_prob = self.conv_dropout_prob if index > 0 else 0
            dropout_prob = min(dropout_prob, 0.99)
            if self.scale_init_std_by_dropout:
                std = std / max((1 - dropout_prob) ** 0.5, 1e-6)
            
            std = max(min(std, 0.75), 1e-6)
            nn.init.normal_(param, std=std)
            
            # Apply dropout mask
            if dropout_prob > 0:
                with torch.no_grad():
                    mask = (torch.rand(param.shape, device=param.device, dtype=param.dtype) < (1 - dropout_prob)).to(param.dtype)
                    param.mul_(mask)
        elif param.dim() == 1:  # Bias initialization
            bias_std = min(self.init_std * 0.1, 0.1)
            nn.init.normal_(param, std=bias_std)

    def _safe_conv1d(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """Apply Conv1d with dynamic padding for small inputs."""
        kernel = conv.kernel_size[0]
        if x.shape[-1] < kernel:
            pad = kernel - x.shape[-1]
            left = pad // 2
            right = pad - left
            x = F.pad(x, (left, right))
        return conv(x)

    def _apply_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply a layer or a Sequential block with safe conv handling."""
        if isinstance(layer, nn.Conv1d):
            return self._safe_conv1d(layer, x)
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                x = self._apply_layer(sub_layer, x)
            return x
        return layer(x)

    @torch.no_grad()
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic (X, y) data."""
        causes = self.xsampler.sample()  # (seq_len, num_causes)
        #print(causes.shape)
        # Reshape for Conv1d: (batch, channels, sequence)
        # Treat each sample as a 1D sequence with 1 input channel
        x = causes.unsqueeze(1) # (seq_len, 1, num_causes)
        
        # Process first layer separately
        x = self._apply_layer(self.layers[0], x)
        
        # Collect intermediate outputs only if needed for causal mode
        if self.is_causal:
            outputs = []
            # Process remaining layers and collect outputs
            for layer in self.layers[1:]:
                x = self._apply_layer(layer, x)
                outputs.append(x)
        else:
            # Non-causal: only need last output
            for layer in self.layers[1:]:
                x = self._apply_layer(layer, x)
            outputs = [x]

        # Handle outputs based on causality
        X, y = self._extract_xy(causes, outputs)

        # Check for NaNs and Infs
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)) or torch.any(torch.isinf(X)) or torch.any(torch.isinf(y)):
            raise RuntimeError("ConvSCM generated NaNs/Infs due to unstable initialization")

        if self.num_outputs == 1:
            y = y.squeeze(-1)
        return X, y

    def _extract_xy(
        self, causes: torch.Tensor, outputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features X and targets y from CNN outputs."""
        if self.is_causal:
            # Flatten spatial dimension: average pool across sequence length
            # Then concatenate channel outputs from all layers
            outputs_pooled = []
            for out in outputs:
                # out shape: (seq_len, channels, spatial_dim)
                # Global average pooling across spatial dimension
                pooled = torch.mean(out, dim=-1)  # (seq_len, channels)
                outputs_pooled.append(pooled)
            
            # Concatenate all intermediate outputs along feature dimension
            outputs_flat = torch.cat(outputs_pooled, dim=-1)
            
            # Joint Selection Strategy (similar to MLPSCM)
            if self.use_joint_covariance_sampling or (not self.y_is_effect):
                total_nodes = outputs_flat.shape[1]
                min_required = self.num_outputs + self.num_features
                
                if total_nodes < min_required:
                    warnings.warn(
                        f"Insufficient output dimensions: got {total_nodes}, need at least {min_required}. "
                        f"Reducing num_features to fit available dimensions.",
                        UserWarning,
                        stacklevel=2
                    )
                    reduction = min_required - total_nodes
                    self.num_features = max(1, self.num_features - reduction)
                    min_required = self.num_outputs + self.num_features
                
                # Randomly select indices without replacement (disjoint sets)
                available_indices = torch.randperm(total_nodes, device=self._dev.device)
                
                indices_y = available_indices[:self.num_outputs]
                indices_X = available_indices[self.num_outputs : self.num_outputs + self.num_features]
                
                if self.sort_features:
                    indices_X, _ = torch.sort(indices_X)
                    X = outputs_flat[:, indices_X]
                else:
                    X = outputs_flat[:, indices_X]
                y = outputs_flat[:, indices_y]
            else:
                # Original behavior: y from later nodes, no overlap
                min_required = 2 * self.num_outputs + self.num_features
                if outputs_flat.shape[-1] < min_required:
                    warnings.warn(
                        f"Insufficient output dimensions: got {outputs_flat.shape[-1]}, "
                        f"need at least {min_required}. Reducing num_features to fit.",
                        UserWarning,
                        stacklevel=2
                    )
                    reduction = min_required - outputs_flat.shape[-1]
                    self.num_features = max(1, self.num_features - reduction)
                    min_required = 2 * self.num_outputs + self.num_features
                
                if self.in_clique:
                    # Clique sampling: contiguous block
                    max_start = outputs_flat.shape[-1] - self.num_outputs - self.num_features - self.num_outputs
                    max_start = max(0, max_start)
                    start = torch.randint(0, max_start + 1, (1,), device=self._dev.device).item()
                    
                    random_perm = start + torch.randperm(self.num_outputs + self.num_features, device=self._dev.device)
                else:
                    # Random sampling excluding last num_outputs
                    random_perm = torch.randperm(outputs_flat.shape[-1] - self.num_outputs, device=self._dev.device)

                indices_X = random_perm[self.num_outputs : self.num_outputs + self.num_features]
                indices_y = list(range(-self.num_outputs, 0))

                if self.sort_features:
                    indices_X, _ = torch.sort(indices_X)

                X = outputs_flat[:, indices_X]
                y = outputs_flat[:, indices_y]
        else:
            # In non-causal mode, use original causes and last layer output
            X = causes
            # Pool the last output
            y = torch.mean(outputs[-1], dim=-1)  # (seq_len, channels)

        return X, y
