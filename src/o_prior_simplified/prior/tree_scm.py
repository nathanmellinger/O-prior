from __future__ import annotations

import random
import warnings
from typing import Optional
import numpy as np
import torch
from torch import nn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

from .utils import GaussianNoise, XSampler

# cuML (RAPIDS) imports for GPU-accelerated tree models
try:
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.ensemble import ExtraTreesRegressor as cuET
    try:
        from cuml.tree import DecisionTreeRegressor as cuDT
        CUML_DT_AVAILABLE = True
    except ImportError:
        cuDT = None
        CUML_DT_AVAILABLE = False
    import cupy as cp
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    cuRF = None
    cuET = None
    cuDT = None
    cp = None


class GPUDecisionTreeLayer(nn.Module):
    """
    Simulates a Random Forest entirely on the GPU without training.
    
    Instead of fitting a tree to learn splits, we:
    1. Randomly select features.
    2. Randomly select thresholds (cuts).
    3. 'Route' samples to leaves using Boolean masks.
    4. Assign random values to leaves.
    
    This preserves the 'step-function' inductive bias of trees but runs 100x faster.
    """
    def __init__(
        self, 
        num_features: int, 
        out_dim: int, 
        n_estimators: int = 10, 
        max_depth: int = 4, 
        device: str = "cpu",
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.num_features = num_features
        self.out_dim = out_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.device = device
        
        # Set random seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # 1. Feature Selection: Which feature does each node split on?
        # Shape: (n_estimators, max_depth)
        self.register_buffer(
            "feature_indices",
            torch.randint(0, num_features, (n_estimators, max_depth), device=device)
        )
        
        # 2. Thresholds: Random cuts in the normalized input space [-3, 3]
        # (assuming standard normal inputs)
        self.register_buffer(
            "thresholds",
            torch.randn(n_estimators, max_depth, device=device)
        )
        
        # 3. Leaf Values: Each path (2^max_depth paths) gets a random weight
        num_leaves = 2 ** max_depth
        self.register_buffer(
            "leaf_values",
            torch.randn(n_estimators, num_leaves, out_dim, device=device) / np.sqrt(n_estimators)
        )
        
        # Powers of 2 for calculating leaf indices [1, 2, 4, 8...]
        self.register_buffer(
            "path_multipliers", 
            2 ** torch.arange(max_depth, device=device)
        )

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            Input features tensor of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Transformed features tensor of shape (batch, out_dim).
        """
        batch_size = X.shape[0]
        
        # Handle NaNs
        X = X.nan_to_num(0.0)
        
        # 1. Gather feature values for splits
        # Expand X to (batch, n_estimators, num_features)
        X_expanded = X.unsqueeze(1).expand(-1, self.n_estimators, -1)
        
        # Gather specific features
        # indices expanded: (batch, n_estimators, max_depth)
        indices_expanded = self.feature_indices.unsqueeze(0).expand(batch_size, -1, -1)
        
        # selected_features: (batch, n_estimators, max_depth)
        selected_features = torch.gather(X_expanded, 2, indices_expanded)
        
        # 2. Compare with thresholds to get routing decision (0 or 1)
        # decisions: (batch, n_estimators, max_depth)
        decisions = (selected_features > self.thresholds).long()
        
        # 3. Compute Leaf Index
        # We treat the sequence of 0/1 decisions as a binary number
        # leaf_indices: (batch, n_estimators)
        leaf_indices = (decisions * self.path_multipliers).sum(dim=2)
        
        # 4. Retrieve Leaf Values
        # Offset indices by tree index to gather from correct tree
        num_leaves = self.leaf_values.shape[1]
        tree_offsets = (torch.arange(self.n_estimators, device=self.device) * num_leaves).unsqueeze(0)
        
        # final_gather_indices: (batch, n_estimators)
        final_gather_indices = leaf_indices + tree_offsets
        
        # output_per_tree: (batch, n_estimators, out_dim)
        # Flatten for gather
        values_flat = self.leaf_values.view(-1, self.out_dim)
        output_per_tree = values_flat[final_gather_indices.view(-1)].view(
            batch_size, self.n_estimators, self.out_dim
        )
        
        # 5. Ensemble: Sum across trees
        output = output_per_tree.sum(dim=1)
        
        if self.out_dim == 1:
            output = output.view(-1, 1)
        
        return output


class DSRFLayer(nn.Module):
    """
    Directly Sampled Random Forest - constructs random p(y|x) via random splits.
    No model fitting required, just random split indices and thresholds.
    """
    def __init__(
        self,
        num_features: int,
        out_dim: int,
        n_estimators: int = 10,
        max_depth: int = 4,
        device: str = "cpu",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.out_dim = out_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.device = device
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # Random split indices and thresholds (sampled once, reused)
        self.register_buffer(
            "feature_indices",
            torch.randint(0, num_features, (n_estimators, max_depth), device=device)
        )
        self.register_buffer(
            "thresholds",
            torch.randn(n_estimators, max_depth, device=device) * 2  # Wider range
        )
        
        # Random leaf values
        num_leaves = 2 ** max_depth
        self.register_buffer(
            "leaf_values",
            torch.randn(n_estimators, num_leaves, out_dim, device=device) / np.sqrt(n_estimators)
        )
        self.register_buffer(
            "path_multipliers",
            2 ** torch.arange(max_depth, device=device)
        )
    
    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply DSRF transformation without fitting."""
        batch_size = X.shape[0]
        X = X.nan_to_num(0.0)
        
        # Route through random splits
        X_expanded = X.unsqueeze(1).expand(-1, self.n_estimators, -1)
        indices_expanded = self.feature_indices.unsqueeze(0).expand(batch_size, -1, -1)
        selected_features = torch.gather(X_expanded, 2, indices_expanded)
        decisions = (selected_features > self.thresholds).long()
        leaf_indices = (decisions * self.path_multipliers).sum(dim=2)
        
        # Retrieve leaf values
        num_leaves = self.leaf_values.shape[1]
        tree_offsets = (torch.arange(self.n_estimators, device=self.device) * num_leaves).unsqueeze(0)
        final_gather_indices = leaf_indices + tree_offsets
        values_flat = self.leaf_values.view(-1, self.out_dim)
        output_per_tree = values_flat[final_gather_indices.view(-1)].view(
            batch_size, self.n_estimators, self.out_dim
        )
        
        output = output_per_tree.sum(dim=1)
        if self.out_dim == 1:
            output = output.view(-1, 1)
        return output


class TreeLayer(nn.Module):
    """ A layer that transforms input features using a tree-based model.

    Inspired from : https://github.com/soda-inria/tabicl/blob/main/src/tabicl/prior/tree_scm.py
    
    Supports GPU-accelerated training via cuML (RAPIDS) when available and on GPU,
    providing 10-50x speedup over sklearn. Automatically falls back to sklearn
    if cuML is unavailable or on CPU.
    
    Parameters
    ----------
    tree_model : str
        The type of tree-based model to use. Options are "decision_tree",
        "extra_trees", "random_forest", "xgboost", "dsrf".

    max_depth : int
        The maximum depth allowed for the individual trees in the model.

    n_estimators : int
        The number of trees in the ensemble.

    out_dim : int
        The desired output dimension for the transformed features. This determines
        the number of target variables (`y_fake`) generated for fitting the
        multi-output regressor.

    device : str or torch.device
        The device ('cpu' or 'cuda') on which to place the output tensor.
    
    use_cuml : bool, default=True
        Whether to use cuML (RAPIDS) for GPU-accelerated tree training when available.
        Automatically disabled if cuML is unavailable or device is CPU.
        Falls back to sklearn on any cuML error.
    """

    def __init__(self, tree_model: str, max_depth: int, n_estimators: int, out_dim: int, device: str, n_jobs: int = 1, random_state: Optional[int] = None, use_cuml: bool = True):
        super(TreeLayer, self).__init__()
        self.tree_model = tree_model
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.out_dim = out_dim
        self.device = device
        
        # Generate random seed if not provided (for diversity across datasets)
        # This ensures reproducibility when random_state is set, but allows diversity when None
        if random_state is None:
            random_state = np.random.randint(0, 2**31)
        self.random_state = random_state

        # Auto-detect cuML availability: only use cuML if available, requested, and on GPU
        self.use_cuml = use_cuml and CUML_AVAILABLE and device != "cpu"

        if tree_model == "dsrf":
            # DSRF doesn't use sklearn - it's directly sampled
            # num_features will be set in forward() based on input
            self.model = None
            self.dsrf_layer = None  # Will be created in forward() when we know num_features
        elif tree_model == "decision_tree":
            if self.use_cuml and CUML_DT_AVAILABLE:
                # cuML DecisionTreeRegressor - GPU-accelerated
                self.model = cuDT(
                    max_depth=max_depth,
                    random_state=random_state,
                )
            else:
                # sklearn DecisionTreeRegressor doesn't support native multi-output, use wrapper
                self.model = MultiOutputRegressor(
                    DecisionTreeRegressor(max_depth=max_depth, splitter="random", random_state=random_state), 
                    n_jobs=n_jobs
                )
        elif tree_model == "extra_trees":
            if self.use_cuml:
                # cuML ExtraTreesRegressor - GPU-accelerated
                self.model = cuET(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=random_state,
                    n_streams=1,  # cuML parameter for parallelism
                )
            else:
                # sklearn ExtraTreesRegressor supports native multi-output (no wrapper needed)
                self.model = ExtraTreesRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=random_state,
                    n_jobs=n_jobs
                )
        elif tree_model == "random_forest":
            if self.use_cuml:
                # cuML RandomForestRegressor - GPU-accelerated
                self.model = cuRF(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=random_state,
                    n_streams=1,  # cuML parameter for parallelism
                )
            else:
                # sklearn RandomForestRegressor supports native multi-output (no wrapper needed)
                self.model = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=random_state,
                    n_jobs=n_jobs
                )
        elif tree_model == "xgboost":
            # Note: XGBoost multi-output trees (multi_strategy="multi_output_tree") do NOT support GPU
            # Error: "GPU is not yet supported for vector leaf"
            # Therefore, we must use CPU for multi-output trees, even if CUDA is available
            # For single-output trees, we can use GPU if available
            
            # Always use "hist" tree method
            tree_method = "hist"
            
            try:
                # Try native multi-output (XGBoost 1.6+)
                # CRITICAL: Multi-output trees require CPU - GPU is not supported
                self.model = XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    tree_method=tree_method,
                    device="cpu",  # Force CPU for multi-output trees
                    multi_strategy="multi_output_tree",
                    n_jobs=n_jobs,
                    random_state=random_state,
                )
            except (TypeError, ValueError) as e:
                # Fallback for older XGBoost versions or if multi_strategy not supported
                import warnings
                warnings.warn(
                    f"XGBoost native multi-output not available, using MultiOutputRegressor wrapper: {e}",
                    UserWarning
                )
                # Determine device for single-output case
                if "cuda" in str(device):
                    device_arg = "cuda"
                else:
                    device_arg = "cpu"
                
                if out_dim == 1:
                    # Single output: no wrapper needed, can use GPU if available
                    self.model = XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        tree_method=tree_method,
                        device=device_arg,
                        n_jobs=n_jobs,
                        random_state=random_state,
                    )
                else:
                    # Multi-output: wrap in MultiOutputRegressor
                    # Note: Even with wrapper, multi-output requires CPU
                    self.model = MultiOutputRegressor(
                        XGBRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            tree_method=tree_method,
                            device="cpu",  # Force CPU for multi-output
                            n_jobs=1,  # Use 1 to avoid nested parallelism issues
                            random_state=random_state,
                        ),
                        n_jobs=n_jobs,
                    )
        else:
            raise ValueError(f"Invalid tree model: {tree_model}")

    @torch.no_grad()
    def forward(self, X):
        """Applies the fitted tree-based transformation to the input features.

        For DSRF: Directly samples without fitting (GPU-native, fastest).
        For cuML models: Fits on GPU using cuML (10-50x faster than sklearn).
        For sklearn models: Fits on CPU (slower, but reliable fallback).

        NOTE: Non-DSRF models train tree models from scratch on each call. With cuML,
        this is 10-50x faster than sklearn, but still slower than DSRF which requires no training.

        Parameters
        ----------
        X : torch.Tensor
            Input features tensor of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Transformed features tensor of shape (n_samples, out_dim).
        """
        if self.tree_model == "dsrf":
            # DSRF: Directly sample without fitting
            num_features = X.shape[1]
            # Create DSRF layer if not exists or if num_features changed
            if self.dsrf_layer is None or self.dsrf_layer.num_features != num_features:
                self.dsrf_layer = DSRFLayer(
                    num_features=num_features,
                    out_dim=self.out_dim,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    device=X.device,  # Use input device
                    random_state=self.random_state,
                )
                self.dsrf_layer = self.dsrf_layer.to(X.device)
            return self.dsrf_layer(X)
        
        # For other tree models: fit and predict
        # Try cuML path first if available, fallback to sklearn
        if self.use_cuml:
            try:
                # cuML path: GPU-accelerated tree training
                # Convert PyTorch tensor to cupy array (stays on GPU)
                X_np = X.nan_to_num(0.0).cpu().numpy()
                X_cp = cp.asarray(X_np)
                
                # Generate fake targets on GPU
                cp.random.seed(self.random_state)
                y_fake_cp = cp.random.randn(X.shape[0], self.out_dim)
                
                # cuML models support multi-output natively (cuML 23.06+)
                # For single output, cuML may return 1D or 2D depending on version
                if self.out_dim == 1:
                    # Single output: cuML may return 1D or 2D, handle both
                    self.model.fit(X_cp, y_fake_cp.ravel())
                    y_cp = self.model.predict(X_cp)
                    # Ensure 2D shape for consistency
                    if y_cp.ndim == 1:
                        y_cp = y_cp.reshape(-1, 1)
                else:
                    # Multi-output: cuML handles natively
                    self.model.fit(X_cp, y_fake_cp)
                    y_cp = self.model.predict(X_cp)
                    # Ensure 2D shape
                    if y_cp.ndim == 1:
                        y_cp = y_cp.reshape(-1, self.out_dim)
                
                # Convert back to PyTorch tensor
                y = torch.as_tensor(y_cp, device=self.device, dtype=torch.float32)
                
                if self.out_dim == 1:
                    y = y.view(-1, 1)
                
                return y
            except Exception as e:
                # Fallback to sklearn on any cuML error (memory, compatibility, etc.)
                import warnings
                warnings.warn(
                    f"cuML failed, falling back to sklearn: {e}",
                    UserWarning
                )
                # Continue to sklearn path below
        
        # sklearn path: CPU-based tree training (fallback or when cuML unavailable)
        X = X.nan_to_num(0.0).cpu()
        y_fake = np.random.randn(X.shape[0], self.out_dim)
        # NOTE: This is the slow operation - training tree models on-the-fly
        # sklearn expects different shapes:
        # - MultiOutputRegressor: always expects 2D (n_samples, n_outputs), even for single output
        # - Native multi-output (ExtraTrees, RandomForest): 1D for single output, 2D for multi-output
        # - XGBoost: depends on multi_strategy, but generally 2D for multi-output
        if isinstance(self.model, MultiOutputRegressor):
            # MultiOutputRegressor always expects 2D, even for out_dim == 1
            # y_fake is already (n_samples, out_dim), so keep it as is
            pass
        elif self.out_dim == 1:
            # For native multi-output regressors, single output should be 1D
            y_fake = y_fake.ravel()  # Convert (n_samples, 1) to (n_samples,)
        # For multi-output (out_dim > 1), y_fake is already correct shape (n_samples, out_dim)
        self.model.fit(X, y_fake)
        y = self.model.predict(X)
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        if self.out_dim == 1:
            y = y.view(-1, 1)

        return y


class TreeSCM(nn.Module):
    """A Tree-based Structural Causal Model for generating synthetic datasets.
    Similar to MLP-based SCM but uses tree-based models (like Random Forests or XGBoost)
    for potentially non-linear feature transformations instead of linear layers.

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
          intermediate outputs of the tree transformations applied to initial causes.
          The `num_causes` parameter controls the number of initial root variables.
        - If `False`, simulates a direct predictive mapping: Initial causes are used
          directly as `X`, and the final output of the tree layers becomes `y`. `num_causes`
          is effectively ignored and set equal to `num_features`.

    num_causes : int, default=10
        The number of initial root 'cause' variables sampled by `XSampler`.
        Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
        set to `num_features`.

    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled from the flattened
        intermediate tree outputs when `is_causal=True`.
        - If `True`, `X` and `y` are selected from a contiguous block of the
          intermediate outputs, potentially creating denser dependencies among them.
        - If `False`, `X` and `y` indices are chosen randomly and independently
          from all available intermediate outputs.

    sort_features : bool, default=True
        Determines whether to sort the features based on their original indices from
        the intermediate tree outputs. Only relevant when `is_causal=True`.

    num_layers : int, default=5
        Number of tree transformation layers.

    hidden_dim : int, default=10
        Output dimension size for intermediate tree transformations.

    tree_model : str, default="xgboost"
        Type of tree model to use. Options:
        - "decision_tree": Axis-aligned discontinuities, piecewise constant (DT weight: 0.08)
        - "extra_trees": Random split thresholds for distinctiveness (ET weight: 0.15)
        - "random_forest": Bagged trees with feature randomness, maximum diversity (RF weight: 0.05)
        - "xgboost": Gradient boosting, additive forward stage-wise fitting (GB weight: 0.10)
        - "dsrf": Directly sampled random forest, no model fitting (DSRF weight: 0.02)
        
        Per MITRA rationale: Tree priors provide distinctiveness and decision-boundary structure
        that SCM-only doesn't cover well. Weights reflect performance + diversity + distinctiveness.

    max_depth_lambda : float, default=0.5
        Lambda parameter for sampling the max_depth for tree models from an exponential distribution.

    n_estimators_lambda : float, default=0.5
        Lambda parameter for sampling the number of estimators (trees) per layer from an exponential distribution.

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
        The base standard deviation for the Gaussian noise added after each tree
        layer's transformation.

    pre_sample_noise_std : bool, default=False
        Controls how the standard deviation for the `GaussianNoise` layers is determined.
        If `True`, the noise standard deviation for each output dimension of a layer
        is sampled from a normal distribution centered at 0 with `noise_std`.
        If `False`, a fixed `noise_std` is used for all dimensions.

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
        in_clique: bool = False,
        sort_features: bool = True,
        num_layers: int = 5,
        hidden_dim: int = 10,
        tree_model: str = "xgboost",
        max_depth_lambda: float = 0.5,
        n_estimators_lambda: float = 0.5,
        sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        use_joint_covariance_sampling: bool = False,
        fast_mode: bool = False,
        device: str = "cpu",
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        use_cuml: bool = True,  # Use cuML (RAPIDS) for GPU-accelerated tree training when available
        **kwargs,
    ):
        super(TreeSCM, self).__init__()
        # Respect user-provided parameters and curriculum
        # If performance is a concern, users can set is_causal=False in config
        # No forced overrides

        # Data Generation Settings
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.use_joint_covariance_sampling = use_joint_covariance_sampling
        self.fast_mode = fast_mode
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_cuml = use_cuml  # Use cuML for GPU-accelerated tree training
        
        # In fast_mode, force DSRF to avoid slow fitting
        if self.fast_mode:
            self.tree_model = "dsrf"
        else:
            self.tree_model = tree_model

        # TabICL-style simplifications for regular trees (not DSRF)
        # Useing fewer layers, smaller dims, and capped depth/estimators for speed
        if not self.fast_mode and self.tree_model != "dsrf":
            # Cap layers and dims for speed (TabICL approach)
            self.num_layers = min(self.num_layers, 3)  # Cap at 3 
            self.hidden_dim = min(self.hidden_dim, 10)  # Cap at 10 
            
        self.tree_depth_lambda = max_depth_lambda
        self.tree_n_estimators_lambda = n_estimators_lambda
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.device = device
        self.n_jobs = n_jobs
        self.random_state = random_state

        if self.is_causal:
            # Ensure enough intermediate variables for sampling X and y
            self.hidden_dim = max(self.hidden_dim, self.num_outputs + 2 * self.num_features)
        else:
            # In non-causal mode, features are the causes
            self.num_causes = self.num_features

        # Define the input sampler
        self.xsampler = XSampler(
            seq_len=self.seq_len,
            num_features=self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        # Build layers
        max_depth = 2 + int(np.random.exponential(1 / self.tree_depth_lambda))
        n_estimators = 1 + int(np.random.exponential(1 / self.tree_n_estimators_lambda))
        
        # TabICL-style caps for regular trees (not DSRF)
        # Capping depth at 4 and estimators at 4 for speed
        if not self.fast_mode and self.tree_model != "dsrf":
            max_depth = min(max_depth, 4)  # Cap at 4 
            n_estimators = min(n_estimators, 4)  # Cap at 4 

        layers = [
            TreeLayer(
                tree_model=self.tree_model,
                max_depth=max_depth,
                n_estimators=n_estimators,
                out_dim=self.hidden_dim,
                device=self.device,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                use_cuml=self.use_cuml,
            )
        ]
        for _ in range(self.num_layers - 1):
            layers.append(self.generate_layer_modules())
        if not self.is_causal:
            layers.append(self.generate_layer_modules(is_output_layer=True))
        self.layers = nn.Sequential(*layers).to(device)
        
        # Set to eval mode for faster generation
        self.eval()

    def generate_layer_modules(self, is_output_layer=False):
        """Generates a layer module with activation, tree-based transformation, and noise."""
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim

        max_depth = 2 + int(np.random.exponential(1 / self.tree_depth_lambda))
        n_estimators = 1 + int(np.random.exponential(1 / self.tree_n_estimators_lambda))
        
        max_depth = min(max_depth, 4)  # Cap at 4 
        n_estimators = min(n_estimators, 4)  # Cap at 4

        # Use actual TreeLayer with specified tree_model
        tree_layer = TreeLayer(
            tree_model=self.tree_model,
            max_depth=max_depth,
            n_estimators=n_estimators,
            out_dim=out_dim,
            device=self.device,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            use_cuml=self.use_cuml,
        )

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(torch.zeros(size=(1, out_dim), device=self.device), float(self.noise_std))
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)

        return nn.Sequential(tree_layer, noise_layer)

    @torch.no_grad()
    def forward(self):
        """Generates synthetic data by sampling input features and applying tree-based transformations."""
        causes = self.xsampler.sample()  # (seq_len, num_causes)

        # Generate outputs through tree layers
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        # Skip the initial causes, keep all tree layer outputs
        outputs = outputs[1:]

        # Handle outputs based on causality
        X, y = self.handle_outputs(causes, outputs)

        # Check for NaNs/Infs and raise error to trigger retry mechanism
        # This is cleaner than returning zeros, which creates invalid datasets that get filtered later
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)) or torch.any(torch.isinf(X)) or torch.any(torch.isinf(y)):
            raise RuntimeError("TreeSCM generated NaNs/Infs due to unstable initialization")

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y

    def handle_outputs(self, causes, outputs):
        """
        Handles outputs based on whether causal or not.

        If causal, sample inputs and target from the graph.
        If not causal, directly use causes as inputs and last output as target.

        Parameters
        ----------
        causes : torch.Tensor
            Causes of shape (seq_len, num_causes)

        outputs : list of torch.Tensor
            List of output tensors from MLP layers

        Returns
        -------
        X : torch.Tensor
            Input features (seq_len, num_features)

        y : torch.Tensor
            Target (seq_len, num_outputs)
        """
        if self.is_causal:
            # CAUSAL INTEGRITY FIX:
            # Instead of flattening everything and mixing indiscriminately, we respect the layer structure.
            # outputs is a list: [causes, layer1_out, layer2_out, ..., layerN_out]
            
            # Joint Selection Strategy (Disjoint Non-Causal Selection)
            if self.use_joint_covariance_sampling:
                # Joint selection style explicitly ignores causal structure for "randomness"
                outputs_flat = torch.cat(outputs, dim=-1)
                total_nodes = outputs_flat.shape[1]
                
                min_required = self.num_outputs + self.num_features
                if total_nodes < min_required:
                    warnings.warn(
                        f"Insufficient output dimensions: got {total_nodes}, need at least {min_required}. "
                        f"Reducing num_features to fit available dimensions.",
                        UserWarning,
                        stacklevel=2
                    )
                    # Reduce num_features to fit available dimensions
                    reduction = min_required - total_nodes
                    self.num_features = max(1, self.num_features - reduction)
                    warnings.warn(
                        f"Reduced num_features to {self.num_features} to fit available dimensions.",
                        UserWarning,
                        stacklevel=2
                    )
                    # Recalculate min_required with new num_features
                    min_required = self.num_outputs + self.num_features
                
                available_indices = torch.randperm(total_nodes, device=self.device)
                indices_y = available_indices[:self.num_outputs]
                indices_X = available_indices[self.num_outputs : self.num_outputs + self.num_features]
                
                if self.sort_features:
                    indices_X, _ = torch.sort(indices_X)
                    X = outputs_flat[:, indices_X]
                else:
                    X = outputs_flat[:, indices_X]
                y = outputs_flat[:, indices_y]
                
            else:
                # Strict Causal Selection or Clique Selection
                if self.in_clique:
                    # Clique selection: sample a contiguous block from flattened outputs
                    # BUT we must still respect the available size
                    outputs_flat = torch.cat(outputs, dim=-1)
                    total_nodes = outputs_flat.shape[1]
                    block_size = self.num_outputs + self.num_features
                    
                    if total_nodes < block_size:
                        warnings.warn(
                            f"Insufficient dimensions for clique: got {total_nodes}, need at least {block_size}. "
                            f"Reducing num_features to fit available dimensions.",
                            UserWarning,
                            stacklevel=2
                        )
                        # Reduce num_features to fit available dimensions
                        reduction = block_size - total_nodes
                        self.num_features = max(1, self.num_features - reduction)
                        warnings.warn(
                            f"Reduced num_features to {self.num_features} for clique selection.",
                            UserWarning,
                            stacklevel=2
                        )
                        # Recalculate block_size with new num_features
                        block_size = self.num_outputs + self.num_features
                    
                    # Ensure clique block matches size constraints
                    # Range for start index: [0, total_nodes - block_size]
                    max_start = total_nodes - block_size
                    start = random.randint(0, max_start)
                    
                    # Get indices for the block
                    block_indices = torch.arange(start, start + block_size, device=self.device)
                    
                    # Within the block, shuffle assignment to X and y
                    perm = torch.randperm(block_size, device=self.device)
                    indices_X = block_indices[perm[:self.num_features]]
                    indices_y = block_indices[perm[self.num_features:]]
                    
                    if self.sort_features:
                        indices_X, _ = torch.sort(indices_X)
                        
                    X = outputs_flat[:, indices_X]
                    y = outputs_flat[:, indices_y]
                    
                else:
                    # STRICT CAUSAL INTEGRITY (Default)
                    # To treat 'y' as effect and 'X' as causes/intermediate, we should force equality:
                    # y comes from LAST layer(s)
                    # X comes from EARLIER layer(s)
                    
                    # Split pools
                    # y candidates: solely from the last layer output to ensure it's the most "downstream"
                    y_pool = outputs[-1] 
                    
                    # X candidates: everything BEFORE the last layer (causes + intermediate layers)
                    # (excluding the last layer to avoid X being caused BY y, or X and y being siblings)
                    # NOTE: If we want "siblings" to be possible, we could include last layer in X pool too,
                    # but strictly speaking, distinct generations are cleaner.
                    # For now, let's include everything up to N-1 for X.
                    # If there's only one layer output, use causes as X pool (causally correct)
                    if len(outputs) == 1:
                        x_pool = causes
                    else:
                        x_pool = torch.cat(outputs[:-1], dim=-1)
                    
                    # Check if we have enough dimensions
                    if y_pool.shape[1] < self.num_outputs:
                         # Fallback: if last layer is too small, borrow from second to last, etc.
                         # But hidden_dim should be > num_outputs ideally.
                         # If strictly failing, we might need to concat last K layers.
                         outputs_flat = torch.cat(outputs, dim=-1)
                         # Fallback to random permutation from flat if structural constraints fail
                         # But let's try to maintain structure first.
                         pass

                    # Select y from last layer
                    if y_pool.shape[1] >= self.num_outputs:
                        indices_y = torch.randperm(y_pool.shape[1], device=self.device)[:self.num_outputs]
                        y = y_pool[:, indices_y]
                    else:
                        # Not enough dims in last layer for y? This is rare if hidden_dim >= num_outputs.
                        # Fallback: take what we can from last, rest from second last...
                        # Easier fallback: use flattened approach for y, but restricted to END of chain
                        outputs_flat = torch.cat(outputs, dim=-1)
                        # Take y from the very end
                        y = outputs_flat[:, -self.num_outputs:]
                    
                    # Select X from earlier layers
                    if x_pool.shape[1] >= self.num_features:
                        indices_X = torch.randperm(x_pool.shape[1], device=self.device)[:self.num_features]
                        if self.sort_features:
                            indices_X, _ = torch.sort(indices_X)
                        X = x_pool[:, indices_X]
                    else:
                        # Not enough earlier dims?
                        # This happens if num_layers is small or hidden_dim is small.
                        # Fallback: Sample X from everything available (excluding y's specific selection if possible)
                        # For robustness, let's just use the flattened pool except the absolute last dims used for y
                        outputs_flat = torch.cat(outputs, dim=-1)
                        # We already took y from somewhere.
                        # Let's simple use random perm logic but prioritize structure where possible.
                        
                        # Use the original logic but with strict bounds
                        total_nodes = outputs_flat.shape[1]
                        # Exclude last num_outputs for y
                        available_for_X = total_nodes - self.num_outputs
                        
                        if available_for_X < self.num_features:
                            warnings.warn(
                                f"Insufficient dimensions for X: {available_for_X} < {self.num_features}. "
                                f"Reducing num_features to {available_for_X}.",
                                UserWarning,
                                stacklevel=2
                            )
                            self.num_features = available_for_X
                             
                        indices_X = torch.randperm(available_for_X, device=self.device)[:self.num_features]
                        if self.sort_features:
                             indices_X, _ = torch.sort(indices_X)
                        X = outputs_flat[:, indices_X]
                        y = outputs_flat[:, -self.num_outputs:]

        else:
            # In non-causal mode, use original causes and last layer output
            X = causes
            y = outputs[-1]

        return X, y
