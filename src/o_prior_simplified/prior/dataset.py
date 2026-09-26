"""
The module offers a flexible framework for creating diverse, realistic tabular datasets
with controlled properties, which can be used for training and evaluating in-context
learning models. Key features include:

- Controlled feature relationships and causal structures via multiple generation methods
- Customizable feature distributions with mixed continuous and categorical variables
- Batch generation capabilities with hierarchical parameter sharing
- Memory-efficient handling of variable-length datasets

The main class is PriorDataset, which provides an iterable interface for generating
an infinite stream of synthetic datasets with diverse characteristics.
"""

from __future__ import annotations

import os
import sys
import math
import warnings
import logging
from typing import Dict, Tuple, Union, Optional, Any, List

import numpy as np
from scipy.stats import loguniform
import joblib

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset

from .mlp_scm import MLPSCM
from .tree_scm import TreeSCM
from .conv_scm import ConvSCM
from .gp_scm import GPSCM
from .linear_scm import LinearSCM
from .hybrid_scm import HybridSCM

from .hp_sampling import HpSamplerList
from .reg2cls import Reg2Cls
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP, TREE_PRIOR_WEIGHTS

# Set up logger for dataset generation
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)

# Feature/meta token ids for downstream embedding pipelines.
FEATURE_TYPE_TO_ID = {
    "continuous": 0,
    "categorical": 1,
    "count": 2,
    "binary": 3,
    "proportion": 4,
    "label": 5,
    "padding": 6,
    "ordinal": 7,
}

MISSING_STATE_TO_ID = {
    "present": 0,
    "missing": 1,
}

IMPUTATION_STRATEGY_TO_ID = {
    "none": 0,
    "mean": 1,
    "median": 2,
    "constant": 3,
    "support_sample": 4,
    "gaussian": 5,
}


def get_prior_weights():
    """Return the fixed prior mixture weights used for tree-family sampling."""
    return {
        'scm': 0.60,
        'et': 0.15,
        'gb': 0.10,
        'dt': 0.08,
        'rf': 0.05,
        'dsrf': 0.02,
    }


def sample_tree_type(weights: Optional[dict] = None) -> str:
    """
    Sample a tree type based on mixture weights.
    
    Parameters
    ----------
    weights : Optional[dict], default=None
        Dictionary with tree type weights. If None, uses TREE_PRIOR_WEIGHTS from config.
        Expected keys: 'et', 'gb', 'dt', 'rf', 'dsrf'
    
    Returns
    -------
    str
        Tree model type: 'extra_trees', 'xgboost', 'decision_tree', 'random_forest', or 'dsrf'
    """
    if weights is None:
        # Use default weights from config
        weights = TREE_PRIOR_WEIGHTS
    
    # Map to tree model names
    tree_types = ['extra_trees', 'xgboost', 'decision_tree', 'random_forest', 'dsrf']
    tree_weights = [
        weights.get('et', TREE_PRIOR_WEIGHTS['et']),
        weights.get('gb', TREE_PRIOR_WEIGHTS['gb']),
        weights.get('dt', TREE_PRIOR_WEIGHTS['dt']),
        weights.get('rf', TREE_PRIOR_WEIGHTS['rf']),
        weights.get('dsrf', TREE_PRIOR_WEIGHTS['dsrf']),
    ]
    
    # Normalize weights
    total = sum(tree_weights)
    if total > 0:
        tree_weights = [w / total for w in tree_weights]
    else:
        # Fallback to uniform
        tree_weights = [1.0 / len(tree_types)] * len(tree_types)
    
    return np.random.choice(tree_types, p=tree_weights)


class Prior:
    """
    Abstract base class for dataset prior generators.

    Defines the interface and common functionality for different types of
    synthetic dataset generators.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    min_features : int, default=1
        Minimum number of features per dataset

    max_features : int, default=99
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 1,
        max_features: int = 99,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        sampling: str = "mixed",
    ):
        self.batch_size = batch_size

        assert min_features <= max_features, "Invalid feature range"
        self.min_features = min_features
        self.max_features = max_features

        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.sampling = sampling

    def sample_seq_len(
        self, min_seq_len: Optional[int], max_seq_len: int, log: bool = False
    ) -> int:
        """
        Selects a random sequence length within the specified range.

        Supports uniform or log-uniform sampling within the configured bounds.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, returns max_seq_len.

        max_seq_len : int
            Maximum sequence length

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes


        Returns
        -------
        int
            The sampled sequence length
        """
        if min_seq_len is None:
            return max_seq_len
            
        if min_seq_len >= max_seq_len:
            return min_seq_len
        
        if log:
            from scipy.stats import loguniform
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
            # Assert to catch bugs immediately
            assert min_seq_len <= seq_len <= max_seq_len, (
                f"seq_len sampling bug: sampled {seq_len} not in [{min_seq_len}, {max_seq_len}]"
            )
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len + 1)  # Fix: +1 for inclusive upper bound

        return seq_len

    def sample_num_features(self, min_feat: int, max_feat: int) -> int:
        """Sample the feature count uniformly between inclusive bounds."""
        return int(np.random.randint(min_feat, max_feat + 1))
    
    @staticmethod
    def enforce_cell_cap(seq_len: int, num_features: int, max_cells: int = 102400) -> int:
        """Enforce standard cell cap: reduce seq_len if total cells exceed max_cells.

        This caps total table cells by reducing sequence length when the number of
        features is large. This prevents memory issues while maintaining diversity.
        Raised from the upstream 75,000 to 102,400 (= 100 features x 1024 rows) so
        that tables match the 1024-row shape LTM1 pre-training expects.
        
        Parameters
        ----------
        seq_len : int
            Current sequence length
        num_features : int
            Number of features
        max_cells : int, default=102400
            Maximum allowed cells
            
        Returns
        -------
        int
            Adjusted sequence length (may be reduced if cap exceeded)
        """
        total_cells = seq_len * num_features
        if total_cells >= max_cells:
            new_seq_len = max_cells // num_features
            return max(new_seq_len, 1)  # Ensure at least 1 sample
        return seq_len
    
    @staticmethod
    def adjust_max_features(seq_len: int, max_features: int) -> int:
        """
        Adjusts the maximum number of features based on the sequence length.

        This method implements an adaptive feature limit that scales inversely
        with sequence length. Longer sequences are restricted to fewer features
        to prevent memory issues and excessive computation times while still
        maintaining dataset diversity and learning difficulty.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of samples)

        max_features : int
            Original maximum number of features

        Returns
        -------
        int
            Adjusted maximum number of features, ensuring computational feasibility
        """
        if seq_len <= 10240:
            return min(100, max_features)
        elif 10240 < seq_len <= 20000:
            return min(80, max_features)
        elif 20000 < seq_len <= 30000:
            return min(60, max_features)
        elif 30000 < seq_len <= 40000:
            return min(40, max_features)
        elif 40000 < seq_len <= 50000:
            return min(30, max_features)
        elif 50000 < seq_len <= 60000:
            return min(20, max_features)
        elif 60000 < seq_len <= 65000:
            return min(15, max_features)
        else:
            return 10

    @staticmethod
    def delete_unique_features(X: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Removes features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant features
        to improve model training efficiency and stability. The removed features are
        replaced with zero padding to maintain tensor dimensions.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H) where:
            - B is batch size
            - T is sequence length
            - H is feature dimensionality

        d : Tensor
            Number of features per dataset of shape (B,), indicating how many
            features are actually used in each dataset (rest is padding)

        Returns
        -------
        tuple
            (X_new, d_new, keep_indices) where:
            - X_new is the filtered tensor with non-informative features removed
            - d_new is the updated feature count per dataset
            - keep_indices is a per-dataset LongTensor of the surviving column
              indices, in their new order, for re-indexing per-column metadata
        """

        def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor, Tensor]:
            """Filters features with only one unique value from a single dataset.

            If all features are constant (d=0), the dataset will be rejected and
            regenerated by the retry mechanism in generate_dataset().
            """
            num_features = xi.shape[-1]
            # Only consider actual features (up to di, ignoring padding)
            xi = xi[:, :di]
            # Identify features with more than one unique value (informative features)
            unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
            di_new = sum(unique_mask)

            # Create new tensor with only informative features, padding the rest
            xi_new = F.pad(xi[:, unique_mask], pad=(0, num_features - di_new), mode="constant", value=0)
            # Kept column indices, in their new left-compacted order, so that
            # per-column metadata can be re-indexed the same way.
            keep_idx = torch.tensor(
                [j for j, keep in enumerate(unique_mask) if keep], device=xi.device, dtype=torch.long
            )
            return xi_new, torch.tensor(di_new, device=xi.device), keep_idx

        # Process each dataset in the batch independently
        filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
        X_new = torch.stack([res[0] for res in filtered_results])
        d_new = torch.stack([res[1] for res in filtered_results])
        keep_indices = [res[2] for res in filtered_results]

        return X_new, d_new, keep_indices

    @staticmethod
    def _reindex_feature_meta(
        feature_meta: Optional[Dict[str, Tensor]], keep_idx: Tensor
    ) -> Optional[Dict[str, Tensor]]:
        """Apply delete_unique_features' column compaction to per-column metadata.

        Without this the metadata keeps the pre-filtering column layout while X has
        been left-compacted, so every entry describes the wrong column.
        """
        if not feature_meta:
            return feature_meta

        pad_values = {
            "feature_type_ids": FEATURE_TYPE_TO_ID["padding"],
            "col_ids": 0,
            "imputation_strategy_ids": IMPUTATION_STRATEGY_TO_ID["none"],
        }
        n_kept = keep_idx.numel()
        reindexed = {}
        for key, value in feature_meta.items():
            if key == "missing_mask":  # (seq_len, max_features)
                new_value = torch.zeros_like(value)
                new_value[:, :n_kept] = value[:, keep_idx]
            else:  # (max_features,)
                new_value = torch.full_like(value, pad_values.get(key, 0))
                new_value[:n_kept] = value[keep_idx]
            reindexed[key] = new_value
        return reindexed

    @staticmethod
    def sanity_check(y: Tensor, min_classes: int = 2, is_regression: bool = False) -> bool:
        """Validate each table's targets without splitting or reordering rows.

        Parameters
        ----------
        y : Tensor
            Target labels of shape (B, T).
        min_classes : int, default=2
            Minimum observed classes per classification table.
        is_regression : bool, default=False
            Check target variation instead of the number of classes.

        Returns
        -------
        bool
            Whether every table has finite targets and sufficient variation or classes.
        """
        for yi in y:
            if not torch.isfinite(yi).all():
                logger.debug("Nonfinite target values detected")
                return False
            if is_regression:
                if yi.numel() < 2 or torch.std(yi) < 1e-6:
                    logger.debug("Regression targets have insufficient variation")
                    return False
            elif torch.unique(yi).numel() < min_classes:
                logger.debug("Classification targets have too few observed classes")
                return False
        return True


class SCMPrior(Prior):
    """
    Generates synthetic datasets using Structural Causal Models (SCM).

    The data generation process follows a hierarchical structure:
    1. Generate a list of parameters for each dataset, respecting group/subgroup sharing.
    2. Process the parameter list to generate datasets, applying necessary transformations and checks.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=1
        Minimum number of features per dataset.

    max_features : int, default=99
        Maximum number of features per dataset.

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset.
        Empirically-aligned: 2048 (uniform distribution up to 2048)

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'conv_scm', 'tree_scm', 'gp_scm',
        'linear_scm', 'hybrid_scm', 'mix_scm',
        'mix_scm_hscm', or 'mix_scm_no_gp'
        'mix_scm' randomly selects between 'linear_scm', 'mlp_scm', 'conv_scm',
        'tree_scm', and 'gp_scm' based on probabilities.
        'mix_scm_no_gp' randomly selects between 'linear_scm', 'mlp_scm',
        'conv_scm', and 'tree_scm' (excludes GP-SCM to avoid
        2048 limit).

    fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed structural configuration parameters

    sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')

    Notes
    -----
    For Reference alignment, consider using:
    - max_seq_len=2048 (Reference-aligned: uniform distribution up to 2048)
    - Cell cap: <= 102,400 cells (automatically enforced via enforce_cell_cap)
    
    Feature counts are sampled uniformly between min_features and max_features,
    including both endpoints.
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 1,
        max_features: int = 99,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        prior_type: str = "mlp_scm",
        fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
        sampling: str = "mixed",
        tree_weights: Optional[Dict[str, float]] = None,  # Custom tree type weights
        tree_model: Optional[str] = None,  # Force specific tree model for TreeSCM
        use_cuml: bool = True,  # Use cuML (RAPIDS) for GPU-accelerated tree training when available
        fast_mode: bool = False,  # Enable fast_mode for TreeSCM
        use_advanced_hybrid_components: bool = False,  # Enable advanced HybridSCM components
        hybrid_sampling_strategy: str = "random",  # HybridSCM sampling strategy
        unstable_activation_threshold: int = 1500,  # Threshold for excluding unstable activations (Exp, Square)
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            sampling=sampling,
        )

        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.seq_len_per_gp = seq_len_per_gp
        self.prior_type = prior_type
        self.fixed_hp = fixed_hp
        self.sampled_hp = sampled_hp
        self.n_jobs = n_jobs
        self.num_threads_per_generate = num_threads_per_generate
        self.device = device
        self.tree_weights = tree_weights  # Custom tree type weights
        self.tree_model = tree_model  # Forced tree model
        self.use_cuml = use_cuml and device != "cpu"  # Use cuML for GPU-accelerated tree training
        self.fast_mode = fast_mode
        self.use_advanced_hybrid_components = use_advanced_hybrid_components
        self.hybrid_sampling_strategy = hybrid_sampling_strategy
        self.unstable_activation_threshold = unstable_activation_threshold

    def hp_sampling(self) -> Dict[str, Any]:
        """Sample core hyperparameters from the configured distributions.

        Returns
        -------
        dict
            Sampled hyperparameters, including callable samplers and activation factories.
        """
        # Filter unstable activations based on max_seq_len
        # Exclude Exp and Square activations when max_seq_len exceeds threshold to prevent numerical instability
        exclude_unstable = self.max_seq_len > self.unstable_activation_threshold
        filtered_hp = self.sampled_hp.copy()
        if exclude_unstable and "mlp_activations" in filtered_hp:
            from .activations import get_activations
            # Get filtered activations
            filtered_activations = get_activations(
                random=True, scale=True, diverse=True, exclude_unstable=True
            )
            # Update the choice_values in the distribution
            filtered_hp["mlp_activations"] = {
                "distribution": "meta_choice_mixed",
                "choice_values": filtered_activations,
            }
        if exclude_unstable and "conv_activations" in filtered_hp:
            from .activations import get_activations
            # Get filtered activations for ConvSCM
            filtered_activations = get_activations(
                random=True, scale=True, diverse=True, exclude_unstable=True
            )
            # Update the choice_values in the distribution
            filtered_hp["conv_activations"] = {
                "distribution": "meta_choice_mixed",
                "choice_values": filtered_activations,
            }
        hp_sampler = HpSamplerList(filtered_hp, device=self.device)
        return hp_sampler.sample()
    
    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Generates a single valid dataset based on the provided parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            num_features, num_classes, prior_type, device, etc.

        Returns
        -------
        tuple
            (X, y, d, feature_meta) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Labels tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
            - feature_meta: per-dataset metadata for embeddings/missingness handling
        """

        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "conv_scm":
            prior_cls = ConvSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        elif params["prior_type"] == "hybrid_scm" or params["prior_type"] == "hierarchy_scm":
            prior_cls = HybridSCM
        elif params["prior_type"] == "gp_scm":
            prior_cls = GPSCM
        elif params["prior_type"] == "linear_scm":
            prior_cls = LinearSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        # Adaptive max attempts: more attempts for difficult cases (long sequences, few features)
        seq_len = params.get("seq_len", 1024)
        num_features = params.get("num_features", 10)
        prior_type = params.get("prior_type", "unknown")
        
        # Log sequence length in debug mode
        logger.debug(f"Generating dataset: prior={prior_type}, requested_seq_len={seq_len}, num_features={num_features}")
        
        # Increase attempts for edge cases: long sequences or very few features
        if isinstance(seq_len, int) and isinstance(num_features, int):
            if seq_len > 1500 or num_features <= 2:
                max_attempts = 50  # More attempts for difficult cases
            else:
                max_attempts = 20  # Standard attempts
        else:
            max_attempts = 20
        
        for attempt in range(max_attempts):
            try:
                # Ensure sampling is passed to SCM
                params["sampling"] = self.sampling

                # Pass advanced HybridSCM flags if applicable
                if prior_cls == HybridSCM:
                    params["use_advanced_components"] = getattr(self, "use_advanced_hybrid_components", False)
                    params["sampling_strategy"] = getattr(self, "hybrid_sampling_strategy", "random")

                X, y = prior_cls(**params)()
                
                # Clean NaNs/Inf from raw SCM output (defensive entry point)
                nan_mask_X = torch.isnan(X) | torch.isinf(X)
                nan_mask_y = torch.isnan(y) | torch.isinf(y)
                if torch.any(nan_mask_X) or torch.any(nan_mask_y):
                    logger.debug(f"Cleaning NaNs/Inf from raw SCM output: X={nan_mask_X.sum().item()}, y={nan_mask_y.sum().item()} NaNs detected")
                    X = self._clean_nan_inf(X)

                    # Replace invalid targets using finite values from the whole table.
                    if torch.any(nan_mask_y):
                        if y.dim() == 0:
                            y = torch.zeros_like(y)
                        else:
                            finite_y = y[torch.isfinite(y)]
                            y_mean = finite_y.mean() if finite_y.numel() > 0 else y.new_zeros(())
                            y_mean = torch.nan_to_num(y_mean, nan=0.0, posinf=0.0, neginf=0.0)
                            y = torch.where(nan_mask_y, y_mean, y)

                # SCM might have clamped seq_len internally (e.g. GPSCM caps at 2048)
                actual_seq_len = X.shape[0]
                
                # Log actual sequence length in debug mode (may differ from requested if SCM clamped it)
                if actual_seq_len != seq_len:
                    logger.debug(f"SCM clamped seq_len: requested={seq_len}, actual={actual_seq_len}, prior={prior_type}")
                else:
                    logger.debug(f"Dataset generated: seq_len={actual_seq_len}, prior={prior_type}")
                
                if self.max_classes == 0 or params.get("num_classes", 0) == 0:
                    # Regression path: keep continuous targets
                    # Normalize using the whole table.
                    X, feature_meta = self._process_features_regression(X, params)

                    target_norm_method = params.get("target_norm_method", "zscore")

                    y = self._normalize_continuous_target(
                        y,
                        norm_method=target_norm_method,
                    )

                else:
                    # Classification path: discretize
                    # Process features before converting continuous targets to classes.

                    X, feature_meta = self._process_features_regression(X, params)

                    try:
                        X, y = Reg2Cls(params, skip_feature_processing=True)(X, y)
                    except Exception as e:
                        logger.debug(f"Reg2Cls conversion failed: {e}")
                        raise

                # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
                X, y = X.unsqueeze(0), y.unsqueeze(0)
                # Record the current active feature count.
                d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

                # Remove constant features, then validate the whole table.
                X, d, keep_indices = self.delete_unique_features(X, d)
                # X's columns were left-compacted, so the per-column metadata has to
                # follow or it would describe the wrong columns downstream.
                feature_meta = self._reindex_feature_meta(feature_meta, keep_indices[0])

                # Determine if this is regression based on max_classes or num_classes
                is_regression = (self.max_classes == 0 or params.get("num_classes", 0) == 0)
                
                # Check validation
                d_valid = (d > 0).all()
                if d_valid:
                    # Validate targets across all rows.
                    sanity_check_passed = self.sanity_check(y, is_regression=is_regression)
                    if sanity_check_passed:
                        if attempt > 0:
                            logger.debug(
                                f"Dataset generation succeeded after {attempt + 1} attempts: "
                                f"prior={prior_type}, seq_len={seq_len}, features={num_features}"
                            )
                        return X.squeeze(0), y.squeeze(0), d.squeeze(0), feature_meta
                    else:
                        if attempt < 5 or attempt % 20 == 0:  # Log first 5 attempts and every 20th
                            # Add GP_SCM-specific logging for debugging
                            if prior_type == "gp_scm":
                                logger.debug(
                                    f"GP_SCM validation failed (attempt {attempt + 1}/{max_attempts}): "
                                    f"seq_len={seq_len}, features={num_features}"
                                )
                            else:
                                logger.debug(
                                    f"Dataset validation failed (attempt {attempt + 1}/{max_attempts}): "
                                    f"prior={prior_type}, seq_len={seq_len}, features={num_features}, "
                                    f"d={d.item()}, is_regression={is_regression}"
                                )
                else:
                    if attempt < 5 or attempt % 20 == 0:
                        logger.debug(
                            f"Dataset has no valid features after filtering (attempt {attempt + 1}/{max_attempts}): "
                            f"prior={prior_type}, seq_len={seq_len}, features={num_features}, d={d.item()}. "
                            f"This usually indicates the prior generated constant features. Retrying..."
                        )
            except RuntimeError as e:
                # Only retry on numerical instability errors
                msg = str(e).lower()
                numerical_error_keywords = ["cholesky", "nan", "inf", "singular", "constant features", "unstable initialization"]
                if any(keyword in msg for keyword in numerical_error_keywords):
                    if attempt < 5 or attempt % 20 == 0:
                        logger.warning(
                            f"Numerical instability during dataset generation (attempt {attempt + 1}/{max_attempts}): "
                            f"prior={prior_type}, seq_len={seq_len}, features={num_features}, error={str(e)}"
                        )
                    if attempt == max_attempts - 1:
                        raise
                    continue
                # Re-raise actual code bugs immediately
                raise
            except ValueError as e:
                # Check if it's a dimension mismatch error (retry-able)
                msg = str(e).lower()
                dimension_keywords = ["not enough", "insufficient", "dimensions", "output dimensions"]
                if any(keyword in msg for keyword in dimension_keywords):
                    # These are retry-able - the SCM will auto-adjust on next attempt
                    if attempt < 5 or attempt % 20 == 0:
                        logger.warning(
                            f"Dimension mismatch during dataset generation (attempt {attempt + 1}/{max_attempts}): "
                            f"prior={prior_type}, seq_len={seq_len}, features={num_features}, error={str(e)}. "
                            f"Will retry with auto-adjusted parameters."
                        )
                    if attempt == max_attempts - 1:
                        raise
                    continue
                # For other ValueErrors, re-raise immediately
                logger.error(
                    f"ValueError during dataset generation (not retrying): "
                    f"prior={prior_type}, seq_len={seq_len}, features={num_features}, error={type(e).__name__}: {str(e)}"
                )
                raise
            except Exception as e:
                # For non-RuntimeError exceptions, log and re-raise immediately
                logger.error(
                    f"Unexpected error during dataset generation (not retrying): "
                    f"prior={prior_type}, seq_len={seq_len}, features={num_features}, error={type(e).__name__}: {str(e)}"
                )
                raise
        
        # If we've exhausted all attempts, raise an error with helpful information
        error_msg = (
            f"Failed to generate valid dataset after {max_attempts} attempts. "
            f"Parameters: seq_len={seq_len}, num_features={num_features}, "
            f"prior_type={prior_type}. "
            f"This may indicate that the validation criteria are too strict for the given parameters. "
            f"Suggestions: "
            f"1. Check the configured sequence-length bounds (min_seq_len and max_seq_len), "
            f"2. For GP_SCM with few features, validation may be stricter, "
            f"3. Consider increasing max_attempts or relaxing validation thresholds."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    @staticmethod
    def _clean_nan_inf(X: Tensor) -> Tensor:
        """Replace nonfinite cells with whole-column finite means, falling back to zero."""
        finite_mask = torch.isfinite(X)
        if torch.all(finite_mask):
            return X
        col_mean = torch.nanmean(torch.where(finite_mask, X, torch.nan), dim=0)
        col_mean = torch.nan_to_num(col_mean, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.where(finite_mask, X, col_mean)

    def _process_features_regression(
        self, X: Tensor, hp: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Process regression and classification features using whole-table statistics.

        Apply outlier handling and z-score normalization uniformly to all rows.

        Parameters
        ----------
        X : Tensor
            Features tensor of shape (seq_len, num_features)
        hp : Dict[str, Any]
            Hyperparameters controlling feature permutation and metadata.
            
        Returns
        -------
        Tuple[Tensor, Dict[str, Tensor]]
            Processed features plus metadata:
            - ``missing_mask``: All-false mask, shape (seq_len, max_features)
            - ``feature_type_ids``: Per-column semantic type ids, shape (max_features,)
            - ``col_ids``: Randomized per-dataset column ids (padding=0), shape (max_features,)
            - ``imputation_strategy_ids``: All "none" ids, shape (max_features,)
        """
        from .reg2cls import outlier_removing
        from .reg2cls import torch_nanstd
        
        if X.shape[0] == 0:
            raise ValueError("Cannot process features with no rows")

        # Repair invalid SCM values before computing whole-table statistics.
        nan_mask = torch.isnan(X) | torch.isinf(X)
        if torch.any(nan_mask):
            logger.debug("Cleaning NaNs/Inf at feature processing entry point")
            X = self._clean_nan_inf(X)

        # Apply the existing outlier treatment to all rows, then standardize.
        X_clean = outlier_removing(X, threshold=4)
        mean = torch.nanmean(X_clean, dim=0)
        std = torch_nanstd(X_clean, dim=0, ddof=1 if X_clean.shape[0] > 1 else 0).clip(min=1e-6)
        X_normalized = (X_clean - mean) / std

        # Step 6: Clip extreme outliers (post-normalization)
        X_normalized = torch.clip(X_normalized, min=-100, max=100)
        
        # Clean NaNs/Inf after clipping (clipping preserves NaNs)
        nan_mask = torch.isnan(X_normalized) | torch.isinf(X_normalized)
        if torch.any(nan_mask):
            X_normalized = self._clean_nan_inf(X_normalized)
        
        # Step 7: Permute features BEFORE truncation to avoid systematic bias
        # When sort_features=True, features are sorted by causal graph position.
        # Permuting before truncation ensures we keep a random subset, not just tail features.
        if hp.get("permute_features", True):
            perm = torch.randperm(X_normalized.shape[1], device=X_normalized.device)
            X_normalized = X_normalized[:, perm]

        # Step 8: Handle feature dimension (truncate or pad)
        # Now truncation removes random features, not just tail features
        num_features = X_normalized.shape[1]
        active_feature_count = min(num_features, self.max_features)
        if num_features > self.max_features:
            # Truncate if too many features (now removes random subset after permutation)
            X_normalized = X_normalized[:, :self.max_features]
        elif num_features < self.max_features:
            # Pad if too few features
            X_normalized = F.pad(X_normalized, (0, self.max_features - num_features))

        # No observations are artificially masked; retain empty metadata for saving and export.
        missing_mask = torch.zeros_like(X_normalized, dtype=torch.bool)
        imputation_strategy_ids = torch.full(
            (self.max_features,),
            IMPUTATION_STRATEGY_TO_ID["none"],
            device=X_normalized.device,
            dtype=torch.long,
        )
        
        # Step 15b: Infer per-column feature type ids for downstream cell embeddings.
        feature_type_ids = torch.full(
            (self.max_features,),
            FEATURE_TYPE_TO_ID["padding"],
            device=X_normalized.device,
            dtype=torch.long,
        )
        cat_max_unique = max(int(hp.get("feature_type_categorical_max_unique", 16)), 2)
        max_cat_frac = float(hp.get("feature_type_categorical_max_frac", 0.12))
        for feat_idx in range(active_feature_count):
            col = X_normalized[:, feat_idx]
            if col.numel() == 0:
                feature_type_ids[feat_idx] = FEATURE_TYPE_TO_ID["continuous"]
                continue

            # Use all rows for feature-type inference.
            col_min = torch.min(col)
            col_max = torch.max(col)
            rounded = torch.round(col)
            integer_like_ratio = ((col - rounded).abs() < 1e-5).to(col.dtype).mean().item()
            is_integer_like = integer_like_ratio > 0.98
            unique_count = torch.unique(col).numel()
            unique_ratio = unique_count / max(int(col.numel()), 1)
            #is_binary = is_integer_like and unique_count <= 2 and col_min >= -1e-5 and col_max <= 1.0 + 1e-5
            is_count = is_integer_like and col_min >= -1e-5 and unique_count > 2
            is_proportion = (col_min >= -1e-5) and (col_max <= 1.0 + 1e-5)
            #is_categorical = unique_count <= cat_max_unique or unique_ratio <= max_cat_frac

            #if is_binary:
            #    t = "binary"
            if is_count:
                t = "binary"
            elif is_count:
                t = "count"
            elif is_proportion:
                t = "proportion"
            #elif is_categorical:
            #    t = "categorical"
            else:
                t = "continuous"
            feature_type_ids[feat_idx] = FEATURE_TYPE_TO_ID[t]

        # Step 15c: Build randomized per-dataset column ids (padding columns use 0).
        col_ids = torch.arange(1, self.max_features + 1, device=X_normalized.device, dtype=torch.long)
        if bool(hp.get("randomize_column_ids", True)):
            col_ids = col_ids[torch.randperm(self.max_features, device=X_normalized.device)]
        if active_feature_count < self.max_features:
            col_ids[active_feature_count:] = 0

        # Step 16: Permutation already done in Step 7 (before truncation)
        # This ensures random feature subset is kept, removing systematic bias
        
        # Final validation: check X_normalized has no NaN/Inf before returning
        # Production preprocessing handles NaNs anyway, so replace with defaults (Tabicl-style)
        # Option 2 (default): Smart replacement with whole-column defaults (mean per feature)
        # Option 1 (fallback): Simple replacement with 0.0 when all values are NaN
        nan_mask = torch.isnan(X_normalized) | torch.isinf(X_normalized)
        if torch.any(nan_mask):
                nan_count = nan_mask.sum().item()
                total_count = nan_mask.numel()
                nan_ratio = nan_count / total_count
                
                # Option 2: Replace NaNs with mean of non-NaN values per feature (whole table)
                for feat_idx in range(X_normalized.shape[1]):
                    feat_col = X_normalized[:, feat_idx]
                    nan_in_col = torch.isnan(feat_col) | torch.isinf(feat_col)
                    if torch.any(nan_in_col):
                        # Get valid (non-NaN/Inf) values for this feature
                        valid_values = feat_col[~nan_in_col]
                        if len(valid_values) > 0:
                            # Option 2: Replace with mean of valid values (preserves feature statistics)
                            replacement = valid_values.mean()
                        else:
                            # Option 1 (fallback): If all values are NaN, use 0.0
                            replacement = 0.0
                        
                        # Replace NaNs/Infs with the replacement value
                        X_normalized[:, feat_idx] = torch.where(
                            nan_in_col,
                            torch.full_like(feat_col, replacement),
                            feat_col
                        )
                
                # Log warning for debugging (but don't fail)
                if nan_ratio > 0.1:  # Only warn if >10% NaNs
                    logger.warning(
                        f"Feature processing produced {nan_count}/{total_count} ({nan_ratio:.1%}) NaN/Inf values. "
                        f"Replaced with whole-column defaults (mean per feature, 0.0 fallback). "
                        f"This may indicate numerical instability in transformations."
                    )
                else:
                    logger.debug(
                        f"Feature processing produced {nan_count}/{total_count} ({nan_ratio:.1%}) NaN/Inf values. "
                        f"Replaced with whole-column defaults."
                    )
        feature_meta = {
            "missing_mask": missing_mask,
            "feature_type_ids": feature_type_ids,
            "col_ids": col_ids,
            "imputation_strategy_ids": imputation_strategy_ids,
        }
        return X_normalized, feature_meta

    def _normalize_continuous_target(
        self,
        y: Tensor,
        norm_method: str = 'zscore',
    ) -> Tensor:
        """Normalize continuous regression targets using whole-table statistics.

        Apply z-score or min-max normalization, then clip to [-10, 10].

        Parameters
        ----------
        y : Tensor
            Target tensor of shape (seq_len,)
        norm_method : str, default='zscore'
            Normalization method: 'zscore' (standard-style) or 'minmax'
            
        Returns
        -------
        Tensor
            Normalized targets clipped to [-10, 10]
        """
        if y.numel() == 0:
            raise ValueError("Cannot normalize an empty target")

        from .reg2cls import torch_nanstd
        mean = torch.nanmean(y)
        # For 1D tensor (target), compute std directly
        # Add dimension for torch_nanstd compatibility, then extract scalar
        y_2d = y.unsqueeze(-1)  # (seq_len,) -> (seq_len, 1)
        std = torch_nanstd(y_2d, dim=0, ddof=1 if len(y) > 1 else 0)
        std = std.squeeze().clip(min=1e-6)  # Extract scalar and ensure > 0
        
        # Apply normalization based on method
        if norm_method == 'minmax':
            # Robust nan-aware min/max
            valid_y = y[~torch.isnan(y)]
            if valid_y.numel() > 0:
                y_min = valid_y.min()
                y_max = valid_y.max()
            else:
                y_min = torch.tensor(0.0, device=y.device)
                y_max = torch.tensor(1.0, device=y.device)
            y_range = (y_max - y_min).clip(min=1e-6)
            y_normalized = (y - y_min) / y_range
        else:  # 'zscore' (default)
            # Handle edge case: constant target (std ≈ 0)
            if std < 1e-6:
                # If the target is constant, center it
                y_normalized = y - mean
            else:
                # Apply whole-table target statistics
                y_normalized = (y - mean) / std
        
        # Safety clipping to prevent extreme outliers from causing gradient explosion
        # Documentation specifies [-10, 10] for safety
        # Normalized targets should be ~[-3, 3] for z-score, [0, 1] for min-max
        # Clipping prevents rare extreme values from destabilizing training
        y_normalized = y_normalized.clamp(min=-10.0, max=10.0)
        
        return y_normalized

    @torch.no_grad()
    def get_batch(
        self, batch_size: Optional[int] = None, **kwargs
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]],
    ]:
        """
        Generates a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size

        Returns
        -------
        X : Tensor or NestedTensor
            Features tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
            If seq_len_per_gp=True, returns a NestedTensor.

        y : Tensor or NestedTensor
            Labels tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len).
            If seq_len_per_gp=True, returns a NestedTensor.

        d : Tensor
            Number of active features per dataset after filtering, shape (batch_size,)

        seq_lens : Tensor
            Sequence length for each dataset, shape (batch_size,)

        feature_meta : Dict[str, Tensor], optional
            Returned when ``return_metadata=True``. Contains:
            ``missing_mask`` (B,T,H), ``feature_type_ids`` (B,H),
            ``col_ids`` (B,H), ``imputation_strategy_ids`` (B,H).
        """
        batch_size = batch_size or self.batch_size
        

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None

        # Determine global seq_len if not per-group
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len, self.max_seq_len, log=self.log_seq_len
            )
            # Apply cell cap at global level to ensure all datasets have same sequence length
            # Use max_features to be conservative (ensures all datasets fit within cap)
            original_seq_len = global_seq_len
            global_seq_len = self.enforce_cell_cap(global_seq_len, self.max_features, max_cells=102400)
            if original_seq_len != global_seq_len:
                logger.debug(
                    f"Cell cap applied: seq_len reduced from {original_seq_len} to {global_seq_len} "
                    f"(max_features={self.max_features})"
                )
            logger.debug(
                f"Global parameters: seq_len={global_seq_len}, "
                f"max_features={self.max_features}"
            )

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            group_sampled_hp = self.hp_sampling()
            # If per-group, sample seq_len for this group. Otherwise, use the global value
            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len
                )
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len = global_seq_len
                gp_max_features = self.max_features

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
                if actual_subgp_size <= 0:
                    break

                # Sample the requested feature count uniformly, including both bounds.
                subgp_num_features = self.sample_num_features(self.min_features, gp_max_features)
                # Subgroups share prior type, number of features, and sampled HPs
                # Pass num_features to avoid problematic priors when features are very few
                subgp_prior_type = self.get_prior(num_features=subgp_num_features)
                # Don't call mlp_activations or conv_activations here - they're factories that should be called per layer
                # This matches the original design where mlp_activations()/conv_activations() is always called in _make_layer_block
                subgp_sampled_hp = {
                    k: (v() if callable(v) and k not in ["mlp_activations", "conv_activations"] else v)
                    for k, v in group_sampled_hp.items()
                }
                
                # For regression tasks, prefer non-causal structure and joint selection
                # This simulates missing value imputation scenarios and aligns with best empirical practices
                if self.max_classes == 0:  # Regression task
                    p_causal = 0.3

                    subgp_sampled_hp["y_is_effect"] = np.random.choice([True, False], p=[p_causal, 1 - p_causal])
                    
                    # Bias toward joint target selection for regression (70% True, 30% False)
                    subgp_sampled_hp["use_joint_covariance_sampling"] = np.random.choice([True, False], p=[0.7, 0.3])

                # Generate parameters for each dataset in this subgroup
                for ds_idx in range(actual_subgp_size):
                    # Each dataset has its own number of classes
                    if self.max_classes == 0:
                        ds_num_classes = 0  # Regression mode
                    elif np.random.random() > 0.5:
                        ds_num_classes = np.random.randint(2, self.max_classes + 1)
                    else:
                        ds_num_classes = 2

                    # Apply cell cap (< 75,000 cells) - reduce seq_len if needed
                    if self.seq_len_per_gp:
                        # When seq_len_per_gp=True, use gp_seq_len directly (no cell cap, like Tabicl)
                        # Nested tensors handle variable lengths, so cell cap is not needed
                        adjusted_seq_len = gp_seq_len
                    else:
                        # When seq_len_per_gp=False, use global seq_len (already cell-capped at global level)
                        # This ensures all datasets have the same sequence length for stacking
                        adjusted_seq_len = gp_seq_len
                    
                    # Create parameters dictionary for this dataset
                    params = {
                        **self.fixed_hp,  # Fixed HPs
                        "seq_len": adjusted_seq_len,
                        # If per-gp setting, use adjusted max features for this group because we use nested tensors
                        # If not per-gp setting, use global max features to fix size for concatenation
                        "max_features": gp_max_features if self.seq_len_per_gp else self.max_features,
                        **subgp_sampled_hp,  # sampled HPs for this group
                        "prior_type": subgp_prior_type,
                        "num_features": subgp_num_features,
                        "num_classes": ds_num_classes,
                        "device": self.device,
                    }
                    # Pass n_jobs only for tree_scm to control TreeLayer parallelism
                    if subgp_prior_type == "tree_scm":
                        params["n_jobs"] = self.n_jobs
                        params["use_cuml"] = self.use_cuml  # Pass use_cuml to TreeSCM
                        params["fast_mode"] = self.fast_mode  # Pass fast_mode to TreeSCM
                        #Hybrid approach - 30% DSRF (fast), 70% regular trees (diversity)
                        if np.random.random() < 0.30:
                            params["tree_model"] = "dsrf"
                        else:
                            # Override tree_model with weighted sampling per mixture weights
                            # This ensures tree types are sampled according to MITRA rationale
                            # If tree_model was already sampled from config, we override it
                            # Use custom tree weights if provided, otherwise use default from get_prior_weights
                            if hasattr(self, 'tree_model') and self.tree_model is not None:
                                # Force specific tree model for performance testing
                                params["tree_model"] = self.tree_model
                            elif self.tree_weights is not None:
                                # Use custom tree weights directly
                                params["tree_model"] = sample_tree_type(self.tree_weights)
                            else:
                                # Use default weights from get_prior_weights (includes SCM weight)
                                weights = get_prior_weights()
                                params["tree_model"] = sample_tree_type(weights)
                    elif "tree_model" not in params:
                        # Backward compatibility: if tree_model not in sampled_hp, use default
                        params["tree_model"] = "xgboost"
                    param_list.append(params)

        # Use joblib to generate datasets in parallel.
        # Note: the 'loky' backend does not support nested parallelism during DDP, whereas the 'threading' backend does.
        # However, 'threading' does not respect `inner_max_num_threads`.
        # Therefore, we stick with the 'loky' backend for parallelism, but this requires generating
        # the prior datasets separately from the training process and loading them from disk,
        # rather than generating them on-the-fly.
        logger.debug(f"Generating {len(param_list)} datasets: n_jobs={self.n_jobs}, device={self.device}")
        import time
        start_time = time.time()
        
        # Parallel generation only benefits CPU execution
        # GPU already parallelizes internally; Python threading adds overhead without benefit
        if self.n_jobs > 1 and self.device == "cpu":
            with joblib.parallel_config(
                n_jobs=self.n_jobs, backend="loky", inner_max_num_threads=self.num_threads_per_generate
            ):
                # Send 16 tables per message: the generator object and each group's
                # activation samplers are then pickled once per message instead of once
                # per table (profiled at ~150 ms per table, the dispatch bottleneck).
                results = joblib.Parallel(batch_size=16)(
                    joblib.delayed(self.generate_dataset)(params) for params in param_list
                )
        else:
            # Sequential for GPU - CUDA handles parallelism internally
            results = [self.generate_dataset(params) for params in param_list]
        
        generation_time = time.time() - start_time
        logger.debug(f"Batch generation completed in {generation_time:.2f}s ({generation_time/len(param_list):.3f}s per dataset)")

        X_list, y_list, d_list, meta_list = zip(*results)

        # Combine Results
        if self.seq_len_per_gp:
            # Use nested tensors for variable sequence lengths
            X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
            y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
        else:
            # Stack into regular tensors for fixed sequence length
            X = torch.stack(X_list).to(self.device)  # (B, T, H)
            y = torch.stack(y_list).to(self.device)  # (B, T)

        # Metadata (always regular tensors)
        d = torch.stack(d_list).to(self.device)  # Actual number of features after filtering out constant ones
        seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)

        return_metadata = bool(kwargs.get("return_metadata", False) or self.fixed_hp.get("return_metadata", False))
        if return_metadata:
            if self.seq_len_per_gp:
                # Variable sequence lengths: keep mask as nested tensor.
                missing_mask = nested_tensor([m["missing_mask"].to(self.device) for m in meta_list], device=self.device)
            else:
                missing_mask = torch.stack([m["missing_mask"] for m in meta_list]).to(self.device)

            feature_meta = {
                "missing_mask": missing_mask,
                "feature_type_ids": torch.stack([m["feature_type_ids"] for m in meta_list]).to(self.device),
                "col_ids": torch.stack([m["col_ids"] for m in meta_list]).to(self.device),
                "imputation_strategy_ids": torch.stack([m["imputation_strategy_ids"] for m in meta_list]).to(self.device),
            }
            return X, y, d, seq_lens, feature_meta
        return X, y, d, seq_lens

    def get_prior(self, num_features: Optional[int] = None) -> str:
        """
        Determine which prior type to use for generation.

        For 'mix_scm' prior type, randomly selects between available priors
        based on configured probabilities. Also avoids problematic priors
        (tree_scm, gp_scm) when feature count is very small.

        Parameters
        ----------
        num_features : int, optional
            Number of features for this dataset. Used to avoid problematic
            priors when features are very few.

        Returns
        -------
        str
            The selected prior type name
        """
        if self.prior_type == "mix_scm":
            # Avoid tree_scm and gp_scm when features are very few (they generate constant features)
            # Use linear_scm and mlp_scm instead for better reliability
            if num_features is not None and num_features <= 3:
                # Very few features: only use linear_scm and mlp_scm (more reliable)
                return np.random.choice(["linear_scm", "mlp_scm"], p=[0.6, 0.4])  # Slightly favor linear
            # Default order: [Linear, MLP, Conv, Tree, GP]
            mix_probas = self.fixed_hp.get("mix_probas", [0.14, 0.20, 0.12, 0.24, 0.12])
            if len(mix_probas) == 4:
                # Legacy order: [Linear, MLP, Tree, GP]
                linear_weight, mlp_total, tree_weight, gp_weight = mix_probas
                mlp_weight = mlp_total * 0.5
                conv_weight = mlp_total * 0.5
                mix_probas = [linear_weight, mlp_weight, conv_weight, tree_weight, gp_weight]
            elif len(mix_probas) != 5:
                mix_probas = [0.14, 0.20, 0.12, 0.24, 0.12]
            total = sum(mix_probas)
            mix_probas = [w / total for w in mix_probas]
            return np.random.choice(
                ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "gp_scm"],
                p=mix_probas,
            )

        elif self.prior_type == "mix_scm_no_gp":
            # Same as mix_scm but excludes gp_scm.
            # Avoid tree_scm when features are very few (it generates constant features)
            if num_features is not None and num_features <= 3:
                # Very few features: only use linear_scm and mlp_scm (more reliable)
                return np.random.choice(["linear_scm", "mlp_scm"], p=[0.6, 0.4])  # Slightly favor linear
            # Default order (no GP): [Linear, MLP, Conv, Tree]
            mix_probas = self.fixed_hp.get("mix_probas")
            if mix_probas is None:
                mix_probas = [0.16, 0.20, 0.12, 0.24]
            elif len(mix_probas) == 5:
                # Full order: [Linear, MLP, Conv, Tree, GP] -> drop GP
                linear_weight, mlp_weight, conv_weight, tree_weight, _ = mix_probas
                mix_probas = [linear_weight, mlp_weight, conv_weight, tree_weight]
            elif len(mix_probas) == 4:
                # Legacy order: [Linear, MLP, Tree, GP]
                linear_weight, mlp_total, tree_weight, _ = mix_probas
                mlp_weight = mlp_total * 0.5
                conv_weight = mlp_total * 0.5
                mix_probas = [linear_weight, mlp_weight, conv_weight, tree_weight]
            elif len(mix_probas) == 3:
                # Legacy order: [Linear, MLP, Tree]
                linear_weight, mlp_total, tree_weight = mix_probas
                mlp_weight = mlp_total * 0.5
                conv_weight = mlp_total * 0.5
                mix_probas = [linear_weight, mlp_weight, conv_weight, tree_weight]
            else:
                mix_probas = [0.16, 0.20, 0.12, 0.24]
            total = sum(mix_probas)
            mix_probas = [w / total for w in mix_probas]
            return np.random.choice(
                ["linear_scm", "mlp_scm", "conv_scm", "tree_scm"],
                p=mix_probas,
            )
        elif self.prior_type == "mix_scm_hscm":
            # mix_scm_hscm includes hybrid_scm in addition to mix_scm priors
            # Avoid tree_scm, gp_scm, and hybrid_scm when features are very few
            if num_features is not None and num_features <= 3:
                # Very few features: only use linear_scm and mlp_scm (more reliable)
                return np.random.choice(["linear_scm", "mlp_scm"], p=[0.6, 0.4])
            # Default order: [Linear, MLP, Conv, Tree, GP, Hybrid]
            mix_probas = [0.14, 0.12, 0.08, 0.20, 0.10, 0.18]
            total = sum(mix_probas)
            return np.random.choice(
                ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "gp_scm", "hybrid_scm"],
                p=[w / total for w in mix_probas],
            )
        elif self.prior_type == "mix_scm_hscm_no_gp":
            # mix_scm_hscm_no_gp: Includes HybridSCM but excludes GP-SCM (for extreme scales)
            # Avoid tree_scm and hybrid_scm when features are very few
            if num_features is not None and num_features <= 3:
                # Very few features: only use linear_scm and mlp_scm (more reliable)
                return np.random.choice(["linear_scm", "mlp_scm"], p=[0.6, 0.4])
            # Default order: [Linear, MLP, Conv, Tree, Hybrid]
            mix_probas = [0.15, 0.12, 0.08, 0.22, 0.20]
            total = sum(mix_probas)
            return np.random.choice(
                ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "hybrid_scm"],
                p=[w / total for w in mix_probas],
            )
        else:
            return self.prior_type


class DummyPrior(Prior):
    """This class creates purely random data. This is useful for testing and debugging
    without the computational overhead of SCM-based generation.

    Parameters
    ----------
    batch_size : int, default=256
        Number of datasets to generate

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    device : str, default="cpu"
        Computation device
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
        )
        self.device = device

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of random datasets for testing purposes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override, if None, uses self.batch_size

        Returns
        -------
        X : Tensor
            Features tensor of shape (batch_size, seq_len, max_features).
            Contains random Gaussian values for all features.

        y : Tensor
            Labels tensor of shape (batch_size, seq_len).
            Contains randomly assigned class labels.

        d : Tensor
            Number of features per dataset of shape (batch_size,).
            Always set to max_features for DummyPrior.

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).
            All datasets share the same sequence length.
        """

        batch_size = batch_size or self.batch_size
        seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)

        X = torch.randn(batch_size, seq_len, self.max_features, device=self.device)

        num_classes = np.random.randint(2, self.max_classes + 1)
        y = torch.randint(0, num_classes, (batch_size, seq_len), device=self.device)

        d = torch.full((batch_size,), self.max_features, device=self.device)
        seq_lens = torch.full((batch_size,), seq_len, device=self.device)

        return X, y, d, seq_lens


class PriorDataset(IterableDataset):
    """
    Main dataset class that provides an infinite iterator over synthetic tabular datasets.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=1
        Minimum number of features per dataset

    max_features : int, default=99
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'conv_scm', 'tree_scm', 'mix_scm', or 'dummy'

        1. SCM-based: Structural causal models with complex feature relationships
         - 'mlp_scm': MLP-based causal models
         - 'conv_scm': Conv-based causal models
         - 'tree_scm': Tree-based causal models
         - 'mix_scm': Probabilistic mix of the above models

        2. Dummy: Randomly generated datasets for debugging

    scm_fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed parameters for SCM-based priors

    scm_sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors)

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 1,
        max_features: int = 99,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        prior_type: str = "mlp_scm",
        scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
        sampling: str = "mixed",
        tree_weights: Optional[Dict[str, float]] = None,  # Custom tree type weights
        tree_model: Optional[str] = None,  # Force specific tree model for TreeSCM
        use_cuml: bool = True,  # Use cuML (RAPIDS) for GPU-accelerated tree training when available
        fast_mode: bool = False,  # Enable fast_mode for TreeSCM
        use_advanced_hybrid_components: bool = False,  # Enable advanced HybridSCM components
        hybrid_sampling_strategy: str = "random",  # HybridSCM sampling strategy
        unstable_activation_threshold: int = 1500,  # Threshold for excluding unstable activations (Exp, Square)
    ):
        super().__init__()
        if prior_type == "dummy":
            self.prior = DummyPrior(
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                device=device,
            )
        elif prior_type == "regression_dummy":
            self.prior = RegressionDummyPrior(
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                device=device,
            )
        elif prior_type in [
            "mlp_scm",
            "conv_scm",
            "tree_scm",
            "gp_scm",
            "linear_scm",
            "hybrid_scm",
            "mix_scm",
            "mix_scm_hscm",
            "mix_scm_no_gp",
            "mix_scm_hscm_no_gp",
        ]:
            self.prior = SCMPrior(
                batch_size=batch_size,
                batch_size_per_gp=batch_size_per_gp,
                batch_size_per_subgp=batch_size_per_subgp,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                seq_len_per_gp=seq_len_per_gp,
                prior_type=prior_type,
                fixed_hp=scm_fixed_hp,
                sampled_hp=scm_sampled_hp,
                n_jobs=n_jobs,
                num_threads_per_generate=num_threads_per_generate,
                device=device,
                sampling=sampling,
                tree_weights=tree_weights,
                tree_model=tree_model,
                use_cuml=use_cuml,
                # fast_mode defaults to False (change to True for speed optimization)
                fast_mode=fast_mode,
                use_advanced_hybrid_components=use_advanced_hybrid_components,
                hybrid_sampling_strategy=hybrid_sampling_strategy,
                unstable_activation_threshold=unstable_activation_threshold,
            )
        else:
            raise ValueError(
                f"Unknown prior type '{prior_type}'. Available options: 'mlp_scm', 'conv_scm', 'tree_scm', 'gp_scm', "
                f"'linear_scm', 'hybrid_scm', 'mix_scm', 'mix_scm_hscm', "
                f"'mix_scm_no_gp', 'mix_scm_hscm_no_gp', or 'dummy'."
            )

        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.device = device
        self.prior_type = prior_type

    def get_batch(
        self, batch_size: Optional[int] = None, **kwargs
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]],
    ]:
        """
        Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call
        **kwargs : dict
            Optional generation settings, such as return_metadata.

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of (batch_size, seq_len, max_features).

        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class labels of (batch_size, seq_len).

        d : Tensor
            Number of active features per dataset of shape (batch_size,).

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).

        feature_meta : Dict[str, Tensor], optional
            Returned by SCM priors when ``return_metadata=True``.
        """
        return self.prior.get_batch(batch_size, **kwargs)

    def __iter__(self) -> "PriorDataset":
        """
        Returns an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator
        """
        return self

    def __next__(
        self,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]],
    ]:
        """
        Returns the next batch from the iterator. Since this is an infinite
        iterator, it never raises StopIteration and instead continuously generates
        new synthetic data batches.
        """
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        return (
            f"PriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max classes: {self.max_classes}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  device: {self.device}\n"
            f")"
        )


class DisablePrinting:
    """Context manager to temporarily suppress printed output."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout


class RegressionDummyPrior(Prior):
    """Simple regression prior generating continuous targets with learnable X-y relationship.
    
    Unlike purely random targets, this prior creates y as a function of X through
    random linear/nonlinear transformations, providing a meaningful signal for training.
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        min_seq_len: int | None = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        device: str = "cpu",
        noise_std: float = 0.1,
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=0,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
        )
        self.device = device
        self.noise_std = noise_std

    def get_batch(self, batch_size: int | None = None):
        bs = batch_size or self.batch_size
        seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)

        # Sample number of active features per dataset
        d = torch.randint(self.min_features, self.max_features + 1, (bs,), device=self.device)
        
        # Generate X
        X = torch.randn(bs, seq_len, self.max_features, device=self.device)
        
        # Generate y as a FUNCTION of X (learnable relationship!)
        # Each dataset in batch gets different random weights
        y_list = []
        for i in range(bs):
            num_feat = d[i].item()
            # Random weight vector scaled by sqrt(num_features) for stable variance
            weights = torch.randn(num_feat, device=self.device) / np.sqrt(num_feat)
            # y = X[:, :num_feat] @ weights + noise
            y_i = X[i, :, :num_feat] @ weights
            # Add nonlinearity with probability 0.5
            if torch.rand(1).item() > 0.5:
                y_i = torch.tanh(y_i) * 2  # Bounded nonlinear transform
            # Add noise
            y_i = y_i + self.noise_std * torch.randn(seq_len, device=self.device)
            # Normalize to zero mean, unit variance
            y_i = (y_i - y_i.mean()) / (y_i.std() + 1e-6)
            y_list.append(y_i)
        
        y = torch.stack(y_list, dim=0)
        
        seq_lens = torch.full((bs,), seq_len, device=self.device)
        return X, y, d, seq_lens
