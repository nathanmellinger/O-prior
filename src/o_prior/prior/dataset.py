"""
The module offers a flexible framework for creating diverse, realistic tabular datasets
with controlled properties, which can be used for training and evaluating in-context
learning models. Key features include:

- Controlled feature relationships and causal structures via multiple generation methods
- Customizable feature distributions with mixed continuous and categorical variables
- Flexible train/test splits optimized for in-context learning evaluation
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
from .time_lagged_scm import TimeLaggedSCM

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


def get_prior_weights(step: Optional[int], total_steps: Optional[int] = None, easy_floor: float = 0.20, warmup_proportion: float = 0.1):
    """
    Get prior mixture weights with easy-floor schedule.
    
    Per REGRESSION_SPEC.md Section 3.2 and 3.3:
    - Default weights: SCM 0.60, ET 0.15, GB 0.10, DT 0.08, RF 0.05, DSRF 0.02
    - Easy-floor ensures at least easy_floor fraction of easy SCM tasks persist
    - Ramp TBP weights gradually from warm start to target
    
    Parameters
    ----------
    step : Optional[int]
        Current training step. If None, returns default weights.
    total_steps : Optional[int], default=None
        Total training steps. Used for ramp calculation. 
        If None and step is provided, assumes warmup completes at step 1000.
    easy_floor : float, default=0.20
        Minimum SCM weight to maintain (easy-floor).
    warmup_proportion : float, default=0.1
        Proportion of total_steps for warmup (10% default).
    
    Returns
    -------
    dict
        Dictionary mapping prior type to weight.
    """
    if step is None:
        # Return default weights without ramp
        return {
            'scm': 0.60,
            'et': 0.15,
            'gb': 0.10,
            'dt': 0.08,
            'rf': 0.05,
            'dsrf': 0.02,
        }
    
    # Calculate ramp: if total_steps not provided, use step-based estimate
    if total_steps is None:
        # Assume warmup completes around step 1000 if not specified
        warmup_steps = max(1000, int(step / warmup_proportion))
        ramp = min(1.0, step / warmup_steps)
    else:
        # Ramp TBP weights gradually from warm start to target
        ramp = min(1.0, step / (total_steps * warmup_proportion))
    
    weights = {
        'scm': 0.60,
        'et': 0.15 * ramp,
        'gb': 0.10 * ramp,
        'dt': 0.08 * ramp,
        'rf': 0.05 * ramp,
        'dsrf': 0.02 * ramp,
    }
    
    # Ensure easy SCM floor
    if weights['scm'] < easy_floor:
        excess = easy_floor - weights['scm']
        tbp_total = sum(w for k, w in weights.items() if k != 'scm')
        if tbp_total > 0:
            for k in ['et', 'gb', 'dt', 'rf', 'dsrf']:
                weights[k] = max(0, weights[k] - excess * (weights[k] / tbp_total))
        weights['scm'] = easy_floor
    
    return weights


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

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets
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
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        use_curriculum: bool = False,
        curriculum_schedule: str = "linear",
        curriculum_warmup_steps: int = 1000,
        curriculum_min_ratio: float = 0.3,
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

        self.validate_train_size_range(min_train_size, max_train_size)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = replay_small
        
        # Curriculum learning parameters
        self.use_curriculum = use_curriculum
        self.curriculum_schedule = curriculum_schedule
        self.curriculum_warmup_steps = curriculum_warmup_steps
        self.curriculum_min_ratio = curriculum_min_ratio
        self.sampling = sampling
        # Cache for curriculum ratio to avoid repeated calculations and logging
        self._curriculum_cache = {}  # {step: ratio}
        self._last_logged_step = None  # Track last logged step to avoid duplicate logs

    def get_curriculum_ratio(self, step: int, log: bool = False, **kwargs) -> float:
        """
        Calculate the curriculum learning ratio based on the current training step.
        Uses caching to avoid repeated calculations and excessive logging.

        Parameters
        ----------
        step : int
            Current training step
        log : bool, default=False
            If True, log the curriculum ratio (only logs once per step to avoid spam)

        Returns
        -------
        float
            Curriculum ratio between min_ratio and 1.0
        """
        if not self.use_curriculum:
            return 1.0

        # Optional overrides from guardrail
        effective_min = kwargs.get("easy_floor", self.curriculum_min_ratio)
        complexity_cap = kwargs.get("complexity_cap", 1.0)

        # Check cache (only if NOT using overrides, or include them in key)
        cache_key = (step, effective_min, complexity_cap)
        if cache_key in self._curriculum_cache:
            return self._curriculum_cache[cache_key]

        # Handle None step (no curriculum progression)
        if step is None:
            ratio = self.curriculum_min_ratio
            self._curriculum_cache[cache_key] = ratio
            return ratio

        # Calculate ratio
        if step >= self.curriculum_warmup_steps:
            ratio = 1.0
        elif self.curriculum_schedule == "linear":
            # Linear interpolation from min_ratio to 1.0
            progress = step / self.curriculum_warmup_steps
            ratio = self.curriculum_min_ratio + (1.0 - self.curriculum_min_ratio) * progress
        elif self.curriculum_schedule == "cosine":
            # Cosine annealing from min_ratio to 1.0
            import math
            progress = step / self.curriculum_warmup_steps
            cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
            ratio = self.curriculum_min_ratio + (1.0 - self.curriculum_min_ratio) * cosine_factor
        elif self.curriculum_schedule == "step":
            # Step-wise increase at milestones
            milestones = [0.25, 0.5, 0.75]
            ratio = 1.0
            for i, milestone in enumerate(milestones):
                if step < milestone * self.curriculum_warmup_steps:
                    ratio = self.curriculum_min_ratio + (1.0 - self.curriculum_min_ratio) * (i / len(milestones))
                    break
        else:
            # Default to linear if unknown schedule
            progress = step / self.curriculum_warmup_steps
            ratio = self.curriculum_min_ratio + (1.0 - self.curriculum_min_ratio) * progress

        # Apply complexity cap and ensure ratio is within [effective_min, complexity_cap]
        ratio = min(complexity_cap, ratio)
        ratio = max(effective_min, ratio)

        # Cache the result
        self._curriculum_cache[cache_key] = ratio

        # Only log once per step to avoid excessive logging
        if log and step != self._last_logged_step:
            progress = step / self.curriculum_warmup_steps if step < self.curriculum_warmup_steps else 1.0
            if step >= self.curriculum_warmup_steps:
                logger.debug(f"Curriculum step {step}: ratio={ratio:.3f} (max reached)")
            else:
                logger.debug(f"Curriculum step {step}/{self.curriculum_warmup_steps}: ratio={ratio:.3f} (progress={progress:.3f})")
            self._last_logged_step = step

        return ratio

    @staticmethod
    def validate_train_size_range(min_train_size: Union[int, float], max_train_size: Union[int, float]) -> None:
        """
        Checks if the training size range is valid.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size (position or ratio)

        max_train_size : int|float
            Maximum training size (position or ratio)

        Raises
        ------
        AssertionError
            If training size range is invalid
        ValueError
            If training size types are mismatched or invalid
        """
        # Check for numeric types only
        if not isinstance(min_train_size, (int, float)) or not isinstance(max_train_size, (int, float)):
            raise TypeError("Training sizes must be int or float")

        # Check for valid ranges based on type
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert 0 < min_train_size < max_train_size, "0 < min_train_size < max_train_size"
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            assert 0 < min_train_size < max_train_size < 1, "0 < min_train_size < max_train_size < 1"
        else:
            raise ValueError("Both training sizes must be of the same type (int or float)")

    def sample_seq_len(
        self, min_seq_len: Optional[int], max_seq_len: int, log: bool = False, replay_small: bool = False, step: Optional[int] = None
    ) -> int:
        """
        Selects a random sequence length within the specified range.

        This method provides flexible sampling strategies for dataset sizes, including
        occasional re-sampling of smaller sequence lengths for better training diversity.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, uses a reasonable default based on max_seq_len.

        max_seq_len : int
            Maximum sequence length

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes

        replay_small : bool, default=False
            If True, occasionally sample smaller sequence lengths with
            specific distributions to ensure model robustness on smaller datasets

        step : int, optional
            Current training step for curriculum learning

        Returns
        -------
        int
            The sampled sequence length
        """
        # Apply curriculum learning if enabled (BEFORE checking min_seq_len)
        # This ensures curriculum works even when min_seq_len is None
        if self.use_curriculum and step is not None:
            ratio = self.get_curriculum_ratio(step)
            # If min_seq_len is None, use a reasonable minimum (e.g., 100 or 10% of max)
            if min_seq_len is None:
                effective_min = max(100, int(max_seq_len * 0.1))
            else:
                effective_min = min_seq_len
            # Scale max_seq_len: start smaller, grow to full
            effective_max = int(effective_min + (max_seq_len - effective_min) * ratio)
            max_seq_len = effective_max
            # Also set min_seq_len for sampling if it was None
            if min_seq_len is None:
                min_seq_len = effective_min
        
        # If min_seq_len is still None after curriculum, return max_seq_len
        # If min_seq_len is still None after curriculum, return max_seq_len
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

        if replay_small:
            # Skip replay for extreme scales (don't override 40K-60K with 410-3750)
            if max_seq_len > 10000:
                return seq_len
            p = np.random.random()
            if p < 0.05:
                # 5% probability: Very small datasets (prevent forgetting simple cases)
                low = min(200, max_seq_len)
                high = min(1000, max_seq_len + 1)
                return np.random.randint(low, high) if low < high else low
            elif p < 0.2:
                # 15% probability: Medium-large datasets (up to max_seq_len)
                # Ensure loguniform bounds are valid (a < b)
                lower = min(1000, max_seq_len)
                upper = max_seq_len
                if lower < upper:
                    from scipy.stats import loguniform
                    return int(loguniform.rvs(lower, upper))
                else:
                    return upper # Fallback if max_seq_len <= 1000
            else:
                return seq_len
        else:
            return seq_len

    def sample_train_size(
        self,
        min_train_size: Union[int, float], 
        max_train_size: Union[int, float], 
        seq_len: int,
        fixed_val_size: Optional[int] = None,
        step: Optional[int] = None,
    ) -> int:
        """
        Selects a random training size within the specified range.

        This method handles both absolute position and fractional ratio approaches
        for determining the training/test split point. Optionally uses fixed validation
        size (Empirically-aligned: 128 samples).

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        max_train_size : int|float
            Maximum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        seq_len : int
            Total sequence length

        fixed_val_size : int, optional, default=None
            If specified, use fixed validation size (Empirically-aligned: 128).
            Training size will be seq_len - fixed_val_size.

        step : int, optional
            Current training step for curriculum learning

        Returns
        -------
        int
            The sampled training size position

        Raises
        ------
        ValueError
            If training size range has incompatible types or fixed_val_size is invalid
        """
        # Fixed validation size
        if fixed_val_size is not None:
            if fixed_val_size >= seq_len:
                raise ValueError(f"fixed_val_size ({fixed_val_size}) must be < seq_len ({seq_len})")
            train_size = seq_len - fixed_val_size
            return max(1, train_size)  # Ensure at least 1 training sample
        
        # Apply curriculum learning if enabled (easier = more training data)
        if self.use_curriculum and step is not None:
            ratio = self.get_curriculum_ratio(step)
            # For easier tasks, use more training data (higher min_train_size)
            if isinstance(min_train_size, float) and isinstance(max_train_size, float):
                # Adjust range: easier tasks get more training data
                # Inverse curriculum: as ratio increases, we allow less training data
                curriculum_min = min_train_size + (max_train_size - min_train_size) * (1.0 - ratio) * 0.3
                curriculum_max = max_train_size
                min_train_size = curriculum_min
                max_train_size = curriculum_max
        
        # Original logic: sample from range
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            train_size = np.random.randint(min_train_size, max_train_size)
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            train_size = int(seq_len * np.random.uniform(min_train_size, max_train_size))
        else:
            raise ValueError("Invalid training size range.")
        return train_size

    def sample_num_features_beta(self, min_feat: int, max_feat: int, step: Optional[int] = None) -> int:
        """Sample number of features using Beta distribution (power-law style).
        
        Standard approach uses Beta distribution for feature sampling, which gives a right-skewed
        distribution favoring smaller feature counts. This creates more diverse datasets.
        
        Parameters
        ----------
        min_feat : int
            Minimum number of features
        max_feat : int
            Maximum number of features
        step : int, optional
            Current training step for curriculum learning
            
        Returns
        -------
        int
            Sampled number of features
        """
        # Apply curriculum learning if enabled
        if self.use_curriculum and step is not None:
            ratio = self.get_curriculum_ratio(step)
            # Scale max_feat: start with fewer features, grow to full
            # Ensure minimum of 5 features early in curriculum to avoid constant feature issues
            # Tree_SCM and GP_SCM often generate constant features with <5 features
            effective_max = int(min_feat + (max_feat - min_feat) * ratio)
            # More aggressive minimum: 5 features early, scales up
            min_safe_features = max(5, int(min_feat + (max_feat - min_feat) * 0.1))
            effective_max = max(min_safe_features, effective_max)
            # CRITICAL FIX: Also enforce minimum on min_feat, not just max_feat
            effective_min = max(5, min_feat)  # Ensure we never sample below 5
            max_feat = effective_max
            min_feat = effective_min  # Update min_feat to enforce minimum
        
        # Beta(2, 5) gives right-skewed distribution favoring smaller values
        # This matches common approaches for feature diversity
        beta_val = np.random.beta(2, 5)
        return int(min_feat + beta_val * (max_feat - min_feat))
    
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
    def sanity_check(X: Tensor, y: Tensor, train_size: int, n_attempts: int = 10, min_classes: int = 2, is_regression: bool = False) -> bool:
        """
        Verifies that both train and test sets contain all classes (classification) or
        have valid target distributions (regression).

        For classification: ensures both train and test sets contain examples from all classes.
        For regression: ensures both train and test sets have sufficient variance and valid ranges.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H)

        y : Tensor
            Target labels tensor of shape (B, T)

        train_size : int
            Position to split the data into train and test sets

        n_attempts : int, default=10
            Number of random permutations to try for fixing invalid splits (classification only)

        min_classes : int, default=2
            Minimum number of classes required in both train and test sets (classification only)

        is_regression : bool, default=False
            If True, performs regression-specific validation instead of classification checks

        Returns
        -------
        bool
            True if all datasets have valid splits, False otherwise
        """

        def is_valid_split_classification(yi: Tensor) -> bool:
            """Check if a single dataset has a valid train/test split for classification."""
            # Guard against invalid train_size
            if train_size <= 0 or train_size >= yi.shape[0]:
                return False

            # A valid split requires both train and test sets to have the same classes
            # and at least min_classes different classes must be present
            unique_tr = torch.unique(yi[:train_size])
            unique_te = torch.unique(yi[train_size:])
            return set(unique_tr.tolist()) == set(unique_te.tolist()) and len(unique_tr) >= min_classes

        def is_valid_split_regression(yi: Tensor) -> bool:
            """Check if a single dataset has a valid train/test split for regression."""
            # Guard against invalid train_size
            if train_size <= 0 or train_size >= yi.shape[0]:
                logger.debug(f"Invalid train_size: {train_size} for seq_len={yi.shape[0]}")
                return False

            y_train = yi[:train_size]
            y_test = yi[train_size:]

            # Check for NaN or Inf values
            if torch.any(torch.isnan(y_train)) or torch.any(torch.isnan(y_test)):
                logger.debug(f"NaN values detected: train={torch.any(torch.isnan(y_train))}, test={torch.any(torch.isnan(y_test))}")
                return False
            if torch.any(torch.isinf(y_train)) or torch.any(torch.isinf(y_test)):
                logger.debug(f"Inf values detected: train={torch.any(torch.isinf(y_train))}, test={torch.any(torch.isinf(y_test))}")
                return False

            # Check that both train and test have sufficient variance (not constant)
            train_std = torch.std(y_train)
            test_std = torch.std(y_test)
            
            # Both should have non-zero variance (not constant)
            if train_std < 1e-6 or test_std < 1e-6:
                logger.debug(f"Insufficient variance: train_std={train_std:.6f}, test_std={test_std:.6f}")
                return False

            # Check that target ranges overlap reasonably (for ICL to work)
            train_min, train_max = torch.min(y_train), torch.max(y_train)
            test_min, test_max = torch.min(y_test), torch.max(y_test)
            
            # Calculate range overlap: intersection of [train_min, train_max] and [test_min, test_max]
            range_overlap = min(train_max, test_max) - max(train_min, test_min)
            train_range = train_max - train_min
            test_range = test_max - test_min
            min_range = min(train_range, test_range)
            
            # If ranges don't overlap, range_overlap will be negative
            if range_overlap <= 0:
                logger.debug(
                    f"No range overlap: train=[{train_min:.3f}, {train_max:.3f}], "
                    f"test=[{test_min:.3f}, {test_max:.3f}], overlap={range_overlap:.3f}"
                )
                return False  # No overlap at all
            
            # Calculate overlap ratio: what fraction of the smaller range is covered by overlap
            if min_range > 0:
                overlap_ratio = range_overlap / min_range
                # Relax threshold for edge cases:
                # - Small datasets (< 50 samples): 5%
                # - Single feature with long sequences (> 1000): 5% (GP edge case)
                # - Default: 10%
                seq_len = yi.shape[0]
                if train_size < 50:
                    threshold = 0.05  # 5% for small datasets
                elif seq_len > 1000:
                    # For very long sequences, be more lenient
                    # This primarily addresses GP_SCM edge cases with uniform coordinates
                    # (now fixed to use random coordinates for long sequences, but kept as safety net)
                    threshold = 0.05  # 5% for long sequences
                else:
                    threshold = 0.1  # 10% default
                    
                if overlap_ratio < threshold:
                    logger.debug(
                        f"Insufficient overlap ratio: {overlap_ratio:.3f} < {threshold:.3f} "
                        f"(train_range={train_range:.3f}, test_range={test_range:.3f}, "
                        f"overlap={range_overlap:.3f}, train_size={train_size}, seq_len={seq_len})"
                    )
                    return False
            else:
                # Both ranges are zero (constant values), but we already checked variance > 0 above
                # This shouldn't happen, but handle it defensively
                logger.debug(f"Zero range detected: train_range={train_range:.3f}, test_range={test_range:.3f}")
                return False

            return True

        # Use regression-specific validation if indicated
        if is_regression:
            for i, (xi, yi) in enumerate(zip(X, y)):
                if not is_valid_split_regression(yi):
                    return False
            return True

        # Classification path: original logic
        for i, (xi, yi) in enumerate(zip(X, y)):
            if is_valid_split_classification(yi):
                continue

            # If the dataset has an invalid split, try to fix it with random permutations
            succeeded = False
            for _ in range(n_attempts):
                # Generate a random permutation of the samples
                perm = torch.randperm(yi.shape[0], device=yi.device)
                yi_perm = yi[perm]
                xi_perm = xi[perm]
                # Check if the permutation results in a valid split
                if is_valid_split_classification(yi_perm):
                    X[i], y[i] = xi_perm, yi_perm
                    succeeded = True
                    break

            if not succeeded:  # No valid split was found after all attempts
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

    min_features : int, default=2
        Minimum number of features per dataset.
        Empirically-aligned: 1 (Beta distribution, range 1-160)

    max_features : int, default=100
        Maximum number of features per dataset.
        Empirically-aligned: 160 (Beta distribution, range 1-160)

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

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'conv_scm', 'tree_scm', 'gp_scm',
        'linear_scm', 'time_lagged_scm', 'hybrid_scm', 'mix_scm',
        'mix_scm_hscm', or 'mix_scm_no_gp'
        'mix_scm' randomly selects between 'linear_scm', 'mlp_scm', 'conv_scm',
        'tree_scm', 'gp_scm', and 'time_lagged_scm' based on probabilities.
        'mix_scm_no_gp' randomly selects between 'linear_scm', 'mlp_scm',
        'conv_scm', 'tree_scm', and 'time_lagged_scm' (excludes GP-SCM to avoid
        2048 limit).
        With curriculum learning, 'mix_scm' uses more linear priors early in training.

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

    fixed_val_size : int, optional, default=None
        If specified, use fixed validation size (Empirically-aligned: 128 samples).
        Training size will be seq_len - fixed_val_size. If None, uses
        min_train_size and max_train_size to sample training size.

    Notes
    -----
    For Reference alignment, consider using:
    - max_seq_len=2048 (Reference-aligned: uniform distribution up to 2048)
    - min_features=1, max_features=160 (Reference-aligned: Beta distribution, range 1-160)
    - Cell cap: <= 102,400 cells (automatically enforced via enforce_cell_cap)
    
    Feature sampling uses Beta(2,5) distribution (power-law style) for right-skewed
    distribution favoring smaller feature counts, creating more diverse datasets.
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
        fixed_val_size: Optional[int] = None,
        use_curriculum: bool = False,
        curriculum_schedule: Optional[str] = None,
        curriculum_warmup_steps: int = 1000,
        curriculum_min_ratio: float = 0.3,
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
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
            use_curriculum=use_curriculum,
            curriculum_schedule=curriculum_schedule,
            curriculum_warmup_steps=curriculum_warmup_steps,
            curriculum_min_ratio=curriculum_min_ratio,
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
        self.fixed_val_size = fixed_val_size  # Fixed validation size (128 samples)
        self.tree_weights = tree_weights  # Custom tree type weights
        self.tree_model = tree_model  # Forced tree model
        self.use_cuml = use_cuml and device != "cpu"  # Use cuML for GPU-accelerated tree training
        self.fast_mode = fast_mode
        self.use_advanced_hybrid_components = use_advanced_hybrid_components
        self.hybrid_sampling_strategy = hybrid_sampling_strategy
        self.unstable_activation_threshold = unstable_activation_threshold

    def hp_sampling(self, step: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Sample hyperparameters for dataset generation with optional curriculum learning.

        Parameters
        ----------
        step : int, optional
            Current training step for curriculum-aware hyperparameter sampling.
        **kwargs : dict
            Optional overrides for curriculum (e.g., easy_floor, complexity_cap)
            If provided and curriculum is enabled, hyperparameters will be easier early
            (lower noise, simpler activations, fewer layers, etc.) and harder later.

        Returns
        -------
        dict
            Dictionary with sampled hyperparameters merged with fixed ones
        """
        # Filter unstable activations based on max_seq_len (for both curriculum and non-curriculum)
        # Exclude Exp and Square activations when max_seq_len exceeds threshold to prevent numerical instability
        exclude_unstable = self.max_seq_len > self.unstable_activation_threshold
        
        # Apply curriculum learning to hyperparameters if enabled
        if self.use_curriculum and step is not None:
            curriculum_hp = self._get_curriculum_hyperparameters(step, exclude_unstable=exclude_unstable, **kwargs)
            # Merge curriculum-adjusted HPs with base sampled_hp
            # Curriculum HPs override base ones where specified
            merged_hp = {**self.sampled_hp, **curriculum_hp}
            hp_sampler = HpSamplerList(merged_hp, device=self.device)
        else:
            # When curriculum is disabled, filter activations from sampled_hp
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
    
    def _get_curriculum_hyperparameters(self, step: int, **kwargs) -> Dict[str, Any]:
        """
        Get curriculum-aware hyperparameter distributions.
        
        Early in curriculum (low ratio): Easier hyperparameters
        - Lower noise (easier to learn)
        - Simpler activations (relu, tanh)
        - Fewer layers, smaller hidden dims
        - Linear GP combinations
        - Fewer causes
        
        Late in curriculum (high ratio): Harder hyperparameters
        - Higher noise (harder to learn)
        - Complex activations (swish, gelu, etc.)
        - More layers, larger hidden dims
        - Quadratic/MLP GP combinations
        - More causes
        
        Parameters
        ----------
        step : int
            Current training step
            
        Returns
        -------
        dict
            Dictionary of curriculum-adjusted hyperparameter distributions
        """
        ratio = self.get_curriculum_ratio(step, **kwargs)
        curriculum_hp = {}
        
        # Noise parameters: keep min easy (Fixed), ramp max (Expanding Window)
        # noise_variance: 0.01 (fixed easy) -> 0.1-0.5 (hard)
        easy_noise_min, easy_noise_max = 0.01, 0.1
        hard_noise_max = 0.5
        noise_min = easy_noise_min # Fixed
        noise_max = easy_noise_max + (hard_noise_max - easy_noise_max) * ratio
        curriculum_hp["noise_variance"] = {
            "distribution": "meta_log_uniform",
            "min": noise_min,
            "max": noise_max,
        }
        
        # noise_std: 0.005 (increased from 0.0001 to ensure diversity) -> 0.05-0.3 (hard)
        easy_noise_std_min, easy_noise_std_max = 0.005, 0.05
        hard_noise_std_max = 0.3
        noise_std_min = easy_noise_std_min # Fixed
        noise_std_max = easy_noise_std_max + (hard_noise_std_max - easy_noise_std_max) * ratio
        curriculum_hp["noise_std"] = {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": noise_std_max,
            "min_mean": noise_std_min,
            "round": False,
            "lower_bound": 0.0,
        }
        
        # GP length_scale: keep 'smooth' available (Expanding Window)
        # Note: Small length_scale = wiggly (hard), Large = smooth (easy)
        easy_length_min, easy_length_max = 2.0, 10.0
        hard_length_min = 0.1
        length_min = easy_length_min - (easy_length_min - hard_length_min) * ratio # Ramp min DOWN (Expanding HARD bounds)
        length_max = easy_length_max # Keep max at 10.0 (Fixed EASY bound)
        curriculum_hp["length_scale"] = {
            "distribution": "meta_log_uniform",
            "min": length_min,
            "max": length_max,
        }
        
        # GP signal_variance: 0.1 (fixed easy) -> 1.0-5.0 (hard)
        easy_signal_min, easy_signal_max = 0.1, 1.0
        hard_signal_max = 5.0
        signal_min = easy_signal_min # Fixed
        signal_max = easy_signal_max + (hard_signal_max - easy_signal_max) * ratio
        curriculum_hp["signal_variance"] = {
            "distribution": "meta_log_uniform",
            "min": signal_min,
            "max": signal_max,
        }
        
        # GP combination: linear always available early, quadratic/mlp ramped in
        if ratio < 0.5:
            # Early: 80% linear, 20% others
            curriculum_hp["gp_combination"] = {
                "distribution": "meta_choice",
                "choice_values": ["linear", "linear", "linear", "linear", "quadratic", "mlp"],
            }
        elif ratio < 0.8:
            # Mid: 40% linear, 30% each quadratic/mlp
            curriculum_hp["gp_combination"] = {
                "distribution": "meta_choice",
                "choice_values": ["linear", "linear", "quadratic", "quadratic", "mlp", "mlp"],
            }
        else:
            # Late: 10% linear, 45% each quadratic/mlp
            curriculum_hp["gp_combination"] = {
                "distribution": "meta_choice",
                "choice_values": ["linear", "quadratic", "quadratic", "quadratic", "mlp", "mlp", "mlp", "mlp"],
            }
        
        # num_layers: 1 (fixed easy) -> 3-6 (hard)
        easy_layers_min, easy_layers_max = 1, 3
        hard_layers_max = 6
        layers_min = easy_layers_min # Fixed
        layers_max = easy_layers_max + (hard_layers_max - easy_layers_max) * ratio
        curriculum_hp["num_layers"] = {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": layers_max,
            "min_mean": layers_min,
            "round": True,
            "lower_bound": 2,
        }
        
        # hidden_dim: 5 (fixed easy) -> 30-130 (hard)
        easy_hidden_min, easy_hidden_max = 5, 30
        hard_hidden_max = 130
        hidden_min = easy_hidden_min # Fixed
        hidden_max = easy_hidden_max + (hard_hidden_max - easy_hidden_max) * ratio
        curriculum_hp["hidden_dim"] = {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": hidden_max,
            "min_mean": hidden_min,
            "round": True,
            "lower_bound": 4,
        }
        
        # num_causes: 1 (fixed easy) -> 4-12 (hard)
        easy_causes_min, easy_causes_max = 1, 4
        hard_causes_max = 12
        causes_min = easy_causes_min # Fixed
        causes_max = easy_causes_max + (hard_causes_max - easy_causes_max) * ratio
        curriculum_hp["num_causes"] = {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": causes_max,
            "min_mean": causes_min,
            "round": True,
            "lower_bound": 1,
        }
        
        # MLP activations: simpler early, complex later
        # Early: prefer relu, tanh (simpler)
        # Late: prefer swish, gelu, etc. (complex)
        from .activations import get_activations
        import torch.nn as nn
        
        # Get exclude_unstable from kwargs (passed from hp_sampling)
        exclude_unstable = kwargs.get("exclude_unstable", False)
        all_activations = get_activations(random=True, scale=True, diverse=True, exclude_unstable=exclude_unstable)
        
        # Simple activations: ReLU, Tanh, Sigmoid, Identity, LeakyReLU
        # Check class names and types
        simple_activation_names = ['relu', 'tanh', 'sigmoid', 'identity', 'leakyrelu', 'elu', 'selu', 'softplus', 'relu6']
        simple_activations = []
        complex_activations = []
        
        for act in all_activations:
            # Get the actual activation class name
            act_name = ""
            if hasattr(act, 'act_class'):
                # Factory class - check the wrapped class
                act_name = act.act_class.__name__.lower()
            elif hasattr(act, '__name__'):
                # Direct class
                act_name = act.__name__.lower()
            elif isinstance(act, type):
                act_name = act.__name__.lower()
            else:
                # Try string representation
                act_name = str(type(act)).lower()
            
            # Check if it's a simple activation
            if any(simple in act_name for simple in simple_activation_names):
                simple_activations.append(act)
            else:
                complex_activations.append(act)
        
        # If we couldn't categorize, use all activations equally
        if not simple_activations and not complex_activations:
            simple_activations = all_activations[:len(all_activations)//2]
            complex_activations = all_activations[len(all_activations)//2:]
        
        if ratio < 0.5:
            # Early: 70% simple, 30% complex
            activation_pool = simple_activations * 7 + complex_activations * 3 if complex_activations else simple_activations
        elif ratio < 0.8:
            # Mid: 50% simple, 50% complex
            activation_pool = simple_activations * 5 + complex_activations * 5 if complex_activations else simple_activations
        else:
            # Late: 30% simple, 70% complex
            activation_pool = simple_activations * 3 + complex_activations * 7 if complex_activations else simple_activations
        
        if activation_pool:
            curriculum_hp["mlp_activations"] = {
                "distribution": "meta_choice_mixed",
                "choice_values": activation_pool,
            }
            curriculum_hp["conv_activations"] = {
                "distribution": "meta_choice_mixed",
                "choice_values": activation_pool,
            }
            
        # TreeSCM Complexity: Shallow/Few -> Deep/Many
        # Depth: 2 (lambda=0.5) -> 8 (lambda=0.125)
        # Estimators: 2 (lambda=0.5) -> 10 (lambda=0.1)
        
        # tree_depth_lambda: 0.5 (easy) -> 0.125 (hard)
        easy_depth_lam, hard_depth_lam = 0.5, 0.125
        curr_depth_lam = easy_depth_lam + (hard_depth_lam - easy_depth_lam) * ratio
        curriculum_hp["tree_depth_lambda"] = {
             "distribution": "meta_trunc_norm_log_scaled",
             "max_mean": curr_depth_lam, # Use current curriculum value as mean
             "min_mean": hard_depth_lam, # Bound by hardest
             "round": False,
             "lower_bound": 0.01,
        }
        # Actually simpler: just set the value directly or tight range based on ratio?
        # The curriculum dict defines distributions. Let's make it simple:
        # We want the MEAN of the distribution to shift.
        # But `meta_trunc_norm_log_scaled` samples a value.
        # Let's use `meta_log_uniform` for lambda to control it better? 
        # Or just scaling the bounds.
        
        # Depth Lambda: Controls tree depth (max_depth = 2 + exp(1/lambda))
        # Smaller lambda = deeper trees.
        # Early: [0.2, 0.4] (Mean depth ~2 + 3.3 = 5.3) - Increased diversity from [0.4, 0.6]
        # Hard: [0.05, 0.15] (Mean depth ~2 + 10 = 12)
        
        depth_lam_min_easy, depth_lam_max_easy = 0.2, 0.4
        depth_lam_min_hard, depth_lam_max_hard = 0.05, 0.15
        
        d_min = depth_lam_min_easy + (depth_lam_min_hard - depth_lam_min_easy) * ratio
        d_max = depth_lam_max_easy + (depth_lam_max_hard - depth_lam_max_easy) * ratio
        
        curriculum_hp["tree_depth_lambda"] = {
            "distribution": "meta_log_uniform",
            "min": d_min,
            "max": d_max,
        }

        # tree_n_estimators_lambda: 0.5 (easy) -> 0.1 (hard)
        est_lam_min_easy, est_lam_max_easy = 0.4, 0.6
        est_lam_min_hard, est_lam_max_hard = 0.05, 0.15
        
        e_min = est_lam_min_easy + (est_lam_min_hard - est_lam_min_easy) * ratio
        e_max = est_lam_max_easy + (est_lam_max_hard - est_lam_max_easy) * ratio
        
        curriculum_hp["tree_n_estimators_lambda"] = {
            "distribution": "meta_log_uniform",
            "min": e_min,
            "max": e_max,
        }
        
        return curriculum_hp

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Generates a single valid dataset based on the provided parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, device, etc.

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
        elif params["prior_type"] == "time_lagged_scm":
            prior_cls = TimeLaggedSCM
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
                elif prior_cls == TimeLaggedSCM:
                    # Map sampled time-lagged HPs to constructor args so temporal
                    # dynamics vary across datasets instead of staying fixed.
                    seq_len_for_lag = max(2, int(params.get("seq_len", seq_len)))
                    lag_order = int(params.get("time_lagged_lag_order", params.get("lag_order", 3)))
                    params["lag_order"] = min(max(1, lag_order), seq_len_for_lag - 1)
                    params["weight_sparsity"] = float(
                        np.clip(
                            params.get("time_lagged_weight_sparsity", params.get("weight_sparsity", 0.7)),
                            0.0,
                            0.99,
                        )
                    )
                    params["output_noise_std"] = float(
                        max(
                            0.0,
                            params.get("time_lagged_output_noise_std", params.get("output_noise_std", 0.01)),
                        )
                    )
                    logger.debug(
                        "TimeLaggedSCM sampled params: lag_order=%s, weight_sparsity=%.3f, noise_std=%.4f, output_noise_std=%.4f",
                        params["lag_order"],
                        params["weight_sparsity"],
                        float(params.get("noise_std", 0.01)),
                        params["output_noise_std"],
                    )

                X, y = prior_cls(**params)()
                
                # Clean NaNs/Inf from raw SCM output (defensive entry point)
                nan_mask_X = torch.isnan(X) | torch.isinf(X)
                nan_mask_y = torch.isnan(y) | torch.isinf(y)
                if torch.any(nan_mask_X) or torch.any(nan_mask_y):
                    logger.debug(f"Cleaning NaNs/Inf from raw SCM output: X={nan_mask_X.sum().item()}, y={nan_mask_y.sum().item()} NaNs detected")
                    train_size_for_clean = params.get("train_size", X.shape[0] // 2)
                    X_support_raw = X[:train_size_for_clean]
                    X_query_raw = X[train_size_for_clean:]
                    X_support_raw, X_query_raw = self._clean_nan_inf_icl_safe(X_support_raw, X_query_raw)
                    X = torch.cat([X_support_raw, X_query_raw], dim=0)
                    
                    # Clean y: replace NaN/Inf with support set mean (ICL-safe)
                    if torch.any(nan_mask_y):
                        if y.dim() == 0:
                            # Scalar case
                            if torch.isnan(y) or torch.isinf(y):
                                y = torch.tensor(0.0, device=y.device, dtype=y.dtype)
                        else:
                            # Vector case
                            y_support_raw = y[:train_size_for_clean] if y.dim() > 0 else y
                            y_query_raw = y[train_size_for_clean:] if y.dim() > 0 else y
                            y_support_mean = torch.nanmean(y_support_raw)
                            if torch.isnan(y_support_mean):
                                y_support_mean = torch.tensor(0.0, device=y.device, dtype=y.dtype)
                            y = torch.where(torch.isnan(y) | torch.isinf(y), y_support_mean, y)
                
                # SCM might have clamped seq_len internally (e.g. GPSCM caps at 2048)
                # Synchronize train_size to prevent ValueError during normalization
                train_size = params["train_size"]
                actual_seq_len = X.shape[0]
                
                # Log actual sequence length in debug mode (may differ from requested if SCM clamped it)
                if actual_seq_len != seq_len:
                    logger.debug(f"SCM clamped seq_len: requested={seq_len}, actual={actual_seq_len}, prior={prior_type}")
                else:
                    logger.debug(f"Dataset generated: seq_len={actual_seq_len}, train_size={train_size}, prior={prior_type}")
                
                if actual_seq_len < train_size:
                    # If clamped, recalculate train_size to maintain ratio or validation size
                    old_train_size = train_size
                    if self.fixed_val_size is not None:
                        train_size = max(1, actual_seq_len - self.fixed_val_size)
                    else:
                        # Fallback: simple ratio adjustment if original ratio is known
                        # or just cap it at 90% of actual seq_len
                        train_size = int(actual_seq_len * 0.9)
                    
                    logger.debug(
                        f"Adjusted train_size from {old_train_size} to {train_size} dues to "
                        f"SCM clamping (seq_len {params.get('seq_len')} -> {actual_seq_len})"
                    )
                    # Update params to use adjusted train_size for consistency
                    params["train_size"] = train_size

                apply_covariate_shift      = params.get("apply_covariate_shift", False)
                apply_seasonal_drift       = params.get("apply_seasonal_drift", False)
                apply_temporal_drift       = params.get("apply_temporal_drift", False)
                temporal_drift_transition  = params.get("temporal_drift_transition", None)
                finance_realism_rate = float(params.get("finance_realism_rate", 0.0) or 0.0)
                params["_finance_regime"] = bool(
                    finance_realism_rate > 0.0 and np.random.random() < finance_realism_rate
                )
                apply_categorical_attr     = params.get("categorical_attr", False)
                apply_censored_targets = params.get("apply_censored_targets", False)
                apply_cross_sectional_rank = params.get("apply_cross_sectional_rank", False)
                use_strictly_positive = params.get("use_strictly_positive_target", False)

                if self.max_classes == 0 or params.get("num_classes", 0) == 0:
                    # Regression path: keep continuous targets
                    # Use support-set-only normalization (ICL-safe)
                    X, feature_meta = self._process_features_regression(X, params, train_size)

                    X, y = self._apply_confounding_and_spurious_features(X, y, train_size, params)

                    # Covariate shift: randomly shift the query-feature distribution
                    if apply_covariate_shift:
                        X = self._apply_covariate_shift(X, train_size, feature_meta=feature_meta)

                    target_norm_method = params.get("target_norm_method", "zscore")
                    add_skewness = params.get("add_skewness", True)

                    # Censored targets (survival analysis)
                    if apply_censored_targets:
                        X, y = self._apply_censored_targets(X, y, train_size, params, feature_meta)

                    y = self._normalize_continuous_target(
                        y,
                        train_size,
                        norm_method=target_norm_method,
                        add_skewness=add_skewness,
                        hp=params,
                        X=X,
                    )

                    # Seasonal drift: add an exogenous sinusoidal cycle to y
                    # (and optionally to a subset of X features) indexed by row
                    if apply_seasonal_drift:
                        X, y = self._apply_seasonal_drift(X, y, train_size, feature_meta=feature_meta)

                    # Temporal drift: covariate drift at changepoints
                    if apply_temporal_drift:
                        X, y = self._apply_temporal_drift(X, y, train_size,
                                                          transition=temporal_drift_transition,
                                                          feature_meta=feature_meta)

                    # Categorical attribute injection - params["num_features"] is incremented here
                    # so that delete_unique_features (below) naturally includes and
                    # preserves the new column.
                    if apply_categorical_attr:
                        X, y = self._inject_categorical_attribute(X, y, train_size, params, feature_meta)
                        # Re-normalise y (support-set only, ICL-safe) so the group
                        # offsets do not break the scale expected by the model.
                        y_mean = y[:train_size].mean()
                        y_std  = y[:train_size].std().clamp(min=1e-6)
                        y = (y - y_mean) / y_std

                    # Cross-sectional rank normalization
                    if apply_cross_sectional_rank:
                        X = self._apply_cross_sectional_rank_normalization(X, feature_meta, train_size, params)
                    
                    # Check if we should use strictly positive target normalization
                    if use_strictly_positive:
                        y = self._normalize_strictly_positive_target(y, train_size, hp=params)

                else:
                    # Classification path: discretize
                    # Apply advanced feature engineering before Reg2Cls
                    # (quantile transform, interactions, SVD, fingerprints)

                    X, feature_meta = self._process_features_regression(X, params, train_size)
                    X, y = self._apply_confounding_and_spurious_features(X, y, train_size, params)

                    # Covariate shift: randomly shift the query-feature distribution
                    if apply_covariate_shift:
                        X = self._apply_covariate_shift(X, train_size, feature_meta=feature_meta)

                    # Seasonal drift: add an exogenous sinusoidal cycle to y
                    # (and optionally to a subset of X features) indexed by row
                    if apply_seasonal_drift:
                        X, y = self._apply_seasonal_drift(X, y, train_size, feature_meta=feature_meta)

                    # Temporal drift: covariate drift at changepoints
                    if apply_temporal_drift:
                        X, y = self._apply_temporal_drift(X, y, train_size,
                                                          transition=temporal_drift_transition,
                                                          feature_meta=feature_meta)

                    # Categorical attribute injection - params["num_features"] is incremented here
                    # so that delete_unique_features (below) naturally includes and
                    # preserves the new column.
                    if apply_categorical_attr:
                        
                        X, y = self._inject_categorical_attribute(X, y, train_size, params, feature_meta)
                        # Re-normalise y (support-set only, ICL-safe) before Reg2Cls
                        # so the group offsets do not distort the class boundaries.
                        y_mean = y[:train_size].mean()
                        y_std  = y[:train_size].std().clamp(min=1e-6)
                        y = (y - y_mean) / y_std

                    # Cross-sectional rank normalization
                    if apply_cross_sectional_rank:
                        X = self._apply_cross_sectional_rank_normalization(X, feature_meta, train_size, params)

                    # Finance-style latent target dynamics for classification too
                    # (e.g., fraud/default regimes), applied before Reg2Cls binning.
                    y = self._apply_finance_target_dynamics(y, train_size, params)

                    try:
                        X, y = Reg2Cls(params, skip_feature_processing=True)(X, y)
                        y = self._apply_classification_label_noise(y, params)
                    except Exception as e:
                        logger.debug(f"Reg2Cls conversion failed: {e}")
                        raise

                # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
                X, y = X.unsqueeze(0), y.unsqueeze(0)
                # params["num_features"] may have been incremented by
                # _inject_categorical_attribute, so d reflects the updated count.
                d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

                # Only keep valid datasets with sufficient features and balanced classes
                X, d, keep_indices = self.delete_unique_features(X, d)
                # X's columns were left-compacted, so the per-column metadata has to
                # follow or it would describe the wrong columns downstream.
                feature_meta = self._reindex_feature_meta(feature_meta, keep_indices[0])

                # Determine if this is regression based on max_classes or num_classes
                is_regression = (self.max_classes == 0 or params.get("num_classes", 0) == 0)
                
                # Check validation
                d_valid = (d > 0).all()
                if d_valid:
                    # Use adjusted train_size (after SCM clamping) for validation
                    sanity_check_passed = self.sanity_check(X, y, train_size, n_attempts=5, is_regression=is_regression)
                    if sanity_check_passed:
                        if attempt > 0:
                            logger.debug(
                                f"Dataset generation succeeded after {attempt + 1} attempts: "
                                f"prior={prior_type}, seq_len={seq_len}, features={num_features}, "
                                f"train_size={train_size}"
                            )
                        return X.squeeze(0), y.squeeze(0), d.squeeze(0), feature_meta
                    else:
                        if attempt < 5 or attempt % 20 == 0:  # Log first 5 attempts and every 20th
                            # Add GP_SCM-specific logging for debugging
                            if prior_type == "gp_scm":
                                logger.debug(
                                    f"GP_SCM validation failed (attempt {attempt + 1}/{max_attempts}): "
                                    f"seq_len={seq_len}, features={num_features}, train_size={train_size}"
                                )
                            else:
                                logger.debug(
                                    f"Dataset validation failed (attempt {attempt + 1}/{max_attempts}): "
                                    f"prior={prior_type}, seq_len={seq_len}, features={num_features}, "
                                    f"train_size={train_size}, d={d.item()}, is_regression={is_regression}"
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
        train_size = params.get('train_size', 'unknown')
        error_msg = (
            f"Failed to generate valid dataset after {max_attempts} attempts. "
            f"Parameters: seq_len={seq_len}, num_features={num_features}, "
            f"train_size={train_size}, prior_type={prior_type}. "
            f"This may indicate that the validation criteria are too strict for the given parameters. "
            f"Suggestions: "
            f"1. Check if curriculum learning is properly scaling sequence length (min_seq_len should be set), "
            f"2. For GP_SCM with few features, validation may be stricter, "
            f"3. Consider increasing max_attempts or relaxing validation thresholds."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    @staticmethod
    def _validate_tensor(X: Tensor, operation_name: str) -> bool:
        """Validate tensor has no NaN/Inf values.
        
        Parameters
        ----------
        X : Tensor
            Tensor to validate
        operation_name : str
            Name of the operation that produced this tensor (for error messages)
        
        Returns
        -------
        bool
            True if tensor is valid (no NaN/Inf), False otherwise
        """
        has_nan = torch.any(torch.isnan(X))
        has_inf = torch.any(torch.isinf(X))
        if has_nan or has_inf:
            logger.warning(f"{operation_name} produced NaN/Inf values. NaN: {has_nan.item()}, Inf: {has_inf.item()}")
            return False
        return True

    @staticmethod
    def _clean_nan_inf_icl_safe(X_support: Tensor, X_query: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Clean NaN/Inf values using ICL-safe replacement (support stats only).
        
        Uses context-aware defaults: mean per feature from support set.
        Falls back to 0.0 if all values are NaN.
        
        Parameters
        ----------
        X_support : Tensor
            Support set (train) tensor of shape (train_size, n_features)
        X_query : Tensor
            Query set (test) tensor of shape (query_size, n_features)
        
        Returns
        -------
        Tuple[Tensor, Tensor]
            Cleaned (X_support, X_query) with NaNs/Inf replaced by feature means from support set
        """
        # Compute replacement values from support set only (ICL-safe)
        col_mean = torch.nanmean(X_support, dim=0)
        col_mean = torch.nan_to_num(col_mean, nan=0.0, posinf=0.0, neginf=0.0)
        
        def clean_tensor(X: Tensor) -> Tensor:
            """Replace NaN/Inf with column means."""
            nan_mask = torch.isnan(X) | torch.isinf(X)
            if not torch.any(nan_mask):
                return X
            X_clean = X.clone()
            for feat_idx in range(X.shape[1]):
                feat_col = X_clean[:, feat_idx]
                nan_in_col = torch.isnan(feat_col) | torch.isinf(feat_col)
                if torch.any(nan_in_col):
                    replacement = col_mean[feat_idx] if not torch.isnan(col_mean[feat_idx]) else 0.0
                    X_clean[:, feat_idx] = torch.where(
                        nan_in_col,
                        torch.full_like(feat_col, replacement),
                        feat_col
                    )
            return X_clean
        
        return clean_tensor(X_support), clean_tensor(X_query)

    @staticmethod
    def _sample_feature_indices(active_dim: int, rate: float, device: torch.device) -> Tensor:
        """Sample feature indices with Bernoulli(rate), enforcing at least one index if rate > 0."""
        if active_dim <= 0 or rate <= 0.0:
            return torch.empty(0, device=device, dtype=torch.long)
        rate = float(min(max(rate, 0.0), 1.0))
        mask = torch.rand(active_dim, device=device) < rate
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            indices = torch.randint(0, active_dim, (1,), device=device)
        return indices

    @staticmethod
    def _normal_cdf(x: Tensor) -> Tensor:
        """Standard normal CDF."""
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _normalize_probabilities(values: List[float], fallback: List[float]) -> List[float]:
        """Normalize a probability vector; fallback if invalid."""
        probs = [float(max(v, 0.0)) for v in values]
        total = sum(probs)
        if total <= 0.0:
            probs = [float(max(v, 0.0)) for v in fallback]
            total = sum(probs)
        return [v / max(total, 1e-12) for v in probs]

    def _get_missingness_mix_probs(self, hp: Dict[str, Any]) -> List[float]:
        """Return normalized MCAR/MAR/MNAR mixture probabilities."""
        default_probs = [0.45, 0.35, 0.20]
        mix = hp.get("missingness_mix_probs", default_probs)
        if isinstance(mix, (tuple, list)) and len(mix) == 3:
            return self._normalize_probabilities([mix[0], mix[1], mix[2]], default_probs)

        mcar_p = hp.get("missingness_mcar_prob", default_probs[0])
        mar_p = hp.get("missingness_mar_prob", default_probs[1])
        mnar_p = hp.get("missingness_mnar_prob", default_probs[2])
        return self._normalize_probabilities([mcar_p, mar_p, mnar_p], default_probs)

    def _sample_imputation_strategy(self, hp: Dict[str, Any]) -> str:
        """Sample one imputation strategy name."""
        strategy = str(hp.get("missing_imputation_strategy", "mixed")).lower()
        if strategy != "mixed":
            if strategy in ("mean", "median", "constant", "support_sample", "gaussian"):
                return strategy
            return "mean"

        mix = hp.get("missing_imputation_mix_probs", (0.35, 0.20, 0.15, 0.20, 0.10))
        if not isinstance(mix, (tuple, list)) or len(mix) != 5:
            mix = (0.35, 0.20, 0.15, 0.20, 0.10)
        probs = self._normalize_probabilities(list(mix), [0.35, 0.20, 0.15, 0.20, 0.10])
        options = ["mean", "median", "constant", "support_sample", "gaussian"]
        return str(np.random.choice(options, p=probs))

    def _impute_missing_columns(
        self,
        X: Tensor,
        missing_mask: Tensor,
        train_size: int,
        hp: Dict[str, Any],
        active_dim: int,
    ) -> Tuple[Tensor, Tensor]:
        """Impute missing values using support-set-only statistics and sampled strategies."""
        if active_dim <= 0:
            strategy_ids = torch.full((X.shape[1],), IMPUTATION_STRATEGY_TO_ID["none"], device=X.device, dtype=torch.long)
            return X, strategy_ids

        X_out = X.clone()
        strategy_ids = torch.full((X.shape[1],), IMPUTATION_STRATEGY_TO_ID["none"], device=X.device, dtype=torch.long)
        support = X_out[:train_size, :active_dim]
        support_missing = missing_mask[:train_size, :active_dim]

        const_low = float(hp.get("missing_constant_min", -1.0))
        const_high = float(hp.get("missing_constant_max", 1.0))
        if const_high < const_low:
            const_low, const_high = const_high, const_low

        for feat_idx in range(active_dim):
            col_mask = missing_mask[:, feat_idx]
            if not torch.any(col_mask):
                continue

            support_col = support[:, feat_idx]
            observed_support = support_col[~support_missing[:, feat_idx]]
            if observed_support.numel() == 0:
                observed_support = support_col[~(torch.isnan(support_col) | torch.isinf(support_col))]
            if observed_support.numel() == 0:
                observed_support = torch.zeros(1, device=X.device, dtype=X.dtype)

            strategy = self._sample_imputation_strategy(hp)
            if strategy == "median":
                med_val = float(torch.median(observed_support).item())
                fill_values = torch.full((int(col_mask.sum().item()),), med_val, device=X.device, dtype=X.dtype)
            elif strategy == "constant":
                c = float(np.random.uniform(const_low, const_high))
                fill_values = torch.full((int(col_mask.sum().item()),), c, device=X.device, dtype=X.dtype)
            elif strategy == "support_sample":
                sample_idx = torch.randint(0, observed_support.numel(), (int(col_mask.sum().item()),), device=X.device)
                fill_values = observed_support[sample_idx]
            elif strategy == "gaussian":
                obs_mean = observed_support.mean()
                obs_std = observed_support.std().clamp(min=1e-6)
                fill_values = obs_mean + torch.randn(int(col_mask.sum().item()), device=X.device, dtype=X.dtype) * obs_std
            else:  # "mean" default
                mean_val = float(observed_support.mean().item())
                fill_values = torch.full((int(col_mask.sum().item()),), mean_val, device=X.device, dtype=X.dtype)
                strategy = "mean"

            X_out[col_mask, feat_idx] = fill_values
            strategy_ids[feat_idx] = IMPUTATION_STRATEGY_TO_ID.get(strategy, IMPUTATION_STRATEGY_TO_ID["mean"])

        return X_out, strategy_ids

    def _apply_heavy_tail_feature_transform(
        self, X: Tensor, train_size: int, hp: Dict[str, Any], active_dim: int
    ) -> Tensor:
        """Apply per-column heavy-tail transforms (Student-t / Pareto-like) to a feature subset."""
        rate = float(hp.get("heavy_tail_feature_rate", 0.0) or 0.0)
        selected = self._sample_feature_indices(active_dim, rate, X.device)
        if selected.numel() == 0:
            return X

        mode = str(hp.get("heavy_tail_feature_distribution", "mixed")).lower()
        support = X[:train_size, selected]
        support_mean = support.mean(dim=0, keepdim=True)
        support_std = support.std(dim=0, keepdim=True).clamp(min=1e-6)
        z = (X[:, selected] - support_mean) / support_std
        u = self._normal_cdf(z).clamp(min=1e-5, max=1.0 - 1e-5)

        transformed_cols = []
        for col_idx in range(selected.numel()):
            col_u = u[:, col_idx]
            dist_name = mode
            if mode == "mixed":
                dist_name = np.random.choice(["student_t", "pareto"])

            if dist_name == "pareto":
                alpha_min = float(hp.get("pareto_alpha_min", 1.5))
                alpha_max = float(hp.get("pareto_alpha_max", 4.0))
                alpha_low = min(alpha_min, alpha_max)
                alpha_high = max(alpha_min, alpha_max)
                alpha = float(np.random.uniform(alpha_low, alpha_high))
                alpha = max(alpha, 1e-3)

                pareto_pos = torch.pow(1.0 - col_u, -1.0 / alpha) - 1.0
                signs = torch.where(
                    torch.rand_like(pareto_pos) < 0.5,
                    -torch.ones_like(pareto_pos),
                    torch.ones_like(pareto_pos),
                )
                col_transformed = signs * pareto_pos
            else:
                nu_min = float(hp.get("feature_student_t_df_min", 2.0))
                nu_max = float(hp.get("feature_student_t_df_max", 8.0))
                nu_low = min(nu_min, nu_max)
                nu_high = max(nu_min, nu_max)
                nu = max(float(np.random.uniform(nu_low, nu_high)), 2.05)

                concentration = torch.tensor(nu / 2.0, device=X.device, dtype=X.dtype)
                gamma = torch.distributions.Gamma(concentration=concentration, rate=concentration).sample((X.shape[0],))
                col_transformed = z[:, col_idx] / torch.sqrt(gamma.clamp(min=1e-6))

            transformed_cols.append(col_transformed.unsqueeze(1))

        transformed = torch.cat(transformed_cols, dim=1)
        t_support = transformed[:train_size]
        t_mean = t_support.mean(dim=0, keepdim=True)
        t_std = t_support.std(dim=0, keepdim=True).clamp(min=1e-6)
        X[:, selected] = (transformed - t_mean) / t_std
        return X

    def _apply_finance_input_tail_transform(
        self, X: Tensor, train_size: int, hp: Dict[str, Any], active_dim: int
    ) -> Tensor:
        """Inject sparse multiplicative Pareto shocks to emulate fat-tailed market factors."""
        if not bool(hp.get("_finance_regime", False)):
            return X

        rate = float(hp.get("finance_input_tail_rate", 0.0) or 0.0)
        selected = self._sample_feature_indices(active_dim, rate, X.device)
        if selected.numel() == 0:
            return X

        alpha = max(float(hp.get("finance_input_pareto_alpha", 1.8)), 1.05)
        shock_prob = float(np.clip(hp.get("finance_input_shock_prob", 0.06), 0.0, 0.5))
        shock_scale = float(max(hp.get("finance_input_shock_scale", 0.8), 0.0))
        neg_prob = float(np.clip(hp.get("finance_input_negative_shock_prob", 0.35), 0.0, 0.95))

        X_out = X.clone()
        seq_len = X.shape[0]
        for col_idx in selected.tolist():
            col = X_out[:, col_idx]
            jump_mask = torch.rand(seq_len, device=X.device) < shock_prob
            if not torch.any(jump_mask):
                continue

            u = torch.rand(seq_len, device=X.device, dtype=X.dtype).clamp(min=1e-6, max=1.0 - 1e-6)
            pareto = torch.pow(1.0 - u, -1.0 / alpha) - 1.0
            sign = torch.where(
                torch.rand(seq_len, device=X.device, dtype=X.dtype) < neg_prob,
                -torch.ones(seq_len, device=X.device, dtype=X.dtype),
                torch.ones(seq_len, device=X.device, dtype=X.dtype),
            )
            mult = (1.0 + shock_scale * sign * pareto).clamp(min=0.05, max=25.0)
            col = torch.where(jump_mask, col * mult, col)
            X_out[:, col_idx] = col

        support = X_out[:train_size, selected]
        mean = support.mean(dim=0, keepdim=True)
        std = support.std(dim=0, keepdim=True).clamp(min=1e-6)
        X_out[:, selected] = ((X_out[:, selected] - mean) / std).clamp(min=-80.0, max=80.0)
        return X_out

    def _apply_count_feature_transform(
        self, X: Tensor, train_size: int, hp: Dict[str, Any], active_dim: int
    ) -> Tensor:
        """Apply Poisson / Negative-Binomial-like count transforms to a feature subset."""
        rate = float(hp.get("count_feature_rate", 0.0) or 0.0)
        selected = self._sample_feature_indices(active_dim, rate, X.device)
        if selected.numel() == 0:
            return X

        mode = str(hp.get("count_feature_distribution", "mixed")).lower()
        support = X[:train_size, selected]
        support_mean = support.mean(dim=0, keepdim=True)
        support_std = support.std(dim=0, keepdim=True).clamp(min=1e-6)
        z = (X[:, selected] - support_mean) / support_std

        lambda_min = max(float(hp.get("count_lambda_min", 1.0)), 1e-4)
        lambda_max = max(float(hp.get("count_lambda_max", 20.0)), lambda_min + 1e-4)
        log_temp = float(hp.get("count_rate_temperature", 0.75))
        lambda_clip = max(float(hp.get("count_lambda_clip", 1e3)), 1.0)

        base_lambdas = torch.empty(selected.numel(), device=X.device, dtype=X.dtype).uniform_(
            math.log(lambda_min), math.log(lambda_max)
        ).exp()

        transformed = torch.empty_like(z)
        for col_idx in range(selected.numel()):
            lam = (base_lambdas[col_idx] * torch.exp(log_temp * z[:, col_idx])).clamp(min=1e-4, max=lambda_clip)
            dist_name = mode
            if mode == "mixed":
                dist_name = np.random.choice(["poisson", "negative_binomial"])

            if dist_name == "negative_binomial":
                total_count = max(float(hp.get("count_nb_total_count", 5.0)), 0.5)
                concentration = torch.tensor(total_count, device=X.device, dtype=X.dtype)
                gamma_rate = concentration / lam.clamp(min=1e-4)
                lam_nb = torch.distributions.Gamma(concentration=concentration, rate=gamma_rate).sample()
                samples = torch.poisson(lam_nb.clamp(min=1e-4))
            else:
                samples = torch.poisson(lam)

            transformed[:, col_idx] = samples

        if bool(hp.get("count_feature_log1p", False)):
            transformed = torch.log1p(transformed)

        X[:, selected] = transformed
        return X

    def _apply_bounded_feature_transform(
        self, X: Tensor, train_size: int, hp: Dict[str, Any], active_dim: int
    ) -> Tensor:
        """Apply bounded/proportion transforms (sigmoid or beta-like via Kumaraswamy inverse CDF)."""
        rate = float(hp.get("bounded_feature_rate", 0.0) or 0.0)
        selected = self._sample_feature_indices(active_dim, rate, X.device)
        if selected.numel() == 0:
            return X

        mode = str(hp.get("bounded_feature_distribution", "mixed")).lower()
        low = float(hp.get("bounded_feature_min", 0.0))
        high = float(hp.get("bounded_feature_max", 1.0))
        if high <= low:
            high = low + 1.0
        span = high - low

        support = X[:train_size, selected]
        support_mean = support.mean(dim=0, keepdim=True)
        support_std = support.std(dim=0, keepdim=True).clamp(min=1e-6)
        z = (X[:, selected] - support_mean) / support_std
        temperature = float(hp.get("bounded_sigmoid_temperature", 1.0))
        u = torch.sigmoid(temperature * z).clamp(min=1e-5, max=1.0 - 1e-5)

        transformed = torch.empty_like(u)
        for col_idx in range(selected.numel()):
            dist_name = mode
            if mode == "mixed":
                dist_name = np.random.choice(["sigmoid", "beta"])

            if dist_name == "beta":
                alpha_min = max(float(hp.get("bounded_alpha_min", 0.5)), 1e-3)
                alpha_max = max(float(hp.get("bounded_alpha_max", 5.0)), alpha_min + 1e-3)
                beta_min = max(float(hp.get("bounded_beta_min", 0.5)), 1e-3)
                beta_max = max(float(hp.get("bounded_beta_max", 5.0)), beta_min + 1e-3)
                alpha = float(np.random.uniform(min(alpha_min, alpha_max), max(alpha_min, alpha_max)))
                beta = float(np.random.uniform(min(beta_min, beta_max), max(beta_min, beta_max)))
                col_val = torch.pow(1.0 - torch.pow(1.0 - u[:, col_idx], 1.0 / beta), 1.0 / alpha)
            else:
                col_val = u[:, col_idx]

            transformed[:, col_idx] = low + span * col_val

        X[:, selected] = transformed
        return X

    def _apply_feature_discretization(
        self, X: Tensor, train_size: int, hp: Dict[str, Any], active_dim: int
    ) -> Tuple[Tensor, Tensor]:
        """Discretize a subset of features into bins and return ordinal-discretized columns."""
        rate = float(hp.get("discretize_feature_rate", 0.0) or 0.0)
        selected = self._sample_feature_indices(active_dim, rate, X.device)
        if selected.numel() == 0:
            return X, torch.empty(0, device=X.device, dtype=torch.long)

        mode = str(hp.get("discretize_feature_mode", "uniform")).lower()
        as_ordinals = bool(hp.get("discretize_by_ordinal", hp.get("discretize_as_ordinals", True)))
        min_bins = max(int(hp.get("discretize_num_bins_min", 3)), 2)
        max_bins = max(int(hp.get("discretize_num_bins_max", 16)), min_bins)

        X_support = X[:train_size]
        for col_idx in selected.tolist():
            n_bins = int(np.random.randint(min_bins, max_bins + 1))
            col = X[:, col_idx]
            col_support = X_support[:, col_idx]
            if torch.std(col_support) < 1e-8:
                continue

            if mode == "quantile":
                q = torch.linspace(0.0, 1.0, n_bins + 1, device=X.device, dtype=X.dtype)
                edges = torch.quantile(col_support, q)
            else:
                cmin = col_support.min()
                cmax = col_support.max()
                if (cmax - cmin).abs() < 1e-8:
                    continue
                edges = torch.linspace(cmin, cmax, n_bins + 1, device=X.device, dtype=X.dtype)

            inner_edges = edges[1:-1].contiguous()
            buckets = torch.bucketize(col.contiguous(), inner_edges, right=False)

            if as_ordinals:
                X[:, col_idx] = buckets.to(X.dtype)
            else:
                centers = 0.5 * (edges[:-1] + edges[1:])
                X[:, col_idx] = centers[buckets.long().clamp(min=0, max=n_bins - 1)]

        ordinal_cols = selected if as_ordinals else torch.empty(0, device=X.device, dtype=torch.long)
        return X, ordinal_cols

    def _inject_missingness(
        self, X: Tensor, train_size: int, hp: Dict[str, Any], active_dim: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Inject feature-wise MCAR/MAR/MNAR missingness and impute with mixed realistic strategies."""
        missing_rate = float(hp.get("missing_rate", 0.0) or 0.0)
        missing_rate = float(min(max(missing_rate, 0.0), 0.95))

        missing_mask = torch.zeros_like(X, dtype=torch.bool)
        strategy_ids = torch.full((X.shape[1],), IMPUTATION_STRATEGY_TO_ID["none"], device=X.device, dtype=torch.long)
        if active_dim <= 0 or missing_rate <= 0.0:
            return X, missing_mask, strategy_ids

        X_active = X[:, :active_dim]
        support = X_active[:train_size]

        miss_type = str(hp.get("missingness_type", "mixed")).lower()
        miss_type = "mixed" if miss_type not in ("mcar", "mar", "mnar", "mixed") else miss_type

        if miss_type == "mixed":
            mix_probs = self._get_missingness_mix_probs(hp)
            per_feature_mech = np.random.choice(["mcar", "mar", "mnar"], size=active_dim, p=mix_probs).tolist()
        else:
            per_feature_mech = [miss_type] * active_dim

        feature_rate_jitter = float(hp.get("missing_rate_feature_jitter", 0.30))
        feature_rate_jitter = max(feature_rate_jitter, 0.0)
        feature_rate = missing_rate * torch.exp(
            torch.empty(active_dim, device=X.device, dtype=X.dtype).uniform_(-feature_rate_jitter, feature_rate_jitter)
        )
        feature_rate = feature_rate.clamp(min=0.0, max=float(hp.get("missing_max_feature_rate", 0.80)))

        mar_slope = float(hp.get("mar_logit_scale", 1.0))
        mnar_slope = float(hp.get("mnar_logit_scale", 1.5))

        probs = torch.zeros_like(X_active)
        feat_ids = torch.arange(active_dim, device=X.device)
        mar_drivers = torch.randint(0, active_dim, (active_dim,), device=X.device)
        same = mar_drivers == feat_ids
        if active_dim > 1:
            mar_drivers[same] = (mar_drivers[same] + 1) % active_dim

        support_mean = support.mean(dim=0, keepdim=True)
        support_std = support.std(dim=0, keepdim=True).clamp(min=1e-6)
        z_self = (X_active - support_mean) / support_std

        if active_dim > 1:
            driver_values = X_active[:, mar_drivers]
            driver_mean = support[:, mar_drivers].mean(dim=0, keepdim=True)
            driver_std = support[:, mar_drivers].std(dim=0, keepdim=True).clamp(min=1e-6)
            z_driver = (driver_values - driver_mean) / driver_std
        else:
            z_driver = z_self

        skew_proxy = torch.mean(torch.clamp((support - support_mean) / support_std, min=-5.0, max=5.0) ** 3, dim=0, keepdim=True)
        direction = torch.where(skew_proxy >= 0, torch.ones_like(skew_proxy), -torch.ones_like(skew_proxy))

        for feat_idx in range(active_dim):
            base_rate = float(feature_rate[feat_idx].item())
            if base_rate <= 0.0:
                continue
            bias = math.log(base_rate / max(1.0 - base_rate, 1e-6))
            mech = per_feature_mech[feat_idx]
            if mech == "mar":
                col_prob = torch.sigmoid(mar_slope * z_driver[:, feat_idx] + bias)
            elif mech == "mnar":
                col_prob = torch.sigmoid(mnar_slope * direction[:, feat_idx] * z_self[:, feat_idx] + bias)
            else:
                col_prob = torch.full((X.shape[0],), base_rate, device=X.device, dtype=X.dtype)
            probs[:, feat_idx] = col_prob

        probs = probs.clamp(min=0.0, max=0.98)
        mask_active = torch.rand_like(X_active) < probs
        missing_mask[:, :active_dim] = mask_active

        X_imputed, strategy_ids = self._impute_missing_columns(X, missing_mask, train_size, hp, active_dim)
        return X_imputed, missing_mask, strategy_ids

    def _apply_confounding_and_spurious_features(
        self, X: Tensor, y: Tensor, train_size: int, hp: Dict[str, Any]
    ) -> Tuple[Tensor, Tensor]:
        """Inject latent confounding and support-specific spurious correlations."""
        if X.dim() != 2 or y.dim() != 1 or X.shape[1] == 0:
            return X, y

        seq_len, num_features = X.shape

        conf_strength = float(hp.get("confounding_strength", 0.0) or 0.0)
        if conf_strength > 0.0:
            conf_feat_rate = float(hp.get("confounded_feature_rate", 0.25))
            conf_idx = self._sample_feature_indices(num_features, conf_feat_rate, X.device)
            if conf_idx.numel() > 0:
                z = torch.randn(seq_len, device=X.device, dtype=X.dtype)
                z_support = z[:train_size]
                z = (z - z_support.mean()) / z_support.std().clamp(min=1e-6)

                feature_scale = conf_strength * float(hp.get("confound_feature_scale", 1.0))
                target_scale = conf_strength * float(hp.get("confound_target_scale", 1.0))
                X[:, conf_idx] = X[:, conf_idx] + feature_scale * z.unsqueeze(1)
                y = y + target_scale * z.to(y.dtype)

        spurious_rate = float(hp.get("spurious_feature_rate", 0.0) or 0.0)
        if spurious_rate > 0.0:
            spur_idx = self._sample_feature_indices(num_features, spurious_rate, X.device)
            if spur_idx.numel() > 0:
                y_support = y[:train_size]
                y_mean = y_support.mean()
                y_std = y_support.std().clamp(min=1e-6)
                y_norm = (y - y_mean) / y_std

                strength = float(hp.get("spurious_corr_strength", 1.5))
                noise_scale = float(hp.get("spurious_noise_scale", 0.25))
                query_flip_prob = float(hp.get("spurious_query_flip_prob", 0.5))
                query_strength_scale = float(hp.get("spurious_query_strength_scale", 0.5))

                y_support_norm = y_norm[:train_size].to(X.dtype)
                y_query_norm = y_norm[train_size:].to(X.dtype)

                for col_idx in spur_idx.tolist():
                    noise = torch.randn(seq_len, device=X.device, dtype=X.dtype) * noise_scale
                    X[:train_size, col_idx] = strength * y_support_norm + noise[:train_size]
                    if y_query_norm.numel() > 0:
                        sign = -1.0 if np.random.random() < query_flip_prob else 1.0
                        q_strength = sign * strength * query_strength_scale
                        X[train_size:, col_idx] = q_strength * y_query_norm + noise[train_size:]

        return X, y

    def _apply_classification_label_noise(self, y: Tensor, hp: Dict[str, Any]) -> Tensor:
        """Flip a random fraction of labels to another observed class."""
        if y.dim() != 1 or y.numel() == 0:
            return y

        flip_rate = float(hp.get("label_flip_rate", 0.0) or 0.0)
        flip_rate = min(max(flip_rate, 0.0), 0.5)
        if flip_rate <= 0.0:
            return y

        classes = torch.unique(y)
        if classes.numel() <= 1:
            return y

        num_flip = int(round(flip_rate * y.shape[0]))
        if num_flip <= 0:
            return y

        flip_idx = torch.randperm(y.shape[0], device=y.device)[:num_flip]
        y_noisy = y.clone()
        for idx in flip_idx.tolist():
            current = y_noisy[idx]
            alternatives = classes[classes != current]
            if alternatives.numel() == 0:
                continue
            repl_idx = torch.randint(0, alternatives.numel(), (1,), device=y.device)
            y_noisy[idx] = alternatives[repl_idx]
        return y_noisy

    def _apply_heteroscedastic_measurement_noise(
        self, X: Tensor, train_size: int, hp: Dict[str, Any], active_dim: int
    ) -> Tensor:
        """Inject per-feature heteroscedastic measurement noise with log-normal std."""
        if active_dim <= 0 or train_size <= 0:
            return X

        feature_rate = float(hp.get("heteroscedastic_feature_rate", 0.0) or 0.0)
        selected = self._sample_feature_indices(active_dim, feature_rate, X.device)
        if selected.numel() == 0:
            return X

        support = X[:train_size, selected]
        support_mean = support.mean(dim=0, keepdim=True)
        support_std = support.std(dim=0, keepdim=True).clamp(min=1e-6)
        z = (X[:, selected] - support_mean) / support_std

        dependence = float(hp.get("heteroscedastic_dependency_strength", 0.5))
        row_scale = torch.exp(dependence * z.abs()).clamp(min=1.0, max=20.0)

        log_std_mean = float(hp.get("heteroscedastic_log_std_mean", -3.0))
        log_std_std = max(float(hp.get("heteroscedastic_log_std_std", 0.5)), 1e-6)
        global_scale = float(hp.get("heteroscedastic_noise_scale", 1.0))

        per_feature_base_std = torch.exp(
            torch.randn(selected.numel(), device=X.device, dtype=X.dtype) * log_std_std + log_std_mean
        ).clamp(min=1e-5, max=5.0)
        per_feature_base_std = per_feature_base_std * global_scale

        noise_std = row_scale * per_feature_base_std.unsqueeze(0)
        noise = torch.randn_like(z) * noise_std
        X[:, selected] = X[:, selected] + noise
        return X

    def _add_near_duplicate_columns(
        self, X: Tensor, train_size: int, hp: Dict[str, Any], active_dim: int
    ) -> Tensor:
        """Append near-duplicate columns (source + small additive noise)."""
        if active_dim <= 0 or train_size <= 0:
            return X

        rate = float(hp.get("near_duplicate_feature_rate", 0.0) or 0.0)
        if rate <= 0.0:
            return X

        num_new = int(round(active_dim * rate))
        if num_new <= 0:
            return X

        noise_scale = float(hp.get("near_duplicate_noise_scale", 0.05))
        duplicates = []
        for _ in range(num_new):
            src_idx = int(torch.randint(0, active_dim, (1,), device=X.device).item())
            src_col = X[:, src_idx]
            src_std = X[:train_size, src_idx].std().clamp(min=1e-6)
            noise = torch.randn_like(src_col) * (noise_scale * src_std)
            duplicates.append((src_col + noise).unsqueeze(1))

        if duplicates:
            X = torch.cat([X, torch.cat(duplicates, dim=1)], dim=1)
        return X

    def _add_linear_combination_columns(
        self, X: Tensor, train_size: int, hp: Dict[str, Any], active_dim: int
    ) -> Tensor:
        """Append columns sampled as linear combinations of random feature pairs."""
        if active_dim <= 0 or train_size <= 0:
            return X

        rate = float(hp.get("linear_combination_feature_rate", 0.0) or 0.0)
        if rate <= 0.0:
            return X

        num_new = int(round(active_dim * rate))
        if num_new <= 0:
            return X

        noise_scale = float(hp.get("linear_combination_noise_scale", 0.05))
        combos = []
        for _ in range(num_new):
            idx_i = int(torch.randint(0, active_dim, (1,), device=X.device).item())
            if active_dim > 1:
                idx_j = int(torch.randint(0, active_dim - 1, (1,), device=X.device).item())
                if idx_j >= idx_i:
                    idx_j += 1
            else:
                idx_j = idx_i

            col_i = X[:, idx_i]
            col_j = X[:, idx_j]
            weights = torch.randn(2, device=X.device, dtype=X.dtype)
            weights = weights / weights.abs().sum().clamp(min=1e-6)

            combo = weights[0] * col_i + weights[1] * col_j
            support_std = combo[:train_size].std().clamp(min=1e-6)
            eps = torch.randn_like(combo) * (noise_scale * support_std)
            combos.append((combo + eps).unsqueeze(1))

        if combos:
            X = torch.cat([X, torch.cat(combos, dim=1)], dim=1)
        return X

    def _apply_rank_normalized_features(
        self, X: Tensor, train_size: int, hp: Dict[str, Any], active_dim: int
    ) -> Tensor:
        """Replace selected columns by support-CDF rank-normalized values in [0,1]."""
        if active_dim <= 0 or train_size <= 1:
            return X

        rate = float(hp.get("rank_feature_rate", 0.0) or 0.0)
        selected = self._sample_feature_indices(active_dim, rate, X.device)
        if selected.numel() == 0:
            return X

        for feat_idx in selected.tolist():
            support_col = X[:train_size, feat_idx]
            if support_col.std() < 1e-8:
                continue
            sorted_support = torch.sort(support_col.contiguous()).values
            col = X[:, feat_idx].contiguous()
            rank = torch.searchsorted(sorted_support, col, right=True).to(X.dtype)
            X[:, feat_idx] = (rank / float(train_size)).clamp(0.0, 1.0)
        return X

    def _inject_gross_outliers(self, X: Tensor, hp: Dict[str, Any], active_dim: int) -> Tensor:
        """Inject gross outliers by multiplying random cells by large constants."""
        if active_dim <= 0:
            return X

        cell_rate = float(hp.get("gross_outlier_cell_rate", 0.0) or 0.0)
        cell_rate = min(max(cell_rate, 0.0), 0.2)
        if cell_rate <= 0.0:
            return X

        active = X[:, :active_dim].clone()
        mask = torch.rand_like(active) < cell_rate
        if not torch.any(mask):
            ridx = int(torch.randint(0, active.shape[0], (1,), device=X.device).item())
            cidx = int(torch.randint(0, active.shape[1], (1,), device=X.device).item())
            mask[ridx, cidx] = True

        low = max(float(hp.get("gross_outlier_multiplier_min", 8.0)), 1.01)
        high = max(float(hp.get("gross_outlier_multiplier_max", 40.0)), low + 1e-6)
        num_cells = int(mask.sum().item())
        multipliers = torch.exp(
            torch.empty(num_cells, device=X.device, dtype=X.dtype).uniform_(math.log(low), math.log(high))
        )
        signs = torch.where(
            torch.rand(num_cells, device=X.device) < 0.15,
            -torch.ones(num_cells, device=X.device, dtype=X.dtype),
            torch.ones(num_cells, device=X.device, dtype=X.dtype),
        )

        active_vals = active[mask]
        active[mask] = active_vals * multipliers * signs
        X[:, :active_dim] = active.clamp(min=-250.0, max=250.0)
        return X

    @staticmethod
    def _sample_group_probs(num_groups: int, group_imbalance: float) -> Optional[np.ndarray]:
        """Return a probability vector over groups, or None for uniform.

        Four regimes are chosen at random:
        - uniform         : all groups equally likely
        - dominant        : one group gets ``group_imbalance`` extra mass
        - geometric decay : p_k ∝ r^k  for r ~ Uniform(0.5, 0.95)
        - zipf-like       : p_k ∝ 1/(k+1)^α  for α ~ Uniform(0.8, 2.0)
        """
        regime = np.random.choice(["uniform", "dominant", "geometric", "zipf"])
        if regime == "uniform" or group_imbalance == 0.0:
            return None
        if regime == "dominant":
            dominant_p = group_imbalance + (1.0 - group_imbalance) / num_groups
            other_p    = (1.0 - dominant_p) / max(num_groups - 1, 1)
            probs = np.array([dominant_p] + [other_p] * (num_groups - 1), dtype=np.float64)
        elif regime == "geometric":
            r     = np.random.uniform(0.5, 0.95)
            probs = np.array([r ** k for k in range(num_groups)], dtype=np.float64)
        else:  # zipf
            alpha = np.random.uniform(0.8, 2.0)
            probs = np.array([1.0 / (k + 1) ** alpha for k in range(num_groups)], dtype=np.float64)
        probs /= probs.sum()
        return probs

    @staticmethod
    def _sample_group_offsets(
        num_groups: int, scale: float, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Draw group offsets from one of four distributions (zero-mean centred).

        All distributions are parameterised to have unit variance before scaling,
        so ``scale`` is directly interpretable as the offset magnitude:

        - gaussian  : N(0, 1)  (light tails, symmetric)
        - laplace   : Laplace(0, 1/√2)  (heavier tails, same variance as N(0,1))
        - uniform   : Uniform(-√3, √3)  (bounded, flat — no extreme groups)
        - student_t : Student-t with df ~ Uniform(2.5, 6)  (very heavy tails)
        """
        dist_name = np.random.choice(["gaussian", "laplace", "uniform", "student_t"])
        if dist_name == "gaussian":
            raw = torch.randn(num_groups, device=device, dtype=dtype)
        elif dist_name == "laplace":
            # Laplace(0, b) with b = 1/√2 → Var = 2b² = 1
            b   = 1.0 / (2.0 ** 0.5)
            raw = torch.distributions.Laplace(0.0, b).sample((num_groups,)).to(device=device, dtype=dtype)
        elif dist_name == "uniform":
            # Uniform(-√3, √3) → Var = 1
            raw = (torch.rand(num_groups, device=device, dtype=dtype) * 2.0 - 1.0) * (3.0 ** 0.5)
        else:  # student_t
            df  = float(np.random.uniform(2.5, 6.0))
            raw = torch.distributions.StudentT(df).sample((num_groups,)).to(device=device, dtype=dtype)
        offsets = raw * scale
        return offsets - offsets.mean()

    def _inject_categorical_attribute(
        self,
        X: Tensor,
        y: Tensor,
        train_size: int,
        hp: Dict[str, Any],
        feature_meta: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Inject a categorical group attribute and optionally condition y on it.

        **Placement contract** — this method must be called on the raw continuous
        ``y`` (after confounding but *before* ``_normalize_continuous_target`` /
        ``Reg2Cls``) so that:

        * The group effect on ``y`` is absorbed by the subsequent target
          normaliser and results in a properly scaled output distribution.
        * For classification, ``Reg2Cls`` discretises the already
          group-conditioned continuous ``y`` into class boundaries, so the
          group signal propagates to the class labels naturally.

        **Column-slot strategy** — after ``_process_features_regression`` the
        feature matrix has shape ``(seq_len, max_features)``.  Positions
        ``[0 : params["num_features"]]`` are checked by ``delete_unique_features``
        (called later); anything at index ≥ ``params["num_features"]`` is
        discarded regardless.  The new group column is therefore written to
        position ``params["num_features"]`` (which would have been discarded
        anyway) and ``params["num_features"]`` is incremented by 1 so the
        column falls inside the checked range.  A random swap within
        ``[0 : new_num_features]`` removes any positional bias.

        Parameters
        ----------
        X : Tensor, shape (seq_len, max_features)
            Normalised feature matrix from ``_process_features_regression``.
            Positions ``[params["num_features"] : max_features]`` are either
            augmented features (interactions / SVD) that will be discarded by
            ``delete_unique_features``, or zero-padding.
        y : Tensor, shape (seq_len,)
            Raw continuous target values (confounding-adjusted, not yet
            normalised).
        train_size : int
            Number of support-set rows; anchors the y-std estimate so the
            group effect is scale-invariant.
        hp : dict
            Hyperparameter dict.  Relevant keys:

            ``categorical_attr_num_cols`` : int ≥ 1, default 1
                How many independent categorical columns to inject.  Each
                column gets its own group assignments and independently
                conditions y.
            ``categorical_attr_num_groups`` : int in [2, 20], default 2
                Number of groups per column (2 = binary attribute).
            ``categorical_attr_effect`` : str, default ``"offset"``
                Which effect to apply.  One of:

                * ``"offset"``     – group-specific additive bias on y
                  (different intercept per group).
                * ``"projection"`` – group-specific linear combination of X
                  added to y (different slope per group).
                * ``"both"``       – offset **and** projection.
                * ``"none"``       – group column injected but y is unchanged.
            ``categorical_attr_effect_strength`` : float ≥ 0, default 0.5
                Effect magnitude as a multiple of the support-set y std.
            ``categorical_attr_group_imbalance`` : float in [0, 1], default 0.0
                Class imbalance; 0 = uniform, higher = dominant-group skew.
        feature_meta : dict
            Metadata from ``_process_features_regression``.
            ``feature_type_ids`` is updated to mark the new column as
            ``"binary"`` or ``"categorical"``.

        Returns
        -------
        X : Tensor, shape (seq_len, max_features)
            Feature matrix with up to ``categorical_attr_num_cols`` group
            columns inserted at random active positions;
            ``params["num_features"]`` is incremented once per injected
            column.
        y : Tensor, shape (seq_len,)
            Group-conditioned continuous target values.
        """

        seq_len = X.shape[0]
        max_features = X.shape[1]
        device = X.device

        # ── 1. Sample shared configuration ────────────────────────────────────
        rate_min = float(hp.get("categorical_attr_num_cols_rate_min", 0.05))
        rate_max = float(hp.get("categorical_attr_num_cols_rate_max", 0.15))
        rate_max = max(rate_min, rate_max)
        rate = np.random.uniform(rate_min, rate_max)
        num_cols = max(1, round(rate * max_features))

        effect = str(hp.get("categorical_attr_effect", "none"))
        apply_offset = effect in ("offset", "both")
        apply_proj = effect in ("projection", "both")
        # Per-column ranges drawn from the same hp bounds at injection time.
        effect_strength_hp = hp.get("categorical_attr_effect_strength", 2.0)
        num_groups_hp = hp.get("categorical_attr_num_groups", 6)
        group_imbalance_hp = hp.get("categorical_attr_group_imbalance", 0.5)

        # Number of original (non-categorical) features — used as the
        # projection basis so we never project onto previously injected
        # categorical columns.
        orig_slot = int(hp["num_features"])

        # ── Resolve injection slots ────────────────────────────────────────────
        # Prefer free padding slots; if not enough, steal from continuous features.
        padding_free   = max(0, max_features - orig_slot)
        n_from_padding = min(num_cols, padding_free)
        n_to_steal     = num_cols - n_from_padding   # still needed after padding


        # Find slots to steal from the most-represented feature type.
        # Only attempt stealing when there are zero padding slots at all.
        steal_slots: list[int] = []
        if n_from_padding == 0 and n_to_steal > 0:
            fti = feature_meta["feature_type_ids"]
            padding_id = FEATURE_TYPE_TO_ID["padding"]
            # Group active slots by type id (exclude padding).
            from collections import Counter
            type_counts: Counter = Counter()
            slots_by_type: dict[int, list[int]] = {}
            for j in range(orig_slot):
                tid = int(fti[j].item())
                if tid != padding_id:
                    type_counts[tid] += 1
                    slots_by_type.setdefault(tid, []).append(j)
            # Pick the most represented type.
            if type_counts:
                dominant_type_id = type_counts.most_common(1)[0][0]
                dominant_slots   = slots_by_type[dominant_type_id]
                # Keep at least 50 % of that type.
                min_keep  = len(dominant_slots) // 2
                can_steal = max(0, len(dominant_slots) - min_keep)
                n_to_steal = min(n_to_steal, can_steal)   # steal at most 2
                if n_to_steal > 0:
                    steal_slots = list(
                        np.random.choice(dominant_slots, size=n_to_steal, replace=False)
                    )
                    logger.debug(
                        f"[CatInject] stealing {n_to_steal} slot(s) "
                        f"{steal_slots} from type_id={dominant_type_id} "
                        f"(most represented, {len(dominant_slots)} cols)."
                    )

        num_cols = n_from_padding + len(steal_slots)
        if num_cols == 0:
            logger.debug(
                "[CatInject] skipped: no free padding and not enough "
                f"features to steal from (num_features={orig_slot})."
            )
            return X, y

        # Build the list of target slots: padding slots first, then stolen slots.
        target_slots = (
            [orig_slot + i for i in range(n_from_padding)] + steal_slots
        )

        # ── 2. Inject each column ─────────────────────────────────────────────
        # Identify eligible continuous-like features for X-correlation injection.
        fti_tensor = feature_meta["feature_type_ids"]
        #eligible_for_corr = torch.where(
        #    (fti_tensor == FEATURE_TYPE_TO_ID["continuous"])
        #    | (fti_tensor == FEATURE_TYPE_TO_ID["count"])
        #    | (fti_tensor == FEATURE_TYPE_TO_ID["proportion"])
        #)[0]
        #eligible_for_corr = eligible_for_corr[eligible_for_corr < orig_slot]  # active only

        for col_idx in range(num_cols):
            slot = target_slots[col_idx]

            # ── 2a. Per-column num_groups, group_imbalance, effect_strength ──
            if isinstance(num_groups_hp, (list, tuple)):
                num_groups = int(np.random.choice(num_groups_hp))
            else:
                num_groups = int(np.random.randint(2, num_groups_hp + 1))
            #num_groups = max(2, min(num_groups, 500))

            if isinstance(group_imbalance_hp, (list, tuple)):
                group_imbalance = float(np.random.choice(group_imbalance_hp))
            else:
                group_imbalance = float(np.random.uniform(0.0, group_imbalance_hp))

            # ── 2b. Sample group labels from one of four distributions ────────
            probs = self._sample_group_probs(num_groups, group_imbalance)
            g_np  = np.random.choice(num_groups, size=seq_len, p=probs)
            g     = torch.from_numpy(g_np).to(device=device, dtype=torch.long)

            # ── 2c. Inject correlation: shift a subset of eligible X features
            #        per group so the categorical is not independent of X ────
            #if eligible_for_corr.numel() > 0:
            #    corr_frac     = np.random.uniform(0.05, 0.30)
            #    n_corr        = max(1, round(corr_frac * eligible_for_corr.numel()))
            #    perm_c        = torch.randperm(eligible_for_corr.numel(), device=device)[:n_corr]
            #    corr_feats    = eligible_for_corr[perm_c]           # feature indices to correlate
            #    corr_strength = float(np.random.uniform(0.01, 0.6))  # fraction of feature std

            #    X_support_corr = X[:train_size, :]
            #    feat_stds = X_support_corr[:, corr_feats].std(dim=0).clamp(min=1e-6)  # (n_corr,)

                # Independent per-group, per-feature shift drawn from Gaussian
            #    group_feat_shift = (
            #        torch.randn(num_groups, n_corr, device=device, dtype=X.dtype)
            #         * corr_strength
            #         * feat_stds.unsqueeze(0)
            #    )
            #    group_feat_shift = group_feat_shift - group_feat_shift.mean(dim=0, keepdim=True)
            #    X = X.clone()
            #    X[:, corr_feats] = X[:, corr_feats] + group_feat_shift[g]  # (seq_len, n_corr)

            # ── 2d. Group-conditioned effect on raw continuous y ─────────────
            if apply_offset or apply_proj:
                y_support_std = y[:train_size].std().clamp(min=1e-6).item()

                if isinstance(effect_strength_hp, (list, tuple)):
                    effect_strength = float(np.random.choice(effect_strength_hp))
                else:
                    effect_strength = float(np.random.uniform(0.0, effect_strength_hp))

                if apply_offset:
                    # Sample offsets from one of four distributions
                    group_offsets = self._sample_group_offsets(
                        num_groups,
                        scale=effect_strength * y_support_std,
                        device=device,
                        dtype=y.dtype,
                    )
                    y = y + group_offsets[g]

                if apply_proj and orig_slot > 0:
                    proj_std = effect_strength * y_support_std / max(1.0, orig_slot ** 0.5)
                    group_weights = (
                        torch.randn(num_groups, orig_slot, device=device, dtype=X.dtype)
                        * proj_std
                    )
                    group_weights = group_weights - group_weights.mean(dim=0, keepdim=True)
                    proj_effect = (X[:, :orig_slot] * group_weights[g]).sum(dim=-1)
                    y = y + proj_effect.to(y.dtype)

            # ── 2e. Write into the resolved slot ─────────────────────────────
            g_encoded = g.float()   # raw codes {0, 1, ..., num_groups-1}
            attr_type_key = "binary" if num_groups == 2 else "categorical"
            X[:, slot] = g_encoded
            feature_meta["feature_type_ids"][slot] = FEATURE_TYPE_TO_ID[attr_type_key]

            # ── 2f. MCAR missingness + imputation for the categorical column ──
            self._inject_categorical_missingness(
                X, g, g_encoded, slot, train_size, feature_meta, col_idx
            )

            logger.debug(
                f"[CatInject] col {col_idx} → slot {slot}: "
                f"num_groups={num_groups}, effect={effect}, "
                f"imbalance={group_imbalance:.2f}"
            )

        # Only padding slots expand the active feature count; stolen slots
        # replace an existing column so the count stays the same.
        hp["num_features"] = orig_slot + n_from_padding

        # ── 4. Shuffle all active columns to remove positional bias ───────────
        #new_num_features = int(hp["num_features"])
        #perm = np.random.permutation(new_num_features)
        #X[:, :new_num_features] = X[:, perm]
        #fti = feature_meta["feature_type_ids"]
        #feature_meta["feature_type_ids"] = fti[torch.from_numpy(perm).to(fti.device)]

        return X, y

    @staticmethod
    def _inject_categorical_missingness(
        X: Tensor,
        g: Tensor,
        g_encoded: Tensor,
        slot: int,
        train_size: int,
        feature_meta: Dict[str, Tensor],
        col_idx: int = 0,
        miss_prob: float = 0.5,
        miss_rate_range: Tuple[float, float] = (0.05, 0.30),
    ) -> None:
        """Inject MCAR missingness into a single injected categorical column and
        update ``feature_meta`` in-place.

        Called once per categorical column injected by
        ``_inject_categorical_attribute``.  With probability ``miss_prob`` a
        random fraction of entries (drawn uniformly from ``miss_rate_range``) is
        masked as missing.  Missing positions are imputed with one of two
        ICL-safe strategies (both derived from the support set only):

        * ``"constant"``       — replace with the mode (most frequent group label
          in the training split).
        * ``"support_sample"`` — replace by sampling uniformly from the training-
          split group codes, preserving empirical class imbalance.

        Parameters
        ----------
        X : Tensor, shape (seq_len, max_features)
            Feature matrix written to in-place at column ``slot``.
        g : Tensor, shape (seq_len,) long
            Raw group assignment for every row (integer codes 0 … num_groups-1).
        g_encoded : Tensor, shape (seq_len,) float
            ``g`` cast to float; already written to ``X[:, slot]`` by the caller.
        slot : int
            Column index of the categorical column inside ``X``.
        train_size : int
            Number of support-set rows; imputation statistics are computed only
            on rows ``[:train_size]``.
        feature_meta : dict
            Shared metadata dict.  Keys ``"missing_mask"`` and
            ``"imputation_strategy_ids"`` are updated when missingness is
            injected.
        col_idx : int
            Column injection index (used only for logging).
        miss_prob : float
            Probability that this column receives any missingness at all.
        miss_rate_range : Tuple[float, float]
            ``(lo, hi)`` for the per-column MCAR rate sampled uniformly.
        """
        if "missing_mask" not in feature_meta:
            return
        if np.random.random() >= miss_prob:
            return

        seq_len = X.shape[0]
        device  = X.device
        cat_miss_rate = float(np.random.uniform(*miss_rate_range))
        miss_mask_col = torch.rand(seq_len, device=device) < cat_miss_rate  # MCAR

        if not miss_mask_col.any():
            return

        g_train = g[:train_size]   # support-set group codes only
        # ICL-safe: compute sentinel value from training set only.
        # Sentinel = max observed group code in training + 1.0
        mode_val = float(g_train.max().item()) + 1.0
        g_imputed = g_encoded.clone()
        g_imputed[miss_mask_col] = mode_val
        # Write imputed values back and update metadata.
        X[:, slot] = g_imputed
        feature_meta["missing_mask"][:, slot] = miss_mask_col
        feature_meta["imputation_strategy_ids"][slot]  = (
            IMPUTATION_STRATEGY_TO_ID['constant']
        )

        logger.debug(
            f"[CatMissingness] col {col_idx} slot {slot}: "
            f"rate={cat_miss_rate:.2f}, "
            f"n_missing={int(miss_mask_col.sum())}, "
            f"strategy=constant"
        )

    @staticmethod
    def _torch_quantile_transform(
        X_support: Tensor, 
        X_query: Tensor, 
        output_distribution: str = 'uniform',
        n_quantiles: Optional[int] = None
    ) -> Tensor:
        """GPU-native quantile transformation equivalent to sklearn QuantileTransformer.
        
        Maps values to quantiles, then to uniform or normal distribution. Stays entirely on GPU.
        
        Parameters
        ----------
        X_support : Tensor
            Support set (train) tensor of shape (train_size, n_features)
        X_query : Tensor
            Query set (test) tensor of shape (query_size, n_features)
        output_distribution : str, default='uniform'
            Output distribution: 'uniform' or 'normal'
        n_quantiles : int, optional
            Number of quantiles to compute. If None, uses train_size
        
        Returns
        -------
        Tensor
            Transformed features concatenated (support + query) of shape (total_size, n_features)
        """
        if n_quantiles is None:
            n_quantiles = X_support.shape[0]
        n_quantiles = min(n_quantiles, X_support.shape[0])
        n_quantiles = max(2, n_quantiles)  # At least 2 quantiles
        
        # Check for constant features before quantile transform
        feature_stds = X_support.std(dim=0)
        constant_features = feature_stds < 1e-6
        
        # Compute quantile values for each feature from support set (ICL-safe)
        quantile_levels = torch.linspace(0, 1, n_quantiles, device=X_support.device, dtype=X_support.dtype)
        quantile_values = torch.quantile(X_support, quantile_levels, dim=0)  # (n_quantiles, n_features)
        
        # Validate quantile values: ensure no NaN after torch.quantile
        if not SCMPrior._validate_tensor(quantile_values, "Quantile transform (quantile_values)"):
            raise ValueError("Quantile transform produced NaN/Inf in quantile_values")
        
        # Map each value to its quantile position using interpolation (vectorized across all features)
        X_combined = torch.cat([X_support, X_query], dim=0)  # (total_size, n_features)
        
        # Vectorized quantile mapping: for each sample and feature, find quantile position
        # Expand dimensions for broadcasting: (total_size, 1, n_features) vs (1, n_quantiles, n_features)
        X_expanded = X_combined.unsqueeze(1)  # (total_size, 1, n_features)
        q_vals_expanded = quantile_values.unsqueeze(0)  # (1, n_quantiles, n_features)
        
        # Find quantile positions using searchsorted (vectorized per feature)
        # Note: searchsorted doesn't support multi-dim, so we process features in parallel batches
        n_features = X_support.shape[1]
        quantile_positions = torch.zeros(X_combined.shape, device=X_combined.device, dtype=X_combined.dtype)
        
        # Process in batches to avoid memory issues with many features
        batch_size = min(32, n_features)  # Process 32 features at a time
        for batch_start in range(0, n_features, batch_size):
            batch_end = min(batch_start + batch_size, n_features)
            batch_features = slice(batch_start, batch_end)
            
            # Get quantile values for this batch of features
            q_vals_batch = quantile_values[:, batch_features]  # (n_quantiles, batch_size)
            x_vals_batch = X_combined[:, batch_features]  # (total_size, batch_size)
            
            # For each feature in batch, find quantile positions
            for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
                q_vals = q_vals_batch[:, local_idx].contiguous()  # (n_quantiles,) - ensure contiguous for searchsorted
                x_vals = x_vals_batch[:, local_idx].contiguous()  # (total_size,) - ensure contiguous for searchsorted
                
                # Find quantile positions using searchsorted
                indices = torch.searchsorted(q_vals, x_vals, right=False)  # (total_size,)
                indices = torch.clamp(indices, 0, n_quantiles - 1)
                
                # Linear interpolation between quantile bins
                lower_idx = torch.clamp(indices - 1, 0, n_quantiles - 1)
                upper_idx = indices
                
                lower_q = quantile_levels[lower_idx]
                upper_q = quantile_levels[upper_idx]
                lower_val = q_vals[lower_idx]
                upper_val = q_vals[upper_idx]
                
                # Handle edge case where lower_val == upper_val (constant feature)
                val_diff = upper_val - lower_val
                mask = val_diff > 1e-10
                
                # For constant features, map to 0.5 quantile (middle of distribution)
                if constant_features[global_idx]:
                    quantile_pos = torch.full_like(x_vals, 0.5)
                else:
                    # Interpolate quantile level
                    quantile_pos = torch.where(
                        mask,
                        lower_q + (upper_q - lower_q) * (x_vals - lower_val) / val_diff,
                        lower_q  # Use lower quantile if no difference
                    )
                
                # Clamp to [0, 1] range
                quantile_pos = torch.clamp(quantile_pos, 0.0, 1.0)
                
                if output_distribution == 'normal':
                    # Map uniform [0,1] to normal using inverse CDF (erfinv)
                    quantile_pos = torch.clamp(quantile_pos, 1e-7, 1 - 1e-7)  # Avoid edge cases
                    quantile_positions[:, global_idx] = torch.erfinv(2 * quantile_pos - 1) * (2 ** 0.5)
                else:  # 'uniform'
                    quantile_positions[:, global_idx] = quantile_pos
        
        # Validate output: ensure no NaN/Inf after quantile transform
        if not SCMPrior._validate_tensor(quantile_positions, "Quantile transform (output)"):
            raise ValueError("Quantile transform produced NaN/Inf in output")
        
        return quantile_positions

    @staticmethod
    def _torch_yeo_johnson_transform(X_support: Tensor, X_query: Tensor) -> Tensor:
        """GPU-native Yeo-Johnson power transformation equivalent to sklearn PowerTransformer.
        
        Implements Yeo-Johnson formula directly in torch. Stays entirely on GPU.
        
        Parameters
        ----------
        X_support : Tensor
            Support set (train) tensor of shape (train_size, n_features)
        X_query : Tensor
            Query set (test) tensor of shape (query_size, n_features)
        
        Returns
        -------
        Tensor
            Transformed features concatenated (support + query) of shape (total_size, n_features)
        """
        # Combine for processing
        X_combined = torch.cat([X_support, X_query], dim=0)  # (total_size, n_features)
        X_transformed = torch.zeros_like(X_combined)
        
        # Estimate lambda using simplified approach (sklearn uses MLE, we use variance minimization)
        # For speed, use a fixed lambda or simple grid search on GPU
        # Remove -2.0 and -1.0 to prevent Inf (use safer range [-0.5, 2.0])
        lambda_candidates = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0], device=X_support.device, dtype=X_support.dtype)
        
        best_lambda = 0.5  # Default
        best_variance = float('inf')
        
        # Simple grid search: find lambda that minimizes variance (simplified MLE)
        for lambda_val in lambda_candidates:
            X_test = X_support.clone()
            X_pos = X_test.clamp(min=0)
            X_neg = X_test.clamp(max=0)
            
            # Apply Yeo-Johnson formula
            if abs(lambda_val) < 1e-6:
                transformed_pos = torch.log1p(X_pos)
            else:
                transformed_pos = (torch.pow(X_pos + 1, lambda_val) - 1) / lambda_val
            
            if abs(2 - lambda_val) < 1e-6:
                transformed_neg = -torch.log1p(-X_neg)
            else:
                transformed_neg = -(torch.pow(-X_neg + 1, 2 - lambda_val) - 1) / (2 - lambda_val)
            
            X_test_transformed = transformed_pos + transformed_neg
            
            # Validate transformed tensor before using in variance calculation
            if not SCMPrior._validate_tensor(X_test_transformed, f"Power transform (lambda={lambda_val.item()})"):
                continue  # Skip this lambda if it produces NaN/Inf
            
            # Check variance (lower is better for normalization)
            var = X_test_transformed.var().item()
            if var < best_variance:
                best_variance = var
                best_lambda = lambda_val.item()
        
        # Apply transformation with best lambda
        # Data should already be cleaned/normalized, so we only clamp the base of pow operations
        # to prevent overflow, not the input values themselves
        X_pos = X_combined.clamp(min=0)
        X_neg = X_combined.clamp(max=0)
        
        if abs(best_lambda) < 1e-6:
            transformed_pos = torch.log1p(X_pos)
        else:
            # Add epsilon to prevent Inf when lambda is negative and x+1 is near 0
            # Clamp base to prevent overflow: (x+1)^lambda can overflow for large x
            # For normalized data, x is typically in [-3, 3], so (x+1) is in [-2, 4]
            # Clamp to reasonable range to prevent overflow with extreme lambda values
            eps = 1e-6
            base_pos = (X_pos + 1 + eps).clamp(min=1e-10, max=1e6)  # Prevent overflow/underflow in pow
            transformed_pos = (torch.pow(base_pos, best_lambda) - 1) / best_lambda
        
        if abs(2 - best_lambda) < 1e-6:
            transformed_neg = -torch.log1p(-X_neg)
        else:
            # Clamp base to prevent overflow: (-x+1)^(2-lambda) can overflow for large negative x
            # For normalized data, -x is typically in [0, 3], so (-x+1) is in [1, 4]
            base_neg = (-X_neg + 1).clamp(min=1e-10, max=1e6)  # Prevent overflow/underflow in pow
            transformed_neg = -(torch.pow(base_neg, 2 - best_lambda) - 1) / (2 - best_lambda)
        
        X_transformed = transformed_pos + transformed_neg
        
        # Validate transformed tensor before standardization
        if not SCMPrior._validate_tensor(X_transformed, "Power transform (before standardization)"):
            raise ValueError("Power transform produced NaN/Inf values")
        
        # Standardize using support set statistics only (ICL-safe, equivalent to sklearn standardize=True)
        mean = X_transformed[:X_support.shape[0]].mean(dim=0, keepdim=True)
        std = X_transformed[:X_support.shape[0]].std(dim=0, keepdim=True).clip(min=1e-6)
        X_transformed = (X_transformed - mean) / std
        
        # Validate final output
        if not SCMPrior._validate_tensor(X_transformed, "Power transform (final output)"):
            raise ValueError("Power transform produced NaN/Inf after standardization")
        
        return X_transformed

    @staticmethod
    def _transform_count_features(
        X_support: Tensor,
        X_query: Tensor,
        distribution: str = 'auto',
        lam: Optional[float] = None,
        r: Optional[float] = None,
        p: Optional[float] = None,
        max_count: int = 100,
        feature_fraction: Optional[float] = None,
        skip_standardize: bool = False,
    ) -> Tensor:
        """Post-SCM count feature transform via Poisson or NegativeBinomial inverse CDF.

        Maps a random subset of continuous SCM features → non-negative integer counts
        by treating the standardized value as a normal quantile and applying the
        discrete distribution's inverse CDF (quantile function). Untransformed
        features are returned unchanged. Fully ICL-safe: all standardization
        statistics are derived from the support set only.

        Parameters
        ----------
        X_support : Tensor
            Support set tensor of shape (train_size, n_features).
        X_query : Tensor
            Query set tensor of shape (query_size, n_features).
        distribution : str, default='auto'
            'poisson', 'negative_binomial', or 'auto' (randomly chosen per call).
        lam : float, optional
            Poisson rate λ. If None, sampled independently per feature from
            Uniform(0.5, 20).
        r : float, optional
            NegativeBinomial total_count r. If None, sampled from Uniform(1, 10).
        p : float, optional
            NegativeBinomial success probability (probs). If None, sampled from
            Uniform(0.2, 0.8).
        max_count : int, default=100
            Maximum count value in the CDF lookup table. Values that would exceed
            this are clamped to max_count.
        feature_fraction : float, optional
            Fraction of features to transform, in (0, 1]. If None, sampled
            from Uniform(0.1, 0.4) so only a realistic subset becomes counts.
        skip_standardize : bool, default=False
            If True, skip the internal standardization step. Set this to True
            when the input is already normalized (e.g. called after
            ``_process_features_regression`` Step 6), to avoid redundant work.

        Returns
        -------
        Tensor
            Features of shape (total_size, n_features). Selected features contain
            count values in {0, …, max_count} (cast to float); the remaining
            features are left as the original SCM output.
        """
        X_combined = torch.cat([X_support, X_query], dim=0)
        n_features = X_combined.shape[1]
        device = X_combined.device
        dtype = X_combined.dtype

        # Decide which features to transform
        if feature_fraction is None:
            feature_fraction = float(np.random.uniform(0.1, 0.4))
        feature_fraction = float(np.clip(feature_fraction, 1.0 / max(n_features, 1), 1.0))
        n_transform = max(1, round(feature_fraction * n_features))
        selected = sorted(np.random.choice(n_features, size=n_transform, replace=False).tolist())

        # Start from a copy of the original combined tensor; only selected cols are overwritten
        out = X_combined.clone()

        X_sel = X_combined[:, selected]                             # (total, n_transform)
        if skip_standardize:
            # Input is already normalized (e.g. post Step-6 in _process_features_regression)
            X_z = X_sel
        else:
            # Standardize using support-set stats only (ICL-safe) so that Φ(z)
            # spreads values across [0,1] rather than collapsing to one extreme.
            mean_s = X_support[:, selected].mean(dim=0, keepdim=True)
            std_s = X_support[:, selected].std(dim=0, keepdim=True).clamp(min=1e-6)
            X_z = (X_sel - mean_s) / std_s

        # Map z-scores → uniform [0,1] via normal CDF Φ(z) = 0.5*(1 + erf(z/√2))
        u = 0.5 * (1.0 + torch.erf(X_z / math.sqrt(2.0)))
        u = u.clamp(1e-7, 1.0 - 1e-7)

        # Resolve distribution
        if distribution == 'auto':
            distribution = np.random.choice(['poisson', 'negative_binomial'])

        # Per-feature inverse CDF via a precomputed CDF lookup table
        counts_grid = torch.arange(0, max_count + 1, device=device, dtype=dtype)  # (max_count+1,)

        for local_idx in range(n_transform):
            if distribution == 'poisson':
                lam_f = lam if lam is not None else float(np.random.uniform(0.5, 20.0))
                dist = torch.distributions.Poisson(
                    rate=torch.tensor(lam_f, device=device, dtype=dtype)
                )
            else:  # negative_binomial
                r_f = r if r is not None else float(np.random.uniform(1.0, 10.0))
                p_f = p if p is not None else float(np.random.uniform(0.2, 0.8))
                dist = torch.distributions.NegativeBinomial(
                    total_count=torch.tensor(r_f, device=device, dtype=dtype),
                    probs=torch.tensor(p_f, device=device, dtype=dtype),
                )

            # Build CDF table: CDF[k] = P(X ≤ k) = Σ_{j=0}^{k} PMF[j]
            log_pmfs = dist.log_prob(counts_grid)           # (max_count+1,)
            pmfs = log_pmfs.exp().clamp(min=0.0)
            cdf = torch.cumsum(pmfs, dim=0)
            cdf = (cdf / cdf[-1].clamp(min=1e-10)).clamp(0.0, 1.0)

            # Inverse CDF: k = min{k : CDF[k] ≥ u}  (right=False → first crossing)
            u_feat = u[:, local_idx].contiguous()
            indices = torch.searchsorted(cdf.contiguous(), u_feat, right=False)
            out[:, selected[local_idx]] = counts_grid[indices.clamp(0, max_count)]

        if not SCMPrior._validate_tensor(out, "Count transform"):
            raise ValueError("Count transform produced NaN/Inf values")

        return out

    @staticmethod
    def _transform_proportion_features(
        X_support: Tensor,
        X_query: Tensor,
        method: str = 'auto',
        alpha: Optional[float] = None,
        beta_param: Optional[float] = None,
        feature_fraction: Optional[float] = None,
        skip_standardize: bool = False,
    ) -> Tensor:
        """Post-SCM proportion/bounded feature transform → [0,1] via sigmoid or Beta ICDF.

        Maps a random subset of continuous SCM features → proportions in [0,1] using
        either:

        - ``'sigmoid'``: standardize then apply σ(z), giving a logistic-shaped
          distribution concentrated away from 0 and 1.
        - ``'beta'``: map z-scores to uniform [0,1] via the normal CDF, then apply
          the Beta(α,β) inverse CDF, giving a Beta-distributed proportion with fully
          controllable skew and concentration.

        Untransformed features are returned unchanged.
        Fully ICL-safe: all standardization statistics come from the support set only.

        Parameters
        ----------
        X_support : Tensor
            Support set tensor of shape (train_size, n_features).
        X_query : Tensor
            Query set tensor of shape (query_size, n_features).
        method : str, default='auto'
            'sigmoid', 'beta', or 'auto' (randomly chosen per call).
        alpha : float, optional
            Beta distribution α shape parameter. If None, sampled independently
            per feature from LogUniform(0.5, 5).
        beta_param : float, optional
            Beta distribution β shape parameter. If None, sampled independently
            per feature from LogUniform(0.5, 5).
        feature_fraction : float, optional
            Fraction of features to transform, in (0, 1]. If None, sampled from
            Uniform(0.1, 0.4) so only a realistic subset becomes proportions.
        skip_standardize : bool, default=False
            If True, skip the internal standardization step. Set this to True
            when the input is already normalized (e.g. called after
            ``_process_features_regression`` Step 6), to avoid redundant work.

        Returns
        -------
        Tensor
            Features of shape (total_size, n_features). Selected features contain
            proportion values in [0,1]; the remaining features are left as the
            original SCM output.
        """
        from scipy.special import betaincinv

        X_combined = torch.cat([X_support, X_query], dim=0)
        n_features = X_combined.shape[1]
        device = X_combined.device
        dtype = X_combined.dtype

        # Decide which features to transform
        if feature_fraction is None:
            feature_fraction = float(np.random.uniform(0.1, 0.4))
        feature_fraction = float(np.clip(feature_fraction, 1.0 / max(n_features, 1), 1.0))
        n_transform = max(1, round(feature_fraction * n_features))
        selected = sorted(np.random.choice(n_features, size=n_transform, replace=False).tolist())

        # Start from a copy of the original combined tensor; only selected cols are overwritten
        out = X_combined.clone()

        X_sel = X_combined[:, selected]                             # (total, n_transform)
        if skip_standardize:
            # Input is already normalized (e.g. post Step-6 in _process_features_regression)
            X_z = X_sel
        else:
            # Standardize using support-set stats only (ICL-safe) so that Φ(z) or σ(z)
            # spans [0,1] meaningfully rather than collapsing to one extreme.
            mean_s = X_support[:, selected].mean(dim=0, keepdim=True)
            std_s = X_support[:, selected].std(dim=0, keepdim=True).clamp(min=1e-6)
            X_z = (X_sel - mean_s) / std_s

        # Resolve method
        if method == 'auto':
            method = np.random.choice(['sigmoid', 'beta'])

        if method == 'sigmoid':
            # σ(z) maps all of ℝ → (0,1) with a logistic shape
            out[:, selected] = torch.sigmoid(X_z)

        else:  # beta
            # Map z-scores → uniform [0,1] via normal CDF
            u = 0.5 * (1.0 + torch.erf(X_z / math.sqrt(2.0)))
            u = u.clamp(1e-7, 1.0 - 1e-7)

            # Apply Beta(α,β) inverse CDF per feature via scipy (betaincinv).
            # betaincinv(a, b, p) solves betainc(a, b, x) = p for x ∈ [0,1].
            # Run on CPU then transfer back to device.
            u_np = u.cpu().numpy()

            for local_idx, global_idx in enumerate(selected):
                a = alpha if alpha is not None else float(
                    np.exp(np.random.uniform(np.log(0.5), np.log(5.0)))
                )
                b = beta_param if beta_param is not None else float(
                    np.exp(np.random.uniform(np.log(0.5), np.log(5.0)))
                )
                feat_vals = betaincinv(a, b, u_np[:, local_idx]).astype(np.float32)
                out[:, global_idx] = torch.from_numpy(feat_vals).to(device=device, dtype=dtype)

        out[:, selected] = out[:, selected].clamp(0.0, 1.0)

        # Normalise so that the selected proportion features sum to 1 per row
        # (compositional constraint: they represent parts of a whole that add to 100%).
        
        #row_sum = out[:, selected].sum(dim=1, keepdim=True).clamp(min=1e-8)
        #out[:, selected] = out[:, selected] / row_sum

        if not SCMPrior._validate_tensor(out, "Proportion transform"):
            raise ValueError("Proportion transform produced NaN/Inf values")

        return out

    def _process_features_regression(
        self, X: Tensor, hp: Dict[str, Any], train_size: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Process features for regression using support-set-only normalization (ICL-safe).
        
        Following best practices: normalize using only support set (train)
        statistics to prevent ICL leakage, then apply same normalization to query set.
        
        CRITICAL: All preprocessing statistics (outlier removal, normalization) are computed
        from support set only to ensure ICL safety.
        
        Parameters
        ----------
        X : Tensor
            Features tensor of shape (seq_len, num_features)
        hp : Dict[str, Any]
            Hyperparameters dictionary. May include:
            - use_quantile_transform (bool, default=False): If True, apply quantile transform
              before standard normalization (MITRA-style)
            - feature_transformation (str, optional): Transformation strategy. If None, randomly samples:
              'quantile_uniform' (30%), 'quantile_normal' (30%), 'power_yeo_johnson' (20%),
              'log_normal' (15%), 'zscore' (5%). Limix-based-Changes
            - add_interaction_features (bool, optional): If None, randomly adds polynomial interactions
              (40-45% probability). Limix-based-Changes
            - add_svd_features (bool, optional): If None, randomly adds SVD-compressed features
              (20-25% probability). Limix-based-Changes
            - add_fingerprint_feature (bool, optional): If None, randomly adds hash fingerprint feature
              (30-35% probability). Limix-based-Changes
        train_size : int
            Size of support set (train portion) for ICL-safe normalization
            
        Returns
        -------
        Tuple[Tensor, Dict[str, Tensor]]
            Processed features plus metadata:
            - ``missing_mask``: Bool mask of imputed cells, shape (seq_len, max_features)
            - ``feature_type_ids``: Per-column semantic type ids, shape (max_features,)
            - ``col_ids``: Randomized per-dataset column ids (padding=0), shape (max_features,)
            - ``imputation_strategy_ids``: Per-column imputation strategy id, shape (max_features,)
        """
        from .reg2cls import outlier_removing
        from .reg2cls import torch_nanstd
        
        # CRITICAL: Split FIRST to prevent any leakage
        # Support set (train) - used for computing ALL statistics
        # Query set (test) - normalized using support stats only
        X_support = X[:train_size]
        X_query = X[train_size:]
        
        # Entry cleaning - catch any NaNs from SCM or previous steps
        nan_mask_support = torch.isnan(X_support) | torch.isinf(X_support)
        nan_mask_query = torch.isnan(X_query) | torch.isinf(X_query)
        if torch.any(nan_mask_support) or torch.any(nan_mask_query):
            logger.debug("Cleaning NaNs/Inf at feature processing entry point")
            X_support, X_query = self._clean_nan_inf_icl_safe(X_support, X_query)
        
        # Edge case: ensure valid train_size
        if train_size <= 0 or train_size >= X.shape[0]:
            raise ValueError(f"Invalid train_size {train_size} for seq_len {X.shape[0]}")
        
        # Step 1: Remove outliers from support set only (ICL-safe)
        X_support_clean = outlier_removing(X_support, threshold=4)
        
        # Step 2: Compute outlier thresholds from support set
        # Then apply same thresholds to query set (ICL-safe)
        mean_support = torch.nanmean(X_support_clean, dim=0)
        std_support = torch_nanstd(X_support_clean, dim=0, ddof=1 if X_support_clean.shape[0] > 1 else 0).clip(min=1e-6)
        threshold = 4.0
        cut_off = std_support * threshold
        lower = mean_support - cut_off
        upper = mean_support + cut_off
        
        # Apply support-set thresholds to query set
        X_query_clean = torch.clamp(X_query, min=lower, max=upper)
        
        # Recombine cleaned data
        X_clean = torch.cat([X_support_clean, X_query_clean], dim=0)
        
        # Step 3: Optional quantile transform (MITRA-style, ICL-safe)
        # If enabled, apply quantile transform before standard normalization
        use_quantile_transform = hp.get("use_quantile_transform", False)
        if use_quantile_transform:
            try:
                # GPU-optimized: Use torch quantile transform instead of sklearn (stays on GPU)
                X_clean = self._torch_quantile_transform(
                    X_support_clean, 
                    X_query_clean,
                    output_distribution='normal',
                    n_quantiles=min(1000, train_size)
                )
                # Update support and query for consistency
                X_support_clean = X_clean[:train_size]
                X_query_clean = X_clean[train_size:]
            except Exception as e:
                # Fallback if transform fails
                logger.warning(f"Quantile transform failed: {e}. Continuing without transform.")
        
        # Step 4-5: Limix-based-Changes - Feature transformation diversity (replaces standard z-score)
        # Real-world tabular data has diverse distribution shapes (normal, log-normal, uniform, skewed)
        # Solution: Use multiple transformation strategies like LimiX (quantile, power, log-normal, etc.)
        transformation_strategy = hp.get("feature_transformation", None)
        if transformation_strategy is None:
            # Sample transformation strategy based on LimiX's distribution (30% quantile, 30% power, 20% log-normal, 20% z-score)
            strategy_rand = np.random.random()
            if strategy_rand < 0.30:
                transformation_strategy = 'quantile_uniform'
            elif strategy_rand < 0.60:
                transformation_strategy = 'quantile_normal'
            elif strategy_rand < 0.75:
                transformation_strategy = 'power_yeo_johnson'
            elif strategy_rand < 0.85:
                transformation_strategy = 'log_normal'
            else:
                transformation_strategy = 'zscore'  # Default (current approach)
        
        # Limix-based-Changes - Apply diverse transformations with robust error handling
        try:
            if transformation_strategy == 'quantile_uniform':
                # GPU-optimized: Use torch quantile transform instead of sklearn (stays on GPU)
                # Limix-based-Changes - Adaptive quantile count (like LimiX: max(n_samples // 10, 2))
                n_quantiles = max(train_size // 10, 2)
                X_normalized = self._torch_quantile_transform(
                    X_support_clean,
                    X_query_clean,
                    output_distribution='uniform',
                    n_quantiles=min(n_quantiles, train_size)
                )
                
            elif transformation_strategy == 'quantile_normal':
                # GPU-optimized: Use torch quantile transform instead of sklearn (stays on GPU)
                # Limix-based-Changes - Adaptive quantile count
                n_quantiles = max(train_size // 10, 2)
                X_normalized = self._torch_quantile_transform(
                    X_support_clean,
                    X_query_clean,
                    output_distribution='normal',
                    n_quantiles=min(n_quantiles, train_size)
                )
                
            elif transformation_strategy == 'power_yeo_johnson':
                # GPU-optimized: Use torch Yeo-Johnson transform instead of sklearn (stays on GPU, no overflow warnings)
                # Limix-based-Changes - PowerTransformer (Yeo-Johnson) like LimiX's RobustPowerTransformer
                # LimiX Insight: Power transforms handle skewed distributions better than z-score
                try:
                    X_normalized = self._torch_yeo_johnson_transform(X_support_clean, X_query_clean)
                    # Check for failures (NaN/Inf)
                    if torch.any(torch.isnan(X_normalized)) or torch.any(torch.isinf(X_normalized)):
                        raise ValueError("Power transform produced NaN/Inf")
                except (ValueError, RuntimeError) as e:
                    # Limix-based-Changes - revert to z-score if power transform fails
                    logger.warning(f"Power transform failed, falling back to z-score: {e}")
                    mean = torch.nanmean(X_support_clean, dim=0)
                    std = torch_nanstd(X_support_clean, dim=0, ddof=1 if X_support_clean.shape[0] > 1 else 0).clip(min=1e-6)
                    X_normalized = (X_clean - mean) / std
                    
            elif transformation_strategy == 'log_normal':
                # Limix-based-Changes - Log-normal transformation like LimiX's logNormal worker
                # LimiX Insight: Many real-world features are log-normally distributed
                # Use more conservative global min to prevent negative values
                global_min = X_support_clean.min()  # Global min across all features
                # Use larger epsilon and ensure positive values
                shift_value = global_min - 1e-6  # More conservative: subtract a bit more
                epsilon = 1e-6  # Larger epsilon for stability
                
                # Shift to positive values (more conservative)
                X_support_shifted = X_support_clean - shift_value + epsilon
                X_query_shifted = X_query_clean - shift_value + epsilon
                
                # Clamp to ensure all values are positive before log
                X_support_shifted = X_support_shifted.clamp(min=epsilon)
                X_query_shifted = X_query_shifted.clamp(min=epsilon)
                
                # Apply log transform (now safe from -inf)
                X_support_log = torch.log(X_support_shifted)
                X_query_log = torch.log(X_query_shifted)
                
                # Check for any remaining issues
                if torch.any(torch.isnan(X_support_log)) or torch.any(torch.isinf(X_support_log)):
                    raise ValueError("Log transform produced NaN/Inf after clamping")
                if torch.any(torch.isnan(X_query_log)) or torch.any(torch.isinf(X_query_log)):
                    raise ValueError("Log transform produced NaN/Inf after clamping")
                
                # Normalize after log transform (ICL-safe: stats from support only)
                mean_log = torch.nanmean(X_support_log, dim=0)
                std_log = torch_nanstd(X_support_log, dim=0, ddof=1 if X_support_log.shape[0] > 1 else 0).clip(min=1e-6)
                X_support_normalized = (X_support_log - mean_log) / std_log
                X_query_normalized = (X_query_log - mean_log) / std_log
                X_normalized = torch.cat([X_support_normalized, X_query_normalized], dim=0)
                
            else:  # 'zscore' (default, current approach)
                # Standard z-score normalization (original approach)
                mean = torch.nanmean(X_support_clean, dim=0)
                std = torch_nanstd(X_support_clean, dim=0, ddof=1 if X_support_clean.shape[0] > 1 else 0).clip(min=1e-6)
                X_normalized = (X_clean - mean) / std
                
        except Exception as e:
            # Limix-based-Changes - fallback to z-score on any transformation failure
            logger.warning(f"Feature transformation '{transformation_strategy}' failed: {e}. Falling back to z-score.")
            mean = torch.nanmean(X_support_clean, dim=0)
            std = torch_nanstd(X_support_clean, dim=0, ddof=1 if X_support_clean.shape[0] > 1 else 0).clip(min=1e-6)
            X_normalized = (X_clean - mean) / std
        
        # Step 6: Clip extreme outliers (post-normalization)
        X_normalized = torch.clip(X_normalized, min=-100, max=100)
        
        # Clean NaNs/Inf after clipping (clipping preserves NaNs)
        nan_mask = torch.isnan(X_normalized) | torch.isinf(X_normalized)
        if torch.any(nan_mask):
            X_support_norm = X_normalized[:train_size]
            X_query_norm = X_normalized[train_size:]
            X_support_norm, X_query_norm = self._clean_nan_inf_icl_safe(X_support_norm, X_query_norm)
            X_normalized = torch.cat([X_support_norm, X_query_norm], dim=0)
        
        # Step 6b: Optionally cast a subset of features to count or proportion type.
        # Applied after normalization (input is already ~N(0,1)) so skip_standardize=True.
        # Mutually exclusive: a dataset gets either count features, proportion features, or neither.
        add_count_features = hp.get("add_count_features", None)
        add_proportion_features = hp.get("add_proportion_features", None)
        skip_standardize = True

        if add_count_features:
            try:
                max_count = np.random.randint(2, 100)
                feature_fraction = np.random.uniform(0.05, 0.2)

                X_normalized = self._transform_count_features(
                    X_normalized[:train_size],
                    X_normalized[train_size:],
                    max_count=max_count,
                    feature_fraction=feature_fraction,
                    skip_standardize=skip_standardize,
                )

                skip_standardize = False
            except Exception as e:
                logger.warning(f"Count transform failed: {e}. Continuing without it.")

        if add_proportion_features:
            try:
                feature_fraction = np.random.uniform(0.05, 0.2)
                
                X_normalized = self._transform_proportion_features(
                    X_normalized[:train_size],
                    X_normalized[train_size:],
                    skip_standardize=skip_standardize,
                )
            except Exception as e:
                logger.warning(f"Proportion transform failed: {e}. Continuing without it.")

        # Step 6c: Optional per-feature heteroscedastic measurement noise.
        X_normalized = self._apply_heteroscedastic_measurement_noise(
            X_normalized, train_size, hp, X_normalized.shape[1]
        )

        # Validate input before interactions/SVD (defensive check)
        nan_mask_pre = torch.isnan(X_normalized) | torch.isinf(X_normalized)
        if torch.any(nan_mask_pre):
            logger.warning("X_normalized contains NaN/Inf before interactions/SVD. Cleaning first.")
            X_support_norm = X_normalized[:train_size]
            X_query_norm = X_normalized[train_size:]
            X_support_norm, X_query_norm = self._clean_nan_inf_icl_safe(X_support_norm, X_query_norm)
            X_normalized = torch.cat([X_support_norm, X_query_norm], dim=0)
        
        # Limix-based-Changes - Add polynomial interaction features (LimiX's PolynomialInteractionGenerator)
        # Real-world data has feature interactions (e.g., price × area = value)
        # Problem: Current SCMs generate mostly additive relationships
        # Solution: Add explicit polynomial interactions (X_i * X_j) like LimiX
        add_interaction_features = hp.get("add_interaction_features", None)
        if add_interaction_features is None:
            # Sample based on LimiX's usage (40-50% of datasets get interactions)
            # GPU-optimized: Use torch.rand instead of numpy (stays on GPU)
            add_interaction_features = torch.rand(1, device=X_normalized.device).item() < 0.45
        
        if add_interaction_features and X_normalized.shape[1] >= 2:
            # Validate input before processing
            nan_mask_input = torch.isnan(X_normalized) | torch.isinf(X_normalized)
            if torch.any(nan_mask_input):
                logger.warning("Skipping interactions: Input contains NaN/Inf after cleaning attempt.")
                add_interaction_features = False  # Skip interactions if input is invalid
            else:
                try:
                    # Limix-based-Changes - Generate polynomial interactions with standardized features
                    n_features = X_normalized.shape[1]
                    if n_features <= 0:
                        raise ValueError(f"Invalid n_features: {n_features}")
                    
                    max_interactions = min(100, int(n_features * 0.3))  # LimiX default: up to 100 interactions
                    
                    # Skip interactions if max_interactions is 0
                    if max_interactions == 0:
                        pass  # No interactions to add
                    else:
                        # Try main method first (torch-based, GPU-optimized)
                        try:
                            # Validate input tensor before processing (check directly to avoid duplicate warning)
                            nan_mask_input = torch.isnan(X_normalized) | torch.isinf(X_normalized)
                            if torch.any(nan_mask_input):
                                raise ValueError("Input tensor contains NaN/Inf")
                            
                            # Standardize before interaction to prevent scale explosion (LimiX approach)
                            X_std_mean = X_normalized.mean(dim=0, keepdim=True)
                            X_std_std = X_normalized.std(dim=0, keepdim=True).clip(min=1e-6)
                            X_standardized = (X_normalized - X_std_mean) / X_std_std
                            
                            # Validate standardized tensor
                            nan_mask_std = torch.isnan(X_standardized) | torch.isinf(X_standardized)
                            if torch.any(nan_mask_std):
                                raise ValueError("Standardized tensor contains NaN/Inf")
                            
                            # Generate random interaction pairs (LimiX uses randomized pairs)
                            # GPU-optimized: Use torch.randint instead of numpy (stays on GPU)
                            primary_indices = torch.randint(0, n_features, size=(max_interactions,), device=X_normalized.device)
                            secondary_indices = torch.randint(0, n_features, size=(max_interactions,), device=X_normalized.device)
                            
                            # Create interaction features (X_i * X_j)
                            interactions = X_standardized[:, primary_indices] * X_standardized[:, secondary_indices]
                            
                            # Validate interactions tensor before concatenation
                            nan_mask_interactions = torch.isnan(interactions) | torch.isinf(interactions)
                            if torch.any(nan_mask_interactions):
                                raise ValueError("Interactions tensor contains NaN/Inf")
                            
                            # Concatenate original + interactions (LimiX approach: FeatureUnion)
                            X_normalized = torch.cat([X_normalized, interactions], dim=1)
                            
                        except ValueError as e:
                            # Fallback: Replace NaN/Inf with feature means, then retry interactions
                            logger.debug(f"Interaction features failed ({e}), replacing NaN/Inf with defaults and retrying")
                            
                            # Replace NaN/Inf with feature means (context-aware, like final output)
                            X_normalized_clean = X_normalized.clone()
                            nan_mask = torch.isnan(X_normalized_clean) | torch.isinf(X_normalized_clean)
                            if torch.any(nan_mask):
                                for feat_idx in range(X_normalized_clean.shape[1]):
                                    feat_col = X_normalized_clean[:, feat_idx]
                                    nan_in_col = torch.isnan(feat_col) | torch.isinf(feat_col)
                                    if torch.any(nan_in_col):
                                        valid_values = feat_col[~nan_in_col]
                                        replacement = valid_values.mean() if len(valid_values) > 0 else 0.0
                                        X_normalized_clean[:, feat_idx] = torch.where(
                                            nan_in_col,
                                            torch.full_like(feat_col, replacement),
                                            feat_col
                                        )
                            
                            # Retry with cleaned tensor
                            X_std_mean = X_normalized_clean.mean(dim=0, keepdim=True)
                            X_std_std = X_normalized_clean.std(dim=0, keepdim=True).clip(min=1e-6)
                            X_standardized = (X_normalized_clean - X_std_mean) / X_std_std
                            
                            # Generate random interaction pairs
                            primary_indices = torch.randint(0, n_features, size=(max_interactions,), device=X_normalized.device)
                            secondary_indices = torch.randint(0, n_features, size=(max_interactions,), device=X_normalized.device)
                            
                            # Create interaction features
                            interactions = X_standardized[:, primary_indices] * X_standardized[:, secondary_indices]
                            
                            # Replace any NaN/Inf in interactions with 0.0 (interactions can be 0 safely)
                            interactions = torch.where(
                                torch.isnan(interactions) | torch.isinf(interactions),
                                torch.zeros_like(interactions),
                                interactions
                            )
                            
                            # Concatenate original + interactions
                            X_normalized = torch.cat([X_normalized_clean, interactions], dim=1)
                
                except Exception as e:
                    # Limix-based-Changes - continue without interactions on failure
                    logger.warning(f"Failed to add interaction features: {e}. Continuing without interactions.")
        
        # Limix-based-Changes - Add SVD-compressed features (LimiX's SVD worker)
        # LimiX Insight: Real-world data often has redundant/correlated features
        # Solution: Add principal components via SVD to capture linear feature combinations
        add_svd_features = hp.get("add_svd_features", None)
        if add_svd_features is None:
            # Sample based on LimiX's usage (20-30% of datasets get SVD features)
            # GPU-optimized: Use torch.rand instead of numpy (stays on GPU)
            add_svd_features = torch.rand(1, device=X_normalized.device).item() < 0.25
        
        if add_svd_features and X_normalized.shape[1] >= 2 and train_size >= 10:
            # Early exit if NaNs/Inf present (avoid expensive sklearn path)
            nan_mask = torch.isnan(X_normalized) | torch.isinf(X_normalized)
            if torch.any(nan_mask):
                logger.debug("Skipping SVD: Input contains NaN/Inf. Continuing without SVD.")
            else:
                try:
                    # GPU-optimized: Use torch SVD instead of sklearn (stays on GPU, no CPU-GPU transfers)
                    # Limix-based-Changes - Adaptive SVD components (like LimiX: max(1, min(n_samples // 10 + 1, n_features // 2)))
                    n_components = max(1, min(train_size // 10 + 1, X_normalized.shape[1] // 2))
                    n_components = min(n_components, train_size - 1)  # Ensure we have enough samples
                    n_components = min(n_components, X_normalized.shape[1])  # Can't have more components than features
                    
                    # Validate n_components
                    if n_components <= 0:
                        raise ValueError(f"Invalid n_components: {n_components}")
                    if train_size < n_components + 1:
                        raise ValueError(f"train_size ({train_size}) too small for {n_components} components")
                    
                    # Center the data (required for SVD to work like PCA/TruncatedSVD)
                    X_centered = X_normalized - X_normalized.mean(dim=0, keepdim=True)
                    
                    # Check for constant features before SVD
                    feature_stds = X_centered.std(dim=0)
                    if feature_stds.min() < 1e-6:
                        raise ValueError("Matrix has constant features, skipping SVD")
                    
                    # Try torch SVD first (GPU-optimized)
                    try:
                        # Compute SVD on GPU (equivalent to TruncatedSVD but faster)
                        # TruncatedSVD computes U * S for first n_components
                        # Suppress convergence warnings (CUDA SVD may use fallback method, which is fine)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*During SVD computation.*", category=UserWarning)
                            U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
                        
                        # Validate SVD output: check singular values for NaN/Inf
                        if not self._validate_tensor(S, "SVD singular values"):
                            raise ValueError("SVD produced NaN/Inf singular values")
                        
                        # Take first n_components (like TruncatedSVD)
                        X_svd = U[:, :n_components] * S[:n_components].unsqueeze(0)
                        
                        # Validate SVD output tensor
                        if not self._validate_tensor(X_svd, "SVD output"):
                            raise ValueError("SVD output contains NaN/Inf")
                    
                    except (RuntimeError, ValueError) as torch_error:
                        # Fallback to sklearn TruncatedSVD if torch SVD fails (e.g., convergence issues)
                        logger.debug(f"Torch SVD failed ({torch_error}), falling back to sklearn TruncatedSVD")
                        from sklearn.decomposition import TruncatedSVD
                        
                        # Convert to numpy for sklearn
                        X_centered_np = X_centered.cpu().numpy()
                        
                        # Clean NaNs/Inf before sklearn (ICL-safe)
                        nan_mask = np.isnan(X_centered_np) | np.isinf(X_centered_np)
                        if np.any(nan_mask):
                            X_support_centered = X_centered[:train_size].cpu().numpy()
                            col_mean = np.nanmean(X_support_centered, axis=0)
                            col_mean = np.nan_to_num(col_mean, nan=0.0, posinf=0.0, neginf=0.0)
                            for feat_idx in range(X_centered_np.shape[1]):
                                feat_col = X_centered_np[:, feat_idx]
                                nan_in_col = np.isnan(feat_col) | np.isinf(feat_col)
                                if np.any(nan_in_col):
                                    replacement = col_mean[feat_idx] if not np.isnan(col_mean[feat_idx]) else 0.0
                                    feat_col[nan_in_col] = replacement
                        
                        # Skip if still has NaNs (shouldn't happen, but defensive)
                        if np.any(np.isnan(X_centered_np)) or np.any(np.isinf(X_centered_np)):
                            raise ValueError("Cannot clean NaNs/Inf from input, skipping SVD")
                        
                        # Use sklearn TruncatedSVD (more robust, handles ill-conditioned matrices better)
                        svd = TruncatedSVD(n_components=n_components, random_state=None)
                        X_svd_np = svd.fit_transform(X_centered_np)
                        
                        # Check for NaN/Inf in sklearn output
                        if np.any(np.isnan(X_svd_np)) or np.any(np.isinf(X_svd_np)):
                            raise ValueError("Sklearn SVD produced NaN/Inf")
                        
                        # Convert back to torch tensor
                        X_svd = torch.from_numpy(X_svd_np).to(X_normalized.device, dtype=X_normalized.dtype)
                    
                    # Concatenate original + SVD features (LimiX's FeatureUnion approach)
                    X_normalized = torch.cat([X_normalized, X_svd], dim=1)
                
                except Exception as e:
                    # Limix-based-Changes - continue without SVD on failure
                    logger.warning(f"Failed to add SVD features: {e}. Continuing without SVD.")
        
        # Limix-based-Changes - Add fingerprint features (LimiX's FingerprintFeatureEncoder)
        # Provide unique row identifiers that help model learn sample-level patterns
        # Add hash-based fingerprint per row
        add_fingerprint_feature = hp.get("add_fingerprint_feature", None)
        if add_fingerprint_feature is None:
            # Sample based on LimiX's usage (30-40% of datasets get fingerprint features)
            add_fingerprint_feature = np.random.random() < 0.35
        
        if add_fingerprint_feature:
            try:
                # Validate input tensor before fingerprint computation (check directly to avoid duplicate warning)
                nan_mask_input = torch.isnan(X_normalized) | torch.isinf(X_normalized)
                if torch.any(nan_mask_input):
                    raise ValueError("Input tensor contains NaN/Inf, skipping fingerprint")
                
                # GPU-optimized: Use torch random projection instead of hash loop (stays on GPU, vectorized)
                # Semantically equivalent: both provide unique row identifiers
                n_samples = X_normalized.shape[0]
                
                # Generate random salt for diversity (LimiX uses salt)
                salt = np.random.randint(0, 65536)
                
                # Use seeded random projection to create unique row fingerprints (GPU-native, vectorized)
                # This is semantically equivalent to hashing but much faster
                torch.manual_seed(salt)
                random_weights = torch.randn(X_normalized.shape[1], device=X_normalized.device, dtype=X_normalized.dtype)
                fingerprints = (X_normalized @ random_weights).unsqueeze(1)
                
                # Validate fingerprints before normalization
                if not self._validate_tensor(fingerprints, "Fingerprint features (before normalization)"):
                    raise ValueError("Fingerprints contain NaN/Inf before normalization")
                
                # Normalize to [0, 1) like LimiX hash approach
                fingerprints_min = fingerprints.min()
                fingerprints_max = fingerprints.max()
                fingerprints_range = fingerprints_max - fingerprints_min
                if fingerprints_range > 1e-10:
                    fingerprints = (fingerprints - fingerprints_min) / fingerprints_range
                else:
                    # Fallback if all values are the same
                    fingerprints = torch.zeros_like(fingerprints)
                
                # Validate fingerprints after normalization
                if not self._validate_tensor(fingerprints, "Fingerprint features (after normalization)"):
                    raise ValueError("Fingerprints contain NaN/Inf after normalization")
                
                # Append as new feature column (LimiX approach)
                X_normalized = torch.cat([X_normalized, fingerprints], dim=1)
                
            except Exception as e:
                # Limix-based-Changes - continue without fingerprint on failure
                logger.debug(f"Failed to add fingerprint feature: {e}. Continuing without fingerprint.")

        # Step 6d: Add redundant/derived columns before permutation and truncation.
        pre_expand_dim = X_normalized.shape[1]
        X_normalized = self._add_near_duplicate_columns(X_normalized, train_size, hp, pre_expand_dim)
        X_normalized = self._add_linear_combination_columns(X_normalized, train_size, hp, pre_expand_dim)
        
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

        # Default metadata placeholders (updated after final transforms).
        missing_mask = torch.zeros_like(X_normalized, dtype=torch.bool)
        imputation_strategy_ids = torch.full(
            (self.max_features,),
            IMPUTATION_STRATEGY_TO_ID["none"],
            device=X_normalized.device,
            dtype=torch.long,
        )
        
        # Step 9: Optional Kumaraswamy warping (standard-style, for realism/challenge)
        use_kumaraswamy_warping = hp.get("use_kumaraswamy_warping", False)
        if use_kumaraswamy_warping:
            try:
                from .warping import apply_kumaraswamy_warping
                # Normalize to [0, 1] first, then apply warping
                # Note: This is applied after standard normalization
                # ICL-safe: Split back into support and query for min/max computation
                X_support_normalized = X_normalized[:train_size]
                X_query_normalized = X_normalized[train_size:]
                
                # Compute min/max from support set only (ICL-safe)
                X_normalized_min = X_support_normalized.min(dim=0, keepdim=True)[0]
                X_normalized_max = X_support_normalized.max(dim=0, keepdim=True)[0]
                X_normalized_range = (X_normalized_max - X_normalized_min).clip(min=1e-6)
                
                # Apply support-set statistics to both support and query
                X_support_normalized_01 = (X_support_normalized - X_normalized_min) / X_normalized_range
                X_query_normalized_01 = (X_query_normalized - X_normalized_min) / X_normalized_range
                X_normalized_01 = torch.cat([X_support_normalized_01, X_query_normalized_01], dim=0)
                
                # Apply Kumaraswamy warping
                # ICL-SAFETY NOTE: This is a deterministic post-processing transformation applied after
                # ICL-safe normalization (min/max computed from support set only). The warping function
                # is applied to both support and query sets, which is acceptable because:
                # 1. All preprocessing statistics (min/max) are computed from support set only
                # 2. The warping is a deterministic, parameter-free transformation (no learned parameters)
                # 3. This matches standard synthetic data generation where transformations are applied
                #    uniformly to the entire dataset after ICL-safe normalization
                a = hp.get("kumaraswamy_a", 2.0)
                b = hp.get("kumaraswamy_b", 5.0)
                X_warped = apply_kumaraswamy_warping(X_normalized_01, a=a, b=b, normalize_first=False)
                
                # Re-normalize back to similar scale (optional, can keep [0,1] range)
                X_normalized = X_warped
            except ImportError:
                # Use module-level warnings import (already imported at top of file)
                warnings.warn("Kumaraswamy warping module not available, skipping")
        
        # Step 10: Heavy-tailed feature realism (Student-t / Pareto-like)
        X_normalized = self._apply_heavy_tail_feature_transform(X_normalized, train_size, hp, active_feature_count)

        # Step 10b: Finance-style sparse Pareto shocks on selected input factors.
        X_normalized = self._apply_finance_input_tail_transform(X_normalized, train_size, hp, active_feature_count)

        # Step 11: Count feature realism (Poisson / Negative-Binomial-like)
        X_normalized = self._apply_count_feature_transform(X_normalized, train_size, hp, active_feature_count)

        # Step 12: Bounded/proportion feature realism (sigmoid / beta-like)
        X_normalized = self._apply_bounded_feature_transform(X_normalized, train_size, hp, active_feature_count)

        # Step 12b: Optional rank-normalized features using support empirical CDF.
        X_normalized = self._apply_rank_normalized_features(X_normalized, train_size, hp, active_feature_count)

        # Step 13: Optional quantization into buckets (standard-style)
        quantize_features = hp.get("quantize_features", False)
        if quantize_features:
            num_buckets = hp.get("num_quantization_buckets", 32)
            # Quantize features into discrete buckets
            # Map continuous values to bucket indices, then back to bucket centers
            # ICL-safe: Split back into support and query for min/max computation
            X_support_normalized = X_normalized[:train_size]
            X_query_normalized = X_normalized[train_size:]
            
            # Compute min/max from support set only (ICL-safe)
            X_min = X_support_normalized.min(dim=0, keepdim=True)[0]
            X_max = X_support_normalized.max(dim=0, keepdim=True)[0]
            X_range = (X_max - X_min).clip(min=1e-6)
            
            # Apply support-set statistics to both support and query
            X_support_normalized_01 = (X_support_normalized - X_min) / X_range
            X_query_normalized_01 = (X_query_normalized - X_min) / X_range
            
            # Quantize both sets
            X_support_normalized_01_quantized = torch.clamp((X_support_normalized_01 * num_buckets).long(), 0, num_buckets - 1)
            X_query_normalized_01_quantized = torch.clamp((X_query_normalized_01 * num_buckets).long(), 0, num_buckets - 1)
            
            bucket_centers_support = (X_support_normalized_01_quantized.float() + 0.5) / num_buckets
            bucket_centers_query = (X_query_normalized_01_quantized.float() + 0.5) / num_buckets
            
            X_support_normalized = bucket_centers_support * X_range + X_min
            X_query_normalized = bucket_centers_query * X_range + X_min
            X_normalized = torch.cat([X_support_normalized, X_query_normalized], dim=0)

        # Step 14: Optional per-column discretization.
        X_normalized, ordinal_discretized_cols = self._apply_feature_discretization(
            X_normalized, train_size, hp, active_feature_count
        )

        # Step 15: Missingness realism (MCAR/MAR/MNAR) with mixed imputation strategies.
        X_normalized, missing_mask, imputation_strategy_ids = self._inject_missingness(
            X_normalized, train_size, hp, active_feature_count
        )

        # Step 15a: Gross outlier injection (data-entry-error style corruption).
        X_normalized = self._inject_gross_outliers(X_normalized, hp, active_feature_count)

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
            col_support = col[:train_size]
            if col_support.numel() == 0:
                feature_type_ids[feat_idx] = FEATURE_TYPE_TO_ID["continuous"]
                continue

            # Use support-only stats for robust type inference.
            col_min = torch.min(col_support)
            col_max = torch.max(col_support)
            rounded = torch.round(col_support)
            integer_like_ratio = ((col_support - rounded).abs() < 1e-5).to(col_support.dtype).mean().item()
            is_integer_like = integer_like_ratio > 0.98
            unique_count = torch.unique(col_support).numel()
            unique_ratio = unique_count / max(int(col_support.numel()), 1)
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

        # Override inferred types for explicitly ordinal discretized columns.
        if ordinal_discretized_cols.numel() > 0:
            valid_ordinal_cols = ordinal_discretized_cols[ordinal_discretized_cols < active_feature_count]
            if valid_ordinal_cols.numel() > 0:
                feature_type_ids[valid_ordinal_cols] = FEATURE_TYPE_TO_ID["ordinal"]

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
        # Option 2 (default): Smart replacement with context-aware defaults (mean per feature)
        # Option 1 (fallback): Simple replacement with 0.0 when all values are NaN
        # Check directly (don't use _validate_tensor to avoid duplicate warning)
        nan_mask = torch.isnan(X_normalized) | torch.isinf(X_normalized)
        if torch.any(nan_mask):
                nan_count = nan_mask.sum().item()
                total_count = nan_mask.numel()
                nan_ratio = nan_count / total_count
                
                # Option 2: Replace NaNs with mean of non-NaN values per feature (context-aware)
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
                        f"Replaced with context-aware defaults (mean per feature, 0.0 fallback). "
                        f"This may indicate numerical instability in transformations."
                    )
                else:
                    logger.debug(
                        f"Feature processing produced {nan_count}/{total_count} ({nan_ratio:.1%}) NaN/Inf values. "
                        f"Replaced with context-aware defaults."
                    )
        feature_meta = {
            "missing_mask": missing_mask,
            "feature_type_ids": feature_type_ids,
            "col_ids": col_ids,
            "imputation_strategy_ids": imputation_strategy_ids,
        }
        return X_normalized, feature_meta

    @staticmethod
    def _apply_covariate_shift(
        X: Tensor,
        train_size: int,
        frac_features_range: Tuple[float, float] = (0.10, 0.50),
        mean_shift_scale_range: Tuple[float, float] = (0.5, 2.0),
        scale_shift_range: Tuple[float, float] = (0.5, 2.0),
        feature_meta: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Apply smooth covariate shift to query (test) features using support-set statistics.

        A random subset of features has their test-set distribution shifted by a
        mean offset and/or a scale multiplier.  Shift magnitude is anchored to
        support-set statistics (mean, std) so this is fully ICL-safe — no
        ground-truth test statistics are ever observed.

        Parameters
        ----------
        X : Tensor
            Features of shape ``(seq_len, num_features)``.  Already normalized.
        train_size : int
            Number of support (train) rows.
        frac_features_range : Tuple[float, float]
            ``(lo, hi)`` fraction of features to shift, sampled uniformly.
        mean_shift_scale_range : Tuple[float, float]
            ``(lo, hi)`` magnitude of the per-feature mean shift expressed in
            units of the support-set standard deviation.  Each selected feature
            receives an *independent* signed shift drawn from
            ``Uniform(-scale, +scale) * σ_support``.
        scale_shift_range : Tuple[float, float]
            ``(lo, hi)`` range for the multiplicative scale applied to shifted
            query features, sampled log-uniformly so values below 1 and above 1
            are equally likely in log-space.

        Returns
        -------
        Tensor
            X with covariate-shifted query features; support rows are unchanged.
        """

        n_features = X.shape[1]
        X_support = X[:train_size]   # (train_size, n_features)
        X_query   = X[train_size:]   # (query_size, n_features)

        if X_query.shape[0] == 0 or n_features == 0:
            return X

        # --- build candidate pool: only continuous / count / proportion columns ---
        _eligible_ids = {FEATURE_TYPE_TO_ID["continuous"], FEATURE_TYPE_TO_ID["count"], FEATURE_TYPE_TO_ID["proportion"]}

        if feature_meta is not None:
            fti = feature_meta["feature_type_ids"]
            candidate_feats = torch.where(
                (fti == FEATURE_TYPE_TO_ID["continuous"])
                | (fti == FEATURE_TYPE_TO_ID["count"])
                | (fti == FEATURE_TYPE_TO_ID["proportion"])
            )[0].to(X.device)
        else:
            candidate_feats = torch.arange(n_features, device=X.device)

        if candidate_feats.numel() == 0:
            return X

        # --- select a random subset of features to shift ---
        frac_lo, frac_hi = frac_features_range
        frac = np.random.uniform(frac_lo, frac_hi)
        n_shift = max(1, int(candidate_feats.numel() * frac))
        perm = torch.randperm(candidate_feats.numel(), device=X.device)[:n_shift]
        shift_feats = candidate_feats[perm]

        # --- support-set statistics (ICL-safe anchor) ---
        sigma_support = X_support[:, shift_feats].std(dim=0).clamp(min=1e-6)  # (n_shift,)

        # --- per-feature signed mean shift in [-shift_scale, +shift_scale] * σ_support ---
        shift_scale = np.random.uniform(*mean_shift_scale_range)
        delta_mean = (
            torch.rand(n_shift, device=X.device) * 2.0 - 1.0
        ) * shift_scale * sigma_support  # (n_shift,)

        # --- log-uniform scale multiplier ---
        log_lo, log_hi = math.log(scale_shift_range[0]), math.log(scale_shift_range[1])
        scale_mult = float(math.exp(np.random.uniform(log_lo, log_hi)))

        # --- apply to query rows only ---
        X_query_shifted = X_query.clone()
        X_query_shifted[:, shift_feats] = (
            X_query[:, shift_feats] * scale_mult + delta_mean
        )
        # Round count features back to integer-like values
        if feature_meta is not None:
            fti = feature_meta["feature_type_ids"]
            count_mask = fti[shift_feats] == FEATURE_TYPE_TO_ID["count"]
            if count_mask.any():
                count_feats = shift_feats[count_mask]
                X_query_shifted[:, count_feats] = X_query_shifted[:, count_feats].round()
        # Re-clip to the same bound used elsewhere in feature processing
        X_query_shifted = X_query_shifted.clamp(-100.0, 100.0)

        return torch.cat([X_support, X_query_shifted], dim=0)

    @staticmethod
    def _apply_seasonal_drift(
        X: Tensor,
        y: Tensor,
        train_size: int,
        amplitude_range: Tuple[float, float] = (0.1, 1.0),
        freq_cycles_range: Tuple[float, float] = (1.0, 8.0),
        apply_to_X_prob: float = 0.7,
        X_frac_range: Tuple[float, float] = (0.05, 0.2),
        X_amplitude_scale_range: Tuple[float, float] = (0.1, 0.5),
        feature_meta: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Add an exogenous sinusoidal seasonal cycle to ``y`` (and optionally ``X``).

        The cycle is indexed purely by row position so it represents an exogenous
        temporal pattern, **not** SCM autocorrelation or feedback from past
        values.  Amplitude and frequency are sampled independently per dataset,
        and a random phase offset ensures the model cannot rely on the cycle
        always starting at zero.

        ``amplitude_range`` is expressed as a fraction of the support-set
        standard deviation of ``y`` (and per-feature std for ``X``), so the
        drift magnitude is meaningful regardless of whether the data has been
        normalised beforehand.

        Parameters
        ----------
        X : Tensor
            Features of shape ``(seq_len, n_features)``.
        y : Tensor
            Targets of shape ``(seq_len,)``.
        train_size : int
            Number of support (train) rows.
        amplitude_range : Tuple[float, float]
            ``(lo, hi)`` for log-uniform amplitude sampling, expressed as a
            fraction of the support-set standard deviation of ``y``.  Works
            correctly whether ``y`` is normalised (std ≈ 1) or raw.
        freq_cycles_range : Tuple[float, float]
            ``(lo, hi)`` number of complete sinusoidal cycles across the full
            sequence length.
        apply_to_X_prob : float
            Probability of also adding drift to a subset of feature columns.
        X_frac_range : Tuple[float, float]
            Fraction of features to drift when feature drift is triggered.
        X_amplitude_scale_range : Tuple[float, float]
            ``(lo, hi)`` for per-feature log-uniform amplitude scale, applied
            on top of the feature's own support-set standard deviation.  Each
            selected feature draws its own independent scale so the drift
            strength varies across features.

        Returns
        -------
        Tuple[Tensor, Tensor]
            ``(X, y)`` with the seasonal drift component added.
        """

        seq_len = y.shape[0]
        device  = y.device

        # --- support-set std of y (ICL-safe; scale anchor for amplitude) ---
        sigma_y = y[:train_size].std().clamp(min=1e-6)

        # --- sample amplitude as a fraction of σ_y (log-uniform) ---
        amp_lo, amp_hi = amplitude_range
        amplitude = float(math.exp(np.random.uniform(math.log(amp_lo), math.log(amp_hi))))
        effective_amplitude = amplitude * float(sigma_y)

        # --- sample frequency: n_cycles full cycles across the entire sequence ---
        n_cycles = np.random.uniform(*freq_cycles_range)
        omega = 2.0 * math.pi * n_cycles / seq_len  # radians per row

        # --- random phase so training prefix doesn't always start at sin(0) ---
        phase = np.random.uniform(0.0, 2.0 * math.pi)

        # --- build the drift signal for every row ---
        t = torch.arange(seq_len, device=device, dtype=y.dtype)
        drift = effective_amplitude * torch.sin(omega * t + phase)  # (seq_len,)

        # --- apply to y (both support and query; model must extrapolate the cycle) ---
        y_drifted = (y + drift).clamp(-10.0, 10.0)  # match existing target safety bounds

        # Restrict to continuous / count / proportion columns only
        if feature_meta is not None:
            fti = feature_meta["feature_type_ids"]
            candidate_feats = torch.where(
                (fti == FEATURE_TYPE_TO_ID["continuous"])
                | (fti == FEATURE_TYPE_TO_ID["count"])
                | (fti == FEATURE_TYPE_TO_ID["proportion"])
            )[0].to(device)
        else:
            return X, y_drifted

        if candidate_feats.numel() == 0:
            return X, y_drifted

        # --- optionally add drift to a subset of X features ---
        if np.random.random() < apply_to_X_prob and candidate_feats.numel() > 0:
            frac     = np.random.uniform(*X_frac_range)
            n_feat   = max(1, int(candidate_feats.numel() * frac))
            perm     = torch.randperm(candidate_feats.numel(), device=device)[:n_feat]
            feat_idx = candidate_feats[perm]

            # Per-feature support-set std
            # If we computed sigma_X from the full sequence (train + query rows), 
            # we would be peeking at the test distribution to decide how strongly to drift the features (data leakage).
            sigma_X = X[:train_size, feat_idx].std(dim=0).clamp(min=1e-6)  # (n_feat,)

            # Per-feature independent amplitude scale (log-uniform)
            sc_lo, sc_hi = X_amplitude_scale_range
            log_scales = torch.empty(n_feat, device=device).uniform_(
                math.log(sc_lo), math.log(sc_hi)
            )
            per_feat_amplitude = torch.exp(log_scales) * sigma_X  # (n_feat,)

            # Per-feature independent phase
            phases_X = torch.rand(n_feat, device=device) * 2.0 * math.pi

            X_drift = (
                per_feat_amplitude.unsqueeze(0)
                * torch.sin(omega * t.unsqueeze(1) + phases_X.unsqueeze(0))
            )  # (seq_len, n_feat)

            X_drifted = X.clone()
            X_drifted[:, feat_idx] = (X[:, feat_idx] + X_drift).clamp(-100.0, 100.0)
            # Round count features back to integer-like values
            if feature_meta is not None:
                fti = feature_meta["feature_type_ids"]
                count_mask = fti[feat_idx] == FEATURE_TYPE_TO_ID["count"]
                if count_mask.any():
                    count_feats = feat_idx[count_mask]
                    X_drifted[:, count_feats] = X_drifted[:, count_feats].round()
            return X_drifted, y_drifted

        return X, y_drifted

    @staticmethod
    def _apply_temporal_drift(
        X: Tensor,
        y: Tensor,
        train_size: int,
        n_changepoints_range: Tuple[int, int] = (1, 5),
        transition: str = "mixed",
        sigmoid_width_frac: float = 0.05,
        cov_frac_features_range: Tuple[float, float] = (0.10, 0.50),
        cov_shift_range: Tuple[float, float] = (0.5, 2.0),
        cov_scale_range: Tuple[float, float] = (0.5, 2.0),
        _cp_out: Optional[list] = None,
        feature_meta: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply abrupt or gradual covariate temporal drift to X at random changepoints.

        At each changepoint the mean and variance of a randomly chosen feature
        subset are shifted.  Each regime draws its own independent shift
        parameters; statistics are computed from the support set only (ICL-safe).

        Abrupt vs. gradual transition
            - ``"abrupt"``: hard step-function — rows are hard-assigned to the
              regime they fall in; no blending.
            - ``"gradual"``: consecutive regimes are blended via a sigmoid window
              of width ``sigmoid_width_frac * seq_len`` centred on each changepoint.
              Multiple regimes are chained through cumulative sigmoid products so
              weights always sum to 1 at every row.
            - ``"mixed"``: choose abrupt or gradual randomly per dataset.

        Parameters
        ----------
        X : Tensor
            Shape ``(seq_len, n_features)``.
        y : Tensor
            Shape ``(seq_len,)``; returned unchanged.
        train_size : int
            Number of support rows; used for ICL-safe statistic computation.
        n_changepoints_range : Tuple[int, int]
            Inclusive ``[lo, hi]`` range for the number of changepoints to sample.
        transition : str
            One of ``"abrupt"``, ``"gradual"``, or ``"mixed"``.
        sigmoid_width_frac : float
            Width of the sigmoid transition window as a fraction of ``seq_len``.
        cov_frac_features_range : Tuple[float, float]
            Fraction of features to shift per regime.
        cov_shift_range : Tuple[float, float]
            Log-uniform ``(lo, hi)`` magnitude multiplier for mean shift (in units
            of the feature's support-set standard deviation).
        cov_scale_range : Tuple[float, float]
            Log-uniform ``(lo, hi)`` scale multiplier applied to shifted features.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Modified ``(X, y)`` — only ``X`` is altered.
        """
        seq_len, n_features = X.shape
        device = X.device
        dtype  = X.dtype

        # --- transition type ---
        if transition == "mixed":
            is_gradual = bool(np.random.random() < 0.5)
        else:
            is_gradual = (transition == "gradual")

        # --- sample changepoints (avoid first/last 15% of the sequence) ---
        margin = max(1, int(0.15 * seq_len))
        n_cp = int(np.random.randint(n_changepoints_range[0], n_changepoints_range[1] + 1))
        pool = np.arange(margin, seq_len - margin)
        if len(pool) < n_cp:
            n_cp = max(1, len(pool))
        cp_positions = sorted(
            np.random.choice(pool, size=n_cp, replace=False).tolist()
        )
        n_regimes = n_cp + 1

        # --- build regime weight tensors: shape (n_regimes, seq_len) ---
        t = torch.arange(seq_len, device=device, dtype=dtype)
        if not is_gradual:
            # Hard step-function assignment
            weights = torch.zeros(n_regimes, seq_len, device=device, dtype=dtype)
            boundaries = [0] + cp_positions + [seq_len]
            for r in range(n_regimes):
                weights[r, boundaries[r]:boundaries[r + 1]] = 1.0
        else:
            # Gradual: chain sigmoid transitions so weights sum to 1 everywhere
            width = max(1.0, sigmoid_width_frac * seq_len)
            cum   = torch.ones(seq_len, device=device, dtype=dtype)
            w_list: list[Tensor] = []
            for r in range(n_regimes - 1):
                cp = float(cp_positions[r])
                s  = torch.sigmoid((t - cp) / width)   # fraction past this CP
                w_list.append(cum * (1.0 - s))          # regime r: mass before CP
                cum = cum * s                            # remaining mass propagates
            w_list.append(cum)                          # last regime
            weights = torch.stack(w_list, dim=0)        # (n_regimes, seq_len)

        # ── Covariate temporal drift ───────────────────────────────────────────
        X_out = X.clone()
        if n_features > 0:
            # Support-set statistics (ICL-safe)
            sig_sup = X[:train_size].std(dim=0).clamp(min=1e-6)

            # Build candidate pool: only continuous / count / proportion columns
            if feature_meta is not None:
                fti = feature_meta["feature_type_ids"]
                candidate_feats = torch.where(
                    (fti == FEATURE_TYPE_TO_ID["continuous"])
                    | (fti == FEATURE_TYPE_TO_ID["count"])
                    | (fti == FEATURE_TYPE_TO_ID["proportion"])
                )[0].to(device)
            else:
                candidate_feats = torch.arange(n_features, device=device)

            # Regime 0: baseline (no shift)
            X_blend = weights[0].unsqueeze(1) * X              # (seq_len, n_features)
            for r in range(1, n_regimes):
                if candidate_feats.numel() == 0:
                    X_blend = X_blend + weights[r].unsqueeze(1) * X
                    continue
                frac = float(np.random.uniform(*cov_frac_features_range))
                n_sh = max(1, int(candidate_feats.numel() * frac))
                perm = torch.randperm(candidate_feats.numel(), device=device)[:n_sh]
                shift_feats = candidate_feats[perm]

                # Signed mean shift in units of support-set std
                log_lo_m = math.log(cov_shift_range[0])
                log_hi_m = math.log(cov_shift_range[1])
                mag  = float(math.exp(np.random.uniform(log_lo_m, log_hi_m)))
                sign = float(np.random.choice([-1.0, 1.0]))
                delta = torch.zeros(n_features, device=device, dtype=dtype)
                delta[shift_feats] = sign * mag * sig_sup[shift_feats]

                # Scale multiplier
                log_lo_s = math.log(cov_scale_range[0])
                log_hi_s = math.log(cov_scale_range[1])
                scale_mult = float(math.exp(np.random.uniform(log_lo_s, log_hi_s)))

                X_regime = X.clone()
                X_regime[:, shift_feats] = (
                    X[:, shift_feats] * scale_mult + delta[shift_feats]
                )
                X_blend = X_blend + weights[r].unsqueeze(1) * X_regime

            X_out = X_blend.clamp(-100.0, 100.0)
            # Round count features back to integer-like values after blending
            if feature_meta is not None:
                fti = feature_meta["feature_type_ids"]
                count_feats = torch.where(fti == FEATURE_TYPE_TO_ID["count"])[0].to(device)
                if count_feats.numel() > 0:
                    X_out[:, count_feats] = X_out[:, count_feats].round()

        if _cp_out is not None:
            _cp_out.extend(cp_positions)

        return X_out, y

    def _apply_finance_target_dynamics(
        self, y: Tensor, train_size: int, hp: Dict[str, Any]
    ) -> Tensor:
        """Apply finance-style target dynamics: GARCH volatility + jump diffusion."""
        if y.dim() != 1 or y.numel() < 3:
            return y
        if not bool(hp.get("_finance_regime", False)):
            return y

        y_out = y.clone()

        garch_rate = float(hp.get("finance_garch_rate", 0.0) or 0.0)
        if garch_rate > 0.0 and np.random.random() < garch_rate:
            omega = max(float(hp.get("finance_garch_omega", 1e-3)), 1e-8)
            alpha = float(np.clip(hp.get("finance_garch_alpha", 0.12), 1e-4, 0.95))
            beta = float(np.clip(hp.get("finance_garch_beta", 0.84), 1e-4, 0.995))
            if alpha + beta >= 0.995:
                scale = 0.995 / (alpha + beta + 1e-8)
                alpha *= scale
                beta *= scale

            support_var = y_out[:train_size].var(unbiased=False).clamp(min=1e-6, max=100.0)
            sigma2 = torch.empty_like(y_out)
            sigma2[0] = support_var
            for t in range(1, y_out.shape[0]):
                eps_prev = y_out[t - 1]
                sigma2[t] = omega + alpha * (eps_prev * eps_prev) + beta * sigma2[t - 1]

            sigma = torch.sqrt(sigma2.clamp(min=1e-8))
            sigma_ref = torch.sqrt(support_var).clamp(min=1e-4)
            vol_mult = (sigma / sigma_ref).clamp(min=0.2, max=8.0)
            y_out = y_out * vol_mult

        jump_rate = float(hp.get("finance_jump_rate", 0.0) or 0.0)
        if jump_rate > 0.0 and np.random.random() < jump_rate:
            jump_lambda = float(np.clip(hp.get("finance_jump_lambda", 0.015), 0.0, 0.5))
            jump_mask = torch.rand_like(y_out) < jump_lambda
            if torch.any(jump_mask):
                log_mu = float(hp.get("finance_jump_log_mu", 0.9))
                log_sigma = max(float(hp.get("finance_jump_log_sigma", 0.5)), 1e-6)
                shocks = torch.exp(log_mu + log_sigma * torch.randn_like(y_out))
                y_out = torch.where(jump_mask, y_out * shocks, y_out)

                crash_prob = float(np.clip(hp.get("finance_jump_crash_prob", 0.15), 0.0, 1.0))
                if crash_prob > 0.0:
                    crash_mask = jump_mask & (torch.rand_like(y_out) < crash_prob)
                    y_out = torch.where(crash_mask, -y_out, y_out)

        return y_out

    def _normalize_continuous_target(
        self,
        y: Tensor,
        train_size: int,
        norm_method: str = 'zscore',
        add_skewness: bool = True,
        hp: Optional[Dict[str, Any]] = None,
        X: Optional[Tensor] = None,
    ) -> Tensor:
        """Normalize continuous regression targets using support-set-only statistics (ICL-safe).
        
        Following best practices:
        - Compute normalization statistics from support set (train) only
        - Apply same normalization to query set (test) using support stats
        - Add controlled variation for diversity across datasets
        
        Parameters
        ----------
        y : Tensor
            Target tensor of shape (seq_len,)
        train_size : int
            Size of support set (train portion) for ICL-safe normalization
        norm_method : str, default='zscore'
            Normalization method: 'zscore' (standard-style) or 'minmax'
        add_skewness : bool, default=True
            If True, apply controlled skewness to targets for realism
            (Real-world targets are often skewed: prices, counts, etc.)
            
        Returns
        -------
        Tensor
            Normalized targets with controlled variation
        """
        # Edge case validation
        if train_size <= 0 or train_size >= y.shape[0]:
            raise ValueError(f"Invalid train_size {train_size} for seq_len {y.shape[0]}")
        
        hp = hp or {}

        # Split into support (train) and query (test) sets
        y_support = y[:train_size]  # Support set for computing statistics
        
        # Compute statistics from support set only (ICL-safe)
        # This prevents leakage: query set doesn't influence normalization
        from .reg2cls import torch_nanstd
        mean = torch.nanmean(y_support)
        # For 1D tensor (target), compute std directly
        # Add dimension for torch_nanstd compatibility, then extract scalar
        y_support_2d = y_support.unsqueeze(-1)  # (seq_len,) -> (seq_len, 1)
        std = torch_nanstd(y_support_2d, dim=0, ddof=1 if len(y_support) > 1 else 0)
        std = std.squeeze().clip(min=1e-6)  # Extract scalar and ensure > 0
        
        # Apply normalization based on method
        if norm_method == 'minmax':
            # Robust nan-aware min/max
            valid_y = y_support[~torch.isnan(y_support)]
            if valid_y.numel() > 0:
                y_min = valid_y.min()
                y_max = valid_y.max()
            else:
                y_min = torch.tensor(0.0, device=y.device)
                y_max = torch.tensor(1.0, device=y.device)
            y_range = (y_max - y_min).clip(min=1e-6)
            y_normalized = (y - y_min) / y_range
        else:  # 'zscore' (default)
            # Handle edge case: constant support set (std ≈ 0)
            if std < 1e-6:
                # If support set is constant, use identity normalization
                y_normalized = y - mean
            else:
                # Apply support-set statistics to both support and query
                y_normalized = (y - mean) / std
        
        # Removed random offsets for invertibility
        # The model learns scale-invariance from diverse input scales in the prior data
        # This ensures exact inversion during inference using stored normalization parameters
        # Previous approach added random offsets that couldn't be inverted:
        #   mean_offset ~ Uniform(-2, 2), std_scale ~ Uniform(0.5, 2.0)
        #   y_varied = y_normalized * std_scale + mean_offset
        # This made exact denormalization impossible during inference
        
        # DATASET QUALITY IMPROVEMENT: Add controlled skewness to targets
        # Limix-based-Changes - Controlled skewness (similar to LimiX's power transformations for targets)
        # Real-world regression targets are often skewed (prices, counts, etc.)
        # Problem: Current normalization removes skewness (z-score centers at 0)
        # Solution: Apply power transform after normalization to add controlled skewness
        if add_skewness:
            # Sample skew parameter from [0.6, 2.0] to add diversity
            # Most real-world targets are right-skewed, so favor that range
            skew_param = np.random.uniform(0.6, 2.0)
            
            if skew_param < 1.0:
                # Left-skew: apply power transform to negative values
                y_neg = y_normalized.clamp(max=0)
                y_pos = y_normalized.clamp(min=0)
                y_normalized = -((-y_neg).clamp(min=-10.0) ** (1.0 / skew_param)) + y_pos
            elif skew_param > 1.0:
                # Right-skew: apply power transform to positive values
                y_neg = y_normalized.clamp(max=0)
                y_pos = y_normalized.clamp(min=0)
                y_normalized = y_neg + (y_pos.clamp(max=10.0) ** skew_param)
            # If skew_param == 1.0, no transformation (already handled by elif)

        # Optional heavy-tailed target noise (Student-t via Gaussian / Gamma scale mixture).
        heavy_tail_target_rate = float(hp.get("heavy_tail_target_rate", 0.0) or 0.0)
        if heavy_tail_target_rate > 0.0 and np.random.random() < heavy_tail_target_rate:
            nu_min = float(hp.get("target_student_t_df_min", 2.0))
            nu_max = float(hp.get("target_student_t_df_max", 5.0))
            nu_low = min(nu_min, nu_max)
            nu_high = max(nu_min, nu_max)
            nu = max(float(np.random.uniform(nu_low, nu_high)), 2.05)

            concentration = torch.tensor(nu / 2.0, device=y.device, dtype=y.dtype)
            gamma = torch.distributions.Gamma(concentration=concentration, rate=concentration).sample(y_normalized.shape)
            t_noise = torch.randn_like(y_normalized) / torch.sqrt(gamma.clamp(min=1e-6))
            noise_scale = float(hp.get("target_student_t_scale", 0.15))
            y_normalized = y_normalized + noise_scale * t_noise

        # Optional multimodal target shaping (two-component Gaussian mixture).
        multimodal_target_rate = float(hp.get("multimodal_target_rate", 0.0) or 0.0)
        if multimodal_target_rate > 0.0 and np.random.random() < multimodal_target_rate:
            mix_weight = float(hp.get("target_mixture_weight", np.random.uniform(0.35, 0.65)))
            mix_weight = min(max(mix_weight, 0.05), 0.95)

            if X is not None and X.dim() == 2 and X.shape[1] > 0:
                gate_idx = int(torch.randint(0, X.shape[1], (1,), device=X.device).item())
                gate_feat = X[:, gate_idx]
                gate_threshold = torch.quantile(gate_feat[:train_size], mix_weight)
                component_high = gate_feat > gate_threshold
            else:
                component_high = torch.rand_like(y_normalized) < mix_weight

            shift = float(hp.get("target_mixture_shift", np.random.uniform(1.0, 3.0)))
            std_low = float(hp.get("target_mixture_std_low", 0.5))
            std_high = float(hp.get("target_mixture_std_high", 1.0))

            comp_low = y_normalized - shift + torch.randn_like(y_normalized) * std_low
            comp_high = y_normalized + shift + torch.randn_like(y_normalized) * std_high
            y_normalized = torch.where(component_high, comp_high, comp_low)

        # Optional finance-style dynamics on targets:
        # - volatility clustering (GARCH-like conditional variance)
        # - Poisson jump diffusion (log-normal shock multipliers)
        y_normalized = self._apply_finance_target_dynamics(y_normalized, train_size, hp)

        # Optional bounded non-linearity (SimpleRegressionPrior-style): tanh(y) * scale.
        bounded_target_rate = float(hp.get("bounded_target_rate", 0.0) or 0.0)
        bounded_target_scale = float(hp.get("bounded_target_scale", 2.0))
        apply_bounded_target = bounded_target_rate > 0.0 and np.random.random() < bounded_target_rate
        if apply_bounded_target:
            y_normalized = torch.tanh(y_normalized) * bounded_target_scale

        # Optional strictly positive log-normal shaping after SCM effects.
        positive_lognormal_rate = float(hp.get("positive_lognormal_target_rate", 0.0) or 0.0)
        positive_lognormal_temperature = float(hp.get("positive_lognormal_temperature", 1.0))
        apply_positive_lognormal = (
            positive_lognormal_rate > 0.0 and np.random.random() < positive_lognormal_rate
        )
        if apply_positive_lognormal:
            y_pre = (positive_lognormal_temperature * y_normalized).clamp(min=-8.0, max=4.0)
            y_normalized = torch.exp(y_pre)
        
        # Safety clipping to prevent extreme outliers from causing gradient explosion
        # Documentation specifies [-10, 10] for safety
        # Normalized targets should be ~[-3, 3] for z-score, [0, 1] for min-max
        # Clipping prevents rare extreme values from destabilizing training
        if apply_positive_lognormal:
            y_normalized = y_normalized.clamp(min=1e-6, max=10.0)
        else:
            y_normalized = y_normalized.clamp(min=-10.0, max=10.0)
        
        return y_normalized

    def _apply_cross_sectional_rank_normalization(
        self,
        X: Tensor,
        feature_meta: Dict[str, Tensor],
        train_size: int,
        hp: Dict[str, Any],
    ) -> Tensor:
        """Apply cross-sectional rank normalization to a proportion of features.
        
        Cross-sectional rank normalization replaces feature columns with their rank
        within the sequence (rank / n), standard in quant finance. When ranking,
        masked cells are placed at the end using the meta features.
        
        Parameters
        ----------
        X : Tensor, shape (seq_len, max_features)
            Feature matrix to transform.
        feature_meta : Dict[str, Tensor]
            Metadata dictionary containing 'missing_mask' of shape (seq_len, max_features).
        train_size : int
            Number of support-set rows (for consistency, not used in ranking).
        hp : Dict[str, Any]
            Hyperparameter dict. Relevant keys:
            - cross_sectional_rank_feature_proportion: float in [0, 1]
              Proportion of features to rank-normalize.
        
        Returns
        -------
        Tensor
            Feature matrix with selected columns rank-normalized, shape (seq_len, max_features).
        """
        seq_len, max_features = X.shape
        device = X.device
        dtype = X.dtype
        
        # Get proportion of features to rank-normalize
        feature_proportion = float(hp.get("cross_sectional_rank_feature_proportion", 0.0))
        if feature_proportion <= 0.0:
            return X
        
        # Get missing mask and feature type ids
        missing_mask = feature_meta.get("missing_mask", torch.zeros_like(X, dtype=torch.bool))
        feature_type_ids = feature_meta.get("feature_type_ids", None)
        
        # Get number of active features
        num_features = int(hp.get("num_features", max_features))
        
        # Filter to only rankable feature types: continuous, count, proportion
        # Exclude: categorical, binary, label, padding
        rankable_type_ids = {FEATURE_TYPE_TO_ID["continuous"], FEATURE_TYPE_TO_ID["count"], FEATURE_TYPE_TO_ID["proportion"]}
        
        # Find rankable features (only from active features)
        if feature_type_ids is not None:
            rankable_mask = torch.zeros(num_features, dtype=torch.bool, device=device)
            for feat_idx in range(num_features):
                feat_type_id = int(feature_type_ids[feat_idx].item())
                if feat_type_id in rankable_type_ids:
                    rankable_mask[feat_idx] = True
            rankable_indices = torch.where(rankable_mask)[0]
        else:
            # Fallback: assume all active features are rankable if feature_type_ids not available
            rankable_indices = torch.arange(num_features, device=device)
        
        n_rankable = rankable_indices.numel()
        if n_rankable == 0:
            # No rankable features, return original X
            return X
        
        # Select features to rank-normalize (only from rankable features)
        n_features_to_rank = max(1, int(feature_proportion * n_rankable))
        n_features_to_rank = min(n_features_to_rank, n_rankable)
        
        # Randomly select which rankable features to rank-normalize
        permuted_rankable = rankable_indices[torch.randperm(n_rankable, device=device)]
        feature_indices = permuted_rankable[:n_features_to_rank]
        
        # Create a copy to modify
        X_ranked = X.clone()
        
        # Apply rank normalization to each selected feature
        for feat_idx in feature_indices:
            feat_idx = feat_idx.item()
            feature_col = X[:, feat_idx].clone()
            col_missing_mask = missing_mask[:, feat_idx]
            
            # Separate valid (non-missing) and missing values
            valid_mask = ~col_missing_mask
            valid_values = feature_col[valid_mask]
            n_valid = valid_values.numel()
            
            if n_valid == 0:
                # All values are missing, skip this feature
                continue
            
            # Check for constant feature (all valid values are the same)
            if n_valid > 1:
                valid_std = valid_values.std()
                if valid_std < 1e-6:
                    # Constant feature: assign uniform rank (middle value)
                    normalized_ranks = torch.full((n_valid,), 0.5, device=device, dtype=dtype)
                else:
                    # Compute ranks for valid values
                    # Use 'average' method: ties get average rank
                    # Sort indices to get ranks
                    sorted_indices = torch.argsort(valid_values)
                    ranks = torch.zeros_like(valid_values, dtype=dtype)
                    
                    # Assign ranks, handling ties by averaging
                    i = 0
                    while i < n_valid:
                        # Find all values equal to current value (ties)
                        current_val = valid_values[sorted_indices[i]]
                        tie_start = i
                        while i < n_valid and valid_values[sorted_indices[i]] == current_val:
                            i += 1
                        tie_end = i
                        # Average rank for tied values: (start_rank + end_rank) / 2
                        # start_rank = tie_start + 1, end_rank = tie_end
                        avg_rank = (tie_start + tie_end + 1) / 2.0
                        ranks[sorted_indices[tie_start:tie_end]] = avg_rank
                    
                    # Normalize ranks: rank / n (where n is number of valid values)
                    # This gives values in [1/n, 1] for valid values
                    normalized_ranks = ranks / n_valid
            else:
                # Single valid value: assign rank 0.5 (middle)
                normalized_ranks = torch.tensor([0.5], device=device, dtype=dtype)
            
            # Create full column: valid values get normalized ranks, missing values go to end (rank = 1.0)
            ranked_col = torch.ones(seq_len, device=device, dtype=dtype)
            ranked_col[valid_mask] = normalized_ranks
            # Update the feature column
            X_ranked[:, feat_idx] = ranked_col
        
        return X_ranked

    def _apply_censored_targets(
        self,
        X: Tensor,
        y: Tensor,
        train_size: int,
        hp: Dict[str, Any],
        feature_meta: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Apply censored targets for survival analysis.
        
        Generates time-to-event data with censoring:
        - Event time T ~ Weibull(λ, k)
        - Censoring time C ~ Exp(η)
        - Observed time y = min(T, C)
        - Censoring indicator c = (T ≤ C) added as feature column
        
        Parameters
        ----------
        X : Tensor, shape (seq_len, max_features)
            Feature matrix.
        y : Tensor, shape (seq_len,)
            Continuous target values (will be transformed to censored times).
        train_size : int
            Number of support-set rows.
        hp : Dict[str, Any]
            Hyperparameter dict. Relevant keys:
            - censored_target_weibull_lambda: float, scale parameter for Weibull (default: 1.0)
            - censored_target_weibull_k: float, shape parameter for Weibull (default: 1.5)
            - censored_target_exp_eta: float, rate parameter for Exponential censoring (default: 0.5)
        feature_meta : Dict[str, Tensor]
            Metadata dictionary. Will be updated with censoring indicator column.
        
        Returns
        -------
        X : Tensor, shape (seq_len, max_features)
            Feature matrix with censoring indicator column added.
        y : Tensor, shape (seq_len,)
            Censored time-to-event values (min(T, C)).
        """
        seq_len = y.shape[0]
        device = y.device
        dtype = y.dtype
        
        # Get hyperparameters
        weibull_lambda = float(hp.get("censored_target_weibull_lambda", 1.0))
        weibull_k = float(hp.get("censored_target_weibull_k", 1.5))
        exp_eta = float(hp.get("censored_target_exp_eta", 0.5))
        
        # Ensure positive parameters
        weibull_lambda = max(0.1, weibull_lambda)
        weibull_k = max(0.1, weibull_k)
        exp_eta = max(0.01, exp_eta)
        
        # Transform y to be strictly positive (needed for Weibull generation)
        # Use support-set statistics for ICL safety
        y_support = y[:train_size]
        y_min = y_support.min()
        y_shift = max(0.0, -y_min.item() + 1e-6)
        y_positive = y + y_shift
        
        # Normalize y to have reasonable scale for survival times
        # Use support-set mean and std
        y_support_pos = y_positive[:train_size]
        y_mean = y_support_pos.mean()
        y_std = y_support_pos.std().clamp(min=1e-6)
        y_normalized = (y_positive - y_mean) / y_std
        
        # Generate event times T ~ Weibull(λ, k)
        # Weibull CDF: F(t) = 1 - exp(-(t/λ)^k)
        # Inverse CDF: t = λ * (-log(1-U))^(1/k) where U ~ Uniform(0,1)
        # Use y_normalized to influence the scale (make it dependent on features)
        u = torch.rand(seq_len, device=device, dtype=dtype)
        # Map normalized y to Weibull scale (ensure positive)
        scale_factor = torch.exp(y_normalized.clamp(min=-3, max=3))
        weibull_scale = weibull_lambda * scale_factor
        T = weibull_scale * ((-torch.log(1 - u + 1e-10)).clamp(min=1e-10) ** (1.0 / weibull_k))
        
        # Generate censoring times C ~ Exp(η)
        # Exponential CDF: F(c) = 1 - exp(-η*c)
        # Inverse CDF: c = -log(1-U) / η
        u_cens = torch.rand(seq_len, device=device, dtype=dtype)
        C = -torch.log(1 - u_cens + 1e-10) / exp_eta
        
        # Observed time: y = min(T, C)
        y_censored = torch.minimum(T, C)
        
        # Censoring indicator: c = (T ≤ C), 1 if event occurred, 0 if censored
        c = (T <= C).to(dtype=dtype)
        
        # Add censoring indicator as a feature column
        # Similar to categorical attribute injection strategy
        max_features = X.shape[1]
        num_features = int(hp.get("num_features", max_features))
        
        # Find a slot for the censoring indicator
        if num_features < max_features:
            # Use padding slot
            slot = num_features
            hp["num_features"] = num_features + 1
        else:
            # Need to steal a slot - use the last feature
            slot = num_features - 1
        
        # Create updated X with censoring indicator
        X_censored = X.clone()
        X_censored[:, slot] = c
        
        # Update feature_meta
        if "feature_type_ids" in feature_meta:
            feature_meta["feature_type_ids"][slot] = FEATURE_TYPE_TO_ID["binary"]

        return X_censored, y_censored

    def _normalize_strictly_positive_target(
        self,
        y: Tensor,
        train_size: int,
        hp: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """Normalize strictly positive targets (prices, volumes, AUM) via log + z-score.

        Pipeline:
            1. Shift y so all values are strictly positive (support-set min only)
            2. Log-transform: log(y + shift)
            3. Z-score using support-set statistics only (ICL-safe)

        The returned values are zero-centered with unit variance on the support set,
        which is what model training expects. Strict positivity of the *input* is
        enforced by the shift; strict positivity of the *output* is not guaranteed
        (nor needed — handle that at prediction time via exp or softplus).

        Parameters
        ----------
        y : Tensor
            Target tensor of shape (seq_len,). May contain non-positive values.
        train_size : int
            Number of support (train) samples. Statistics are computed on these only.
        hp : dict, optional
            Reserved for future extensions.

        Returns
        -------
        Tensor
            Normalized targets of shape (seq_len,), approximately N(0, 1) on support.
        """
        if train_size <= 0 or train_size >= y.shape[0]:
            raise ValueError(
                f"train_size must be in (0, {y.shape[0]}), got {train_size}"
            )

        epsilon = 1e-6

        # Step 1: Shift so all values are strictly positive.
        # Use only the support min to stay ICL-safe (no query leakage).
        y_support = y[:train_size]
        # NaN-aware min: filter out NaNs, then compute min
        valid_y = y_support[~torch.isnan(y_support)]
        if valid_y.numel() > 0:
            support_min = valid_y.min()
        else:
            # Fallback if all values are NaN
            support_min = torch.tensor(0.0, device=y.device, dtype=y.dtype)

        shift = (-support_min + epsilon) if support_min <= 0 else 0.0
        y_positive = y + shift  # query set shifted by the same constant — fine

        # Step 2: Log-transform.
        log_y = torch.log(y_positive.clamp(min=epsilon))

        # Step 3: Z-score using support-set statistics only.
        from .reg2cls import torch_nanstd

        log_support = log_y[:train_size]
        mean_log = torch.nanmean(log_support)

        ddof = 1 if train_size > 1 else 0
        std_log = torch_nanstd(
            log_support.unsqueeze(-1), dim=0, ddof=ddof
        ).squeeze().clamp(min=epsilon)

        # Handle edge case: constant support set (std ≈ 0)
        if std_log < 1e-6:
            # If support set is constant in log-space, use identity normalization
            log_y_normalized = log_y - mean_log
        else:
            # Apply support-set statistics to both support and query
            log_y_normalized = (log_y - mean_log) / std_log
        
        # Step 4: Apply exp() to return to strictly positive normalized values
        # Clamp normalized log values to prevent exp() overflow
        # exp(10) ≈ 22026, exp(-10) ≈ 4.5e-5, so [-10, 10] is a safe range
        log_y_normalized = log_y_normalized.clamp(min=-10.0, max=10.0)
        y_normalized_positive = torch.exp(log_y_normalized)
        
        # Final safety check: ensure all values are strictly positive
        y_normalized_positive = y_normalized_positive.clamp(min=epsilon)
        
        return y_normalized_positive

    @torch.no_grad()
    def get_batch(
        self, batch_size: Optional[int] = None, step: Optional[int] = None, **kwargs
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]],
    ]:
        """
        Generates a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size
        step : int, optional
            Current training step for curriculum learning. If None, curriculum is not applied.

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

        train_sizes : Tensor
            Position for train/test split for each dataset, shape (batch_size,)

        feature_meta : Dict[str, Tensor], optional
            Returned when ``return_metadata=True``. Contains:
            ``missing_mask`` (B,T,H), ``feature_type_ids`` (B,H),
            ``col_ids`` (B,H), ``imputation_strategy_ids`` (B,H).
        """
        batch_size = batch_size or self.batch_size
        
        if self.use_curriculum and step is not None:
            ratio = self.get_curriculum_ratio(step, log=True, **kwargs)
            logger.debug(
                f"Generating batch: step={step}, curriculum_ratio={ratio:.3f}, "
                f"batch_size={batch_size}, device={self.device}"
            )
            
            # Easy Batch Interleaving (Replay) strategy
            # 10% chance to generate an "easy" batch (Linear/GP, few features, clean noise)
            # This prevents catastrophic forgetting of simple patterns and maintains "first principles"
            is_easy_batch = np.random.random() < 0.1
            if is_easy_batch:
                logger.debug(f"REPLAY: Generating EASY BATCH for step {step} (Linear/GP, low features)")
        else:
            is_easy_batch = False

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None
        global_train_size = None

        # Determine global seq_len/train_size if not per-group
        if not self.seq_len_per_gp:
            if is_easy_batch:
                # Easy batch: moderate sequence length for clear signal, floored at
                # min_seq_len. Upstream ignored min_seq_len here, so every easy batch
                # landed under LTM1's 1000-row minimum and was silently dropped --
                # taking the anti-forgetting replay with it.
                global_seq_len = max(self.min_seq_len, np.random.randint(200, 1000))
            else:
                global_seq_len = self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small, step=step
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
            global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len, self.fixed_val_size, step=step)
            logger.debug(
                f"Global parameters: seq_len={global_seq_len}, train_size={global_train_size}, "
                f"max_features={self.max_features}"
            )

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            group_sampled_hp = self.hp_sampling(step=step, **kwargs)
            # If per-group, sample seq_len and train_size for this group. Otherwise, use global ones
            if self.seq_len_per_gp:
                if is_easy_batch:
                    gp_seq_len = max(self.min_seq_len, np.random.randint(200, 1000))
                else:
                    gp_seq_len = self.sample_seq_len(
                        self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small, step=step
                    )
                gp_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len, self.fixed_val_size, step=step)
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_features = self.max_features

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
                if actual_subgp_size <= 0:
                    break

                # Feature sampling uses Beta distribution (right-skewed)
                if is_easy_batch:
                    # Easy batch: small number of features [2, 10]
                    subgp_num_features = np.random.randint(2, 11)
                    # Easy batch: Force Linear or GP (simple functions)
                    subgp_prior_type = np.random.choice(["linear_scm", "gp_scm"])
                else:
                    subgp_num_features = self.sample_num_features_beta(self.min_features, gp_max_features, step=step)
                    # Subgroups share prior type, number of features, and sampled HPs
                    # Pass num_features to avoid problematic priors when features are very few
                    subgp_prior_type = self.get_prior(step=step, num_features=subgp_num_features)
                # Don't call mlp_activations or conv_activations here - they're factories that should be called per layer
                # This matches the original design where mlp_activations()/conv_activations() is always called in _make_layer_block
                subgp_sampled_hp = {
                    k: (v() if callable(v) and k not in ["mlp_activations", "conv_activations"] else v)
                    for k, v in group_sampled_hp.items()
                }
                
                # For regression tasks, prefer non-causal structure and joint selection
                # This simulates missing value imputation scenarios and aligns with best empirical practices
                if self.max_classes == 0:  # Regression task
                    # Bias toward non-causal for regression
                    # Early Curriculum: Prefer Causal (True) structure (CLEANER forward relationship X->y)
                    # Late Curriculum: Prefer Non-Causal (False) (More reverse causality/confounding)
                    if self.use_curriculum:
                        # Ratio 0.0 (Easy) -> p_causal = 0.8
                        # Ratio 1.0 (Hard) -> p_causal = 0.3
                        # We linearly interpolate between 0.8 and 0.3 based on ratio
                        ratio = self.get_curriculum_ratio(step, **kwargs)
                        p_causal = 0.8 - (0.8 - 0.3) * ratio
                    else:
                        p_causal = 0.3 # Default hard setting

                    subgp_sampled_hp["y_is_effect"] = np.random.choice([True, False], p=[p_causal, 1 - p_causal])
                    
                    # Bias toward joint target selection for regression (70% True, 30% False)
                    subgp_sampled_hp["use_joint_covariance_sampling"] = np.random.choice([True, False], p=[0.7, 0.3])

                if is_easy_batch:
                    # Overrides for Easy Batch mode
                    # 1. Clean noise
                    # GPU-optimized: Use torch.rand instead of numpy (stays on GPU)
                    subgp_sampled_hp["noise_std"] = torch.rand(1, device=self.device).item() * (0.05 - 0.0001) + 0.0001
                    subgp_sampled_hp["noise_variance"] = torch.rand(1, device=self.device).item() * (0.05 - 0.01) + 0.01
                    # 2. Smooth functions (for GP)
                    # Use moderate length_scale to avoid numerical instability
                    # Large values (>2.0) relative to coord range [0,1] cause nearly-singular kernel matrices
                    # GPU-optimized: Use torch.rand instead of numpy (stays on GPU)
                    subgp_sampled_hp["length_scale"] = torch.rand(1, device=self.device).item() * (2.0 - 0.5) + 0.5
                    # 3. Simple structure
                    subgp_sampled_hp["y_is_effect"] = True # Causal (X->y) is easier
                    # 4. Deterministic targets (no fancy node selection)
                    subgp_sampled_hp["use_joint_covariance_sampling"] = False

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
                        adjusted_train_size = gp_train_size
                    else:
                        # When seq_len_per_gp=False, use global seq_len (already cell-capped at global level)
                        # This ensures all datasets have the same sequence length for stacking
                        adjusted_seq_len = gp_seq_len
                        adjusted_train_size = gp_train_size
                    
                    # Create parameters dictionary for this dataset
                    params = {
                        **self.fixed_hp,  # Fixed HPs
                        "seq_len": adjusted_seq_len,
                        "train_size": adjusted_train_size,
                        # If per-gp setting, use adjusted max features for this group because we use nested tensors
                        # If not per-gp setting, use global max features to fix size for concatenation
                        "max_features": gp_max_features if self.seq_len_per_gp else self.max_features,
                        **subgp_sampled_hp,  # sampled HPs for this group
                        "prior_type": subgp_prior_type,
                        "num_features": subgp_num_features,
                        "num_classes": ds_num_classes,
                        "device": self.device,
                    }
                    # Finance stage controls are intentionally fixed per run/stage.
                    # Sampled HP can still drive other realism dimensions.
                    finance_keys = (
                        "finance_realism_rate",
                        "finance_input_tail_rate",
                        "finance_input_pareto_alpha",
                        "finance_input_shock_prob",
                        "finance_input_shock_scale",
                        "finance_input_negative_shock_prob",
                        "finance_garch_rate",
                        "finance_garch_omega",
                        "finance_garch_alpha",
                        "finance_garch_beta",
                        "finance_jump_rate",
                        "finance_jump_lambda",
                        "finance_jump_log_mu",
                        "finance_jump_log_sigma",
                        "finance_jump_crash_prob",
                    )
                    for key in finance_keys:
                        if key in self.fixed_hp:
                            params[key] = self.fixed_hp[key]
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
                                total_steps = getattr(self, 'total_steps', None)
                                weights = get_prior_weights(step, total_steps)
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
                results = joblib.Parallel()(joblib.delayed(self.generate_dataset)(params) for params in param_list)
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
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list], device=self.device, dtype=torch.long
        )

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
            return X, y, d, seq_lens, train_sizes, feature_meta
        return X, y, d, seq_lens, train_sizes

    def get_prior(self, step: Optional[int] = None, num_features: Optional[int] = None) -> str:
        """
        Determine which prior type to use for generation.

        For 'mix_scm' prior type, randomly selects between available priors
        based on configured probabilities. With curriculum learning, uses
        more linear priors early in training. Also avoids problematic priors
        (tree_scm, gp_scm) when feature count is very small.

        Parameters
        ----------
        step : int, optional
            Current training step for curriculum-aware weighting
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
                if self.use_curriculum and step is not None:
                    ratio = self.get_curriculum_ratio(step)
                    # Early: 65% linear, 35% mlp
                    # Late: 45% linear, 55% mlp
                    linear_weight = 0.65 - 0.2 * ratio
                    mlp_weight = 1.0 - linear_weight
                    return np.random.choice(["linear_scm", "mlp_scm"], p=[linear_weight, mlp_weight])
                else:
                    return np.random.choice(["linear_scm", "mlp_scm"], p=[0.6, 0.4])  # Slightly favor linear
            
            # Curriculum-aware weighting with a realism tilt over time.
            if self.use_curriculum and step is not None:
                ratio = self.get_curriculum_ratio(step)
                # Early (ratio~0.2): more linear/simple; late (ratio~1.0): more tree/time-lagged.
                linear_weight = 0.415 - 0.275 * ratio         # 0.36 -> 0.14
                mlp_total = 0.235 - 0.075 * ratio             # 0.22 -> 0.16
                mlp_weight = mlp_total * 0.5
                conv_weight = mlp_total * 0.5
                tree_weight = 0.105 + 0.175 * ratio           # 0.14 -> 0.28
                gp_weight = 0.07 + 0.05 * ratio               # 0.08 -> 0.12
                time_lagged_weight = 0.175 + 0.125 * ratio    # 0.20 -> 0.30

                total = (
                    linear_weight + mlp_weight + conv_weight
                    + tree_weight + gp_weight + time_lagged_weight
                )
                mix_probas = [
                    linear_weight / total,
                    mlp_weight / total,
                    conv_weight / total,
                    tree_weight / total,
                    gp_weight / total,
                    time_lagged_weight / total,
                ]
                return np.random.choice(
                    ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "gp_scm", "time_lagged_scm"],
                    p=mix_probas,
                )
            else:
                # Default order: [Linear, MLP, Conv, Tree, GP, TimeLagged]
                mix_probas = self.fixed_hp.get("mix_probas", [0.14, 0.20, 0.12, 0.24, 0.12, 0.18])
                if len(mix_probas) == 4:
                    # Legacy order: [Linear, MLP, Tree, GP]
                    linear_weight, mlp_total, tree_weight, gp_weight = mix_probas
                    mlp_weight = mlp_total * 0.5
                    conv_weight = mlp_total * 0.5
                    time_lagged_weight = mlp_total * 0.5
                    mix_probas = [linear_weight, mlp_weight, conv_weight, tree_weight, gp_weight, time_lagged_weight]
                elif len(mix_probas) == 5:
                    # Previous default order: [Linear, MLP, Conv, Tree, GP]
                    linear_weight, mlp_weight, conv_weight, tree_weight, gp_weight = mix_probas
                    time_lagged_weight = conv_weight
                    mix_probas = [linear_weight, mlp_weight, conv_weight, tree_weight, gp_weight, time_lagged_weight]
                elif len(mix_probas) != 6:
                    mix_probas = [0.14, 0.20, 0.12, 0.24, 0.12, 0.18]
                total = sum(mix_probas)
                mix_probas = [w / total for w in mix_probas]
                return np.random.choice(
                    ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "gp_scm", "time_lagged_scm"],
                    p=mix_probas,
                )

        elif self.prior_type == "mix_scm_no_gp":
            # Same as mix_scm but excludes gp_scm.
            # Avoid tree_scm when features are very few (it generates constant features)
            if num_features is not None and num_features <= 3:
                # Very few features: only use linear_scm and mlp_scm (more reliable)
                if self.use_curriculum and step is not None:
                    ratio = self.get_curriculum_ratio(step)
                    # Early: 65% linear, 35% mlp
                    # Late: 45% linear, 55% mlp
                    linear_weight = 0.65 - 0.2 * ratio
                    mlp_weight = 1.0 - linear_weight
                    return np.random.choice(["linear_scm", "mlp_scm"], p=[linear_weight, mlp_weight])
                else:
                    return np.random.choice(["linear_scm", "mlp_scm"], p=[0.6, 0.4])  # Slightly favor linear
            
            # Curriculum-aware weighting (no GP): shift toward tree/time-lagged later.
            if self.use_curriculum and step is not None:
                ratio = self.get_curriculum_ratio(step)
                linear_weight = 0.4625 - 0.3125 * ratio       # 0.40 -> 0.15
                mlp_total = 0.2625 - 0.1125 * ratio           # 0.24 -> 0.15
                mlp_weight = mlp_total * 0.5
                conv_weight = mlp_total * 0.5
                tree_weight = 0.1125 + 0.2375 * ratio         # 0.16 -> 0.35
                time_lagged_weight = 0.1625 + 0.1875 * ratio  # 0.20 -> 0.35

                total = linear_weight + mlp_weight + conv_weight + tree_weight + time_lagged_weight
                mix_probas = [
                    linear_weight / total,
                    mlp_weight / total,
                    conv_weight / total,
                    tree_weight / total,
                    time_lagged_weight / total,
                ]
                return np.random.choice(
                    ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "time_lagged_scm"],
                    p=mix_probas,
                )
            else:
                # Default order (no GP): [Linear, MLP, Conv, Tree, TimeLagged]
                mix_probas = self.fixed_hp.get("mix_probas")
                if mix_probas is None:
                    mix_probas = [0.16, 0.20, 0.12, 0.24, 0.28]
                elif len(mix_probas) == 6:
                    # Current full order: [Linear, MLP, Conv, Tree, GP, TimeLagged] -> drop GP
                    linear_weight, mlp_weight, conv_weight, tree_weight, _, time_lagged_weight = mix_probas
                    mix_probas = [linear_weight, mlp_weight, conv_weight, tree_weight, time_lagged_weight]
                elif len(mix_probas) == 5:
                    # Previous order: [Linear, MLP, Conv, Tree, GP]
                    linear_weight, mlp_weight, conv_weight, tree_weight, _ = mix_probas
                    time_lagged_weight = conv_weight
                    mix_probas = [linear_weight, mlp_weight, conv_weight, tree_weight, time_lagged_weight]
                elif len(mix_probas) == 4:
                    # Legacy order: [Linear, MLP, Tree, GP]
                    linear_weight, mlp_total, tree_weight, _ = mix_probas
                    mlp_weight = mlp_total * 0.5
                    conv_weight = mlp_total * 0.5
                    time_lagged_weight = mlp_total * 0.5
                    mix_probas = [linear_weight, mlp_weight, conv_weight, tree_weight, time_lagged_weight]
                elif len(mix_probas) == 3:
                    # Legacy order: [Linear, MLP, Tree]
                    linear_weight, mlp_total, tree_weight = mix_probas
                    mlp_weight = mlp_total * 0.5
                    conv_weight = mlp_total * 0.5
                    time_lagged_weight = mlp_total * 0.5
                    mix_probas = [linear_weight, mlp_weight, conv_weight, tree_weight, time_lagged_weight]
                else:
                    mix_probas = [0.16, 0.20, 0.12, 0.24, 0.28]
                total = sum(mix_probas)
                mix_probas = [w / total for w in mix_probas]
                return np.random.choice(
                    ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "time_lagged_scm"],
                    p=mix_probas,
                )
        elif self.prior_type == "mix_scm_hscm":
            # mix_scm_hscm includes hybrid_scm in addition to mix_scm priors
            # Avoid tree_scm, gp_scm, and hybrid_scm when features are very few
            if num_features is not None and num_features <= 3:
                # Very few features: only use linear_scm and mlp_scm (more reliable)
                if self.use_curriculum and step is not None:
                    ratio = self.get_curriculum_ratio(step)
                    linear_weight = 0.65 - 0.2 * ratio
                    mlp_weight = 1.0 - linear_weight
                    return np.random.choice(["linear_scm", "mlp_scm"], p=[linear_weight, mlp_weight])
                else:
                    return np.random.choice(["linear_scm", "mlp_scm"], p=[0.6, 0.4])
            
            # Curriculum-aware weighting for mix_scm_hscm
            if self.use_curriculum and step is not None:
                ratio = self.get_curriculum_ratio(step)
                linear_weight = 0.30 - 0.20 * ratio           # 0.26 -> 0.10
                mlp_total = 0.155 - 0.075 * ratio             # 0.14 -> 0.08
                mlp_weight = mlp_total * 0.5
                conv_weight = mlp_total * 0.5
                tree_weight = 0.095 + 0.125 * ratio           # 0.12 -> 0.22
                gp_weight = 0.075 + 0.025 * ratio             # 0.08 -> 0.10
                time_lagged_weight = 0.135 + 0.125 * ratio    # 0.16 -> 0.26
                hybrid_weight = 0.24                          # Constant weight for hybrid

                total = (
                    linear_weight + mlp_weight + conv_weight + tree_weight
                    + gp_weight + time_lagged_weight + hybrid_weight
                )
                mix_probas = [
                    linear_weight / total,
                    mlp_weight / total,
                    conv_weight / total,
                    tree_weight / total,
                    gp_weight / total,
                    time_lagged_weight / total,
                    hybrid_weight / total,
                ]
                return np.random.choice(
                    ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "gp_scm", "time_lagged_scm", "hybrid_scm"],
                    p=mix_probas,
                )
            else:
                # Default order: [Linear, MLP, Conv, Tree, GP, TimeLagged, Hybrid]
                return np.random.choice(
                    ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "gp_scm", "time_lagged_scm", "hybrid_scm"],
                    p=[0.14, 0.12, 0.08, 0.20, 0.10, 0.18, 0.18],
                )
        elif self.prior_type == "mix_scm_hscm_no_gp":
            # mix_scm_hscm_no_gp: Includes HybridSCM but excludes GP-SCM (for extreme scales)
            # Avoid tree_scm and hybrid_scm when features are very few
            if num_features is not None and num_features <= 3:
                # Very few features: only use linear_scm and mlp_scm (more reliable)
                if self.use_curriculum and step is not None:
                    ratio = self.get_curriculum_ratio(step)
                    linear_weight = 0.65 - 0.2 * ratio
                    mlp_weight = 1.0 - linear_weight
                    return np.random.choice(["linear_scm", "mlp_scm"], p=[linear_weight, mlp_weight])
                else:
                    return np.random.choice(["linear_scm", "mlp_scm"], p=[0.6, 0.4])
            
            # Curriculum-aware weighting for mix_scm_hscm_no_gp (no GP-SCM)
            if self.use_curriculum and step is not None:
                ratio = self.get_curriculum_ratio(step)
                linear_weight = 0.325 - 0.225 * ratio         # 0.28 -> 0.10
                mlp_total = 0.175 - 0.075 * ratio             # 0.16 -> 0.10
                mlp_weight = mlp_total * 0.5
                conv_weight = mlp_total * 0.5
                tree_weight = 0.115 + 0.125 * ratio           # 0.14 -> 0.24
                time_lagged_weight = 0.145 + 0.175 * ratio    # 0.18 -> 0.32
                hybrid_weight = 0.24                          # Keep hybrid stable

                total = linear_weight + mlp_weight + conv_weight + tree_weight + time_lagged_weight + hybrid_weight
                mix_probas = [
                    linear_weight / total,
                    mlp_weight / total,
                    conv_weight / total,
                    tree_weight / total,
                    time_lagged_weight / total,
                    hybrid_weight / total,
                ]
                return np.random.choice(
                    ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "time_lagged_scm", "hybrid_scm"],
                    p=mix_probas,
                )
            else:
                # Default order: [Linear, MLP, Conv, Tree, TimeLagged, Hybrid]
                return np.random.choice(
                    ["linear_scm", "mlp_scm", "conv_scm", "tree_scm", "time_lagged_scm", "hybrid_scm"],
                    p=[0.15, 0.12, 0.08, 0.22, 0.23, 0.20],
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

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

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
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        device: str = "cpu",
        use_curriculum: bool = False,
        curriculum_schedule: str = "linear",
        curriculum_warmup_steps: int = 1000,
        curriculum_min_ratio: float = 0.3,
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=False,
            use_curriculum=use_curriculum,
            curriculum_schedule=curriculum_schedule,
            curriculum_warmup_steps=curriculum_warmup_steps,
            curriculum_min_ratio=curriculum_min_ratio,
        )
        self.device = device

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
            All datasets share the same split position.
        """

        batch_size = batch_size or self.batch_size
        seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)
        train_size = self.sample_train_size(self.min_train_size, self.max_train_size, seq_len)

        X = torch.randn(batch_size, seq_len, self.max_features, device=self.device)

        num_classes = np.random.randint(2, self.max_classes + 1)
        y = torch.randint(0, num_classes, (batch_size, seq_len), device=self.device)

        d = torch.full((batch_size,), self.max_features, device=self.device)
        seq_lens = torch.full((batch_size,), seq_len, device=self.device)
        train_sizes = torch.full((batch_size,), train_size, device=self.device)

        return X, y, d, seq_lens, train_sizes


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

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

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

    use_curriculum : bool, default=False
        Enable curriculum learning during data generation

    curriculum_schedule : str, default="linear"
        Curriculum schedule type: "linear", "cosine", or "step"

    curriculum_warmup_steps : int, default=1000
        Number of training steps to reach full difficulty

    curriculum_min_ratio : float, default=0.3
        Minimum difficulty ratio (0.0-1.0). Lower values = easier tasks at start
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
        use_curriculum: bool = False,
        curriculum_schedule: str = "linear",
        curriculum_warmup_steps: int = 1000,
        curriculum_min_ratio: float = 0.3,
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
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                device=device,
                use_curriculum=use_curriculum,
                curriculum_schedule=curriculum_schedule,
                curriculum_warmup_steps=curriculum_warmup_steps,
                curriculum_min_ratio=curriculum_min_ratio,
            )
        elif prior_type == "regression_dummy":
            self.prior = RegressionDummyPrior(
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                device=device,
                use_curriculum=use_curriculum,
                curriculum_schedule=curriculum_schedule,
                curriculum_warmup_steps=curriculum_warmup_steps,
                curriculum_min_ratio=curriculum_min_ratio,
            )
        elif prior_type in [
            "mlp_scm",
            "conv_scm",
            "tree_scm",
            "gp_scm",
            "linear_scm",
            "time_lagged_scm",
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
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                replay_small=replay_small,
                prior_type=prior_type,
                fixed_hp=scm_fixed_hp,
                sampled_hp=scm_sampled_hp,
                n_jobs=n_jobs,
                num_threads_per_generate=num_threads_per_generate,
                device=device,
                use_curriculum=use_curriculum,
                curriculum_schedule=curriculum_schedule,
                curriculum_warmup_steps=curriculum_warmup_steps,
                curriculum_min_ratio=curriculum_min_ratio,
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
                f"'linear_scm', 'time_lagged_scm', 'hybrid_scm', 'mix_scm', 'mix_scm_hscm', "
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
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.prior_type = prior_type

    def get_batch(
        self, batch_size: Optional[int] = None, step: Optional[int] = None, **kwargs
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]],
    ]:
        """
        Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call
        step : int, optional
            Current training step for curriculum learning. If None, curriculum is not applied.
        **kwargs : dict
            Optional overrides for curriculum (e.g., easy_floor, complexity_cap)

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

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
        """
        return self.prior.get_batch(batch_size, step=step, **kwargs)

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
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]],
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
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
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
        min_train_size: int | float = 0.1,
        max_train_size: int | float = 0.9,
        device: str = "cpu",
        noise_std: float = 0.1,
        use_curriculum: bool = False,
        curriculum_schedule: str = "linear",
        curriculum_warmup_steps: int = 1000,
        curriculum_min_ratio: float = 0.3,
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=0,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=False,
            use_curriculum=use_curriculum,
            curriculum_schedule=curriculum_schedule,
            curriculum_warmup_steps=curriculum_warmup_steps,
            curriculum_min_ratio=curriculum_min_ratio,
        )
        self.device = device
        self.noise_std = noise_std

    def get_batch(self, batch_size: int | None = None):
        bs = batch_size or self.batch_size
        seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=False)
        train_size = self.sample_train_size(self.min_train_size, self.max_train_size, seq_len)

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
        train_sizes = torch.full((bs,), train_size, device=self.device)
        return X, y, d, seq_lens, train_sizes
