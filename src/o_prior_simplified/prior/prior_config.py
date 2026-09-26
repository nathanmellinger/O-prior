from .activations import get_activations

# Tree prior weights per MITRA rationale (REGRESSION_SPEC.md Section 3.2)
# These weights reflect performance + diversity + distinctiveness
TREE_PRIOR_WEIGHTS = {
    'et': 0.15,   # Extra Trees: Distinctiveness booster (hard for SCM to mimic)
    'gb': 0.10,   # Gradient Boosting: Realistic boosting structure
    'dt': 0.05,   # Decision Tree: Axis-aligned discontinuities
    'rf': 0.08,   # Random Forest: Maximum diversity, weak standalone
    'dsrf': 0.02, # Directly Sampled RF: Optional overlap, keep small
}
# Note: SCM gets 0.60 weight, so tree priors total = 0.40


DEFAULT_FIXED_HP = {
    # SCMPrior
    "mix_probs": (0.7, 0.3),  # For backward compatibility (MLP, Tree)
    # Realism-oriented default mix for mix_scm:
    # [Linear, MLP, Conv, Tree, GP]
    "mix_probas": [0.15, 0.18, 0.05, 0.44, 0.01],
    # Reg2Cls
    "balanced": False,
    "multiclass_ordered_prob": 0.0,
    "scale_by_max_features": False,
    "permute_features": True,
    "permute_labels": True,
    # Feature metadata
    "randomize_column_ids": True,
    "return_metadata": False,
    "feature_type_categorical_max_unique": 24,
    "feature_type_categorical_max_frac": 0.12,
}

DEFAULT_SAMPLED_HP = {
    # Reg2Cls
    "multiclass_type": {"distribution": "meta_choice", "choice_values": ["value", "rank"]},
    # MLPSCM
    "mlp_activations": {
        "distribution": "meta_choice_mixed",
        "choice_values": get_activations(random=True, scale=True, diverse=True),
    },
    "block_wise_dropout": {"distribution": "meta_choice", "choice_values": [True, False]},
    "mlp_dropout_prob": {"distribution": "meta_beta", "scale": 0.9, "min": 0.1, "max": 5.0},
    # ConvSCM
    "conv_activations": {
        "distribution": "meta_choice_mixed",
        "choice_values": get_activations(random=True, scale=True, diverse=True),
    },
    "conv_dropout_prob": {"distribution": "meta_beta", "scale": 0.9, "min": 0.1, "max": 5.0},
    "hidden_channels": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10,
        "min_mean": 3,
        "round": True,
        "lower_bound": 3,
    },
    "kernel_size": {"distribution": "meta_choice", "choice_values": [3, 5, 7]},
    "stride": {"distribution": "meta_choice", "choice_values": [1, 2]},
    "padding": {"distribution": "meta_choice", "choice_values": [0, 1, 2]},
    # MLPSCM and TreeSCM
    "is_causal": {"distribution": "meta_choice", "choice_values": [True, False]},
    "num_causes": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 1,
        "round": True,
        "lower_bound": 1,
    },
    "y_is_effect": {"distribution": "meta_choice", "choice_values": [True, False]},
    "in_clique": {"distribution": "meta_choice", "choice_values": [True, False]},
    "sort_features": {"distribution": "meta_choice", "choice_values": [True, False]},
    "num_layers": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 3,
        "min_mean": 2.0,
        "round": True,
        "lower_bound": 2,
    },
    "hidden_dim": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10,
        "min_mean": 3,
        "round": True,
        "lower_bound": 3,
    },
    "init_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 0.5,  # Reduced from 2.0 to 0.5 for better stability with Exp/Square activations
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
    "noise_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 0.3,
        "min_mean": 0.0001,
        "round": False,
        "lower_bound": 0.0,
    },
    "sampling": {"distribution": "meta_choice", "choice_values": ["normal", "mixed", "uniform", "beta"]},
    "pre_sample_cause_stats": {"distribution": "meta_choice", "choice_values": [True, False]},
    "pre_sample_noise_std": {"distribution": "meta_choice", "choice_values": [True, False]},
    # GPSCM
    "length_scale": {
        "distribution": "meta_log_uniform",
        "min": 0.1,
        "max": 3.0,  # Reduced from 10.0 to prevent numerical instability with coords in [0,1]
    },
    "signal_variance": {
        "distribution": "meta_log_uniform",
        "min": 0.1,
        "max": 5.0,
    },
    "noise_variance": {
        "distribution": "meta_log_uniform",
        "min": 0.01,
        "max": 0.5,
    },
    "gp_combination": {
        "distribution": "meta_choice",
        "choice_values": ["linear", "quadratic", "mlp"],
    },
    # Random node selection strategy (disjoint X, y sampling from joint distribution)
    "use_joint_covariance_sampling": {
        "distribution": "meta_choice",
        "choice_values": [True, False]  # Mix of both strategies
    },
    # TreeSCM: Tree type selection
    # Weights per MITRA rationale: ET 0.15, GB 0.10, DT 0.08, RF 0.05, DSRF 0.02
    # (SCM gets 0.60, so tree priors total = 0.40)
    "tree_model": {
        "distribution": "meta_choice",
        "choice_values": ["extra_trees", "xgboost", "decision_tree", "random_forest", "dsrf"],
    },
    # Regression: Target Normalization Method
    "target_norm_method": {
        "distribution": "meta_choice",
        "choice_values": ["zscore", "minmax"],
    },
    # HybridSCM
    "max_parents": {
        "distribution": "meta_choice",
        "choice_values": [1, 2, 3, 4]
    },
    "num_roots": {
        "distribution": "meta_choice",
        "choice_values": [5, 10, 15]
    },
}

