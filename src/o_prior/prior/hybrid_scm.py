from __future__ import annotations

import math
import random
from typing import Any, List, Optional, Tuple, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, SpectralClustering

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .utils import GaussianNoise, XSampler
from .mlp_scm import MLPSCM
from .tree_scm import TreeSCM

# --- Edge Functions ---

class EdgeFunction(nn.Module):
    """Abstract base class for edge functions g(X_parent) -> X_transformed."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class IdentityEdge(EdgeFunction):
    """Linear/Identity transformation."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        if input_dim != output_dim:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class MLPEdge(EdgeFunction):
    """Non-linear transformation using a small MLP."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, num_layers: int = 2, activation: str = ''):
        super().__init__(input_dim, output_dim)
        layers = []
        curr_dim = input_dim

        # Resolve activation
        act_fn = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'sigmoid': nn.Sigmoid,
            'sin': torch.sin
        }

        if activation == '':
            activation = torch.randint(len(act_fn), (1,)).item()
            activation = list(act_fn.keys())[activation]
            act_fn = act_fn[activation]
        else:
            act_fn = act_fn[activation]

        if activation == 'sin':
            act_module = lambda: SinActivation()
        else:
            act_module = act_fn

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(act_module())
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ConvEdge(EdgeFunction):
    """Non-linear transformation using a small Convolutional Layer."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, num_layers: int = 2, activation: str = '', kernel_size: int = 3, stride: int = 1, padding: int = 0):
        super().__init__(input_dim, output_dim)
        layers = []
        curr_dim = input_dim

        # Resolve activation
        act_fn = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'gelu': nn.GELU,
        }

        if activation == '':
            activation = torch.randint(len(act_fn), (1,)).item()
            activation = list(act_fn.keys())[activation]
            act_module = act_fn[activation]
        else:
            act_module = act_fn[activation]

        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(curr_dim, hidden_dim,  kernel_size=kernel_size,  stride=stride, padding=padding))
            layers.append(act_module())
            curr_dim = hidden_dim

        layers.append(nn.Conv1d(curr_dim, output_dim,  kernel_size=kernel_size,  stride=stride, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect (batch, channels) or (batch, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        y = self.net(x)
        # Collapse spatial dimension back to (batch, features)
        if y.dim() == 3:
            if y.shape[2] > 1:
                y = y.mean(dim=2)
            else:
                y = y.squeeze(2)
        return y

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class TreeEdge(EdgeFunction):
    """
    Simulation of a decision tree edge function.
    Since we need differentiability/torch compatibility, we simulate this
    using a hard-coded random decision path or a soft decision tree approximation.
    For simplicity and efficiency here, we use a randomly initialized high-frequency
    step-function-like MLP or a lookup-table approximation to mimic tree discrete nature.
    """
    def __init__(self, input_dim: int, output_dim: int, depth: int = 3):
        super().__init__(input_dim, output_dim)
        # Using a specialized MLP to approximate tree-like boundaries (sharp transitions)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32 * depth),
            nn.Tanh(), # Tanh scaling to approximate step
            nn.Linear(32 * depth, 32 * depth),
            nn.ReLU(),
            nn.Linear(32 * depth, output_dim)
        )
        # Initialize with high gain to create sharp boundaries
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize output to simulate leaf values
        out = self.net(x)
        return torch.round(out * 5) / 5 # Discrete steps

class TraditionalTreeEdge(EdgeFunction):
    """
    Tree-inspired edge that mixes Random Forest / XGBoost / Decision Tree
    models per layer. Uses real sklearn/xgboost models fit on synthetic data.
    """
    def __init__(self, input_dim: int, output_dim: int, tree_depth_lambda: int = 0.5, tree_n_estimators_lambda: int = 0.5):
        super().__init__(input_dim, output_dim)
        model_choices = ["random_forest", "xgboost", "decision_tree"]
        model_type = random.choice(model_choices)

        rng = np.random.RandomState(random.randint(0, 1000000))
        train_size = random.randint(256, 1024)

        max_depth = 2 + int(np.random.exponential(1 / tree_depth_lambda))
        n_estimators = 1 + int(np.random.exponential(1 / tree_n_estimators_lambda))

        self.model = self._build_model(model_type, max_depth, n_estimators)
        X_train, y_train = self._make_synthetic_data(rng, input_dim, output_dim, train_size)
        self.model.fit(X_train, y_train)

    def _build_model(self, model_type: str, depth: int, n_estimators: int):
        if model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=depth,
                random_state=random.randint(0, 1000000),
            )
        if model_type == "decision_tree":
            return DecisionTreeRegressor(
                max_depth=depth,
                random_state=random.randint(0, 1000000),
            )
        
        if model_type == "xgboost":
            return XGBRegressor(
                n_estimators=n_estimators,
                max_depth=depth,
                verbosity=0,
                tree_method="hist",
            )
        
        raise ValueError(f"Invalid model type: {model_type}")

    def _make_synthetic_data(self, rng: np.random.RandomState, in_dim: int, out_dim: int, train_size: int):
        X = rng.normal(size=(train_size, in_dim))
        W = rng.normal(size=(in_dim, out_dim))
        y = np.tanh(X @ W) + 0.1 * rng.normal(size=(train_size, out_dim))
        if out_dim == 1:
            y = y.ravel()
        return X, y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Defensive cleanup: sklearn/xgboost can't handle NaN/Inf or huge values
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp(min=-1e3, max=1e3)
        x_np = x.detach().cpu().numpy()
        y_np = self.model.predict(x_np)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        return torch.tensor(y_np, device=x.device, dtype=x.dtype)

class PolynomialEdge(EdgeFunction):
    """Higher-order polynomial transformation: g(x) = sum(w_k * x^k)."""
    def __init__(self, input_dim: int, output_dim: int, degree: int = 2):
        super().__init__(input_dim, output_dim)
        self.degree = degree
        self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(degree)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = 0
        for i, lin in enumerate(self.linears):
            res = res + lin(torch.pow(x, i + 1))
        return res

class CrossInteractionEdge(EdgeFunction):
    """Captures interaction between parents: g(x1, x2) = w * (x1 * x2).
    In this design, we treat this as a multi-input edge.
    """
    def __init__(self, input_dim: int, output_dim: int):
        # input_dim here should be the sum of parent dims or similar
        super().__init__(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, parents: List[torch.Tensor]) -> torch.Tensor:
        # This is a bit special as it takes multiple parents
        # For simplicity in the generic LCS loop, we'll handle this in Aggregation
        # or as a special Edge that expects concatenated/stacked input
        if len(parents) < 2:
            return self.linear(parents[0])

        # Simple interaction: x1 * x2
        interaction = parents[0]
        for i in range(1, len(parents)):
            interaction = interaction * parents[i]
        return self.linear(interaction)

# --- Aggregation Functions ---

class AggregationFunction(nn.Module):
    """Combines transformed parent values."""
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

class MeanAggregation(AggregationFunction):
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if not inputs:
            return torch.zeros_like(inputs[0]) # Should not happen if parents exist
        return torch.stack(inputs).mean(dim=0)

class WeightedAggregation(AggregationFunction):
    def __init__(self, num_parents: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_parents))

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        w = F.softmax(self.weights, dim=0)
        return sum(w[i] * inputs[i] for i in range(len(inputs)))

class MLPAggregation(AggregationFunction):
    def __init__(self, input_total_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_total_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        cat_inputs = torch.cat(inputs, dim=-1)
        return self.net(cat_inputs)

class ProductAggregation(AggregationFunction):
    """Multiplicative aggregation: f(x) = prod(g_i(x_i))."""
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if not inputs:
            return torch.zeros(1)
        res = inputs[0]
        for i in range(1, len(inputs)):
            res = res * inputs[i]
        return res

class MaxAggregation(AggregationFunction):
    """Non-smooth aggregation: f(x) = max(g_i(x_i))."""
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if not inputs:
            return torch.zeros(1)
        return torch.stack(inputs).max(dim=0).values

class ConcatAggregation(AggregationFunction):
    """Concatenate inputs along the feature dimension without reduction."""
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if not inputs:
            return torch.zeros(1)
        
        t = torch.cat(inputs, dim=-1)
        return t

# --- Local Causal Structure ---

class LCS(nn.Module):
    """Represents a single node generation process."""
    def __init__(self,
                 node_idx: int,
                 parent_indices: List[int],
                 parent_dims: List[int],
                 output_dim: int,
                 edge_types: List[str],
                 agg_type: str,
                 noise_std: float = 0.01):
        super().__init__()
        self.node_idx = node_idx
        self.parent_indices = parent_indices
        self.output_dim = output_dim
        self.noise_std = noise_std
        self.agg_type = agg_type
        self.edges = nn.ModuleList()

        if 'tree_traditional' in edge_types:
            self.edges.append(TraditionalTreeEdge(len(parent_indices), output_dim))
        
        else:
            for p_dim, e_type in zip(parent_dims, edge_types):
                if e_type == 'mlp':
                    nb_layers = random.randint(2, 8)
                    hidden_dim = random.choice([16, 32, 64])
                    self.edges.append(MLPEdge(p_dim, output_dim, hidden_dim=hidden_dim, num_layers=nb_layers))
                elif e_type == 'tree':
                    depth = random.randint(3, 5)
                    self.edges.append(TreeEdge(p_dim, output_dim, depth=depth))
                #elif e_type == 'tree_traditional':
                #    self.edges.append(TraditionalTreeEdge(p_dim, output_dim))
                elif e_type == 'conv':
                    nb_layers = random.randint(2, 8)
                    hidden_dim = random.choice([16, 32, 64])
                    # Inputs are shaped (batch, channels, length=1), so kernel must be 1
                    kernel_size = 1
                    stride = 1
                    padding = 0
                    self.edges.append(ConvEdge(p_dim, output_dim, hidden_dim=hidden_dim, num_layers=nb_layers, kernel_size=kernel_size, stride=stride, padding=padding))
                elif e_type == 'polynomial':
                    degree = random.randint(2, 5)
                    self.edges.append(PolynomialEdge(p_dim, output_dim, degree=degree))
                else:
                    self.edges.append(IdentityEdge(p_dim, output_dim))

        if agg_type == 'weighted':
            self.agg = WeightedAggregation(len(parent_indices))
        elif agg_type == 'mlp':
            # MLP aggregation takes concatenated output of edges (which all map to output_dim)
            self.agg = MLPAggregation(len(parent_indices) * output_dim, output_dim)
        elif agg_type == 'product':
            self.agg = ProductAggregation()
        elif agg_type == 'max':
            self.agg = MaxAggregation()
        elif agg_type == 'concat':
            self.agg = ConcatAggregation()
        else:
            self.agg = MeanAggregation()

    def forward(self, all_nodes: Dict[int, torch.Tensor]) -> torch.Tensor:
        parent_vals = [all_nodes[i] for i in self.parent_indices]

        if self.agg_type == 'concat':
            agg_val = self.agg(parent_vals)
            agg_val = self.edges[0](agg_val)
        else:
            transformed = [edge(val) for edge, val in zip(self.edges, parent_vals)]
            agg_val = self.agg(transformed)

        # Add noise
        noise = torch.randn_like(agg_val) * self.noise_std
        return agg_val + noise

# --- Hybrid SCM ---

class HybridSCM(nn.Module):
    """
    Generates datasets using Hierarchical LCS with mixed edge functions.
    """
    def __init__(self,
                 seq_len: int = 1024,
                 num_features: int = 100, # This serves as 'num_nodes' in the DAG
                 num_outputs: int = 1, # Target nodes
                 num_roots: int = 10,
                 max_parents: int = 4,
                 noise_std: float = 0.01,
                 device: str = "cpu",
                 use_advanced_components: bool = False,
                 sampling_strategy: str = "kmeans", # "random" or "graph_aware"
                 min_nodes_multiplier: int = 2,
                 max_nodes_multiplier: int = 6,
                 **kwargs):
        
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_roots = num_roots
        self.max_parents = max_parents
        self.noise_std = noise_std
        self.device = device
        self.use_advanced_components = use_advanced_components
        self.sampling_strategy = sampling_strategy
        self.min_nodes_multiplier = min_nodes_multiplier
        self.max_nodes_multiplier = max_nodes_multiplier

        # Register dummy buffer for device tracking
        self.register_buffer("_dev", torch.empty(0, device=device))

        # Root sampler
        self.root_sampler = XSampler(seq_len, num_roots, device=device)

        # Build DAG structure
        self.lcs_modules = nn.ModuleList()
        self.node_dims = {i: 1 for i in range(num_roots)} # Roots have dim 1

        # Define remaining nodes (features + output)
        # We generate more nodes than needed and select a subset, or just generate exactly num_features + num_outputs
        min_mult = max(2, self.min_nodes_multiplier)
        max_mult = max(min_mult, self.max_nodes_multiplier)
        self.node_multiplier = random.randint(min_mult, max_mult)
        total_generated_nodes = self.node_multiplier * num_features + num_outputs

        # Current available nodes to be parents (starts with roots)
        available_indices = list(range(num_roots))

        for i in range(total_generated_nodes):
            # Node ID for this new node
            node_idx = num_roots + i
            # Select parents
            num_parents = random.randint(1, min(len(available_indices), max_parents))
            parent_indices = random.sample(available_indices, num_parents)
            parent_dims = [self.node_dims[p] for p in parent_indices]

            # Sample edge types
            # Probabilities could be configurable, hardcoded for now based on newfile.md
            if self.use_advanced_components:
                e_choices = ['mlp', 'tree', 'tree_traditional', 'conv', 'identity', 'polynomial']
                if num_parents == 1:
                    e_weights = [0.25, 0.2, 0, 0.25, 0.1, 0.2]
                else:
                    e_weights = [0.2, 0.15, 0.2, 0.2, 0.1, 0.15]
            else:
                e_choices = ['mlp', 'tree', 'conv', 'identity']
                e_weights = [0.4, 0.2, 0.2, 0.2]
            edge_types = random.choices(e_choices, weights=e_weights, k=num_parents)

            # Sample aggregation type
            if self.use_advanced_components:
                a_choices = ['mean', 'weighted', 'mlp', 'product', 'max']
                a_weights = [0.25, 0.2, 0.2, 0.2, 0.15]
            else:
                a_choices = ['mean', 'weighted', 'mlp']
                a_weights = [0.4, 0.3, 0.3]
            agg_type = random.choices(a_choices, weights=a_weights, k=1)[0]

            if num_parents == 1 and 'tree_traditional' in edge_types:
                edge_types = ['tree']

            # Create LCS
            current_dim = 1 # We keep distinct feature dimension as 1 for simplicity in tabular

            if 'tree_traditional' in edge_types:
                edge_types = ['tree_traditional'] * len(edge_types)
                agg_type = 'concat'

            lcs = LCS(node_idx, parent_indices, parent_dims, current_dim, edge_types, agg_type, noise_std)
            self.lcs_modules.append(lcs)

            # Update state
            self.node_dims[node_idx] = current_dim
            available_indices.append(node_idx)

        #print('nb nodes', len(self.lcs_modules))
        self.to(device)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate roots
        roots = self.root_sampler.sample() # (B, num_roots)

        # Store all node values
        nodes = {}
        for i in range(self.num_roots):
            nodes[i] = roots[:, i:i+1]

        # Forward pass through LCS (topo order is guaranteed by construction loop)
        for lcs in self.lcs_modules:
            val = lcs(nodes)
            nodes[lcs.node_idx] = val

        # Collect indices based on sampling strategy
        if self.sampling_strategy == "graph_aware":
            selected_feature_indices, target_indices = self.get_graph_aware_sample(nodes)
        elif self.sampling_strategy == "kmeans":
            selected_feature_indices, target_indices = self.get_kmeans_sample(nodes)
        elif self.sampling_strategy == "farthest":
            selected_feature_indices, target_indices = self.get_farthest_point_sample(nodes)
        elif self.sampling_strategy == "entropy":
            selected_feature_indices, target_indices = self.get_entropy_sample(nodes)
        elif self.sampling_strategy in {"community", "community_leiden", "community_louvain"}:
            selected_feature_indices, target_indices = self.get_community_sample(nodes)
        else:
            # Default: Random/Traditional logic
            all_generated_indices = list(range(self.num_roots, self.num_roots + self.num_features + self.num_outputs))
            target_indices = all_generated_indices[-self.num_outputs:]
            feature_pool_indices = all_generated_indices[:-self.num_outputs]
            feature_pool_indices = list(range(self.num_roots)) + feature_pool_indices

            if len(feature_pool_indices) < self.num_features:
                 selected_feature_indices = feature_pool_indices
            else:
                 selected_feature_indices = sorted(random.sample(feature_pool_indices, self.num_features))

        X_list = [nodes[i] for i in selected_feature_indices]
        y_list = [nodes[i] for i in target_indices]

        X = torch.cat(X_list, dim=1) # (B, num_features)
        y = torch.cat(y_list, dim=1) # (B, num_outputs)

        if self.num_outputs == 1:
            y = y.squeeze(1)

        return X, y

    def get_graph_aware_sample(self, nodes: Dict[int, torch.Tensor]) -> Tuple[List[int], List[int]]:
        """
        Implements graph-aware sampling:
        1. Select target nodes from terminal/intermediate nodes.
        2. Find ancestors of targets.
        3. Sample features from ancestors (informative) and non-ancestors (distractors).
        """
        all_node_indices = list(range(self.num_roots + self.node_multiplier * self.num_features + self.num_outputs))

        # 1. Target selection: prefer nodes added later (more ancestors)
        # We'll take the last num_outputs nodes as targets for topo consistency
        target_indices = all_node_indices[-self.num_outputs:]

        # 2. Find ancestors
        ancestors = set()
        to_visit = list(target_indices)

        # Create a mapping for quick lookup of parents
        parent_map = {lcs.node_idx: lcs.parent_indices for lcs in self.lcs_modules}

        while to_visit:
            curr = to_visit.pop()
            if curr in parent_map:
                for p in parent_map[curr]:
                    if p not in ancestors:
                        ancestors.add(p)
                        to_visit.append(p)

        # Also include roots in ancestors if they are parents
        ancestors.update([i for i in range(self.num_roots) if any(i in p_list for p_list in parent_map.values())])

        # 3. Feature sampling
        # We need num_features total.
        # Strategy: Sample 80% from ancestors, 20% from random rest
        ancestor_list = sorted(list(ancestors))
        non_ancestors = sorted(list(set(all_node_indices) - ancestors - set(target_indices)))

        num_ancestor_features = int(self.num_features * 0.8)
        num_distractor_features = self.num_features - num_ancestor_features

        if len(ancestor_list) < num_ancestor_features:
            selected_informative = ancestor_list
            num_distractor_features = self.num_features - len(selected_informative)
        else:
            selected_informative = random.sample(ancestor_list, num_ancestor_features)

        if len(non_ancestors) < num_distractor_features:
            selected_distractors = non_ancestors
        else:
            selected_distractors = random.sample(non_ancestors, num_distractor_features)

        selected_features = sorted(selected_informative + selected_distractors)

        # Final safety check: if we still don't have enough features, fill from remaining
        if len(selected_features) < self.num_features:
            remaining = sorted(list(set(all_node_indices) - set(target_indices) - set(selected_features)))
            needed = self.num_features - len(selected_features)
            selected_features += random.sample(remaining, min(needed, len(remaining)))

        return sorted(selected_features[:self.num_features]), target_indices

    def get_kmeans_sample(self, nodes: Dict[int, torch.Tensor]) -> Tuple[List[int], List[int]]:
        """
        K-means sampling:
        1. Keep target_indices.
        2. Cluster remaining nodes into num_features/2 clusters.
        3. Select two nodes from each cluster.
        4. Safety fill if not enough features.
        """
        all_node_indices = list(range(self.num_roots + self.node_multiplier * self.num_features + self.num_outputs))
        target_indices = all_node_indices[-self.num_outputs:]
        remaining = all_node_indices[:-self.num_outputs]

        # Cluster into num_features/2 groups
        n_clusters = max(2, self.num_features // 2)

        if len(remaining) < n_clusters:
            selected_features = remaining
        else:
            # Build per-node embeddings for clustering (mean over time/feature dims)
            embeddings = []
            for idx in remaining:
                v = nodes[idx]
                if v.dim() > 1:
                    v = v.mean(dim=-1)
                embeddings.append(v.flatten())
            X = torch.stack(embeddings, dim=0).detach().cpu().numpy()
        
            kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=None)
            labels = kmeans.fit_predict(X)

            selected_features = []
            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                if not cluster_indices:
                    continue

                # Pick two random nodes from the cluster
                if len(cluster_indices) <= 2:
                    picked = cluster_indices
                else:
                    picked = random.sample(cluster_indices, 2)

                selected_features.extend([remaining[i] for i in picked])

        # Final safety: fill from remaining if not enough
        if len(selected_features) < self.num_features:
            remaining_pool = [i for i in remaining if i not in selected_features]
            needed = self.num_features - len(selected_features)
            if remaining_pool:
                extra = random.sample(remaining_pool, min(needed, len(remaining_pool)))
                selected_features.extend(extra)

        return sorted(selected_features[:self.num_features]), target_indices

    def get_farthest_point_sample(self, nodes: Dict[int, torch.Tensor]) -> Tuple[List[int], List[int]]:
        """
        Farthest-point sampling:
        1. Keep target_indices.
        2. Pick one random node from remaining.
        3. Iteratively pick the node farthest from the last chosen node.
        4. Safety fill if not enough features.
        """
        all_node_indices = list(range(self.num_roots + self.node_multiplier * self.num_features + self.num_outputs))
        target_indices = all_node_indices[-self.num_outputs:]
        remaining = all_node_indices[:-self.num_outputs]

        # Build embeddings
        embeddings = []
        for idx in remaining:
            v = nodes[idx]
            if v.dim() > 1:
                v = v.mean(dim=-1)
            embeddings.append(v.flatten())
            
        X = torch.stack(embeddings, dim=0).detach().cpu().numpy()

        # Start from a random node
        selected_features = []
        current_idx = random.randrange(len(remaining))
        selected_features.append(remaining[current_idx])

        # Iteratively select farthest from the centroid of chosen points
        chosen_mask = np.zeros(len(remaining), dtype=bool)
        chosen_mask[current_idx] = True

        while len(selected_features) < self.num_features:
            centroid = X[chosen_mask].mean(axis=0)
            distances = np.linalg.norm(X - centroid, axis=1)
            distances[chosen_mask] = -1.0
            next_idx = int(np.argmax(distances))
            selected_features.append(remaining[next_idx])
            chosen_mask[next_idx] = True

        # Final safety: fill from remaining if not enough
        if len(selected_features) < self.num_features:
            remaining_pool = [i for i in remaining if i not in selected_features]
            needed = self.num_features - len(selected_features)
            if remaining_pool:
                extra = random.sample(remaining_pool, min(needed, len(remaining_pool)))
                selected_features.extend(extra)

        return sorted(selected_features[:self.num_features]), target_indices

    def get_entropy_sample(self, nodes: Dict[int, torch.Tensor]) -> Tuple[List[int], List[int]]:
        """
        Entropy-based sampling:
        1. Keep target_indices.
        2. Compute entropy for remaining nodes.
        3. Select top num_features by entropy.
        4. Safety fill if not enough features.
        """
        all_node_indices = list(range(self.num_roots + self.node_multiplier * self.num_features + self.num_outputs))
        target_indices = all_node_indices[-self.num_outputs:]
        remaining = all_node_indices[:-self.num_outputs]

        if not remaining:
            return [], target_indices

        entropies = []
        k = 5
        for idx in remaining:
            v = nodes[idx]
            if v.dim() > 1:
                v = v.mean(dim=-1)
            #v = v.flatten()
            v_np = v.detach().cpu().numpy().reshape(-1, 1)

            if v_np.shape[0] <= k:
                entropy = 0.0
            else:
                nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
                nn.fit(v_np)
                distances, _ = nn.kneighbors(v_np, return_distance=True)
                # Distance to k-th neighbor (skip self at index 0)
                kth = distances[:, k]
                entropy = float(np.mean(np.log(kth + 1e-12)))
                
            entropies.append(entropy)

        # Select top entropy nodes
        sorted_idx = np.argsort(entropies)[::-1]
        selected_features = [remaining[i] for i in sorted_idx[: self.num_features]]

        # Final safety: fill from remaining if not enough
        if len(selected_features) < self.num_features:
            remaining_pool = [i for i in remaining if i not in selected_features]
            needed = self.num_features - len(selected_features)
            if remaining_pool:
                extra = random.sample(remaining_pool, min(needed, len(remaining_pool)))
                selected_features.extend(extra)

        return sorted(selected_features[:self.num_features]), target_indices

    def get_community_sample(self, nodes: Dict[int, torch.Tensor]) -> Tuple[List[int], List[int]]:
        """
        Community-based sampling using the built graph:
        1. Keep target_indices.
        2. Find undirected connected components as communities.
        3. Sample up to two nodes per community.
        4. Safety fill if not enough features.
        """
        all_node_indices = list(range(self.num_roots + self.node_multiplier * self.num_features + self.num_outputs))
        target_indices = all_node_indices[-self.num_outputs:]
        remaining = all_node_indices[:-self.num_outputs]

        # Build undirected adjacency matrix from LCS parent map
        parent_map = {lcs.node_idx: lcs.parent_indices for lcs in self.lcs_modules}
        idx_map = {node: i for i, node in enumerate(remaining)}
        n = len(remaining)
        adj = np.zeros((n, n), dtype=float)
        for child, parents in parent_map.items():
            if child not in idx_map:
                continue
            for p in parents:
                if p not in idx_map:
                    continue
                i, j = idx_map[child], idx_map[p]
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        selected_features = []
        n_clusters = max(2, self.num_features // 2)
        n_clusters = min(n_clusters, n)

        if adj.sum() == 0 or n_clusters == 0:
            #No edge founde, use random sampling
            selected_features = random.sample(remaining, min(self.num_features, len(remaining)))
        else:
            labels = None

            if self.sampling_strategy in {"community", "community_leiden"}:
                import igraph as ig
                import leidenalg

                edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] > 0]
                g = ig.Graph(n=n, edges=edges, directed=False)
                part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
                labels = np.array(part.membership, dtype=int)

            elif self.sampling_strategy in {"community", "community_louvain"}:
                import networkx as nx
                import community as community_louvain  # python-louvain

                G = nx.Graph()
                G.add_nodes_from(range(n))
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j] > 0:
                            G.add_edge(i, j)
                partition = community_louvain.best_partition(G)
                labels = np.array([partition[i] for i in range(n)], dtype=int)

            if labels is None:
                print('No community labels found, using random sampling\n\n')

            for cluster_id in range(n_clusters):
                cluster_nodes = [remaining[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
                if not cluster_nodes:
                    continue
                if len(cluster_nodes) <= 2:
                    picked = cluster_nodes
                else:
                    picked = random.sample(cluster_nodes, 2)
                
                selected_features.extend(picked)
                if len(selected_features) >= self.num_features:
                    break

        # Final safety: fill from remaining if not enough
        if len(selected_features) < self.num_features:
            remaining_pool = [i for i in remaining if i not in selected_features]
            needed = self.num_features - len(selected_features)
            if remaining_pool:
                extra = random.sample(remaining_pool, min(needed, len(remaining_pool)))
                selected_features.extend(extra)

        return sorted(selected_features[:self.num_features]), target_indices
