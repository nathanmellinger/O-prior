from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .genload import LoadPriorDataset


@dataclass(frozen=True)
class _TaskSpec:
    task_id: int
    data_dir: str


class MultiTaskEpisodeIterableDataset(IterableDataset):
    """Mix datasets from multiple task prior folders into one episodic stream.

    Each yielded batch is composed of samples drawn across task folders according
    to a sampling strategy, then re-packed as an episode with a support/query split.
    """

    def __init__(
        self,
        task_prior_dirs: Sequence[str],
        batch_size: int = 512,
        support_size_candidates: Sequence[int] = (8, 16, 32, 64, 128),
        support_sampling_strategy: str = "random",
        task_sampling_strategy: str = "balanced",
        task_sampling_weights: Optional[Sequence[float]] = None,
        rng_seed: int = 42,
        load_batch_size: int = 512,
        load_start: int = 0,
        load_timeout: int = 2,
        loop: bool = False,
        ddp_world_size: int = 1,
        ddp_rank: int = 0,
        source_label: str = "prior",
        return_task_ids: bool = False,
    ) -> None:
        super().__init__()
        if not task_prior_dirs:
            raise ValueError("task_prior_dirs must contain at least one directory.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0.")

        self.batch_size = int(batch_size)
        self.support_size_candidates = [int(x) for x in support_size_candidates if int(x) > 0]
        if not self.support_size_candidates:
            raise ValueError("support_size_candidates must contain at least one positive integer.")

        self.support_sampling_strategy = str(support_sampling_strategy).strip().lower()
        if self.support_sampling_strategy not in {"random", "sequential"}:
            raise ValueError("support_sampling_strategy must be one of {'random', 'sequential'}.")

        self.task_sampling_strategy = str(task_sampling_strategy).strip().lower()
        if self.task_sampling_strategy not in {"balanced", "weighted"}:
            raise ValueError("task_sampling_strategy must be one of {'balanced', 'weighted'}.")

        self.return_task_ids = bool(return_task_ids)
        self.rng = np.random.default_rng(int(rng_seed) + int(ddp_rank))
        self.source_label = str(source_label)

        self.task_specs = self._parse_task_specs(task_prior_dirs)
        self.task_ids = np.array([s.task_id for s in self.task_specs], dtype=np.int64)
        self.task_probs = self._build_task_probs(self.task_specs, self.task_sampling_strategy, task_sampling_weights)

        self._loaders: Dict[int, LoadPriorDataset] = {}
        self._iters: Dict[int, Iterable] = {}
        self._buffers: Dict[int, Dict[str, object]] = {}
        self._seq_offsets: Dict[int, int] = {}

        for spec in self.task_specs:
            loader = LoadPriorDataset(
                data_dir=spec.data_dir,
                batch_size=int(load_batch_size),
                ddp_world_size=int(ddp_world_size),
                ddp_rank=int(ddp_rank),
                start_from=int(load_start),
                timeout=int(load_timeout),
                loop=bool(loop),
                delete_after_load=False,
                device="cpu",
                return_metadata=False,
                save_format="auto",
            )
            self._loaders[spec.task_id] = loader
            self._iters[spec.task_id] = iter(loader)
            self._buffers[spec.task_id] = {}
            self._seq_offsets[spec.task_id] = 0

    @staticmethod
    def _parse_task_specs(task_prior_dirs: Sequence[str]) -> List[_TaskSpec]:
        specs: List[_TaskSpec] = []
        for raw in task_prior_dirs:
            item = str(raw).strip()
            if not item:
                continue
            if "=" in item:
                task_raw, path_raw = item.split("=", 1)
                task_id = int(task_raw.strip())
                data_dir = path_raw.strip()
            else:
                data_dir = item
                task_id = int(os.path.basename(os.path.normpath(data_dir)))
            if not os.path.isdir(data_dir):
                raise FileNotFoundError(f"Task prior dir does not exist: {data_dir}")
            specs.append(_TaskSpec(task_id=task_id, data_dir=data_dir))

        if not specs:
            raise ValueError("No valid task_prior_dirs provided.")
        return specs

    @staticmethod
    def _build_task_probs(
        specs: Sequence[_TaskSpec],
        strategy: str,
        weights: Optional[Sequence[float]],
    ) -> np.ndarray:
        n = len(specs)
        if strategy == "balanced":
            return np.full((n,), 1.0 / n, dtype=np.float64)

        if not weights:
            raise ValueError("task_sampling_weights is required when task_sampling_strategy='weighted'.")
        vals = np.array([float(w) for w in weights], dtype=np.float64)
        if vals.shape[0] != n:
            raise ValueError(f"Expected {n} task_sampling_weights, got {vals.shape[0]}.")
        if np.any(vals < 0):
            raise ValueError("task_sampling_weights must be non-negative.")
        s = float(vals.sum())
        if s <= 0:
            raise ValueError("Sum of task_sampling_weights must be > 0.")
        return vals / s

    def __iter__(self):
        return self

    def _refill(self, task_id: int) -> None:
        batch = next(self._iters[task_id])
        if len(batch) < 5:
            raise RuntimeError(f"Task {task_id} loader returned malformed batch.")

        X, y, d, seq_lens, train_sizes = batch[:5]
        file_batch_size = int(d.shape[0])
        order = self.rng.permutation(file_batch_size)
        self._buffers[task_id] = {
            "X": X,
            "y": y,
            "d": d,
            "seq_lens": seq_lens,
            "train_sizes": train_sizes,
            "order": order,
            "pos": 0,
        }

    @staticmethod
    def _slice_row(t: torch.Tensor, idx: int):
        if isinstance(t, torch.Tensor) and t.is_nested:
            return t[idx]
        return t[idx]

    def _sample_support_size(self, train_size: int, seq_len: int) -> int:
        max_support = max(1, min(int(train_size), int(seq_len) - 1))
        valid = [k for k in self.support_size_candidates if k <= max_support]
        if not valid:
            return max_support
        return int(self.rng.choice(valid))

    def _episode_reorder(
        self,
        task_id: int,
        X: torch.Tensor,
        y: torch.Tensor,
        train_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq_len = int(X.shape[0])
        if seq_len <= 1:
            return X, y, 1

        train_size = max(1, min(int(train_size), seq_len - 1))
        support_size = self._sample_support_size(train_size=train_size, seq_len=seq_len)

        support_idx = np.arange(train_size, dtype=np.int64)
        if self.support_sampling_strategy == "random":
            picked_np = self.rng.choice(support_idx, size=support_size, replace=False)
            picked_np.sort()
        else:
            start = self._seq_offsets[task_id] % train_size
            picked_np = (start + np.arange(support_size, dtype=np.int64)) % train_size
            self._seq_offsets[task_id] = (start + support_size) % max(1, train_size)

        picked = torch.from_numpy(picked_np).long()
        all_support = torch.arange(train_size, dtype=torch.long)
        keep_mask = torch.ones(train_size, dtype=torch.bool)
        keep_mask[picked] = False
        remaining_support = all_support[keep_mask]
        query_idx = torch.arange(train_size, seq_len, dtype=torch.long)
        order = torch.cat([picked, query_idx, remaining_support], dim=0)

        X_new = X.index_select(0, order.to(X.device))
        y_new = y.index_select(0, order.to(y.device))
        return X_new, y_new, support_size

    def _pop_sample(self, task_id: int):
        buf = self._buffers[task_id]
        if not buf or int(buf["pos"]) >= int(len(buf["order"])):
            self._refill(task_id)
            buf = self._buffers[task_id]

        row = int(buf["order"][int(buf["pos"])])
        buf["pos"] = int(buf["pos"]) + 1

        X_all = buf["X"]
        y_all = buf["y"]
        d_all = buf["d"]
        seq_all = buf["seq_lens"]
        train_all = buf["train_sizes"]

        d_i = int(d_all[row].item())
        seq_i = int(seq_all[row].item())
        train_i = int(train_all[row].item())

        x_i = self._slice_row(X_all, row)
        y_i = self._slice_row(y_all, row)
        x_i = x_i[:seq_i, :d_i].to(torch.float32)
        y_i = y_i[:seq_i]

        x_i, y_i, train_i = self._episode_reorder(task_id, x_i, y_i, train_i)
        seq_i = int(x_i.shape[0])
        return x_i, y_i, d_i, seq_i, int(train_i)

    def __next__(self):
        task_positions = self.rng.choice(len(self.task_specs), size=self.batch_size, p=self.task_probs)

        X_rows: List[torch.Tensor] = []
        y_rows: List[torch.Tensor] = []
        d_rows: List[int] = []
        seq_rows: List[int] = []
        train_rows: List[int] = []
        task_rows: List[int] = []

        for pos in task_positions:
            task_id = int(self.task_ids[int(pos)])
            x_i, y_i, d_i, seq_i, train_i = self._pop_sample(task_id)
            X_rows.append(x_i.cpu())
            y_rows.append(y_i.cpu())
            d_rows.append(int(d_i))
            seq_rows.append(int(seq_i))
            train_rows.append(int(train_i))
            task_rows.append(int(task_id))

        y_is_float = any(torch.is_floating_point(yi) for yi in y_rows)
        y_dtype = torch.float32 if y_is_float else torch.long
        X_out = torch.nested.nested_tensor([x.to(torch.float32) for x in X_rows], dtype=torch.float32, device="cpu")
        y_out = torch.nested.nested_tensor([y.to(y_dtype) for y in y_rows], dtype=y_dtype, device="cpu")
        d_out = torch.tensor(d_rows, dtype=torch.uint16, device="cpu")
        seq_out = torch.tensor(seq_rows, dtype=torch.int32, device="cpu")
        train_out = torch.tensor(train_rows, dtype=torch.int32, device="cpu")

        if self.return_task_ids:
            task_out = torch.tensor(task_rows, dtype=torch.long, device="cpu")
            return X_out, y_out, d_out, seq_out, train_out, task_out
        return X_out, y_out, d_out, seq_out, train_out

