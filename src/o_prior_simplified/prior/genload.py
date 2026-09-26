from __future__ import annotations

import copy
import time
import json
import warnings
import argparse
import logging
import sys
import threading
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from torch.utils.data import IterableDataset
try:
    import h5py  # Optional dependency used only when save_format=h5
except ImportError:
    h5py = None

from .dataset import PriorDataset
from .prior_config import (
    DEFAULT_FIXED_HP,
    DEFAULT_SAMPLED_HP,
)

# Set up logger for batch generation
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

_MISSING_MASK_CODEC = "bitpack_bool_v1"


def _pack_bool_tensor(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack a bool tensor into uint8 bits (8x smaller than dense bool storage)."""
    flat = mask.reshape(-1).to(torch.uint8).cpu().numpy()
    packed_np = np.packbits(flat, bitorder="little")
    packed = torch.from_numpy(packed_np.astype(np.uint8, copy=False))
    return packed, int(flat.size)


def _unpack_bool_tensor(
    packed: torch.Tensor,
    numel: int,
    shape: tuple[int, ...],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Unpack uint8-packed bits back to a bool tensor with the provided shape."""
    packed_np = packed.detach().cpu().numpy().astype(np.uint8, copy=False)
    unpacked_np = np.unpackbits(packed_np, bitorder="little", count=int(numel)).astype(np.bool_, copy=False)
    return torch.from_numpy(unpacked_np.reshape(shape)).to(device=device)


def _encode_missing_mask_bitpack(missing_mask: torch.Tensor) -> dict:
    """Encode dense/nested bool missing_mask into a compact bitpacked payload."""
    if not isinstance(missing_mask, torch.Tensor):
        raise TypeError(f"missing_mask must be a torch.Tensor, got {type(missing_mask)}")

    if missing_mask.is_nested:
        packed_list = []
        shapes = []
        numels = []
        # Use unbind() to avoid calling size/len on NestedTensor (unsupported).
        for mask_i in missing_mask.unbind():
            mask_i = mask_i.to(torch.bool)
            packed_i, numel_i = _pack_bool_tensor(mask_i)
            packed_list.append(packed_i)
            shapes.append(list(mask_i.shape))
            numels.append(numel_i)
        return {
            "_codec": _MISSING_MASK_CODEC,
            "nested": True,
            "packed": packed_list,
            "shapes": shapes,
            "numels": numels,
        }

    packed, numel = _pack_bool_tensor(missing_mask.to(torch.bool))
    return {
        "_codec": _MISSING_MASK_CODEC,
        "nested": False,
        "packed": packed,
        "shape": list(missing_mask.shape),
        "numel": numel,
    }


def _decode_missing_mask_bitpack(mask_obj, device: str | torch.device = "cpu"):
    """Decode bitpacked missing_mask payload; pass through legacy dense masks."""
    if not isinstance(mask_obj, dict) or mask_obj.get("_codec") != _MISSING_MASK_CODEC:
        return mask_obj

    is_nested = bool(mask_obj.get("nested", False))
    if is_nested:
        packed_list = mask_obj.get("packed", [])
        shapes = mask_obj.get("shapes", [])
        numels = mask_obj.get("numels", [])
        decoded = [
            _unpack_bool_tensor(packed, numel, tuple(shape), device=device)
            for packed, shape, numel in zip(packed_list, shapes, numels)
        ]
        return torch.nested.nested_tensor(decoded, device=device)

    return _unpack_bool_tensor(
        mask_obj["packed"],
        int(mask_obj["numel"]),
        tuple(mask_obj["shape"]),
        device=device,
    )


def _compact_feature_meta_dtypes(feature_meta: dict) -> dict:
    """Downcast feature metadata integer tensors to compact dtypes."""
    compact = dict(feature_meta)
    if isinstance(compact.get("feature_type_ids"), torch.Tensor):
        compact["feature_type_ids"] = compact["feature_type_ids"].to(torch.uint8)
    if isinstance(compact.get("imputation_strategy_ids"), torch.Tensor):
        compact["imputation_strategy_ids"] = compact["imputation_strategy_ids"].to(torch.uint8)
    if isinstance(compact.get("col_ids"), torch.Tensor):
        compact["col_ids"] = compact["col_ids"].to(torch.uint16)
    return compact


def _decode_and_compact_feature_meta(feature_meta: Optional[dict], device: str | torch.device = "cpu"):
    """Decode compressed metadata from disk and enforce compact integer dtypes."""
    if feature_meta is None or not isinstance(feature_meta, dict):
        return feature_meta

    decoded = dict(feature_meta)
    if "missing_mask" in decoded:
        decoded["missing_mask"] = _decode_missing_mask_bitpack(decoded["missing_mask"], device=device)

    return _compact_feature_meta_dtypes(decoded)


def _ensure_h5py_available() -> None:
    """Raise a clear error if h5 support is requested without h5py installed."""
    if h5py is None:
        raise ImportError("h5py is required for save_format='h5'. Install it with: pip install h5py")


def _h5_write_tensor(group, name: str, tensor: torch.Tensor) -> None:
    """Write a tensor to HDF5 with lightweight gzip compression."""
    arr = tensor.detach().cpu().contiguous().numpy()
    group.create_dataset(name, data=arr, compression="gzip", compression_opts=4)


def _h5_write_nested(group, name: str, nested_tensor: torch.Tensor) -> None:
    """Write a nested tensor as an indexed HDF5 group."""
    sub = group.create_group(name)
    sub.attrs["nested"] = True
    tensors = list(nested_tensor.unbind())
    sub.attrs["count"] = len(tensors)
    for i, t in enumerate(tensors):
        _h5_write_tensor(sub, f"{i:06d}", t)


def _h5_read_tensor(dataset, device: str | torch.device = "cpu") -> torch.Tensor:
    """Read an HDF5 dataset into a torch tensor on target device."""
    return torch.from_numpy(dataset[()]).to(device=device)


def _h5_read_nested(group, device: str | torch.device = "cpu") -> torch.Tensor:
    """Read an indexed nested tensor group from HDF5."""
    count = int(group.attrs.get("count", len(group.keys())))
    tensors = []
    for i in range(count):
        key = f"{i:06d}"
        if key not in group:
            key = str(i)
        tensors.append(_h5_read_tensor(group[key], device=device))
    return torch.nested.nested_tensor(tensors, device=device)


def _h5_write_feature_meta(parent_group, feature_meta: dict) -> None:
    """Write feature_meta payload (including bitpacked missing mask) to HDF5."""
    meta = parent_group.create_group("feature_meta")
    for key, val in feature_meta.items():
        if key == "missing_mask" and isinstance(val, dict):
            mm = meta.create_group("missing_mask")
            mm.attrs["_codec"] = val.get("_codec", "")
            nested = bool(val.get("nested", False))
            mm.attrs["nested"] = nested
            if nested:
                packed_list = val.get("packed", [])
                shapes = val.get("shapes", [])
                numels = val.get("numels", [])
                mm.attrs["count"] = len(packed_list)
                for i, (packed, shape, numel) in enumerate(zip(packed_list, shapes, numels)):
                    _h5_write_tensor(mm, f"packed_{i:06d}", packed)
                    mm.attrs[f"shape_{i}"] = list(shape)
                    mm.attrs[f"numel_{i}"] = int(numel)
            else:
                _h5_write_tensor(mm, "packed", val["packed"])
                mm.attrs["shape"] = list(val["shape"])
                mm.attrs["numel"] = int(val["numel"])
        elif isinstance(val, torch.Tensor):
            _h5_write_tensor(meta, key, val)


def _h5_read_feature_meta(parent_group, device: str | torch.device = "cpu") -> Optional[dict]:
    """Read feature_meta payload from HDF5 and reconstruct codec dictionaries."""
    if "feature_meta" not in parent_group:
        return None
    meta_group = parent_group["feature_meta"]
    feature_meta: dict = {}
    for key in meta_group.keys():
        if key == "missing_mask":
            mm = meta_group["missing_mask"]
            nested = bool(mm.attrs.get("nested", False))
            codec = mm.attrs.get("_codec", _MISSING_MASK_CODEC)
            if nested:
                count = int(mm.attrs.get("count", 0))
                packed_list = []
                shapes = []
                numels = []
                for i in range(count):
                    key = f"packed_{i:06d}"
                    if key not in mm:
                        key = f"packed_{i}"
                    packed_list.append(_h5_read_tensor(mm[key], device=device))
                    shapes.append(list(mm.attrs[f"shape_{i}"]))
                    numels.append(int(mm.attrs[f"numel_{i}"]))
                feature_meta["missing_mask"] = {
                    "_codec": codec,
                    "nested": True,
                    "packed": packed_list,
                    "shapes": shapes,
                    "numels": numels,
                }
            else:
                feature_meta["missing_mask"] = {
                    "_codec": codec,
                    "nested": False,
                    "packed": _h5_read_tensor(mm["packed"], device=device),
                    "shape": list(mm.attrs["shape"]),
                    "numel": int(mm.attrs["numel"]),
                }
        else:
            feature_meta[key] = _h5_read_tensor(meta_group[key], device=device)
    return feature_meta


def dense2sparse(
    dense_tensor: torch.Tensor, row_lengths: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert a dense tensor with trailing zeros into a compact 1D representation.

    Parameters
    ----------
    dense_tensor : torch.Tensor
        Input tensor of shape (num_rows, num_cols) where each row may contain
        trailing zeros beyond the valid entries

    row_lengths : torch.Tensor
        Tensor of shape (num_rows,) specifying the number of valid entries
        in each row of the dense tensor

    dtype : torch.dtype, default=torch.float32
        Output data type for the sparse representation

    Returns
    -------
    torch.Tensor
        1D tensor of shape (sum(row_lengths),) containing only the valid entries
    """

    assert dense_tensor.dim() == 2, "dense_tensor must be 2D"
    num_rows, num_cols = dense_tensor.shape
    assert row_lengths.shape[0] == num_rows, "row_lengths must match number of rows"
    assert (row_lengths <= num_cols).all(), "row_lengths cannot exceed number of columns"

    indices = torch.arange(num_cols, device=dense_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    sparse = dense_tensor[mask].to(dtype)

    return sparse


def sparse2dense(
    sparse_tensor: torch.Tensor,
    row_lengths: torch.Tensor,
    max_len: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a dense tensor from its sparse representation.

    This function is the inverse of dense2sparse, reconstructing a padded dense
    tensor from a compact 1D representation and the corresponding row lengths.
    Unused entries in the output are filled with zeros.

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        1D tensor containing the valid entries from the original dense tensor

    row_lengths : torch.Tensor
        Number of valid entries for each row in the output tensor

    max_len : Optional[int], default=None
        Maximum length for each row in the output. If None, uses max(row_lengths)

    dtype : torch.dtype, default=torch.float32
        Output data type for the dense representation

    Returns
    -------
    torch.Tensor
        Dense tensor of shape (num_rows, max_len) with zeros padding
    """

    assert sparse_tensor.dim() == 1, "data must be 1D"
    assert row_lengths.sum() == len(sparse_tensor), "data length must match sum of row_lengths"

    num_rows = len(row_lengths)
    max_len = max_len or row_lengths.max().item()
    dense = torch.zeros(num_rows, max_len, dtype=dtype, device=sparse_tensor.device)
    indices = torch.arange(max_len, device=sparse_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    dense[mask] = sparse_tensor.to(dtype)

    return dense


class SliceNestedTensor:
    """A wrapper for nested tensors that supports slicing along the first dimension.

    This class wraps PyTorch's nested tensor and provides slicing operations
    along the first dimension, which are not natively supported by nested tensors.
    It maintains compatibility with other nested tensor operations by forwarding
    attribute access to the wrapped tensor.

    Parameters
    ----------
    nested_tensor : torch.Tensor
        A nested tensor to wrap
    """

    def __init__(self, nested_tensor):
        self.nested_tensor = nested_tensor
        self.is_nested = nested_tensor.is_nested

    def __getitem__(self, idx):
        """Support slicing operations along the first dimension."""
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = self.nested_tensor.size(0) if idx.stop is None else idx.stop
            step = 1 if idx.step is None else idx.step

            indices = list(range(start, stop, step))
            return SliceNestedTensor(torch.nested.nested_tensor([self.nested_tensor[i] for i in indices]))
        elif isinstance(idx, int):
            return self.nested_tensor[idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getattr__(self, name):
        """Forward attribute access to the wrapped nested tensor."""
        return getattr(self.nested_tensor, name)

    def __len__(self):
        """Return the length of the first dimension."""
        return self.nested_tensor.size(0)

    def to(self, *args, **kwargs):
        """Support the to() method for device/dtype conversion."""
        return SliceNestedTensor(self.nested_tensor.to(*args, **kwargs))


def cat_slice_nested_tensors(tensors: List, dim=0) -> SliceNestedTensor:
    """Concatenate a list of SliceNestedTensor objects along dimension dim.

    Parameters
    ----------
    tensors : List
        List of tensors to concatenate

    dim : int, default=0
        Dimension along which to concatenate

    Returns
    -------
    SliceNestedTensor
        Concatenated tensor wrapped in SliceNestedTensor
    """
    # Extract the wrapped nested tensors
    nested_tensors = [t.nested_tensor if isinstance(t, SliceNestedTensor) else t for t in tensors]
    return SliceNestedTensor(torch.cat(nested_tensors, dim=dim))


class LoadPriorDataset(IterableDataset):
    """Loads pre-generated prior data sequentially for distributed training.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the batch files

    batch_size : int, default=512
        Number of datasets to return in each iteration

    ddp_world_size : int, default=1
        Total number of distributed processes

    ddp_rank : int, default=0
        Rank of current process

    start_from : int, default=0
        Batch index to start loading from

    max_batches : int, optional
        Maximum number of batches to load. If None, load indefinitely.

    timeout : int, default=60
        Maximum time in seconds to wait for a batch file

    delete_after_load : bool, default=False
        Whether to delete batch files after loading them

    device : str, default='cpu'
        Device to load tensors to
    """

    def __init__(
        self,
        data_dir,
        batch_size=512,
        ddp_world_size=1,
        ddp_rank=0,
        start_from=0,
        max_batches=None,
        timeout=60,
        delete_after_load=False,
        device="cpu",
        loop=False,
        return_metadata=False,
        save_format="auto",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.current_idx = ddp_rank + start_from
        self.max_batches = max_batches
        self.timeout = timeout
        self.delete_after_load = delete_after_load
        self.device = device
        self.loop = loop
        self.return_metadata = return_metadata
        self.save_format = str(save_format).strip().lower()
        if self.save_format not in {"auto", "pt", "h5"}:
            raise ValueError("save_format must be one of: auto, pt, h5")

        # Load metadata if available
        self.metadata = None
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load or parse metadata.json: {e}")

        # Resolve actual on-disk batch format.
        if self.save_format == "auto":
            meta_fmt = ""
            if isinstance(self.metadata, dict):
                meta_fmt = str(self.metadata.get("save_format", "")).strip().lower()
            if meta_fmt in {"pt", "h5"}:
                self.save_format = meta_fmt
            else:
                pt_candidate = self.data_dir / f"batch_{self.current_idx:06d}.pt"
                h5_candidate = self.data_dir / f"batch_{self.current_idx:06d}.h5"
                self.save_format = "pt" if pt_candidate.exists() else ("h5" if h5_candidate.exists() else "pt")
        self.batch_suffix = ".h5" if self.save_format == "h5" else ".pt"

        # Initial verification: Ensure the first batch exists to prevent "silent failures"
        # We wait a few seconds in case the generator is JUST about to write it.
        first_batch = self.data_dir / f"batch_{self.current_idx:06d}{self.batch_suffix}"
        if not first_batch.exists():
            time.sleep(2)
            if not first_batch.exists():
                raise FileNotFoundError(
                    f"First batch file not found: {first_batch}. "
                    "Please ensure the data directory is correct and has at least one batch."
                )

        # Buffer for storing datasets that haven't been returned yet
        self.buffer_X = None
        self.buffer_y = None
        self.buffer_d = None
        self.buffer_seq_lens = None
        self.buffer_feature_meta = None
        self.buffer_size = 0

    def __iter__(self):
        return self

    @staticmethod
    def _slice_first_dim(value, start: int, end: int):
        """Slice along first dim for dense, SliceNestedTensor, and NestedTensor."""
        if isinstance(value, SliceNestedTensor):
            return value[start:end]
        if isinstance(value, torch.Tensor) and value.is_nested:
            chunks = value.unbind()
            sliced = list(chunks[start:end])
            if len(sliced) == 0:
                return torch.nested.nested_tensor([], dtype=value.dtype, device=value.device)
            return torch.nested.nested_tensor(sliced, dtype=value.dtype, device=value.device)
        return value[start:end]

    def _load_batch_file(self, retry_count=0, max_retries=10):
        """Load a single batch file from disk.

        Parameters
        ----------
        retry_count : int
            Current retry count (for recursive calls)
        max_retries : int
            Maximum number of retries before raising error

        Returns
        -------
        tuple
            A tuple containing X, y, d, seq_lens, optional metadata, and batch size
        """
        if retry_count >= max_retries:
            raise RuntimeError(f"Failed to load valid batch file after {max_retries} retries. Check data integrity.")
            
        batch_file = self.data_dir / f"batch_{self.current_idx:06d}{self.batch_suffix}"

        # Try loading the file for up to timeout seconds
        # If file doesn't exist and we've waited, check if we should loop or stop
        wait_time = 0
        while not batch_file.exists():
            if wait_time >= self.timeout:
                # Check if we should loop back to start
                if self.loop:
                    # Loop back to start (batch 0 + ddp_rank)
                    self.current_idx = self.ddp_rank
                    batch_file = self.data_dir / f"batch_{self.current_idx:06d}{self.batch_suffix}"
                    if not batch_file.exists():
                        raise RuntimeError(f"No batch files found starting from {batch_file}")
                    # Reset wait time for the new file
                    wait_time = 0
                else:
                    raise RuntimeError(f"Reached end of data and loop=False. Missing: {batch_file}")
            else:
                time.sleep(1)  # Faster polling
                wait_time += 1

        if self.save_format == "h5":
            _ensure_h5py_available()
            with h5py.File(batch_file, "r") as f:
                if "X" in f and isinstance(f["X"], h5py.Group) and bool(f["X"].attrs.get("nested", False)):
                    X = _h5_read_nested(f["X"], device=self.device)
                else:
                    X = _h5_read_tensor(f["X"], device=self.device)

                if "y" in f and isinstance(f["y"], h5py.Group) and bool(f["y"].attrs.get("nested", False)):
                    y = _h5_read_nested(f["y"], device=self.device)
                else:
                    y = _h5_read_tensor(f["y"], device=self.device)

                d = _h5_read_tensor(f["d"], device=self.device)
                seq_lens = _h5_read_tensor(f["seq_lens"], device=self.device)
                batch_size = int(f.attrs["batch_size"])
                feature_meta = _h5_read_feature_meta(f, device=self.device)
            feature_meta = _decode_and_compact_feature_meta(feature_meta, device=self.device)
        else:
            try:
                batch = torch.load(batch_file, map_location=self.device, weights_only=True)
                # Backward compatibility: some legacy dumps used "ys" instead of "y".
                # Keep "y" as canonical and only fall back to "ys" when needed.
                y_key = "y" if "y" in batch else ("ys" if "ys" in batch else None)
                required_keys = {"X", "d", "seq_lens", "batch_size"}
                missing_keys = [k for k in required_keys if k not in batch]
                if y_key is None:
                    missing_keys = ["y"] + missing_keys
                if missing_keys:
                    raise KeyError(
                        f"Missing key(s) {missing_keys} in {batch_file.name}. "
                        f"Available keys: {sorted(batch.keys())}"
                    )
                X = batch["X"]
                y = batch[y_key]
                d = batch["d"]
                seq_lens = batch["seq_lens"]
                feature_meta = _decode_and_compact_feature_meta(batch.get("feature_meta", None), device=self.device)
                batch_size = batch["batch_size"]
            except Exception as e:
                # Skip malformed/unreadable files instead of crashing the dataloader.
                print(f"[WARNING] Skipping malformed batch file {batch_file.name}: {e}")
                if self.delete_after_load and batch_file.exists():
                    batch_file.unlink()
                self.current_idx += self.ddp_world_size
                return self._load_batch_file(retry_count=retry_count + 1, max_retries=max_retries)

        # Compact integer dtypes on load (for new and legacy files).
        d = d.to(torch.uint16)
        seq_lens = seq_lens.to(torch.int32)

        if X.is_nested:
            # Wrap nested tensors with SliceNestedTensor
            X = SliceNestedTensor(X)
            y = SliceNestedTensor(y)
        else:
            # Convert sparse tensor to dense
            # Note: All datasets in a batch should have the same seq_len when seq_len_per_gp=False
            # But we use seq_lens[0] as a safe assumption
            # Use global max_features from metadata to ensure consistent dimensions across batches.
            # Cast d to signed int before reduction because some torch builds do not
            # implement max reduction for uint16 tensors.
            max_features_meta = self.metadata.get("max_features", None)
            if max_features_meta is not None:
                max_features = int(max_features_meta)
            else:
                max_features = int(d.to(torch.int32).max().item()) if d.numel() > 0 else 100
            try:
                seq_len0 = int(seq_lens[0].item())
                row_lengths = d.to(torch.long).repeat_interleave(seq_len0)
                X = sparse2dense(X, row_lengths, max_len=max_features, dtype=torch.float32).view(batch_size, seq_len0, max_features)
            except AssertionError as e:
                # Skip corrupted batch file and try next one
                print(f"[WARNING] Skipping corrupted batch file {batch_file.name}: {e}")
                # Delete corrupted file if requested
                if self.delete_after_load and batch_file.exists():
                    batch_file.unlink()
                # Try next file
                self.current_idx += self.ddp_world_size
                return self._load_batch_file(retry_count=retry_count + 1, max_retries=max_retries)            
            # Ensure y is properly shaped (should already be (batch_size, seq_len) from generation)
            if y.dim() == 1:
                # If y is flattened, reshape it
                y = y.view(batch_size, int(seq_lens[0].item()))

        # Delete file if requested
        if self.delete_after_load and batch_file.exists():
            batch_file.unlink()

        # Prepare next index for this process
        self.current_idx += self.ddp_world_size

        return X, y, d, seq_lens, feature_meta, batch_size

    def __next__(self):
        """Load datasets until we have at least batch_size, then return exactly batch_size.

        This method accumulates datasets from multiple files if necessary to return
        the exact number of datasets specified in batch_size. Any extra datasets are
        kept in a buffer for the next iteration.

        Returns
        -------
        tuple
            A tuple containing:
            - X: Input features [batch_size, seq_len, features] or nested tensor
            - y: Target labels [batch_size, seq_len] or nested tensor
            - d: Number of features per dataset
            - seq_lens: Sequence length for each dataset
            - feature_meta: Feature metadata, only when return_metadata=True
        """
        # Check if we've reached the maximum number of batches and have no buffered data
        if self.max_batches is not None and self.current_idx >= self.max_batches and (self.buffer_size == 0):
            raise StopIteration

        # Initialize or use existing buffer
        if self.buffer_size == 0:
            # Load the first batch
            X, y, d, seq_lens, feature_meta, file_batch_size = self._load_batch_file()
            self.buffer_X = X
            self.buffer_y = y
            self.buffer_d = d
            self.buffer_seq_lens = seq_lens
            self.buffer_feature_meta = feature_meta
            if self.return_metadata and self.buffer_feature_meta is None:
                raise RuntimeError("return_metadata=True but loaded batch does not contain feature_meta.")
            self.buffer_size = file_batch_size

        # Keep loading files until we have enough data or no more files
        while self.buffer_size < self.batch_size:
            # Check if we've reached max_batches
            if self.max_batches is not None and self.current_idx >= self.max_batches:
                # If we can't get a full batch, return what we have
                break

            try:
                # Load another batch and append to buffer
                X, y, d, seq_lens, feature_meta, file_batch_size = self._load_batch_file()

                # Concatenate with existing buffer
                if self.buffer_X is None:
                    # If buffer is empty, directly assign
                    self.buffer_X = X
                    self.buffer_y = y
                    self.buffer_d = d
                    self.buffer_seq_lens = seq_lens
                    self.buffer_feature_meta = feature_meta
                    if self.return_metadata and self.buffer_feature_meta is None:
                        raise RuntimeError("return_metadata=True but loaded batch does not contain feature_meta.")
                    self.buffer_size = file_batch_size
                else:
                    # Otherwise concatenate, handling SliceNestedTensor if needed
                    if isinstance(X, SliceNestedTensor):
                        self.buffer_X = cat_slice_nested_tensors([self.buffer_X, X], dim=0)
                        self.buffer_y = cat_slice_nested_tensors([self.buffer_y, y], dim=0)
                    else:
                        self.buffer_X = torch.cat([self.buffer_X, X], dim=0)
                        self.buffer_y = torch.cat([self.buffer_y, y], dim=0)

                    self.buffer_d = torch.cat([self.buffer_d, d], dim=0)
                    self.buffer_seq_lens = torch.cat([self.buffer_seq_lens, seq_lens], dim=0)
                    if self.return_metadata:
                        if self.buffer_feature_meta is None or feature_meta is None:
                            raise RuntimeError("return_metadata=True but some batch files do not contain feature_meta.")
                        self.buffer_feature_meta = {
                            k: torch.cat([self.buffer_feature_meta[k], feature_meta[k]], dim=0)
                            for k in self.buffer_feature_meta
                        }
                    self.buffer_size += file_batch_size
            except Exception as e:
                # If we can't load more files, use what we have
                print(f"Warning: Could not load more files: {str(e)}")
                break

        # Extract batch_size datasets (or all if we have fewer)
        output_size = min(self.batch_size, self.buffer_size)

        # Prepare output
        X_out = self.buffer_X[:output_size]
        y_out = self.buffer_y[:output_size]
        d_out = self.buffer_d[:output_size]
        seq_lens_out = self.buffer_seq_lens[:output_size]
        feature_meta_out = None
        if self.return_metadata and self.buffer_feature_meta is not None:
            feature_meta_out = {
                k: self._slice_first_dim(v, 0, output_size) for k, v in self.buffer_feature_meta.items()
            }

        # Update buffer with remaining data
        if output_size < self.buffer_size:
            self.buffer_X = self.buffer_X[output_size:]
            self.buffer_y = self.buffer_y[output_size:]
            self.buffer_d = self.buffer_d[output_size:]
            self.buffer_seq_lens = self.buffer_seq_lens[output_size:]
            if self.return_metadata and self.buffer_feature_meta is not None:
                self.buffer_feature_meta = {
                    k: self._slice_first_dim(v, output_size, self.buffer_size)
                    for k, v in self.buffer_feature_meta.items()
                }
            self.buffer_size -= output_size
        else:
            # Buffer is now empty
            self.buffer_X = None
            self.buffer_y = None
            self.buffer_d = None
            self.buffer_seq_lens = None
            self.buffer_feature_meta = None
            self.buffer_size = 0

        if isinstance(X_out, SliceNestedTensor):
            X_out = X_out.nested_tensor
            y_out = y_out.nested_tensor

        if self.return_metadata:
            return X_out, y_out, d_out, seq_lens_out, feature_meta_out
        return X_out, y_out, d_out, seq_lens_out

    def __repr__(self) -> str:
        """
        Returns a string representation of the LoadPriorDataset.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        repr_str = (
            f"LoadPriorDataset(\n"
            f"  data_dir: {self.data_dir}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  ddp_world_size: {self.ddp_world_size}\n"
            f"  ddp_rank: {self.ddp_rank}\n"
            f"  start_from: {self.current_idx - self.ddp_rank}\n"
            f"  max_batches: {self.max_batches or 'Infinite'}\n"
            f"  timeout: {self.timeout}\n"
            f"  delete_after_load: {self.delete_after_load}\n"
            f"  save_format: {self.save_format}\n"
            f"  device: {self.device}\n"
        )
        if self.metadata:
            repr_str += "  Loaded Metadata:\n"
            repr_str += f"    prior_type: {self.metadata.get('prior_type', 'N/A')}\n"
            repr_str += f"    batch_size (generated): {self.metadata.get('batch_size', 'N/A')}\n"
            repr_str += f"    batch_size_per_gp: {self.metadata.get('batch_size_per_gp', 'N/A')}\n"
            repr_str += f"    min features: {self.metadata.get('min_features', 'N/A')}\n"
            repr_str += f"    max features: {self.metadata.get('max_features', 'N/A')}\n"
            repr_str += f"    max classes: {self.metadata.get('max_classes', 'N/A')}\n"
            repr_str += f"    seq_len: {self.metadata.get('min_seq_len', 'N/A') or 'None'} - {self.metadata.get('max_seq_len', 'N/A')}\n"
            repr_str += f"    log_seq_len: {self.metadata.get('log_seq_len', 'N/A')}\n"
            repr_str += f"    sequence length varies across groups: {self.metadata.get('seq_len_per_gp', 'N/A')}\n"
        repr_str += ")"

        return repr_str


class SavePriorDataset:
    """Generates and saves batches of prior datasets to disk.

    The datasets are saved as individual batch files in the specified directory
    using an atomic file writing pattern to ensure data integrity.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration for dataset generation
    """

    def __init__(self, args):
        self.args = args
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_format = str(getattr(self.args, "save_format", "pt")).strip().lower()
        if self.save_format not in {"pt", "h5"}:
            raise ValueError("save_format must be one of: pt, h5")
        if self.save_format == "h5":
            _ensure_h5py_available()
        
        self.save_metadata()

        # Build custom tree weights dict if any tree weight args are provided
        tree_weights = None
        if any([
            getattr(self.args, 'tree_weight_et', None) is not None,
            getattr(self.args, 'tree_weight_gb', None) is not None,
            getattr(self.args, 'tree_weight_dt', None) is not None,
            getattr(self.args, 'tree_weight_rf', None) is not None,
            getattr(self.args, 'tree_weight_dsrf', None) is not None,
        ]):
            from .prior_config import TREE_PRIOR_WEIGHTS
            tree_weights = {
                'et': getattr(self.args, 'tree_weight_et', TREE_PRIOR_WEIGHTS['et']),
                'gb': getattr(self.args, 'tree_weight_gb', TREE_PRIOR_WEIGHTS['gb']),
                'dt': getattr(self.args, 'tree_weight_dt', TREE_PRIOR_WEIGHTS['dt']),
                'rf': getattr(self.args, 'tree_weight_rf', TREE_PRIOR_WEIGHTS['rf']),
                'dsrf': getattr(self.args, 'tree_weight_dsrf', TREE_PRIOR_WEIGHTS['dsrf']),
            }
            logger.info(f"Using custom tree weights: {tree_weights}")
        
        # Build fixed hyperparameter overrides
        fixed_hp_overrides = {}
        if hasattr(self.args, 'return_metadata'):
            fixed_hp_overrides["return_metadata"] = self.args.return_metadata

        sampled_hp_cfg = copy.deepcopy(DEFAULT_SAMPLED_HP)

        self.prior = PriorDataset(
            batch_size=self.args.batch_size,
            batch_size_per_gp=self.args.batch_size_per_gp,
            min_features=self.args.min_features,
            max_features=self.args.max_features,
            max_classes=self.args.max_classes,
            min_seq_len=self.args.min_seq_len,
            max_seq_len=self.args.max_seq_len,
            log_seq_len=self.args.log_seq_len,
            seq_len_per_gp=self.args.seq_len_per_gp,
            prior_type=self.args.prior_type,
            scm_fixed_hp={**DEFAULT_FIXED_HP, **fixed_hp_overrides},
            scm_sampled_hp=sampled_hp_cfg,
            n_jobs=self.args.n_jobs,
            num_threads_per_generate=self.args.num_threads_per_generate,
            device=self.args.device,
            sampling=getattr(self.args, 'sampling', 'mixed'),
            tree_weights=tree_weights,
            tree_model=getattr(self.args, 'tree_model', None),
            use_cuml=getattr(self.args, 'use_cuml', True),
            use_advanced_hybrid_components=getattr(self.args, 'use_advanced_hybrid_components', False),
            hybrid_sampling_strategy=getattr(self.args, 'hybrid_sampling_strategy', 'random'),
            unstable_activation_threshold=getattr(self.args, 'unstable_activation_threshold', 1500),
        )
        print(self.prior)

    def save_metadata(self):
        """Save metadata about the dataset generation configuration to a JSON file."""
        metadata = {
            "save_format": getattr(self, "save_format", "pt"),
            "prior_type": self.args.prior_type,
            "batch_size": self.args.batch_size,
            "batch_size_per_gp": self.args.batch_size_per_gp,
            "min_seq_len": self.args.min_seq_len,
            "max_seq_len": self.args.max_seq_len,
            "log_seq_len": self.args.log_seq_len,
            "seq_len_per_gp": self.args.seq_len_per_gp,
            "min_features": self.args.min_features,
            "max_features": self.args.max_features,
            "max_classes": self.args.max_classes,
            "sampling": getattr(self.args, 'sampling', 'beta'),
            "return_metadata": getattr(self.args, 'return_metadata', False),
            "tree_weights": {
                "et": getattr(self.args, 'tree_weight_et', None),
                "gb": getattr(self.args, 'tree_weight_gb', None),
                "dt": getattr(self.args, 'tree_weight_dt', None),
                "rf": getattr(self.args, 'tree_weight_rf', None),
                "dsrf": getattr(self.args, 'tree_weight_dsrf', None),
            } if any([
                getattr(self.args, 'tree_weight_et', None) is not None,
                getattr(self.args, 'tree_weight_gb', None) is not None,
                getattr(self.args, 'tree_weight_dt', None) is not None,
                getattr(self.args, 'tree_weight_rf', None) is not None,
                getattr(self.args, 'tree_weight_dsrf', None) is not None,
            ]) else None,
        }
        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def save_batch_sparse(self, batch_idx, X, y, d, seq_lens, feature_meta=None):
        """Save batch data in sparse format for efficient storage.

        This method handles the conversion between dense and sparse tensor formats
        when appropriate and saves the batch data to a PyTorch file. It uses an atomic
        write pattern (writing to a temporary file and then renaming) to ensure data
        integrity even if the process is interrupted during saving.

        All tensors are moved to CPU before saving to ensure portability (files can be
        loaded on machines without CUDA).

        Parameters
        ----------
        batch_idx : int
            Index of the current batch used for file naming

        X : torch.Tensor
            Input features tensor, either in dense format [batch_size, seq_len, features]
            or in nested tensor format for variable sequence lengths

        y : torch.Tensor
            Target labels tensor

        d : torch.Tensor
            Number of features for each dataset

        seq_lens : torch.Tensor
            Sequence length for each dataset

        feature_meta : dict, optional
            Feature metadata to store with the batch
        """
        # Move all tensors to CPU before processing (needed when generating on CUDA)
        # This ensures saved files are portable and don't require CUDA when loading
        if X.device.type != 'cpu':
            X = X.cpu()
        if y.device.type != 'cpu':
            y = y.cpu()
        if d.device.type != 'cpu':
            d = d.cpu()
        if seq_lens.device.type != 'cpu':
            seq_lens = seq_lens.cpu()
        if feature_meta is not None:
            feature_meta = {
                k: (v.cpu() if isinstance(v, torch.Tensor) and v.device.type != 'cpu' else v)
                for k, v in feature_meta.items()
            }

        if self.args.seq_len_per_gp:
            # X and y are nested tensors and they are already sparse
            B = len(d)
        else:
            B, T, H = X.shape
            X = dense2sparse(X.view(-1, H), d.to(torch.long).repeat_interleave(T), dtype=torch.float32)

        # Compact integer tensors for storage.
        d = d.to(torch.uint16)
        seq_lens = seq_lens.to(torch.int32)

        # Compact and encode feature metadata.
        if feature_meta is not None:
            feature_meta = _compact_feature_meta_dtypes(feature_meta)
            if "missing_mask" in feature_meta:
                feature_meta = dict(feature_meta)
                feature_meta["missing_mask"] = _encode_missing_mask_bitpack(feature_meta["missing_mask"])

        if self.save_format == "h5":
            _ensure_h5py_available()
            batch_file = self.save_dir / f"batch_{batch_idx:06d}.h5"
            temp_file = self.save_dir / f"batch_{batch_idx:06d}.h5.tmp"
            with h5py.File(temp_file, "w") as f:
                f.attrs["batch_size"] = int(B)
                if isinstance(X, torch.Tensor) and X.is_nested:
                    _h5_write_nested(f, "X", X)
                else:
                    _h5_write_tensor(f, "X", X)
                if isinstance(y, torch.Tensor) and y.is_nested:
                    _h5_write_nested(f, "y", y)
                else:
                    _h5_write_tensor(f, "y", y)
                _h5_write_tensor(f, "d", d)
                _h5_write_tensor(f, "seq_lens", seq_lens)
                if feature_meta is not None:
                    _h5_write_feature_meta(f, feature_meta)
            temp_file.replace(batch_file)
        else:
            # Create temporary file first
            batch_file = self.save_dir / f"batch_{batch_idx:06d}.pt"
            temp_file = self.save_dir / f"batch_{batch_idx:06d}.pt.tmp"
            payload = {"X": X, "y": y, "d": d, "seq_lens": seq_lens, "batch_size": B}
            if feature_meta is not None:
                payload["feature_meta"] = feature_meta
            torch.save(payload, temp_file)
            # Atomic rename to ensure file integrity
            temp_file.replace(batch_file)

    def run(self):
        """Generate and save batches of prior datasets."""
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(f"Generating {self.args.num_batches} batches starting from index {self.args.resume_from}")
        logger.info("Note: First batch may take longer due to initialization. Generation is in progress...")

        total_start_time = time.time()
        batch_times = []

        # Async saving: save in background while generating next batch
        save_thread = None

        def async_save(batch_idx, X, y, d, seq_lens, feature_meta=None):
            """Save batch in background thread."""
            try:
                self.save_batch_sparse(batch_idx, X, y, d, seq_lens, feature_meta=feature_meta)
            except Exception as e:
                logger.error(f"Error saving batch {batch_idx}: {e}")

        for batch_idx in tqdm(
            range(self.args.resume_from, self.args.resume_from + self.args.num_batches),
            desc="Generating batches",
            mininterval=1.0,
        ):
            try:
                batch_start_time = time.time()

                # Wait for previous save to complete before generating next
                # This ensures we don't accumulate too many pending saves
                if save_thread is not None and save_thread.is_alive():
                    save_thread.join()

                if batch_idx == self.args.resume_from:
                    logger.info(f"Starting batch generation: batch_idx={batch_idx}")

                batch = self.prior.get_batch()
                if len(batch) == 5:
                    X, y, d, seq_lens, feature_meta = batch
                else:
                    X, y, d, seq_lens = batch
                    feature_meta = None

                generation_time = time.time() - batch_start_time
                batch_times.append(generation_time)

                # Log batch statistics periodically
                if batch_idx == self.args.resume_from or (batch_idx + 1) % 100 == 0:
                    avg_time = sum(batch_times[-100:]) / min(100, len(batch_times))
                    d_stats = d.to(torch.int32)
                    logger.info(
                        f"Batch {batch_idx}: generated in {generation_time:.2f}s "
                        f"(avg last 100: {avg_time:.2f}s, "
                        f"seq_len_range=[{seq_lens.min().item()}, {seq_lens.max().item()}], "
                        f"features_range=[{d_stats.min().item()}, {d_stats.max().item()}])"
                    )

                # Move tensors to CPU before saving (needed when generating on CUDA)
                if X.device.type != 'cpu':
                    X = X.cpu()
                if y.device.type != 'cpu':
                    y = y.cpu()
                if d.device.type != 'cpu':
                    d = d.cpu()
                if seq_lens.device.type != 'cpu':
                    seq_lens = seq_lens.cpu()
                if feature_meta is not None:
                    feature_meta = {
                        k: (v.cpu() if isinstance(v, torch.Tensor) and v.device.type != 'cpu' else v)
                        for k, v in feature_meta.items()
                    }

                # Start async save (overlap saving with next batch generation)
                save_thread = threading.Thread(
                    target=async_save,
                    args=(batch_idx, X, y, d, seq_lens, feature_meta),
                )
                save_thread.start()

                if generation_time > 10.0:  # Log slow batches
                    logger.warning(f"Slow batch {batch_idx}: generation={generation_time:.2f}s")

            except Exception as e:
                logger.error(f"Error generating batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                raise

        # Wait for final save to complete
        if save_thread is not None and save_thread.is_alive():
            save_thread.join()
        
        total_time = time.time() - total_start_time
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        logger.info(
            f"Generation complete: {self.args.num_batches} batches in {total_time:.2f}s "
            f"(avg: {avg_batch_time:.2f}s/batch, "
            f"min: {min(batch_times):.2f}s, max: {max(batch_times):.2f}s)"
        )


if __name__ == "__main__":

    def str2bool(value):
        return value.lower() == "true"

    parser = argparse.ArgumentParser(description="Generate training prior datasets")
    parser.add_argument("--save_dir", type=str, default="data", help="Directory to save the generated data")
    parser.add_argument("--save_format", type=str, default="pt", choices=["pt", "h5"],
                        help="Output batch file format. 'pt' preserves current behavior; 'h5' writes HDF5.")
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--num_batches", type=int, default=10000, help="Number of batches to generate")
    parser.add_argument("--resume_from", type=int, default=0, help="Resume generation from this batch index")
    parser.add_argument("--batch_size", type=int, default=512, help="Total batch size")
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=1, help="Minimum number of features (inclusive)")
    parser.add_argument("--max_features", type=int, default=99, help="Maximum number of features (inclusive)")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        default="mix_scm",
        choices=["mlp_scm", "conv_scm", "tree_scm", "gp_scm", "linear_scm", "hybrid_scm", "mix_scm", "mix_scm_hscm", "mix_scm_no_gp", "mix_scm_hscm_no_gp"],
        help="Type of prior to use",
    )
    parser.add_argument("--sampling", type=str, default="mixed", choices=["normal", "mixed", "uniform", "beta"],
                        help="Feature sampling strategy for SCM priors ('beta' recommended for diversity, matches TabPFN alignment)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing")
    parser.add_argument("--num_threads_per_generate", type=int, default=1, help="Threads per generation")
    
    # Advanced data generation flags
    parser.add_argument("--use_advanced_hybrid_components", default=True, type=str2bool,
                        help="Enable advanced edge/aggregation functions in HybridSCM")
    parser.add_argument("--hybrid_sampling_strategy", type=str, default="random",
                        choices=["random", "graph_aware"],
                        help="Sampling strategy for features in HybridSCM ('graph_aware' recommended for proper dependency ordering)")
    parser.add_argument("--unstable_activation_threshold", type=int, default=1500,
                        help="Sequence length threshold above which unstable activations (Exp, Square) are excluded. Default: 1500")
    parser.add_argument("--return_metadata", default=False, type=str2bool,
                        help="If True, save feature metadata (missing_mask/type_ids/col_ids/imputation_ids) with each batch.")
    
    # Tree type weights (for tree_scm prior)
    parser.add_argument("--tree_weight_et", type=float, default=None, help="Weight for extra_trees (default: 0.15)")
    parser.add_argument("--tree_weight_gb", type=float, default=None, help="Weight for xgboost/gradient_boosting (default: 0.10)")
    parser.add_argument("--tree_weight_dt", type=float, default=None, help="Weight for decision_tree (default: 0.08)")
    parser.add_argument("--tree_weight_rf", type=float, default=None, help="Weight for random_forest (default: 0.05)")
    parser.add_argument("--tree_weight_dsrf", type=float, default=None, help="Weight for dsrf (default: 0.02)")
    parser.add_argument("--tree_model", type=str, default=None,
                        choices=["decision_tree", "extra_trees", "random_forest", "xgboost", "dsrf"],
                        help="Force specific tree model for tree_scm prior. None = random sampling by weights (default)")
    parser.add_argument("--use_cuml", action="store_true", default=True,
                        help="Use cuML (RAPIDS) for GPU-accelerated tree training when available (default: True). "
                             "Provides 10-50x speedup over sklearn. Automatically disabled on CPU or if cuML unavailable.")
    parser.add_argument("--no_cuml", action="store_false", dest="use_cuml",
                        help="Explicitly disable cuML, force sklearn for all tree models")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for generation"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (DEBUG for detailed debugging, INFO for normal output)"
    )

    args = parser.parse_args()
    
    # Set logging level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    # Also set level for dataset logger
    from .dataset import logger as dataset_logger
    dataset_logger.setLevel(log_level)
    
    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)
    saver = SavePriorDataset(args)
    saver.run()
