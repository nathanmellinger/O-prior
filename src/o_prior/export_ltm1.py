"""Convert O-Prior batches into LTM1 TaskFormatV0 h5 collections.

O-Prior stores a batch as one stacked array (num_tasks, seq_len, max_features) with
zero padding past each task's active feature count. LTM1 wants one group per task,
each holding its own x/y at its real width. This script does that conversion.

    python -m o_prior.export_ltm1 --in <batch_dir> --out <collection_dir>

Missing values are restored as NaN from feature_meta["missing_mask"], so generation
must have run with --return_metadata True.
"""
import os

import argparse
import json
import pathlib

import h5py
import numpy as np
import torch

from o_prior.prior.genload import (
    _decode_and_compact_feature_meta,
    _h5_read_feature_meta,
    _h5_read_tensor,
    sparse2dense,
)

MAX_CLASSES = 10  # LTM1 silently clamps labels above this
BUCKET = f"fun-research-datasets-{os.environ.get('AWS_REGION', 'us-west-1')}"

def _read_batch(path: pathlib.Path, max_features: int):
    """Densify one O-Prior batch file."""
    with h5py.File(path, "r") as f:
        if isinstance(f["X"], h5py.Group):
            raise ValueError(f"{path.name}: nested batches (seq_len_per_gp) are not supported")
        X = _h5_read_tensor(f["X"])
        y = _h5_read_tensor(f["y"])
        d = _h5_read_tensor(f["d"]).to(torch.long)
        seq_lens = _h5_read_tensor(f["seq_lens"]).to(torch.long)
        train_sizes = _h5_read_tensor(f["train_sizes"]).to(torch.long)
        num_tasks = int(f.attrs["batch_size"])
        feature_meta = _decode_and_compact_feature_meta(_h5_read_feature_meta(f))

    if feature_meta is None or "missing_mask" not in feature_meta:
        raise ValueError(f"{path.name}: no missing_mask; regenerate with --return_metadata True")

    seq_len = int(seq_lens[0])
    X = sparse2dense(X, d.repeat_interleave(seq_len), max_len=max_features).view(
        num_tasks, seq_len, max_features
    )
    return X, y, d, seq_lens, train_sizes, feature_meta["missing_mask"], num_tasks


def _tasks(path: pathlib.Path, batch_idx: int, max_features: int, batch_size: int, discrete_y: bool):
    """Yield (x, y, metadata) per task, trimmed to real width with NaN restored."""
    X, y, d, seq_lens, train_sizes, mask, num_tasks = _read_batch(path, max_features)

    for i in range(num_tasks):
        width, rows = int(d[i]), int(seq_lens[i])
        if width == 0:
            continue  # every column was constant

        x_i = X[i, :rows, :width].numpy().astype(np.float32)
        x_i[mask[i, :rows, :width].numpy()] = np.nan
        y_i = y[i, :rows].numpy().astype(np.float32).reshape(-1, 1)

        if not np.isfinite(y_i).all() or not np.isfinite(x_i[~np.isnan(x_i)]).all():
            continue  # NaN/inf targets give NaN gradients; inf features overflow downstream

        meta = {
            "discrete_y": discrete_y,
            "index_split": int(train_sizes[i]),
            "id_generated": batch_idx * batch_size + i,
        }
        if discrete_y:
            labels = np.unique(y_i)
            num_classes = int(labels.max()) + 1
            if not np.array_equal(labels, np.arange(num_classes)) or num_classes > MAX_CLASSES:
                continue  # label_permuter indexes by label, so they must be contiguous 0..K-1
            meta["num_classes"] = num_classes

        yield x_i, y_i, meta


def _write_shard(tasks: list, path: pathlib.Path) -> None:
    """Write one LTM1 collection file. Scalars go in attrs: as datasets they load back
    as 0-d arrays and break label_permuter."""
    with h5py.File(path, "w") as f:
        for x, y, meta in tasks:
            group = f.create_group(f"task_{meta['id_generated']:07d}")
            group.create_dataset("x", data=x)
            group.create_dataset("y", data=y)
            metadata = group.create_group("metadata")
            for key, value in meta.items():
                metadata.attrs[key] = value

def _upload(path: pathlib.Path, prefix: str, overwrite: bool) -> None:
    """Upload one collection file to S3, keeping the local copy."""
    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client("s3")
    key = f"{prefix}/{path.name}"
    if not overwrite:
        try:
            client.head_object(Bucket=BUCKET, Key=key)
            exists = True
        except ClientError as err:
            if err.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise
            exists = False
        if exists:
            raise SystemExit(f"s3://{BUCKET}/{key} already exists (pass --overwrite)")
    client.upload_file(str(path), BUCKET, key)
    print(f"uploaded s3://{BUCKET}/{key}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", required=True, help="Directory of O-Prior batch_*.h5")
    parser.add_argument("--out", dest="out_dir", required=True, help="Directory for the collection")
    parser.add_argument("--tasks-per-file", type=int, default=512, help="Tasks per output shard")
    parser.add_argument("--s3-prefix", help="Upload each file to s3://<bucket>/<prefix>/")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting S3 keys")
    args = parser.parse_args()

    in_dir, out_dir = pathlib.Path(args.in_dir), pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((in_dir / "metadata.json").read_text())
    max_features, batch_size = int(config["max_features"]), int(config["batch_size"])
    discrete_y = int(config["max_classes"]) > 0

    batch_files = sorted(in_dir.glob("batch_*.h5"))
    if not batch_files:
        raise SystemExit(f"No batch_*.h5 in {in_dir}")

    def flush(tasks: list, index: int) -> None:
        path = out_dir / f"oprior-{index:06d}.h5"
        _write_shard(tasks, path)
        if args.s3_prefix:
            _upload(path, args.s3_prefix.strip("/"), args.overwrite)

    buffer, shard, total, seen = [], 0, 0, 0
    for path in batch_files:
        batch_idx = int(path.stem.split("_")[1])
        seen += batch_size
        for task in _tasks(path, batch_idx, max_features, batch_size, discrete_y):
            buffer.append(task)
            if len(buffer) == args.tasks_per_file:
                flush(buffer, shard)
                total += len(buffer)
                buffer, shard = [], shard + 1
        # Close the shard at every generation-batch boundary. All tasks in one
        # O-Prior batch share an index_split, so a shard that straddles two of
        # them would hand LTM1 a training batch cut at two different rows.
        if buffer:
            flush(buffer, shard)
            total += len(buffer)
            buffer, shard = [], shard + 1

    print(f"{total} tasks ({seen - total} skipped) -> {shard} file(s)")

if __name__ == "__main__":
    main()
