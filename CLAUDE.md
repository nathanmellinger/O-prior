# O-Prior for CubICL pre-training

## Goal

Generate synthetic tabular data with the O-Prior SCM (Lexsi Labs, *Shaping The Prior*),
upload it to S3, and pre-train CubICL on it inside LTM1. Then compare training and
validation loss plus final eval (`client_eval_v0-2`) against TabICLv1, TabICLv2 and the
LTM1 SCM.

Success criterion: O-Prior data trains CubICL through the existing LTM1 pipeline with no
compatibility workarounds.

## Repo boundary

- `/Users/nathanmellinger/O-prior` (this repo): **all changes go here**.
- `/Users/nathanmellinger/LTM1/worktrees/oprior`: **read-only reference.** Never edit.
  Consult it to confirm what the training pipeline expects. LTM1-side changes (registry
  entry, training config, curriculum support) are Nathan's, done separately.

## Environment

Use `.venv/bin/python` (Python 3.12; system python is 3.14, outside O-Prior's
`>=3.9,<3.13`). Created with `uv venv --python 3.12 && uv pip install -e .`.

**macOS local runs must set `OMP_NUM_THREADS=1`:**

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m o_prior.prior.genload ...
```

Without it, generation segfaults non-deterministically. Cause: `import torch` followed by
fitting any XGBoost model crashes on Apple Silicon (duplicate OpenMP runtimes).
`hybrid_scm`'s `TreeEdge` picks xgboost 1 time in 3, so the crash is seed-dependent.
`KMP_DUPLICATE_LIB_OK=TRUE` does **not** help; `OMP_NUM_THREADS=1` does. This is a
macOS/environment issue, not an O-Prior bug, and is not expected on the Linux cluster.

LTM1's own venv (`/Users/nathanmellinger/LTM1/worktrees/oprior/.venv/bin/python`) imports
fine from here. Use it to validate exports against LTM1's **real** loader and processor
chain rather than reimplementing either.

## What LTM1 expects

`.h5` collections of `TaskFormatV0`, read by
`ltm1/train/data_generators/utils.py:load_collection_v0`:

- root groups `task_0`, `task_1`, ... read back with `sorted(keys)`, so **lexicographic**.
  Zero-pad (`task_000000`) or generation order is lost.
- per group: `x` `(n_rows, n_cols)` float32 (never int), `y` `(n_rows, 1)` float32 single
  output, and a `metadata` group (must exist).
- classification `y` must be contiguous ints `0..K-1` with `K <= 10`. Labels `>= 10` are
  silently clamped to class 9, not rejected.

**Metadata scalars must be h5 attrs, never datasets.** `create_dataset` round-trips as a
0-d ndarray and `label_permuter` then dies with `AxisError`. LTM1's own
`_save_collection_v0` gets this wrong, so do not copy it. Required keys: `discrete_y`
(bool) and, for classification, `num_classes` (int, `> max(y)`, `<= 10`). Ours:
`index_split`, `id_generated`. Must **not** contain `num_context`. Omit `x_column_names`.

Because the processor chain runs inside `concurrent_reader` worker processes, a bad task
kills a worker and training **hangs** rather than crashing. Validate exports locally.

Sizing: `to_batch_v1` needs exactly 1000 rows (`num_context + num_pred` is constant), so
1024 works and anything under 1000 is silently skipped. Max 100 columns; more is silently
truncated, categoricals first. Reference baseline `tabicl_phase1_reg_class_v0`
(`s3://fun-research-datasets-us-west-1/tabicl-big/tabicl/batch*/*.h5`) is ~512 tasks per
file, 1024 rows, 93-100 columns, ~213 MB/file. Match that shard size.

### LTM1-side handoff (Nathan, not this repo)

**`filter_unlearnable` must be removed from the O-Prior branch of the config.** Measured on
a 32-task sample:

| | passes `filter_unlearnable` |
|---|---|
| our export (NaN restored) | 1/32 (3%) |
| same tasks, NaN imputed away | 12/32 (38%) |

The first row is why the filter has to go: it does `dropna(axis=1, how="any")` and O-Prior
applies a non-zero missing rate to every active column.

The second row is a **confound to state in the report**: even with no missingness, ~62% of
O-Prior tasks fail LTM1's learnability test, so the O-Prior arm trains on tasks the
baseline arms never see. Sample is small (±9pp). The comparison number to look for is
TabICLv1's own rate, which existing training logs already print as
`"Proportion of learnable tasks"` every 10k tasks.

Two of the three known frictions are config-only on the LTM1 side, no code change:
`to_batch_v1` supports `split_type: "chronological"`, and `from_s3` has
`shuffle_tasks_in_file` / `shuffle_file_order`. Honouring the exact `index_split` needs an
LTM1 change, and must happen at `to_batch_v1` or earlier because metadata is dropped at
`to_cubicl_input`.

## What O-Prior gives us

`scripts/generate_rq4.sh` is the "full setup" (hierarchical SCM mix, realism curriculum
mild to hard, covariate/seasonal/temporal shift, up to 100 features). It writes
`batch_XXXXXX.h5` containing `X` (sparse-packed, unpack with `sparse2dense`), `y`,
`d` (active features per task), `seq_lens`, `train_sizes`, and `feature_meta`.

With the raised cell cap (below) it produces 1024-row tasks. `train_sizes` is the
context/prediction boundary we want, and is uniform across a generation group
(`batch_size_per_gp`), not per task. `X` is fully imputed and NaN-free; missingness lives
only in `feature_meta["missing_mask"]`, which needs `--return_metadata True`.

## Decisions taken

| Topic | Decision |
|---|---|
| Realism | Run RQ4 with `--return_metadata True` and re-insert NaN into `x` from `feature_meta["missing_mask"]`. **This requires dropping `filter_unlearnable` from the O-Prior branch of the LTM1 config**: it does `dropna(axis=1, how="any")` and O-Prior applies a non-zero missing rate to every active column, so 0/10 tasks survive otherwise. Known confound versus TabICLv1, which keeps that filter; measure and report the size of the effect. |
| Task types | Two passes: RQ4 as published (classification, `max_classes=10`), plus a regression pass (`max_classes=0`), to match the reg+class mix of the TabICLv1 baseline. |
| Compute | Local CPU smoke test first to validate the format end to end, then the full run on the cluster. |
| S3 | `s3://fun-research-datasets-us-west-1/oprior/v0/{classification,regression}/oprior-000000.h5` |
| Metadata | `discrete_y`, `num_classes`, plus `index_split` (from `train_sizes`) and `id_generated` (global task counter across the run, so realism increases with the number). |
| Naming | Zero-padded everywhere: `oprior-000000.h5`, `task_000000`. |

## Deliberate non-changes

**Easy-batch replay is left as upstream wrote it**, and ~10% of generated tasks are
therefore unusable. When `use_curriculum=True` (which RQ4 sets), `get_batch` has a 10%
chance per batch of producing an "easy" replay batch (`dataset.py:5132`): forced
`linear_scm`/`gp_scm`, 2-10 features, near-zero noise, and `seq_len` drawn from
`randint(200, 1000)`. That last draw **ignores `--min_seq_len`** and its bound is
exclusive, so every easy batch lands below LTM1's 1000-row minimum and is silently
skipped by `to_batch_v1`. Measured 4/25 batches on one run, consistent with the 10% design
rate.

Two consequences:
- **Over-generate by ~11%** to hit a target count of trainable tasks.
- The mechanism exists to stop the model forgetting simple patterns. Since exactly those
  tasks are the ones dropped, that anti-forgetting replay is **inactive** in our runs and
  only the harder tasks survive. This is a property of our setup, not of O-Prior; note it
  in the report.

## Changes to upstream O-Prior

Keep this list current: the cell cap in particular is a real deviation to report.

| Change | Why |
|---|---|
| `scripts/generate_rq4.sh`: `TIME_LAGGED_WEIGHT_SPARSITY` / `TIME_LAGGED_OUTPUT_NOISE_STD` defaults `float` -> empty | Upstream passed the literal string `float` to an argparse `type=float`, aborting on batch 0. |
| `scripts/generate_rq4.sh`: `"${EXTRA_ARGS[@]}"` -> `${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}` | macOS bash 3.2 treats an empty array expansion as unbound under `set -u`. Masked by the bug above until it was fixed. |
| `scripts/generate_rq4.sh`: added `--return_metadata True` | Needed for the missing mask. |
| `dataset.py`: `delete_unique_features` returns keep-indices; new `_reindex_feature_meta` applied at the call site | It dropped constant columns from `X` and left-compacted survivors without re-indexing `feature_meta`, so `missing_mask` described the wrong columns. Would have corrupted the NaN re-insertion. |
| `dataset.py`: `enforce_cell_cap` max_cells 75,000 -> 102,400 | 100 features x 1024 rows was truncated to 750 rows, below LTM1's 1000-row minimum, so every task would have been silently skipped. LTM1 has no cell cap at all and its own generators routinely produce 100k-1.1M-cell tables; TabICLv1 is ~102,400. **Deviates from O-Prior's default; note in the report.** |
| `scripts/generate_rq4.sh`: seeds now `NP_SEED + batch_idx` / `TORCH_SEED + batch_idx` | The script runs one process per batch and re-seeded to the same value each time, so every batch drew identical batch-level parameters: `train_size` was constant across the whole dataset (409, 409, 409 instead of 409, 196, 786). O-Prior assumes one process generating many batches, with the RNG advancing. Still fully reproducible and resume-safe: batch N always gets seed 42+N. |

## Exporter

`src/o_prior/export_ltm1.py` converts O-Prior batches into an LTM1 collection. O-Prior
stores a batch as one stacked array `(num_tasks, seq_len, max_features)` with zero padding
past each task's active feature count; LTM1 wants one group per task at its real width.

```bash
export OMP_NUM_THREADS=1                      # macOS only, see Environment
GENLOAD_CMD=".venv/bin/python -u -m o_prior.prior.genload" \
  DATA_ROOT=data/oprior_batches NUM_BATCHES=2 BATCH_SIZE=8 N_JOBS=1 \
  bash scripts/generate_rq4.sh
.venv/bin/python -m o_prior.export_ltm1 --in data/oprior_batches --out data/oprior_ltm1
```

Each task is trimmed to its real width (no padding, and `d` is dropped: it is just
`x.shape[1]`; LTM1 strips zero padding anyway via `remove_constant_columns`), NaN is
restored from the missing mask, and scalars are written as attrs. Tasks are skipped if
they have no columns, non-finite `y`, or non-contiguous / out-of-range labels.

Validated end to end against LTM1's real `load_collection_v0` and the full `cubicl_v4`
processor chain (minus `filter_unlearnable`).

## Remaining

1. Regression pass (`max_classes=0`), then full generation on the cluster and S3 upload.
2. Nathan adds the LTM1 registry entry and training config, runs pre-training and evals.

## Working style

- **Explain every change and wait for explicit approval before making it.** Plan approval
  is not blanket approval for the edits inside it.
- Ask whenever a key decision comes up or something is unclear. Do not guess.
- Keep changes minimal against the existing repo. Prefer a small script over new modules,
  classes or abstractions.
- If a task looks like it needs a lot of code, stop and check for a simpler route first;
  if the volume is genuinely necessary, pause and split it into subtasks.
- Conciseness, simplicity, clarity. No overcoding, no over-commenting.
