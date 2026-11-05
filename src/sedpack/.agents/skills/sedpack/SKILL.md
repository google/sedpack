---
name: sedpack
description: >-
  Creates, writes, inspects, and iterates multi-shard machine learning and
  side-channel analysis datasets using the sedpack library (sedpack.io). Use
  when creating a sedpack Dataset, defining Metadata, DatasetStructure, or
  Attribute schemas, writing examples with dataset.filler(), DatasetFiller,
  or dataset.write_multiprocessing(), iterating data for JAX, PyTorch, Keras,
  TensorFlow, or NumPy (as_numpy_iterator, as_numpy_iterator_concurrent,
  as_numpy_iterator_async, as_numpy_iterator_rust,
  as_numpy_iterator_rust_batched, as_tfdataset), filtering or balancing
  shards via custom_metadata, or converting raw datasets (HDF5, NPZ, traces)
  into sedpack format. Don't use for modifying sedpack's internal Rust or
  FlatBuffers compiler implementation unless high-level API semantics are
  needed.
license: Apache-2.0
---

# Using `sedpack` (`sedpack.io`)

`sedpack` stores and streams sharded datasets using a hierarchical metadata tree
(`dataset_info.json` to `<split>/shards_list.json` to shard files) backed by
FlatBuffers (`"fb"`), NPZ (`"npz"`), or TFRecords (`"tfrec"`).

## 1. Core Imports

Top-level `import sedpack` only exposes `sedpack.__version__`. Import API
classes from `sedpack.io` and type/split constants from `sedpack.io.types`:

```python
import itertools
from pathlib import Path
import numpy as np

from sedpack.io import (
    Attribute, Dataset, DatasetFiller, DatasetStructure, Metadata,
)
from sedpack.io.errors import DatasetExistsError
from sedpack.io.iteration import RustBatchedGenerator, RustGenerator
from sedpack.io.shard_file_metadata import ShardInfo, ShardListInfo
from sedpack.io.shard_info_iterator import CachedShardInfoIterator
from sedpack.io.shard_info_iterator.balanced_iterator import (
    BalancedShardInfoIterator,
)
from sedpack.io.types import (
    HOLDOUT_SPLIT, TEST_SPLIT, TRAIN_SPLIT, BatchT, ExampleT, SplitT,
)
```

## 2. Creating vs. Loading a Dataset

-   **Create a new dataset**: Call
    `Dataset.create(path=dataset_path, metadata=metadata, ...)`
    -   Creates `dataset_path` and writes `dataset_info.json`.
    -   Raises `DatasetExistsError` (a subclass of `FileExistsError`) if
        `dataset_info.json` already exists at `dataset_path`.
-   **Load an existing dataset (to read or append)**: Call
    `dataset = Dataset(dataset_path)`
    -   There is **no** `Dataset.load()` or `Dataset.open()` method. Always call
        `Dataset(path)`.
-   **Valid Splits (`SplitT`)**:
    -   Strictly `"train"`, `"test"`, or `"holdout"` (`TRAIN_SPLIT`,
        `TEST_SPLIT`, `HOLDOUT_SPLIT`).
    -   **Gotcha**: Passing `"val"` or `"validation"` raises
        `ValueError: Not a known split value`. In `sedpack`, use `"test"` for
        validation during training and `"holdout"` for final evaluation.

## 3. Defining the Schema (`Metadata`, `DatasetStructure`, `Attribute`)

```python
metadata = Metadata(
    description="AES power traces dataset",
    dataset_license="Apache-2.0",
    dataset_version="1.0.0",
    custom_metadata={"capture_scope": "ChipWhisperer"},
)

dataset_structure = DatasetStructure(
    saved_data_description=[
        Attribute(name="trace", dtype="float32", shape=(5000,)),
        Attribute(name="plaintext", dtype="uint8", shape=(16,)),
        Attribute(name="key_id", dtype="int", shape=()),
        Attribute(name="label", dtype="uint8", shape=()),
        Attribute(name="raw_packet", dtype="bytes", shape=()),
        Attribute(name="fixed_nonce", dtype="bytes", shape=(12,)),
        Attribute(name="notes", dtype="str", shape=()),
    ],
    compression="LZ4",          # Default is "GZIP"; "LZ4" or "ZSTD" recommended for speed
    examples_per_shard=256,     # Default is 256
    shard_file_type="fb",       # "fb" (default, fastest), "npz", or "tfrec"
    hash_checksum_algorithms=("sha256",),
)

dataset_path = Path("datasets/aes_traces")
dataset = Dataset.create(
    path=dataset_path,
    metadata=metadata,
    dataset_structure=dataset_structure,
)
```

### `Attribute` `dtype` and `shape` Rules

1.  **Positive dimensions only**: Every integer in `shape` must be **strictly
    positive (`> 0`)**. Passing `(-1,)`, `(0,)`, or `(None,)` raises
    `ValueError`.
2.  **Use `shape=()` for scalars and variable-length types**:
    -   NumPy scalar: `Attribute(name="label", dtype="uint8", shape=())`
    -   Arbitrary 64-bit signed Python `int`:
        `Attribute(name="index", dtype="int", shape=())`
    -   UTF-8 variable-length string:
        `Attribute(name="notes", dtype="str", shape=())`
    -   Variable-length byte sequence:
        `Attribute(name="raw_packet", dtype="bytes", shape=())`
        (`has_variable_size() == True`)
3.  **Fixed-size vs. variable-size `"bytes"`**:
    -   `dtype="bytes", shape=()`: Variable-length `bytes`.
    -   `dtype="bytes", shape=(N,)`: Fixed-length `N`-byte sequence
        (`has_variable_size() == False`). On write, accepts
        `bytes`/`bytearray`/`memoryview` of length `N` or a 1D integer array of
        shape `(N,)` with values in `[0, 255]`. On read, always returns `bytes`
        of length `N`.
4.  **Supported `dtype` strings**:
    -   Numeric/boolean arrays or scalars: `"float16"`, `"float32"`,
        `"float64"`, `"int8"`, `"uint8"`, `"int16"`, `"uint16"`, `"int32"`,
        `"uint32"`, `"int64"`, `"uint64"`, `"bool"`.
    -   Special scalar/sequence types: `"int"` (`shape=()`), `"str"`
        (`shape=()`), `"bytes"` (`shape=()` or `shape=(N,)`).

### Format (`shard_file_type`) & Compression Compatibility Matrix

Selecting an unsupported `(shard_file_type, compression)` pair raises an
`AssertionError` or `ValueError`:

| `shard_file_type` | Supported `compression` values | Key Notes |
| :--- | :--- | :--- |
| `"fb"` (FlatBuffers, default) | `""`, `"LZ4"`, `"ZSTD"`, `"GZIP"`, `"ZLIB"`, `"BZ2"`, `"LZMA"` | **Recommended**. Rust iterators (`as_numpy_iterator_rust*`) support `""`, `"LZ4"`, `"ZSTD"`, `"GZIP"`, `"ZLIB"` (do **not** use `"BZ2"` or `"LZMA"` with Rust iterators). Does **not** support `"ZIP"`. |
| `"npz"` (NumPy Zip) | `""`, `"ZIP"` | **Gotcha**: Default `compression="GZIP"` fails with `shard_file_type="npz"`. You **must** pass `compression="ZIP"` or `compression=""`. |
| `"tfrec"` (TFRecord) | `""`, `"GZIP"`, `"ZLIB"` | Requires `tensorflow` (`sedpack[tf]` extra). Note: `"str"` attributes are returned as UTF-8 `bytes` when read via `as_numpy_iterator*`. |

## 4. Writing Examples to a Dataset

### Strict `dtype` and `shape` Checking on Write

For `"fb"` shards, `write_example` enforces
`np.can_cast(value, attribute.dtype, casting="safe")` and exact `shape` equality
(`np.asarray(value).shape == attribute.shape`).

-   **Gotcha**: Passing a default NumPy `float64` array to a `"float32"`
    attribute, or a Python/NumPy `int64` to a `"uint8"`/`"int32"` attribute,
    raises `ValueError: Cannot cast value of dtype ...`.
-   **Fix**: Explicitly cast values before writing (for example,
    `trace.astype(np.float32)`, `np.asarray(label, dtype=np.uint8)`).

### Pattern A: Standard Single-Coordinator Writing (`dataset.filler`)

Use `with dataset.filler(concurrency=4) as dataset_filler:` to stream examples.
Full shards are serialized, compressed, and hashed asynchronously in a
background `ProcessPoolExecutor` with `max(1, concurrency)` workers.

```python
def make_example(i: int) -> ExampleT:
    return {
        "trace": np.zeros((5000,), dtype=np.float32),
        "plaintext": np.zeros((16,), dtype=np.uint8),
        "key_id": i,
        "label": np.asarray(i % 256, dtype=np.uint8),
        "raw_packet": b"variable_len_bytes",
        "fixed_nonce": b"0123456789ab",
        "notes": f"example_{i}",
    }

with dataset.filler(concurrency=4) as dataset_filler:
    for i in range(500):
        dataset_filler.write_example(
            values=make_example(i),
            split=TRAIN_SPLIT,
            custom_metadata={"device": i % 2},  # Optional per-shard metadata
        )
```

-   **Shard buffering & `custom_metadata` grouping**:
    -   Examples with different `custom_metadata` (or `None`/`{}` vs. non-empty
        `custom_metadata`) are never mixed in the same shard.
    -   Up to `max(1, concurrency)` open shards are buffered in memory
        simultaneously across `(split, custom_metadata)` pairs using an LRU
        policy. Grouping writes by `(split, custom_metadata)` avoids flushing
        partially filled shards when writing many distinct `custom_metadata`
        values.
-   **Appending to an existing dataset**:
    -   Load `dataset = Dataset(path)` and open another
        `with dataset.filler() as filler:` block. Exiting the `with` block
        merges the new shards into `<split>/shards_list.json` and
        `dataset_info.json`.
-   **Updating `dataset.metadata` or `Attribute.custom_metadata`**:
    -   `dataset.filler()` saves `dataset_info.json` on `__exit__`. If you
        modify `dataset.metadata.custom_metadata` or
        `dataset.dataset_structure.saved_data_description[i].custom_metadata`
        *after* the `with dataset.filler():` block (for example, recording
        global min/max statistics), you **must** call `dataset.write_config()`
        afterwards to persist those updates.
-   **Converting existing NPZ, HDF5, or raw trace files**:
    -   Stream records into `with dataset.filler(concurrency=4) as filler:` and
        cast each array to the schema `dtype` (`np.asarray(x, dtype=...)`).
    -   When reading `.npz`/`.npy` files, pass `allow_pickle=False`
        (`with np.load(npz_path, allow_pickle=False) as archive:`) to prevent
        unsafe `pickle` deserialization. For HDF5 (`h5py.File(h5_path, "r")`),
        slice rows incrementally rather than loading the entire file into RAM.

### Pattern B: Built-In Multiprocessing (`dataset.write_multiprocessing`)

When generating or converting examples is CPU-bound, use
`dataset.write_multiprocessing`. Each worker automatically receives an isolated
`DatasetFiller` with a unique `relative_path_from_split`.

```python
def feed_worker(
    dataset_filler: DatasetFiller,
    worker_items: list[int],
    *,
    split: SplitT,
) -> int:
    # MUST enter the context manager inside feed_writer
    with dataset_filler:
        for item in worker_items:
            dataset_filler.write_example(values=make_example(item), split=split)
    return len(worker_items)

# Spawn 2 workers (len(custom_arguments) determines the number of tasks)
counts = dataset.write_multiprocessing(
    feed_writer=feed_worker,
    custom_arguments=[([0, 1, 2],), ([3, 4, 5],)],
    custom_kwarguments=[{"split": TRAIN_SPLIT}, {"split": TEST_SPLIT}],
    consistency_check=True,
    single_process=False,
)
```

-   **Gotcha**: The first positional argument of `feed_writer` is **always**
    `dataset_filler: DatasetFiller`, and `feed_writer` **must** use
    `with dataset_filler:`.

### Pattern C: Custom Multi-Worker / Distributed Writing

If you manage your own multi-process or distributed workers:

1.  Every worker must open `Dataset(dataset_path)` and construct `DatasetFiller`
    with a **non-empty, mutually disjoint** `relative_path_from_split` (no
    worker path may be a parent prefix of another; for example, use
    `Path("worker_0")`, `Path("worker_1")`).
2.  Each worker writes inside `with filler:` and returns
    `filler.get_updated_infos()`.
3.  After all workers finish, a single coordinator calls
    `dataset.write_config(updated_infos=all_updated_infos)`.

```python
# Inside worker `worker_id`:
worker_ds = Dataset(dataset_path)
filler = DatasetFiller(
    dataset=worker_ds,
    concurrency=2,
    relative_path_from_split=Path(f"worker_{worker_id}"),
)
with filler:
    filler.write_example(values=make_example(worker_id), split=TRAIN_SPLIT)
worker_infos: list[ShardListInfo] = filler.get_updated_infos()

# On the coordinator after all workers complete:
dataset.write_config(updated_infos=all_worker_infos)
```

> **Critical**: Never use default `relative_path_from_split=Path(".")` across
> concurrent workers, even if workers write to different splits—exiting a root
> `DatasetFiller` immediately writes `dataset_info.json` and overwrites other
> splits.

## 5. Reading & Iterating a Dataset

### Critical Defaults Gotcha: `repeat=True` and `shuffle=1_000`

Every `dataset.as_*` iterator defaults to **`repeat=True` (infinite stream)**
and **`shuffle=1_000`**:

-   **For training**: Keep `repeat=True, shuffle=1_000` (or larger) and bound
    epochs with `itertools.islice(it, steps_per_epoch)` or `steps_per_epoch` in
    `model.fit`.
-   **For evaluation, validation, testing, or single-pass statistics (such as
    SNR or accuracy)**: You **must** pass **`repeat=False, shuffle=0`**.
    Otherwise a `for batch in dataset.as_numpy_iterator...(split=...)` loop
    runs forever.
-   **Keyword-only `split`**: Always pass `split=` as a keyword argument
    (`split=TRAIN_SPLIT`).

### Choosing the Right Iterator

| Method | Yields / Returns | Formats | Record Transform Arg | When to Use |
| :--- | :--- | :--- | :--- | :--- |
| `as_numpy_iterator_rust_batched` | `Iterator[BatchT]` | `"fb"` (`""`, `"LZ4"`, `"ZSTD"`, `"GZIP"`, `"ZLIB"`) | `process_batch` | **Best for JAX, PyTorch, Keras, and fast batched evaluation**. |
| `as_numpy_iterator_rust` | `Iterator[ExampleT]` | `"fb"` (`""`, `"LZ4"`, `"ZSTD"`, `"GZIP"`, `"ZLIB"`) | `process_record` | Fast unbatched iteration over `"fb"` shards. |
| `as_tfdataset` | `tf.data.Dataset` | `"tfrec"`, `"fb"`, `"npz"` | `process_record` | TensorFlow / `tf.data` pipelines. |
| `as_numpy_iterator` | `Iterator[ExampleT]` | `"fb"`, `"npz"`, `"tfrec"` | `process_record` | Simple single-process iteration, debugging, or `"npz"`/`"tfrec"` reading. |
| `as_numpy_iterator_concurrent` | `Iterator[ExampleT]` | `"fb"`, `"npz"`, `"tfrec"` | `process_record` | Multi-worker Python unbatched iteration (supports `custom_metadata_type_limit`). |
| `as_numpy_iterator_async` | `AsyncIterator[ExampleT]` | `"fb"`, `"npz"` | `process_record` | `asyncio` pipelines (`async for example in ...`). |

### 1. Fast Batched Iteration (`as_numpy_iterator_rust_batched`)

```python
# Training (infinite shuffled stream of batches)
train_batches = dataset.as_numpy_iterator_rust_batched(
    split=TRAIN_SPLIT,
    batch_size=64,
    repeat=True,
    shuffle=10_000,
    process_batch=lambda b: (b["trace"], b["label"]),
)

# Single-pass evaluation (finite, deterministic order)
for batch in dataset.as_numpy_iterator_rust_batched(
    split=HOLDOUT_SPLIT,
    batch_size=64,
    repeat=False,
    shuffle=0,
):
    traces = batch["trace"]          # np.ndarray of shape (64, 5000), dtype float32
    labels = batch["label"]          # np.ndarray of shape (64,), dtype uint8
    packets = batch["raw_packet"]    # list[bytes] of length 64
    notes = batch["notes"]           # list[str] of length 64
    key_ids = batch["key_id"]        # list[int] of length 64
```

-   **Always pass `batch_size` explicitly**: `as_numpy_iterator_rust_batched`
    defaults to `batch_size=1` (unlike `as_tfdataset`, which defaults to
    `batch_size=32`).
-   **Batched value types (`BatchT`)**:
    -   Numeric/boolean NumPy attributes have shape
        `(batch_size, *attribute.shape)`.
    -   `"bytes"`, `"str"`, and `"int"` attributes are returned as
        `list[bytes]`, `list[str]`, and `list[int]` of length `batch_size`.
-   **Trailing remainder batches**: When `repeat=False`, if the total number of
    examples `N` is not divisible by `batch_size`, the trailing
    `N % batch_size` examples are dropped by `as_numpy_iterator_rust_batched`.
    To visit every single example when `N % batch_size != 0`, use unbatched
    `as_numpy_iterator_rust(split=..., repeat=False, shuffle=0)`.

### 2. Unbatched Iteration (`as_numpy_iterator_rust`, `as_numpy_iterator`, `as_numpy_iterator_concurrent`)

```python
for example in dataset.as_numpy_iterator_rust(
    split=TEST_SPLIT,
    repeat=False,
    shuffle=0,
    process_record=None,
):
    trace = example["trace"]  # np.ndarray of shape (5000,)
    label = example["label"]  # np.ndarray scalar of shape ()
```

-   **Pickleable `process_record` in `as_numpy_iterator_concurrent`**: When
    `shuffle > 0`, `as_numpy_iterator_concurrent` executes `process_record`
    inside `multiprocessing.Pool` workers, so `process_record` **must be a
    top-level function or `functools.partial`** (not a `lambda` or local
    closure).

### 3. TensorFlow Dataset (`as_tfdataset`)

```python
tf_ds = dataset.as_tfdataset(
    split=TRAIN_SPLIT,
    batch_size=32,   # Pass batch_size=0 to disable batching
    repeat=True,
    shuffle=1_000,
)
# When reshaping or converting dictionaries to (x, y) tuples for Keras on "fb"
# or "npz" datasets, apply .map() on the returned tf.data.Dataset rather than
# changing structure inside process_record (which must match
# saved_data_description):
tf_ds = tf_ds.map(lambda ex: (ex["trace"], ex["label"]))
```

## 6. Inspecting Metadata, Filtering, Balancing, & Checking Integrity

### Inspect Example & Shard Counts Without Reading Shards

```python
dataset = Dataset(dataset_path)

# Check if split exists before indexing dataset.dataset_info.splits
if TRAIN_SPLIT in dataset.dataset_info.splits:
    split_meta = dataset.dataset_info.splits[TRAIN_SPLIT]
    print(split_meta.number_of_examples, split_meta.number_of_shards)

# Look up an Attribute schema by name
trace_attr = dataset.dataset_structure.attribute_by_name("trace")
print(trace_attr.dtype, trace_attr.shape, trace_attr.custom_metadata)
```

### Filter or Limit Shards by `ShardInfo` Metadata

All `as_*` iterators accept `shard_filter: Callable[[ShardInfo], bool]` and
`shards: int | None` (evaluated purely from `shards_list.json` without reading
shard files):

```python
device_0_it = dataset.as_numpy_iterator_rust_batched(
    split=TRAIN_SPLIT,
    batch_size=32,
    repeat=False,
    shuffle=0,
    shards=10,  # Optional: read at most the first 10 matching shards
    shard_filter=lambda shard_info: shard_info.custom_metadata.get("device") == 0,
)
```

-   **Empty filter result**: If `shard_filter` matches zero shards (or the
    split has zero shards), `as_numpy_iterator`, `as_numpy_iterator_rust`, and
    `as_numpy_iterator_rust_batched` yield zero items, whereas
    `as_numpy_iterator_concurrent`, `as_numpy_iterator_async`, `as_tfdataset`,
    and `shard_paths_dataset` raise
    `ValueError: The list of shards is empty. Try less restrictive filtering.`
-   **Limiting shards per distinct `custom_metadata`**: `as_numpy_iterator`,
    `as_numpy_iterator_concurrent`, and `as_tfdataset` also accept
    `custom_metadata_type_limit=K` to read at most `K` shards for each unique
    `ShardInfo.custom_metadata` dictionary.
-   **Iterating shard metadata directly**: Use
    `for shard_info in dataset.shard_info_iterator(TRAIN_SPLIT): ...` (pass
    `None` to iterate across all present splits).

### Balanced Iteration Across Shard Categories (`BalancedShardInfoIterator`)

To sample shards with equal (or custom-weighted) probability across metadata
attributes regardless of shard count imbalance:

```python
def by_device(shard_info: ShardInfo) -> int:
    return shard_info.custom_metadata["device"]

# Optional custom weight per category:
# by_device.weight = lambda shard_info: 3.0 if shard_info.custom_metadata["device"] == 0 else 1.0

balanced_shards = BalancedShardInfoIterator(
    dataset_path=dataset.path,
    dataset_info=dataset.dataset_info,
    split=TRAIN_SPLIT,
    repeat=True,
    shuffle=100,
    balance_by=(by_device,),
    shard_filter=None,
)

# Option A: Fast Rust batched iteration over balanced shards ("fb" format)
with RustBatchedGenerator(
    dataset_path=dataset.path,
    dataset_structure=dataset.dataset_structure,
    shard_iterator=balanced_shards,
    batch_size=64,
    process_batch=None,
    shuffle_buffer_size=1_000,
) as rust_generator:
    for batch in itertools.islice(rust_generator(), 100):
        traces = batch["trace"]

# Option B: Python unbatched iteration over balanced shards (all formats)
balanced_examples = dataset.example_iterator_from_shard_iterator(
    shard_iterator=balanced_shards,
    process_record=None,
    shuffle=1_000,
)
for example in itertools.islice(balanced_examples, 100):
    trace = example["trace"]
```

### Verify Dataset Integrity (`dataset.check()`)

```python
# Validates dataset_info.json, all shards_list.json files, shard hashes, and example counts
dataset.check(show_progressbar=False)

# Get root dataset_info.json checksums (call after write_config)
checksums: tuple[str, ...] = dataset.current_metadata_checksums()
```

## 7. Quick Troubleshooting Table

| Error / Symptom | Cause | Solution |
| :--- | :--- | :--- |
| Evaluation loop hangs / never finishes | Default `repeat=True` creates an infinite iterator | Pass `repeat=False, shuffle=0` to `dataset.as_*` |
| `ValueError: Not a known split value` | Passed `"val"` or `"validation"` as `split` | Use `"train"`, `"test"` (`TEST_SPLIT`), or `"holdout"` (`HOLDOUT_SPLIT`) |
| `ValueError: The list of shards is empty. Try less restrictive filtering.` | `shard_filter` matched zero shards or split has no shards | Relax `shard_filter` or check `split in dataset.dataset_info.splits` first |
| `ValueError: Cannot cast value of dtype ...` | Passed `float64` to `"float32"` or `int64` to `"uint8"`/`"int32"` in `"fb"` writer | Cast explicitly before `write_example`: `trace.astype(np.float32)` |
| `ValueError: Shape requires positive values` | Used `shape=()` with `0` or `-1`, such as `(-1, 128)` or `(0,)` | Use `shape=()` for scalars/variable-length `"str"`/`"bytes"`, or fixed positive dimensions |
| `AssertionError` when creating/writing `"npz"` dataset | Default `compression="GZIP"` is unsupported for `shard_file_type="npz"` | Set `compression="ZIP"` or `compression=""` in `DatasetStructure` |
| `TypeError: ... takes 1 positional argument` on `as_numpy_iterator*` | Passed `split` by position (`dataset.as_numpy_iterator("train")`) | Pass `split` as a keyword argument (`dataset.as_numpy_iterator(split="train")`) |
| `TypeError: ... got an unexpected keyword argument 'process_record'` on `as_numpy_iterator_rust_batched` | `as_numpy_iterator_rust_batched` transforms batches, not individual records | Use `process_batch=...` instead of `process_record=...` |
| Attribute `custom_metadata` changes lost after `with dataset.filler():` | `dataset.filler()` writes `dataset_info.json` upon exiting the `with` block | Call `dataset.write_config()` after mutating `dataset.metadata` or `Attribute.custom_metadata` |
| `AttributeError: Can't pickle local object` in `as_numpy_iterator_concurrent` | Passed a `lambda` or closure as `process_record` while `shuffle > 0` | Define `process_record` as a top-level function or use `as_numpy_iterator_rust` |
