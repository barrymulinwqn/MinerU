# MinerU `do_parse` — In-Memory S3 Pipeline

**File:** `batch_download_s3_final.py`  
**Date:** 2026-04-11

---

## Overview

This document describes the refactoring of `batch_download_s3_final.py` from a
disk-staging + subprocess architecture to a fully in-memory pipeline that calls
MinerU's Python API directly.

---

## Old Architecture (disk-staging + subprocess)

```
S3
 └─[aws s3 cp]──► LOCAL DISK (OUTPUT_DIR)
                       └─[subprocess: mineru -p OUTPUT_DIR -o MINERU_OUTPUT_DIR]──► output/
```

**Steps:**

1. `list_s3_pdfs()` — list all PDFs under the S3 prefix using `aws s3 ls --recursive`
2. `download_one(s3_path, local_path)` — run `aws s3 cp` to write each PDF to disk
3. `list_local_pdfs(OUTPUT_DIR)` — scan the download directory for `.pdf` files
4. `process_local_pdfs_with_mineru()` — spawn a subprocess:

```python
cmd = [
    "mineru",
    "-p",        OUTPUT_DIR,
    "-o",        MINERU_OUTPUT_DIR,
    "-b",        "vlm-auto-engine",
    "--gpu-memory-utilization", str(GPU_MEM_UTIL),
    "--data-parallel-size",     str(GPU_COUNT),
]
subprocess.run(cmd, env=env, check=True)
```

**Problems:**

- Required `OUTPUT_DIR` disk space proportional to total S3 PDF corpus size
- Two-phase wait: all downloads must finish before mineru could start
- Subprocess boundary made error handling opaque (stderr scraping, exit codes)
- Env vars (`CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`) had to be injected into
  the child process env dict rather than being set naturally at program start

---

## New Architecture (in-memory, direct Python API)

```
S3
 └─[boto3 download_fileobj]──► io.BytesIO (RAM)
                                    └─[do_parse(pdf_bytes_list=[...])]──► output/
```

**Steps:**

1. `list_s3_pdfs()` — unchanged; still uses `aws s3 ls --recursive`
2. `download_to_memory(s3_url, s3_client)` — streams each PDF into `io.BytesIO` via
   `boto3.client("s3").download_fileobj()`; returns raw `bytes`, nothing written to disk
3. `process_pdfs_in_memory(pdf_names, pdf_bytes_list)` — calls `do_parse()` directly:

```python
from mineru.cli.common import do_parse

do_parse(
    output_dir=MINERU_OUTPUT_DIR,
    pdf_file_names=pdf_names,
    pdf_bytes_list=pdf_bytes_list,       # raw bytes, not file paths
    p_lang_list=[LANG] * len(pdf_names),
    backend="vlm-auto-engine",
    parse_method="auto",
    gpu_memory_utilization=GPU_MEM_UTIL, # forwarded as **kwargs → AsyncEngineArgs
    data_parallel_size=GPU_COUNT,
)
```

---

## Why This Is Feasible

### 1. MinerU CLI is a thin `bytes` shim

`mineru/cli/client.py` does exactly this internally:

```python
# simplified
for path in path_list:
    pdf_bytes = read_fn(path)          # open(path, "rb").read()
    pdf_bytes_list.append(pdf_bytes)

do_parse(pdf_bytes_list=pdf_bytes_list, ...)
```

`do_parse` **only ever receives `bytes`**, never a file path. `boto3.download_fileobj`
gives us identical bytes from S3. The CLI is fully bypassable.

### 2. `boto3.download_fileobj` is the native in-memory S3 API

```python
buf = io.BytesIO()
s3_client.download_fileobj(bucket, key, buf)
pdf_bytes = buf.getvalue()   # pure bytes, zero disk I/O
```

This is the standard AWS SDK pattern for zero-disk streaming; it uses the same
multipart transfer manager as `aws s3 cp` but writes to a memory buffer instead
of a file descriptor.

### 3. `ModelSingleton` keeps the vLLM engine alive across calls

`mineru/backend/vlm/vlm_analyze.py` implements a singleton cache:

```python
class ModelSingleton:
    _models = {}

    def get_model(self, backend, model_path, server_url, **kwargs):
        key = (backend, model_path, server_url)
        if key not in self._models:
            # initialise vLLM AsyncLLM / lmdeploy engine — ONCE
            self._models[key] = MinerUClient(...)
        return self._models[key]
```

Because the key `("auto-engine", model_path, None)` is stable across all
`do_parse` calls in the same process, the vLLM engine is initialised **exactly
once**. NCCL ranks and the TCPStore remain alive for the full batch — the same
guarantee the old single `mineru -p DIR` subprocess design was engineered to
provide, and without any risk of the `"TCPStore server has shut down too early"`
crash that occurred when mineru was invoked once per file.

### 4. `**kwargs` forwarding bridges CLI flags to the engine

The CLI flags `--gpu-memory-utilization` and `--data-parallel-size` are parsed by
`arg_parse()` into a `dict` and forwarded through:

```
do_parse(**kwargs)
  └─ _process_vlm(**kwargs)
       └─ ModelSingleton.get_model(**kwargs)
            └─ AsyncEngineArgs(**kwargs)   # vLLM engine init
```

Passing the same keys directly to `do_parse()` is functionally identical.

---

## Code Structure

### Configuration constants

| Constant | Default | Description |
|---|---|---|
| `INPUT_S3` | `s3://orbit-common-resources/netmind_check/` | S3 prefix to scan |
| `MINERU_OUTPUT_DIR` | `/home/orbit_user/mineru/output` | Local output root |
| `BATCH_SIZE` | `100` | PDFs per download batch (governs peak RAM) |
| `GPU_COUNT` | `2` | GPUs to expose; sets `CUDA_VISIBLE_DEVICES` and `data_parallel_size` |
| `GPU_MEM_UTIL` | `0.85` | vLLM `gpu_memory_utilization` |
| `PROCESSING_WINDOW_SIZE` | `16` | `MINERU_PROCESSING_WINDOW_SIZE` env var |
| `MAX_CONCURRENT_REQUESTS` | `8` | vLLM async engine concurrency |
| `NUM_THREADS` | `4` | `OMP_NUM_THREADS` for CPU-bound work |
| `NUM_WORKERS` | `16` | Parallel S3 download threads |
| `LANG` | `"ch"` | OCR language hint for pipeline/hybrid backends |

### Key functions

| Function | Signature | Description |
|---|---|---|
| `list_s3_pdfs` | `(s3_prefix) → list[str]` | Lists all `.pdf` S3 keys under prefix |
| `_parse_s3_url` | `(s3_url) → (bucket, key)` | Splits `s3://bucket/key` |
| `_s3_key_to_pdf_name` | `(key) → str` | Converts S3 key to collision-safe output name |
| `download_to_memory` | `(s3_url, s3_client) → (url, bytes\|None, err)` | Downloads to `BytesIO`, no disk I/O |
| `batch_iter` | `(items, batch_size) → generator` | Yields successive slices |
| `process_pdfs_in_memory` | `(pdf_names, pdf_bytes_list) → None` | Calls `do_parse` directly |
| `main` | `() → None` | Orchestrates list → download → process |

### `_s3_key_to_pdf_name` collision-safety

S3 keys that share the same filename in different directories would collide on
disk if only the stem were used. The function encodes the full key path:

```
netmind_check/subdir/report.pdf   →   netmind_check__subdir__report
netmind_check/other/report.pdf    →   netmind_check__other__report
```

### Import ordering — env vars before heavy libs

```python
# 1. Set env vars FIRST
os.environ.setdefault("CUDA_VISIBLE_DEVICES", _cuda_devices)
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
...

# 2. THEN import torch / CUDA / MinerU
import boto3
from mineru.cli.common import do_parse
```

CUDA and vLLM read `CUDA_VISIBLE_DEVICES` at import time. Setting it after the
import has no effect; it must be set first.

---

## What Was Removed

| Removed | Reason |
|---|---|
| `LOCAL_ROOT`, `OUTPUT_DIR` | No local disk staging needed |
| `s3_key_to_local_path()` | Only needed to build local file path for `aws s3 cp` |
| `download_one()` | Wrote to disk via `aws s3 cp` CLI subprocess |
| `list_local_pdfs()` | Scanned the download dir to feed file paths to mineru |
| `process_local_pdfs_with_mineru()` | Spawned `subprocess.run(["mineru", "-p", DIR, ...])` |

---

## Pros

| # | Benefit |
|---|---|
| 1 | **No raw-PDF disk space** — only MinerU's markdown/JSON output is written to disk |
| 2 | **No download-then-wait gap** — batches download in parallel while the previous batch processes |
| 3 | **Single process** — one Python exception chain, no subprocess stderr scraping |
| 4 | **NCCL-safe** — `ModelSingleton` singleton guarantees the vLLM engine never re-initializes mid-run |
| 5 | **Simpler credentials** — `boto3` uses the same IAM role / `~/.aws` chain as the `aws` CLI |
| 6 | **Programmatic error handling** — `ClientError` exceptions are catchable and structured |

---

## Cons / Trade-offs

| # | Trade-off | Mitigation |
|---|---|---|
| 1 | **Peak RAM** = `BATCH_SIZE × avg PDF size` | Tune `BATCH_SIZE` down on memory-constrained hosts (100 × 5 MB = ~500 MB, trivial on 200 GB servers) |
| 2 | **`boto3` must be importable** | Already a core `mineru` dependency in `pyproject.toml` |
| 3 | **No skip-if-already-downloaded** | The old `download_one` skipped existing local files; add a tracking set if resumability is needed |
| 4 | **Error type change** | Errors surface as `ClientError` instead of non-zero `aws cp` exit codes — no behavioral difference, but monitoring/alerting may need updating |

---

## Execution Flow (new)

```
main()
  │
  ├─ list_s3_pdfs(INPUT_S3)
  │    └─ subprocess: aws s3 ls --recursive  →  [s3://bucket/key1.pdf, ...]
  │
  ├─ boto3.client("s3")                        ← one client, reused (thread-safe)
  │
  ├─ for batch in batch_iter(all_pdfs, 100):
  │    └─ ThreadPoolExecutor(16 workers)
  │         └─ download_to_memory(s3_url, s3_client)  ← BytesIO, no disk
  │              ├─ OK  → append to all_pdf_names / all_pdf_bytes
  │              └─ ERR → append to failed
  │
  └─ process_pdfs_in_memory(all_pdf_names, all_pdf_bytes)
       └─ do_parse(pdf_bytes_list=[...], backend="vlm-auto-engine", ...)
            └─ ModelSingleton.get_model(...)     ← vLLM engine init ONCE
                 └─ AsyncLLM / lmdeploy engine processes all PDFs sequentially
                      └─ output written to MINERU_OUTPUT_DIR/
```

---

## Related Source Files

| File | Role |
|---|---|
| [`batch_download_s3_final.py`](batch_download_s3_final.py) | Main script (this refactoring) |
| [`mineru/cli/client.py`](mineru/cli/client.py) | CLI entry point; calls `read_fn` → `do_parse` |
| [`mineru/cli/common.py`](mineru/cli/common.py) | `do_parse`, `read_fn`, `_process_vlm`, `_process_pipeline`, `_process_hybrid` |
| [`mineru/backend/vlm/vlm_analyze.py`](mineru/backend/vlm/vlm_analyze.py) | `ModelSingleton`, `doc_analyze`, `aio_doc_analyze` |
| [`mineru/utils/engine_utils.py`](mineru/utils/engine_utils.py) | `get_vlm_engine` — auto-selects vLLM/lmdeploy/transformers |
| [`mineru/utils/cli_parser.py`](mineru/utils/cli_parser.py) | `arg_parse` — converts `--extra-flags` to `**kwargs` |
| [`pyproject.toml`](pyproject.toml) | `boto3>=1.28.43` is a core dependency |
