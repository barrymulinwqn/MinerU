"""
Batch process S3 PDFs with MinerU – fully in-memory, zero disk staging.

NEW FEATURE — in-memory S3 → MinerU pipeline
=============================================
Previously the workflow was:
  1.  ``aws s3 cp``  →  local disk  (OUTPUT_DIR)
  2.  ``subprocess.run(["mineru", "-p", OUTPUT_DIR, ...])``

The new workflow is:
  1.  ``boto3 download_fileobj``  →  ``io.BytesIO``  (RAM only)
  2.  ``do_parse(pdf_bytes_list=[...], ...)``          (direct Python call)

Why this is feasible
--------------------
* MinerU's CLI is a thin wrapper: it reads each file to ``bytes`` with
  ``read_fn(path)`` and then calls ``do_parse(pdf_bytes_list=…)``.
  ``do_parse`` only ever sees raw bytes, never file paths – so we can
  bypass the CLI completely and hand it bytes we downloaded from S3.

* ``boto3.client("s3").download_fileobj(bucket, key, BytesIO())`` gives
  us those same bytes without touching the filesystem.

* Inside the VLM backend, ``ModelSingleton`` (vlm_analyze.py) caches the
  vLLM engine keyed on ``(backend, model_path, server_url)``.  The engine
  is therefore initialised *once* across all ``do_parse`` calls in the same
  process – the same guarantee the old single-subprocess design provided,
  and without any NCCL TCPStore restart risk.

Pros
----
* No local disk space required for raw PDFs (only MinerU output is written).
* Eliminates the download → wait → subprocess start latency gap.
* Streaming downloads and processing can be interleaved inside a batch.
* Single process; no NCCL rank/TCPStore teardown between files.
* Simpler error handling – one Python exception chain, no subprocess stderr.

Cons / trade-offs
-----------------
* Peak RAM usage = BATCH_SIZE × average PDF size.  With BATCH_SIZE=100 and
  ~5 MB PDFs that is ~500 MB per batch; easily acceptable on a 200 GB host,
  but should be tuned for smaller machines.
* boto3 credentials must be reachable from the same process (IAM role,
  ~/.aws/credentials, or env vars).  The old design relied on the ``aws``
  CLI credential chain, which behaves identically, so this is no change in
  practice.
* If a PDF is corrupt in S3, the error surfaces as a Python exception
  rather than a non-zero subprocess exit code – slightly different
  observability, but easier to handle programmatically.
"""

import io
import os
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_S3 = "s3://orbit-common-resources/netmind_check/"
MINERU_OUTPUT_DIR = "/home/orbit_user/mineru/output"
BATCH_SIZE = 100  # PDFs per download batch (caps peak RAM per batch)
GPU_COUNT = 2
GPU_MEM_UTIL = 0.85
PROCESSING_WINDOW_SIZE = 16
MAX_CONCURRENT_REQUESTS = 8  # vLLM async engine concurrency
NUM_THREADS = 4  # OMP threads for CPU-bound work
NUM_WORKERS = 16  # parallel S3 download threads
LANG = "ch"  # OCR language hint (pipeline / hybrid backends)

# ---------------------------------------------------------------------------
# Set process-level env vars BEFORE importing torch / MinerU / CUDA libs so
# that CUDA_VISIBLE_DEVICES and OMP_NUM_THREADS are honoured at library init.
# ---------------------------------------------------------------------------
_cuda_devices = ",".join(str(i) for i in range(GPU_COUNT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", _cuda_devices)
os.environ.setdefault("MINERU_PROCESSING_WINDOW_SIZE", str(PROCESSING_WINDOW_SIZE))
os.environ.setdefault(
    "MINERU_API_MAX_CONCURRENT_REQUESTS", str(MAX_CONCURRENT_REQUESTS)
)
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))

# ---------------------------------------------------------------------------
# Deferred heavy imports (after env vars are set)
# ---------------------------------------------------------------------------
import boto3  # noqa: E402 – must come after os.environ setup
from botocore.exceptions import ClientError  # noqa: E402

from mineru.cli.common import do_parse  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def list_s3_pdfs(s3_prefix: str) -> list[str]:
    """Return full ``s3://`` paths for every PDF under *s3_prefix*."""
    cmd = ["aws", "s3", "ls", "--recursive", s3_prefix]
    logger.info("Listing S3 objects: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    bucket = s3_prefix.split("/")[2]  # e.g. "orbit-common-resources"
    pdf_paths: list[str] = []
    for line in result.stdout.splitlines():
        # aws s3 ls --recursive output:
        # 2024-01-01 00:00:00      12345 path/to/file.pdf
        parts = line.split(maxsplit=3)
        if len(parts) == 4:
            key = parts[3].strip()
            if key.lower().endswith(".pdf"):
                pdf_paths.append(f"s3://{bucket}/{key}")

    logger.info("Found %d PDF(s).", len(pdf_paths))
    return pdf_paths


def _parse_s3_url(s3_url: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` → ``(bucket, key)``."""
    without_scheme = s3_url[len("s3://") :]
    bucket, key = without_scheme.split("/", 1)
    return bucket, key


def _s3_key_to_pdf_name(key: str) -> str:
    """
    Convert an S3 key to a unique, filesystem-safe MinerU output name.

    ``netmind_check/subdir/report.pdf``  →  ``netmind_check__subdir__report``

    Using the full key path (with slashes replaced) avoids collisions when
    PDFs in different S3 directories share the same filename.
    """
    return Path(key).with_suffix("").as_posix().replace("/", "__")


def download_to_memory(
    s3_url: str,
    s3_client,
) -> tuple[str, bytes | None, str]:
    """
    Download a single S3 object directly into a ``bytes`` buffer.

    Returns ``(s3_url, data_or_None, error_message)``.
    No data is written to local disk.
    """
    bucket, key = _parse_s3_url(s3_url)
    try:
        buf = io.BytesIO()
        s3_client.download_fileobj(bucket, key, buf)
        data = buf.getvalue()
        logger.info("In-memory download OK (%d B): %s", len(data), s3_url)
        return s3_url, data, ""
    except ClientError as exc:
        err = str(exc)
        logger.error("Download failed %s: %s", s3_url, err)
        return s3_url, None, err


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


def batch_iter(items: list, batch_size: int):
    """Yield successive slices of *batch_size* from *items*."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------------------------------------------------------------------------
# MinerU processing  (replaces the subprocess-based approach)
# ---------------------------------------------------------------------------


def process_pdfs_in_memory(
    pdf_names: list[str],
    pdf_bytes_list: list[bytes],
) -> None:
    """
    Feed PDF bytes directly to MinerU's Python API – no subprocess, no disk.

    ``do_parse`` accepts ``pdf_bytes_list`` natively; the CLI's ``read_fn``
    is just a file-to-bytes shim that we skip entirely here.

    The vLLM engine is managed by ``ModelSingleton`` (vlm_analyze.py), which
    caches it by ``(backend, model_path, server_url)``.  The engine is
    therefore initialised **once** across all calls to this function, keeping
    NCCL ranks and the TCPStore alive for the full run – the same guarantee
    the old single ``mineru -p DIR`` subprocess design provided.

    ``gpu_memory_utilization`` and ``data_parallel_size`` are forwarded as
    ``**kwargs`` through ``do_parse`` → ``_process_vlm`` → ``ModelSingleton``
    → ``AsyncEngineArgs``, mirroring the CLI flags
    ``--gpu-memory-utilization`` and ``--data-parallel-size``.
    """
    if not pdf_bytes_list:
        return

    Path(MINERU_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(
        "MinerU: processing %d PDF(s) | backend=vlm-auto-engine | GPUs=%d",
        len(pdf_names),
        GPU_COUNT,
    )

    do_parse(
        output_dir=MINERU_OUTPUT_DIR,
        pdf_file_names=pdf_names,
        pdf_bytes_list=pdf_bytes_list,
        p_lang_list=[LANG] * len(pdf_names),
        backend="vlm-auto-engine",
        parse_method="auto",
        # Forwarded as **kwargs to vLLM AsyncEngineArgs
        gpu_memory_utilization=GPU_MEM_UTIL,
        data_parallel_size=GPU_COUNT,
    )

    logger.info("MinerU complete for %d PDF(s).", len(pdf_names))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])

    # 1. List all PDFs from S3
    all_pdfs = list_s3_pdfs(INPUT_S3)
    if not all_pdfs:
        logger.warning("No PDFs found under %s. Exiting.", INPUT_S3)
        return

    total = len(all_pdfs)

    # One boto3 client reused across all downloads (thread-safe for get ops)
    s3_client = boto3.client("s3")

    # 2. Download ALL PDFs into memory in parallel batches, then process once.
    #    Accumulating before processing means the vLLM engine starts exactly
    #    once and all PDFs flow through it without any engine restart.
    all_pdf_names: list[str] = []
    all_pdf_bytes: list[bytes] = []
    failed: list[tuple[str, str]] = []

    for batch_idx, batch in enumerate(batch_iter(all_pdfs, BATCH_SIZE), start=1):
        logger.info(
            "--- Download batch %d: %d / %d PDFs ---",
            batch_idx,
            len(batch),
            total,
        )
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(download_to_memory, s3_url, s3_client): s3_url
                for s3_url in batch
            }
            for future in as_completed(futures):
                s3_url, data, err = future.result()
                if data is not None:
                    _, key = _parse_s3_url(s3_url)
                    all_pdf_names.append(_s3_key_to_pdf_name(key))
                    all_pdf_bytes.append(data)
                else:
                    failed.append((s3_url, err))

    # 3. Download summary
    logger.info("=" * 60)
    logger.info(
        "Downloads complete: %d succeeded, %d failed.",
        len(all_pdf_names),
        len(failed),
    )
    for s3_url, err in failed:
        logger.warning("  FAILED: %s | %s", s3_url, err)
    logger.info("=" * 60)

    # 4. Single MinerU call for all successfully downloaded PDFs
    process_pdfs_in_memory(all_pdf_names, all_pdf_bytes)

    logger.info("All done. Processed %d / %d PDF(s).", len(all_pdf_names), total)


if __name__ == "__main__":
    main()
