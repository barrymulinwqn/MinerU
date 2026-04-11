"""
调有代码 最终版
Batch download PDF files from S3 to local storage.
Lists all PDFs under INPUT_S3 and downloads them in parallel workers.
"""

import os
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_S3 = "s3://orbit-common-resources/netmind_check/"
# OUTPUT_S3 = "s3://orbit-common-resources/netmind_check/output/"
LOCAL_ROOT = "/home/orbit_user/mineru"
OUTPUT_DIR = "/home/orbit_user/mineru/netmind_check"
MINERU_OUTPUT_DIR = "/home/orbit_user/mineru/output"
BATCH_SIZE = 100  # number of files per batch
GPU_COUNT = 2
GPU_MEM_UTIL = 0.85
PROCESSING_WINDOW_SIZE = 16
MAX_CONCURRENT_REQUESTS = (
    8  # mineru API max concurrent requests (vLLM async engine concurrency)
)
NUM_THREADS = 4  # threads per mineru worker for CPU-bound tasks
NUM_WORKERS = 16  # parallel download threads

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
# Helpers
# ---------------------------------------------------------------------------


def list_s3_pdfs(s3_prefix: str) -> list[str]:
    """Return a list of full s3:// paths for every PDF under s3_prefix."""
    cmd = ["aws", "s3", "ls", "--recursive", s3_prefix]
    logger.info("Listing S3 objects: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    pdf_paths = []
    bucket = s3_prefix.split("/")[2]  # e.g. "orbit-common-resources"
    for line in result.stdout.splitlines():
        # aws s3 ls --recursive output format:
        # 2024-01-01 00:00:00      12345 path/to/file.pdf
        parts = line.split(maxsplit=3)
        if len(parts) == 4:
            key = parts[3].strip()
            if key.lower().endswith(".pdf"):
                pdf_paths.append(f"s3://{bucket}/{key}")

    logger.info("Found %d PDF file(s).", len(pdf_paths))
    return pdf_paths


def s3_key_to_local_path(s3_path: str, local_root: str) -> str:
    """
    Map an s3:// path to a local file path, preserving the key structure
    relative to the bucket root.

    Example:
        s3://orbit-common-resources/netmind_check/foo/bar.pdf
        -> /home/orbit_user/mineru/netmind_check/foo/bar.pdf
    """
    # Strip scheme and bucket name, keep everything after
    without_scheme = s3_path[len("s3://") :]  # bucket/key
    key = without_scheme.split("/", 1)[1]  # key only
    return os.path.join(local_root, key)


def download_one(s3_path: str, local_path: str) -> tuple[str, bool, str]:
    """
    Download a single S3 object to local_path.
    Returns (s3_path, success, error_message).
    """
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(local_path).exists():
        logger.debug("Already exists, skipping: %s", local_path)
        return s3_path, True, ""

    cmd = ["aws", "s3", "cp", s3_path, local_path]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Downloaded: %s -> %s", s3_path, local_path)
        return s3_path, True, ""
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.strip()
        logger.error("Failed: %s | %s", s3_path, err)
        return s3_path, False, err


def batch_iter(items: list, batch_size: int):
    """Yield successive batches of `batch_size` from `items`."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def list_local_pdfs(directory: str) -> list[str]:
    """Return a sorted list of absolute paths for every PDF under *directory*."""
    pdf_paths = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, fname))
    pdf_paths.sort()
    logger.info("Found %d local PDF file(s) under %s.", len(pdf_paths), directory)
    return pdf_paths


def process_local_pdfs_with_mineru():
    """
    Run mineru once on the entire OUTPUT_DIR directory.

    Passing a directory to ``mineru -p`` processes all PDFs inside it with a
    single engine initialisation, keeping the vLLM async engine (and its NCCL
    rank0/rank1 TCPStore) alive for the whole batch.  This avoids the
    per-PDF engine restart that causes the "TCPStore server has shut down too
    early" NCCL crash.

    All available GPUs are exposed via CUDA_VISIBLE_DEVICES and
    ``--data-parallel-size`` is set to GPU_COUNT so that every GPU is used.
    """
    pdf_files = list_local_pdfs(OUTPUT_DIR)
    if not pdf_files:
        logger.warning(
            "No local PDF files found under %s. Skipping mineru step.", OUTPUT_DIR
        )
        return

    Path(MINERU_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Expose all GPUs (fix: min(1, GPU_COUNT-1) always capped at 1 regardless
    # of GPU_COUNT, so GPUs 2-N were never visible).
    cuda_devices = ",".join(str(i) for i in range(GPU_COUNT))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["MINERU_PROCESSING_WINDOW_SIZE"] = str(PROCESSING_WINDOW_SIZE)
    env["MINERU_API_MAX_CONCURRENT_REQUESTS"] = str(MAX_CONCURRENT_REQUESTS)
    env["OMP_NUM_THREADS"] = str(NUM_THREADS)

    # Process the whole directory in one mineru call so the vLLM engine is
    # initialised exactly once and all PDFs are fed through it sequentially.
    cmd = [
        "mineru",
        "-p",
        OUTPUT_DIR,
        "-o",
        MINERU_OUTPUT_DIR,
        "-b",
        "vlm-auto-engine",
        "--gpu-memory-utilization",
        str(GPU_MEM_UTIL),
        "--data-parallel-size",
        str(GPU_COUNT),
    ]
    logger.info(
        "Running mineru on directory: CUDA_VISIBLE_DEVICES=%s %s",
        cuda_devices,
        env["MINERU_PROCESSING_WINDOW_SIZE"],
        " ".join(cmd),
    )
    try:
        subprocess.run(cmd, env=env, check=True)
        logger.info("mineru complete for directory: %s", OUTPUT_DIR)
    except subprocess.CalledProcessError as exc:
        logger.error("mineru failed: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # 1. List all PDFs
    all_pdfs = list_s3_pdfs(INPUT_S3)
    if not all_pdfs:
        logger.warning("No PDF files found under %s. Exiting.", INPUT_S3)
        return

    total = len(all_pdfs)
    succeeded, failed = 0, []

    # 2. Process in batches
    for batch_idx, batch in enumerate(batch_iter(all_pdfs, BATCH_SIZE), start=1):
        batch_total = len(batch)
        logger.info(
            "--- Batch %d: processing %d / %d files ---",
            batch_idx,
            batch_total,
            total,
        )

        # Build (s3_path, local_path) pairs for this batch
        work_items = [
            (s3_path, s3_key_to_local_path(s3_path, LOCAL_ROOT)) for s3_path in batch
        ]

        # 3. Download in parallel with NUM_WORKERS threads
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(download_one, s3_path, local_path): s3_path
                for s3_path, local_path in work_items
            }
            for future in as_completed(futures):
                s3_path, ok, err = future.result()
                if ok:
                    succeeded += 1
                else:
                    failed.append((s3_path, err))

    # 4. Summary
    logger.info("=" * 60)
    logger.info("Download complete. Success: %d / %d", succeeded, total)
    if failed:
        logger.warning("Failed (%d):", len(failed))
        for s3_path, err in failed:
            logger.warning("  %s | %s", s3_path, err)
    logger.info("=" * 60)

    # 5. Process downloaded PDFs with mineru
    process_local_pdfs_with_mineru()


if __name__ == "__main__":
    main()
