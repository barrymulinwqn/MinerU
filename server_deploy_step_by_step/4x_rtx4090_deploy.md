# MinerU Production Deployment — 4× RTX 4090 Server

## Target Hardware

| Component | Specification |
|---|---|
| **CPU** | AMD Ryzen Threadripper PRO 5955WX (16 cores / 32 threads) |
| **GPU** | NVIDIA GeForce RTX 4090 × 4 |
| **GPU VRAM** | 24 GB per card — **96 GB total** |
| **CPU Memory** | 256 GB DDR4 ECC |
| **System Disk** | 4 TB NVMe |
| **Network** | Aquantia AQC113C 10 GbE + Intel I210 1 GbE |

---

## Architecture Decision & Rationale

### Why Split the Stack Into Two Tiers

MinerU has two fundamentally different compute workloads:

| Tier | Workload | Bottleneck |
|---|---|---|
| **VLM inference** | The 1.2B-parameter Qwen2-VL model (`MinerU2.5-2509-1.2B`) processes page images through a neural network | GPU compute + VRAM bandwidth |
| **Pipeline post-processing** | OCR, layout parsing, formula detection/recognition, table recognition, reading-order sorting | CPU + moderate GPU (CV models) |

Running both in one process on one GPU wastes three idle GPUs. The architecture below separates concerns:

```
                         ┌─────────────────────────────────────────────┐
                         │               Server (256 GB RAM)            │
                         │                                              │
  Clients (PDF/image)    │  ┌─────────────────────────────────────────┐ │
  ──────────────────►    │  │  Load Balancer (nginx / mineru process) │ │
  10 GbE AQC113C         │  └───────────────────┬─────────────────────┘ │
                         │                      │                       │
                         │         ┌────────────┼────────────┐          │
                         │         ▼            ▼            ▼          │
                         │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
                         │  │ Worker 0 │ │ Worker 1 │ │ Worker N │...  │
                         │  │ GPU 0    │ │ GPU 1    │ │ GPU 2/3  │     │
                         │  │ pipeline │ │ pipeline │ │ hybrid   │     │
                         │  └──────────┘ └──────────┘ └──────────┘     │
                         │                                              │
                         │  ┌───────────────────────────────────────┐  │
                         │  │  VLM OpenAI-Compatible Server (vllm)  │  │
                         │  │  GPUs 2–3, data-parallel-size=2       │  │
                         │  │  port 30000                           │  │
                         │  └───────────────────────────────────────┘  │
                         └─────────────────────────────────────────────┘
```

### Recommended Deployment Strategy: Hybrid Split Mode

**GPU 0–1: Pipeline workers** (LitServe, 2 workers per GPU = 4 parallel pipeline jobs)  
**GPU 2–3: VLM/Hybrid backend** (vllm data-parallel-size=2, serving the hybrid-http-client workers)

**Why this split maximises throughput on this hardware:**

1. **RTX 4090 has 24 GB VRAM.** `gpu_memory_utilization=0.80` reserves 80% of total card VRAM for the **entire vllm process** (model weights + KV cache combined): 24 × 0.80 = **19.2 GB** total vllm budget. The ~2.5 GB loaded model weights are subtracted from that budget, leaving ~**16.7 GB** for the KV cache; the remaining ~4.8 GB covers the CUDA context and PyTorch allocator overhead. Pipeline CV models consume an additional ~1–2 GB on GPU 0–1. There is ample headroom for multiple concurrent workers per GPU.

2. **16 CPU cores / 32 threads.** The pipeline post-processing (OCR text assembly, table merging, reading-order, markdown rendering) is CPU-bound. 16 cores can comfortably serve 4–8 concurrent pipeline workers without saturation.

3. **256 GB RAM.** No concern about RAM pressure even when hosting 8+ workers and model checkpoints in shared memory.

4. **vllm data-parallel-size=2 on GPU 2–3** gives the VLM server two independent inference streams. Each stream independently handles page batches, doubling VLM throughput. This is superior to tensor-parallel (which splits one model across GPUs) because the 1.2B model fits easily on one card — tensor parallelism would only add synchronisation overhead.

5. **10 GbE network (AQC113C)** supports ~1.25 GB/s ingress — sufficient for high-frequency PDF uploads. Bind the production API to this interface.

---

## Step 0 — OS & Driver Prerequisites

```bash
# 1. Verify Ubuntu 22.04 LTS or 24.04 LTS
lsb_release -a

# 2. Install NVIDIA driver 570+ (Ada Lovelace / RTX 4090 requires ≥ 525)
sudo apt-get install -y linux-headers-$(uname -r)
```

**Option A — Ubuntu package (simplest):**

```bash
sudo apt-get install -y nvidia-driver-570

# The nouveau open-source driver must be explicitly blacklisted.
# The apt package creates /etc/modprobe.d/nvidia-installer-disable-nouveau.conf
# but on some Ubuntu variants this is insufficient.
echo 'blacklist nouveau\noptions nouveau modeset=0' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u

# A full reboot is REQUIRED — the NVIDIA kernel module is not active until then.
sudo reboot
# After reboot, confirm nouveau is gone and NVIDIA is loaded:
lsmod | grep nouveau   # should return empty
nvidia-smi             # should show 4× RTX 4090
```

**Option B — NVIDIA runfile (recommended for production, cleaner uninstall path):**

```bash
# Download from https://www.nvidia.com/Download/index.aspx
# sudo bash NVIDIA-Linux-x86_64-570.*.run --no-opengl-files
# sudo reboot
```

```bash
# 3. Verify all 4 GPUs visible after reboot
nvidia-smi
# Expected: 4× RTX 4090, driver ≥ 525, CUDA ≥ 12.1

# 4. Install CUDA Toolkit 12.1+ (for building custom CUDA ops if needed)
sudo apt-get install -y cuda-toolkit-12-4

# 5. Enable persistence mode (prevents GPU init latency per request)
sudo nvidia-smi -pm 1

# 6. Set GPU power limit and clocks for sustained throughput
sudo nvidia-smi --power-limit=450 -i 0,1,2,3     # RTX 4090 TDP is 450W
sudo nvidia-smi --auto-boost-default=0 -i 0,1,2,3
```

---

## Step 1 — System Dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    fonts-noto-core fonts-noto-cjk fontconfig \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential git curl wget htop nvtop \
    nginx

# Rebuild font cache (required for PDF rendering)
sudo fc-cache -fv
```

---

## Step 2 — Python Environment

```bash
# Create a dedicated venv (avoid system-wide pip installs)
python3.11 -m venv /opt/mineru/venv
source /opt/mineru/venv/bin/activate

# Upgrade pip and install uv for fast dependency resolution
pip install --upgrade pip
pip install uv

# Install MinerU core (layout models, OCR, pipeline stack)
uv pip install -U "mineru[core]"

# Install vllm explicitly — mineru[core] does NOT pull it in automatically.
# Pin to the same version used in the production Dockerfile.
# Verify the pin first: grep 'vllm' docker/global/Dockerfile
uv pip install "vllm==0.10.1.1"

# Install LitServe for the multi-GPU worker server
uv pip install litserve aiohttp loguru

# Verify vllm can see all 4 GPUs
python -c "import torch; print(torch.cuda.device_count())"
# Expected: 4
```

---

## Step 3 — Download Models

```bash
activate /opt/mineru/venv
source /opt/mineru/venv/bin/activate

# Set a shared model cache directory BEFORE downloading.
# This ensures the paths written into mineru.json point to a location
# that the mineru service user can read (ownership set below).
export HF_HOME=/opt/mineru/models
sudo mkdir -p /opt/mineru/models

# Download all models (pipeline + VLM)
mineru-models-download -s huggingface -m all

# Verify the config was written correctly
cat ~/.mineru.json
# "models-dir" paths should reference /opt/mineru/models/...

# Stage the config in the shared location used by the systemd service user.
# The environment variable MINERU_TOOLS_CONFIG_JSON (set in systemd unit files
# in Step 9) points here.
sudo cp ~/.mineru.json /opt/mineru/mineru.json

# Grant the mineru service user ownership of all shared assets.
sudo chown -R mineru:mineru /opt/mineru/models /opt/mineru/mineru.json
```

> **Tip — China network:** Replace `-s huggingface` with `-s modelscope` and set:
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

**Disk space required:**
| Models | Size |
|---|---|
| `PDF-Extract-Kit-1.0` (pipeline models) | ~1.5 GB |
| `MinerU2.5-2509-1.2B` (VLM) | ~2.5 GB |
| **Total** | **~4 GB** |

---

## Step 4 — Deployment Architecture Detail

### 4.1 Service Layout on 4 GPUs

```
GPU 0  →  LitServe Worker A  (pipeline backend, CPU rendering threads 0–7)
GPU 1  →  LitServe Worker B  (pipeline backend, CPU rendering threads 8–15)
GPU 2  →  vllm data-parallel shard 0  (VLM inference)
GPU 3  →  vllm data-parallel shard 1  (VLM inference)
```

The LitServe workers use `hybrid-http-client` backend, pointing at the vllm server running on GPUs 2–3. This means:

- **Concurrency:** 2 pipeline workers (one per GPU) handle page rendering and OCR simultaneously
- **VLM server:** vllm with `data-parallel-size=2` dispatches page-image batches across GPU 2 and GPU 3
- **CPU:** 16 cores / 32 threads handle PDF rendering (`MINERU_PDF_RENDER_THREADS=8` per worker) and post-processing

> **Why not 4 pipeline workers (one per GPU)?**  
> The VLM model is the throughput bottleneck per document. Dedicating 2 GPUs to vllm and running them in data-parallel is more efficient than 4 isolated pipeline+VLM instances where each GPU waits for its own VLM inference. Data-parallel vllm shares the KV cache scheduler and can pipeline across the two shards.

> **Optional tuning:** For pure-text/native PDFs where VLM is rarely invoked (parse_method=`txt`), switch all 4 GPUs to LitServe pipeline workers with `workers_per_device=2` for maximum OCR throughput.

---

## Step 5 — Start the VLM Server (GPUs 2–3)

```bash
source /opt/mineru/venv/bin/activate

# Pin VLM server to GPU 2 and 3
CUDA_VISIBLE_DEVICES=2,3 \
MINERU_MODEL_SOURCE=local \
mineru-openai-server \
    --host 127.0.0.1 \
    --port 30000 \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.80
```

**Parameter rationale:**

| Parameter | Value | Reason |
|---|---|---|
| `CUDA_VISIBLE_DEVICES=2,3` | GPU 2 and 3 | Isolate VLM server; GPU 0–1 reserved for pipeline |
| `--data-parallel-size 2` | 2 | One shard per GPU; doubles VLM throughput vs single GPU |
| `--gpu-memory-utilization` | 0.80 | 24 × 0.80 = 19.2 GB total vllm budget (weights + KV cache). ~2.5 GB used by model weights → ~16.7 GB for KV cache. Remaining ~4.8 GB for CUDA context overhead. |
| `--host 127.0.0.1` | loopback | Never expose VLM inference port to public network |

**Health check:**

```bash
curl -s http://127.0.0.1:30000/health
# {"status":"healthy"}
```

---

## Step 6 — Start the Pipeline Worker Server (GPUs 0–1, LitServe)

Create `/opt/mineru/server.py`:

```python
import os
import uuid
import base64
import tempfile
from pathlib import Path
import litserve as ls
from fastapi import HTTPException
from loguru import logger

from mineru.cli.common import do_parse, read_fn
from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram


class MinerUAPI(ls.LitAPI):
    def __init__(self, output_dir="/tmp/mineru_output", vlm_server_url="http://127.0.0.1:30000/v1"):
        super().__init__()
        self.output_dir = output_dir
        self.vlm_server_url = vlm_server_url

    def setup(self, device):
        logger.info(f"Worker initialising on device: {device}")
        if not os.getenv("MINERU_DEVICE_MODE"):
            os.environ["MINERU_DEVICE_MODE"] = device if device != "auto" else get_device()
        device_mode = os.environ["MINERU_DEVICE_MODE"]
        if not os.getenv("MINERU_VIRTUAL_VRAM_SIZE"):
            vram = get_vram(device_mode) if device_mode.startswith(("cuda", "npu")) else 1
            # workers_per_device=2: two workers share the same physical GPU (24 GB).
            # Report half so MinerU's batch-size logic does not over-allocate
            # and trigger OOM when both workers run concurrently.
            os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = str(max(1, vram // 2))
        os.environ.setdefault("MINERU_MODEL_SOURCE", "local")
        os.environ.setdefault("MINERU_PDF_RENDER_THREADS", "8")
        logger.info(
            f"device={device_mode}  VRAM={os.environ['MINERU_VIRTUAL_VRAM_SIZE']} GB  "
            f"render_threads={os.environ['MINERU_PDF_RENDER_THREADS']}"
        )

    def decode_request(self, request):
        file_bytes = base64.b64decode(request["file"])
        opts = request.get("options", {})
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(file_bytes)
        tmp.close()
        return {
            "input_path": tmp.name,
            "backend": opts.get("backend", "hybrid-http-client"),
            "method": opts.get("method", "auto"),
            "lang": opts.get("lang", "ch"),
            "formula_enable": opts.get("formula_enable", True),
            "table_enable": opts.get("table_enable", True),
            "start_page_id": opts.get("start_page_id", 0),
            "end_page_id": opts.get("end_page_id", None),
            # Point at the local vllm server
            "server_url": opts.get("server_url", self.vlm_server_url),
        }

    def predict(self, inputs):
        input_path = Path(inputs["input_path"])
        output_dir = Path(self.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Append a short unique ID so concurrent workers processing files
            # with the same filename never write to the same subdirectory.
            job_id = str(uuid.uuid4())[:8]
            unique_name = f"{input_path.stem}_{job_id}"
            do_parse(
                output_dir=str(output_dir),
                pdf_file_names=[unique_name],
                pdf_bytes_list=[read_fn(input_path)],
                p_lang_list=[inputs["lang"]],
                backend=inputs["backend"],
                parse_method=inputs["method"],
                formula_enable=inputs["formula_enable"],
                table_enable=inputs["table_enable"],
                server_url=inputs["server_url"],
                start_page_id=inputs["start_page_id"],
                end_page_id=inputs["end_page_id"],
            )
            return str(output_dir / unique_name)
        except Exception as exc:
            logger.error(f"Processing failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            if input_path.exists():
                input_path.unlink()

    def encode_response(self, response):
        return {"output_dir": response}


if __name__ == "__main__":
    server = ls.LitServer(
        MinerUAPI(
            output_dir="/tmp/mineru_output",
            vlm_server_url="http://127.0.0.1:30000/v1",
        ),
        accelerator="cuda",
        devices=[0, 1],          # bind only to GPU 0 and GPU 1
        workers_per_device=2,    # 2 workers × 2 GPUs = 4 concurrent pipeline jobs
        timeout=False,
    )
    logger.info("Starting MinerU LitServe pipeline server on port 8000")
    server.run(port=8000, generate_client_file=False)
```

```bash
# Pin pipeline server to GPU 0 and 1 via env var as an extra safety net
CUDA_VISIBLE_DEVICES=0,1 \
MINERU_MODEL_SOURCE=local \
python /opt/mineru/server.py
```

**`workers_per_device=2` rationale:**
- Each pipeline worker occupies ~1.5–2 GB VRAM on its assigned GPU (CV models)
- RTX 4090 has 24 GB → 2 workers use 3–4 GB, well within headroom
- 2 workers × 2 GPUs = **4 concurrent PDF-to-Markdown jobs**
- Each job spawns `MINERU_PDF_RENDER_THREADS=8` render threads; 4 jobs × 8 threads = 32 threads maps neatly onto the 32-thread CPU

---

## Step 7 — FastAPI Gateway (optional, recommended for HTTP clients)

The built-in `mineru-api` server provides a REST endpoint that accepts file uploads and returns JSON/ZIP. Run it as a proxy in front of the LitServe workers using `hybrid-http-client` mode:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MINERU_MODEL_SOURCE=local \
MINERU_API_MAX_CONCURRENT_REQUESTS=8 \
mineru-api \
    --host 0.0.0.0 \
    --port 8080
```

Or — for full multi-GPU utilisation — keep the LitServe server (Step 6) as the main entry point, since LitServe already handles queuing and load-balancing across workers.

---

## Step 8 — Nginx Reverse Proxy

Bind the production API to the 10 GbE interface (AQC113C). The Intel I210 1 GbE can serve an admin/monitoring interface.

Create `/etc/nginx/sites-available/mineru`:

```nginx
upstream mineru_workers {
    # LitServe handles internal load-balancing; single upstream entry is sufficient
    server 127.0.0.1:8000;
    keepalive 64;
}

server {
    # Bind ONLY to the 10 GbE interface (AQC113C) by replacing _ with the NIC's IP.
    # "listen 80;" without an IP binds on all interfaces, including the 1 GbE Intel
    # I210 and any Docker bridge. In production, set this to the specific IP:
    #   listen 10.x.x.x:80;     # Replace with actual AQC113C IP
    # Or terminate TLS here (strongly recommended):
    #   listen 10.x.x.x:443 ssl;
    listen 80;
    server_name _;

    client_max_body_size 512M;   # Allow large PDF uploads
    client_body_timeout 900s;
    # 900s allows a 500-page scientific PDF with formulas/tables to complete.
    # LitServe workers have timeout=False; nginx is the binding constraint.
    proxy_read_timeout  900s;
    proxy_send_timeout  900s;

    location / {
        proxy_pass         http://mineru_workers;
        proxy_http_version 1.1;
        proxy_set_header   Connection "";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/mineru /etc/nginx/sites-enabled/mineru
sudo nginx -t && sudo systemctl reload nginx
```

---

## Step 9 — Systemd Service Files

### 9.1 VLM Server (GPUs 2–3)

Create `/etc/systemd/system/mineru-vlm.service`:

```ini
[Unit]
Description=MinerU VLM Server (vllm, GPUs 2-3)
After=network.target
Wants=network.target

[Service]
Type=simple
User=mineru
Group=mineru
WorkingDirectory=/opt/mineru
Environment=PATH=/opt/mineru/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=CUDA_VISIBLE_DEVICES=2,3
Environment=MINERU_MODEL_SOURCE=local
Environment=MINERU_TOOLS_CONFIG_JSON=/opt/mineru/mineru.json
ExecStart=/opt/mineru/venv/bin/mineru-openai-server \
    --host 127.0.0.1 \
    --port 30000 \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.80
Restart=on-failure
RestartSec=10s
StandardOutput=journal
StandardError=journal
LimitNOFILE=65536
LimitMEMLOCK=infinity
LimitSTACK=67108864

[Install]
WantedBy=multi-user.target
```

### 9.2 Pipeline Worker Server (GPUs 0–1)

Create `/etc/systemd/system/mineru-pipeline.service`:

```ini
[Unit]
Description=MinerU Pipeline Worker Server (LitServe, GPUs 0-1)
After=network.target mineru-vlm.service
# Hard dependency: if the VLM service is not running, this service will not start.
Requires=mineru-vlm.service

[Service]
Type=simple
User=mineru
Group=mineru
WorkingDirectory=/opt/mineru
Environment=PATH=/opt/mineru/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=CUDA_VISIBLE_DEVICES=0,1
Environment=MINERU_MODEL_SOURCE=local
Environment=MINERU_TOOLS_CONFIG_JSON=/opt/mineru/mineru.json
Environment=MINERU_PDF_RENDER_THREADS=8
# Report half the physical VRAM (12 GB) because workers_per_device=2 means
# two workers share the same 24 GB GPU. This prevents batch-size over-allocation.
Environment=MINERU_VIRTUAL_VRAM_SIZE=12
# Block startup until the VLM server's /health endpoint responds.
# vllm model loading takes 30-60 s; without this probe, early requests fail.
ExecStartPre=/bin/bash -c \
    'until curl -sf http://127.0.0.1:30000/health; do echo "Waiting for VLM server..."; sleep 5; done'
ExecStart=/opt/mineru/venv/bin/python /opt/mineru/server.py
Restart=on-failure
RestartSec=15s
StandardOutput=journal
StandardError=journal
LimitNOFILE=65536
LimitMEMLOCK=infinity
LimitSTACK=67108864

[Install]
WantedBy=multi-user.target
```

```bash
# Create the shared directory first, then create the service user.
# (useradd --home sets /etc/passwd but does NOT create the directory)
sudo mkdir -p /opt/mineru
sudo useradd --system --home /opt/mineru --shell /usr/sbin/nologin mineru

# Transfer ownership of all shared assets downloaded in Step 3.
sudo chown -R mineru:mineru /opt/mineru

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable --now mineru-vlm.service
# The pipeline service waits automatically via ExecStartPre health probe.
sudo systemctl enable --now mineru-pipeline.service

# Verify
sudo systemctl status mineru-vlm.service
sudo systemctl status mineru-pipeline.service
journalctl -u mineru-vlm.service -f
```

---

## Step 10 — Docker Alternative (Self-Contained)

If you prefer Docker over bare-metal Python:

### 10.0 Prerequisites — nvidia-container-toolkit

Docker GPU passthrough requires `nvidia-container-toolkit` on the host:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# Verify Docker can see GPUs:
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 10.1 Build the Image (bake models in)

```bash
cd /path/to/MinerU/docker/global
docker build -t mineru:prod .
```

### 10.2 `docker-compose.yml` for 4× RTX 4090

```yaml
services:
  mineru-vlm:
    image: mineru:prod
    container_name: mineru-vlm
    restart: always
    entrypoint: mineru-openai-server
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "30000"
      - --data-parallel-size
      - "2"
      - --gpu-memory-utilization
      - "0.80"
    environment:
      MINERU_MODEL_SOURCE: local
    ports:
      - "127.0.0.1:30000:30000"   # Not exposed externally
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    # Healthcheck gates mineru-api startup: the VLM model takes 30-60 s to load.
    # Without this, mineru-api starts sending requests before vllm is ready.
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:30000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 60s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2", "3"]
              capabilities: [gpu]

  mineru-api:
    image: mineru:prod
    container_name: mineru-api
    restart: always
    depends_on:
      mineru-vlm:
        condition: service_healthy   # Wait until VLM model finishes loading
    entrypoint: mineru-api
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    environment:
      MINERU_MODEL_SOURCE: local
      MINERU_API_MAX_CONCURRENT_REQUESTS: "4"
      # Pass server_url per request: server_url=http://mineru-vlm:30000/v1
      # Docker bridge DNS resolves the container name "mineru-vlm".
    ports:
      - "8000:8000"
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0", "1"]
              capabilities: [gpu]
```

```bash
docker compose up -d
docker compose logs -f
```

> **Docker network note:** In the Docker setup, the `mineru-api` container uses `hybrid-http-client` backend and must pass `server_url=http://mineru-vlm:30000/v1` in each request (Docker bridge DNS resolves `mineru-vlm`). For the LitServe multi-worker approach, use the bare-metal setup in Steps 5–6 instead, as LitServe's multi-GPU binding does not work well inside a single Docker container with NVML device isolation.

---

## Step 11 — Configuration Tuning Reference

### Key Environment Variables for This Hardware

```bash
# Worker isolation
CUDA_VISIBLE_DEVICES=0,1          # pipeline server
CUDA_VISIBLE_DEVICES=2,3          # VLM server

# VRAM per pipeline worker — with workers_per_device=2, two workers share each
# physical GPU (24 GB). Report half to prevent batch-size logic from over-allocating.
MINERU_VIRTUAL_VRAM_SIZE=12       # 24 GB ÷ 2 workers per GPU

# Pipeline batch: 384 pages/batch (default) is good; increase if RAM allows
MINERU_MIN_BATCH_INFERENCE_SIZE=512

# PDF render threads per worker (8 per worker × 4 workers = 32 = full CPU)
MINERU_PDF_RENDER_THREADS=8

# Enable Chinese math formula model (slightly better for Chinese papers)
MINERU_FORMULA_CH_SUPPORT=false   # set true only if workload is Chinese STEM docs

# API concurrency gate
MINERU_API_MAX_CONCURRENT_REQUESTS=8

# Log verbosity (use WARNING in production)
MINERU_LOG_LEVEL=WARNING
```

### vllm GPU Utilisation Guide

`gpu_memory_utilization` controls the **total** vllm VRAM budget (model weights + KV cache). Subtract ~2.5 GB model weight overhead to get the actual KV cache size.

| `--gpu-memory-utilization` | Total vllm VRAM budget (24 GB card) | KV cache (budget − 2.5 GB weights) | When to use |
|---|---|---|---|
| 0.90 | ~21.6 GB | ~19.1 GB | Single large-document jobs, no concurrent requests |
| **0.80** | **~19.2 GB** | **~16.7 GB** | **Recommended — 2–4 concurrent page batches** |
| 0.70 | ~16.8 GB | ~14.3 GB | Conservative; extra OS/CUDA headroom |
| 0.50 | ~12.0 GB | ~9.5 GB | Debugging / memory leak investigation |

---

## Step 12 — Monitoring & Observability

```bash
# Real-time GPU utilisation across all 4 cards
nvtop
# or
watch -n 1 nvidia-smi

# Service logs
journalctl -u mineru-vlm.service -f
journalctl -u mineru-pipeline.service -f
```

**Load test — LitServe server (Step 6):** uses `POST /predict` with a JSON body.

```bash
FILE_B64=$(base64 -w 0 demo/pdfs/demo3.pdf)
for i in 1 2 3 4; do
    curl -s -X POST http://localhost:8000/predict \
        -H 'Content-Type: application/json' \
        -d "{\"file\": \"$FILE_B64\", \"options\": {\"backend\": \"hybrid-http-client\", \"server_url\": \"http://127.0.0.1:30000/v1\"}}" &
done
wait
```

**Load test — FastAPI server (Step 7):** uses `POST /file_parse` with multipart form.

```bash
for i in 1 2 3 4; do
    curl -s -X POST http://localhost:8080/file_parse \
        -F "files=@demo/pdfs/demo3.pdf" \
        -F "backend=hybrid-http-client" \
        -F "server_url=http://127.0.0.1:30000/v1" &
done
wait
```

**Expected GPU utilisation at full load:**
- GPU 0, 1: ~60–80% SM utilisation (pipeline CV models + OCR)
- GPU 2, 3: ~85–95% SM utilisation (VLM token generation)
- VRAM usage: GPU 0–1 ~2–4 GB, GPU 2–3 ~14–19 GB

---

## Summary — GPU Allocation Table

| GPU | Role | Backend | Workers | VRAM Budget | SM Target |
|---|---|---|---|---|---|
| 0 | Pipeline + CV models | LitServe worker A1 & A2 | 2 | ~4 GB | 60–80% |
| 1 | Pipeline + CV models | LitServe worker B1 & B2 | 2 | ~4 GB | 60–80% |
| 2 | VLM inference shard 0 | vllm data-parallel | 1 | 19.2 GB budget / ~16.7 GB KV cache | 85–95% |
| 3 | VLM inference shard 1 | vllm data-parallel | 1 | 19.2 GB budget / ~16.7 GB KV cache | 85–95% |
| **Total** | | | **4 concurrent jobs** | **~42 GB / 96 GB** | |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `CUDA out of memory` on GPU 2/3 | KV cache too large | Reduce `--gpu-memory-utilization` to 0.70 |
| VLM server slow to start | Model checkpoint load from disk | First-start takes ~30–60 s; use `After=` in systemd |
| Pipeline workers stall | All 32 CPU threads contended | Reduce `MINERU_PDF_RENDER_THREADS` to 4 per worker |
| `HTTP 503` from API | `MINERU_API_MAX_CONCURRENT_REQUESTS` too low | Increase to 16 or 0 (unlimited) |
| GPU 0/1 idle, GPU 2/3 saturated | All requests use VLM-only path | Mix in `pipeline` backend requests or increase `workers_per_device` |
| `nvidia-smi` shows only 1 GPU in container | Missing `device_ids` in compose | Add all required IDs under `deploy.resources.reservations.devices` |

---

## Step 13 — Post-Deployment Quick Testing

Run these checks in order after completing Steps 0–12. Each section is self-contained — failures here pinpoint exactly which layer has a problem.

### 13.1 Hardware & Driver Layer

```bash
# All 4 GPUs must appear
nvidia-smi
# Expected: 4× RTX 4090, CUDA Version ≥ 12.1, driver ≥ 525

# All GPUs must show zero memory used (nothing running yet)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
# Expected: each row ~0 MiB used, ~24576 MiB free

# Confirm persistence mode is active
nvidia-smi -q | grep 'Persistence Mode'
# Expected: Enabled (all 4 GPUs)
```

### 13.2 Python / CUDA Layer

```bash
source /opt/mineru/venv/bin/activate

# PyTorch must see all 4 GPUs
python -c "import torch; print('GPUs:', torch.cuda.device_count()); \
    [print(f'  GPU {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
# Expected: GPUs: 4

# vllm must import cleanly
python -c "import vllm; print('vllm:', vllm.__version__)"

# MinerU must import cleanly
python -c "from mineru.cli.common import do_parse; print('MinerU OK')"
```

### 13.3 VLM Server (GPUs 2–3)

```bash
# Check the service is running
sudo systemctl status mineru-vlm.service

# Health endpoint must return {"status":"healthy"}
curl -s http://127.0.0.1:30000/health
# Expected: {"status":"healthy"}

# List available models (confirms vllm loaded MinerU2.5 successfully)
curl -s http://127.0.0.1:30000/v1/models | python3 -m json.tool
# Expected: a models list containing the MinerU2.5 model id

# GPU memory check: GPU 2 and 3 should show ~19 GB used (vllm loaded)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# Expected: GPU 2 ~19000 MiB, GPU 3 ~19000 MiB, GPU 0/1 near 0
```

### 13.4 Single PDF Parse — VLM Path

```bash
# Parse one page using the VLM server directly (vlm-http-client)
mineru \
    -p demo/pdfs/demo3.pdf \
    -o /tmp/test_vlm_output \
    -b vlm-http-client \
    -u http://127.0.0.1:30000/v1 \
    -e 0          # first page only for speed

# Check output exists
ls /tmp/test_vlm_output/demo3/
# Expected: demo3.md  plus images/ directory

# Spot-check the markdown is non-empty
wc -l /tmp/test_vlm_output/demo3/demo3.md
# Expected: > 5 lines
```

### 13.5 Single PDF Parse — Pipeline Path

```bash
# Parse one page using pipeline backend only (CV models, no VLM)
MINERU_VIRTUAL_VRAM_SIZE=12 \
CUDA_VISIBLE_DEVICES=0 \
mineru \
    -p demo/pdfs/demo3.pdf \
    -o /tmp/test_pipeline_output \
    -b pipeline \
    -e 0

ls /tmp/test_pipeline_output/demo3/
wc -l /tmp/test_pipeline_output/demo3/demo3.md
```

### 13.6 LitServe Worker Server

```bash
# Start the pipeline server in the foreground for testing
CUDA_VISIBLE_DEVICES=0,1 \
MINERU_MODEL_SOURCE=local \
MINERU_VIRTUAL_VRAM_SIZE=12 \
python /opt/mineru/server.py &
SERVER_PID=$!
sleep 5   # wait for LitServe to bind

# Send one request
FILE_B64=$(base64 -w 0 demo/pdfs/demo3.pdf)
RESPONSE=$(curl -s -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d "{\"file\": \"$FILE_B64\", \"options\": {\
        \"backend\": \"hybrid-http-client\",\
        \"server_url\": \"http://127.0.0.1:30000/v1\"\
    }}")
echo "Response: $RESPONSE"
# Expected: {"output_dir": "/tmp/mineru_output/demo3_XXXXXXXX"}

# Verify the output directory exists and contains markdown
OUTDIR=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['output_dir'])")
ls "$OUTDIR"

kill $SERVER_PID
```

### 13.7 Concurrent Load Test (4 PDFs in Parallel)

```bash
# Use the systemd service (must be running from Step 9)
FILE_B64=$(base64 -w 0 demo/pdfs/demo3.pdf)

START=$(date +%s)
for i in 1 2 3 4; do
    curl -s -X POST http://localhost:8000/predict \
        -H 'Content-Type: application/json' \
        -d "{\"file\": \"$FILE_B64\", \"options\": {\
            \"backend\": \"hybrid-http-client\",\
            \"server_url\": \"http://127.0.0.1:30000/v1\"\
        }}" > /tmp/result_$i.json &
done
wait
END=$(date +%s)
echo "4 concurrent jobs finished in $((END - START)) seconds"

# All 4 must have succeeded
for i in 1 2 3 4; do
    cat /tmp/result_$i.json
done

# During the run, GPU utilisation should match the Summary table:
# GPU 0,1: 60-80% SM  GPU 2,3: 85-95% SM
# Check live in a second terminal:
#   watch -n 1 nvidia-smi
```

### 13.8 Service Restart Resilience

```bash
# Simulate VLM server crash and verify pipeline auto-recovers
sudo systemctl restart mineru-vlm.service

# Pipeline service must not start accepting requests until VLM is healthy.
# journalctl output should show "Waiting for VLM server..." lines from ExecStartPre.
journalctl -u mineru-pipeline.service -f --since "1 min ago"
# Expected log sequence:
#   Waiting for VLM server...
#   Waiting for VLM server...   (repeats ~6-12 times)
#   Starting MinerU LitServe pipeline server on port 8000

# Confirm both services are Active after restart
sudo systemctl status mineru-vlm.service mineru-pipeline.service
```

### 13.9 Nginx Gateway (if deployed)

```bash
# Confirm nginx resolves to the LitServe server
curl -s -o /dev/null -w "%{http_code}" http://localhost/
# Expected: 200 (LitServe root) or 404 (path not found, but nginx connected)

# Full end-to-end via nginx
FILE_B64=$(base64 -w 0 demo/pdfs/demo3.pdf)
curl -s -X POST http://localhost/predict \
    -H 'Content-Type: application/json' \
    -d "{\"file\": \"$FILE_B64\", \"options\": {\"backend\": \"hybrid-http-client\", \"server_url\": \"http://127.0.0.1:30000/v1\"}}"
```

### 13.10 Quick Checklist

| # | Test | Pass Condition |
|---|---|---|
| 1 | `nvidia-smi` shows 4 GPUs | All 4 RTX 4090 visible, driver ≥ 525 |
| 2 | `torch.cuda.device_count()` | Returns `4` |
| 3 | `vllm` imports | No `ModuleNotFoundError` |
| 4 | VLM server `/health` | `{"status":"healthy"}` |
| 5 | VLM `/v1/models` | MinerU2.5 model listed |
| 6 | GPU 2/3 VRAM after VLM load | ~19 GB used per card |
| 7 | Single PDF — VLM path | Markdown output ≥ 5 lines |
| 8 | Single PDF — pipeline path | Markdown output ≥ 5 lines |
| 9 | LitServe single request | `output_dir` in JSON response |
| 10 | 4 concurrent jobs | All return success, ≤ 4× sequential time |
| 11 | Service restart | Pipeline waits for VLM; no early failures |
| 12 | Nginx gateway | HTTP 200 end-to-end |
