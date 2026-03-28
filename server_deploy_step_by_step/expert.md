# MinerU Expert Deployment Guide
## Server: AMD Threadripper PRO 5955WX · 4× RTX 4090 · 256 GB RAM

---

## 1. Server Hardware Profile

| Component | Spec | MinerU Relevance |
|---|---|---|
| CPU | AMD Ryzen Threadripper PRO 5955WX, 16-cores / 32-threads | PDF rendering (up to 4 parallel workers), ONNX thread pools |
| GPU | 4× NVIDIA RTX 4090 (Ada Lovelace, Compute Capability 8.9) | Full feature support: vLLM v1 engine, BF16, custom logits processors |
| CPU RAM | 256 GB DDR4 ECC | Large shared-memory for vLLM/PyTorch IPC, many concurrent tasks |
| GPU VRAM | 24 GB × 4 = 96 GB total | Ample for hybrid: ~9–10 GB/GPU; can run 4 parallel inference workers |
| System Disk | 4 TB NVMe | Model storage, output cache, Docker volumes |
| 10 GbE NIC | Aquantia AQC113C (10 Gbps) | Primary API traffic — saturates at ~1.25 GB/s, well above API payload sizes |
| 1 GbE NIC | Intel I210 | Management / out-of-band access |

**RTX 4090 GPU class:** Ada Lovelace, CC 8.9 → qualifies for all MinerU optimizations:
- vLLM v1 engine enabled (VLLM_USE_V1 active)
- BF16 inference (CC ≥ 8.0)
- Custom logits processors enabled (CC ≥ 8.0, vLLM ≥ 0.10.1)
- lmdeploy `pytorch` backend (Linux, CC ≥ 8.0)
- Default Docker base image: `vllm/vllm-openai:v0.10.1.1` (Ampere/Ada/Hopper range)

---

## 2. Recommended Adoption Strategy: 4-GPU Split Service Layout

### Why This Strategy Maximises Capacity

With 4× 24 GB GPUs (96 GB VRAM total), the optimal layout is to run **independent services on each GPU** using `CUDA_VISIBLE_DEVICES` isolation rather than pooling all GPUs into a single vLLM data-parallel job. This approach:

- **Maximises concurrency**: 4 fully independent inference workers serving 4 parallel client streams simultaneously
- **Eliminates VRAM contention**: each GPU is isolated, no cross-GPU locking overhead
- **Enables tiered service**: one GPU can be dedicated to a heavier `hybrid-auto-engine` (FastAPI) service while the other three serve `vlm-http-client` traffic
- **Resilience**: failure of one service does not cascade to others
- **Network utilisation**: the 10 GbE link can deliver >800 MB/s sustained; 4 parallel API streams fully exploit this

### GPU Assignment Map

| GPU | `CUDA_VISIBLE_DEVICES` | Service | Port | Role |
|---|---|---|---|---|
| GPU 0 | `0` | `mineru-openai-server` (vLLM) | 30000 | VLM inference — primary |
| GPU 1 | `1` | `mineru-openai-server` (vLLM) | 30001 | VLM inference — secondary |
| GPU 2 | `2` | `mineru-api` (FastAPI, hybrid-auto-engine) | 8000 | REST API with local hybrid backend |
| GPU 3 | `3` | `mineru-api` (FastAPI, pipeline backend) | 8001 | Fast pipeline-only REST API |

**Optional alternative**: Assign GPUs 0+1 to a single `mineru-openai-server` with `--data-parallel-size 2` for higher single-stream throughput on very large documents (tensor-parallel VLM). GPUs 2+3 then run two independent FastAPI services.

### VRAM Budget per GPU (RTX 4090, 24 GB)

| Backend | Min VRAM | Typical Usage | Headroom |
|---|---|---|---|
| `vlm-vllm` (openai-server) | 8 GB | ~16–18 GB (Qwen2-VL) | 6–8 GB KV cache |
| `hybrid-auto-engine` | 10 GB | ~18–20 GB | 4–6 GB |
| `pipeline` | 3–6 GB | ~4–5 GB | 18–20 GB free |

GPU memory utilisation flag for 24 GB cards: use `--gpu-memory-utilization 0.75` (18 GB allocated, leaves 6 GB safety margin for KV cache spikes).

---

## 3. Server Setup

### 3.1 Prerequisites

```bash
# Verify NVIDIA driver and CUDA
nvidia-smi
# Expected: Driver ≥ 535.xx, CUDA ≥ 12.1

# Install Docker + NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 3.2 Build the MinerU Docker Image

```bash
# Pull Dockerfile (global, uses HuggingFace models)
wget https://raw.githubusercontent.com/opendatalab/MinerU/master/docker/global/Dockerfile

# Build — this bakes all models into the image (~20 GB)
docker build -t mineru:latest -f Dockerfile .
# Build time: ~30–60 min depending on network speed
```

> **Note:** The base `vllm/vllm-openai:v0.10.1.1` is correct for RTX 4090 (Ada Lovelace, CC 8.9). Do not change the base image.

### 3.3 Custom docker-compose.yml for 4-GPU Split Layout

Create `/opt/mineru/docker-compose.yml`:

```yaml
services:
  # GPU 0 — Primary VLM OpenAI-compatible inference server
  mineru-openai-server-0:
    image: mineru:latest
    restart: always
    profiles: ["openai-server"]
    entrypoint: mineru-openai-server
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "30000"
      - --gpu-memory-utilization
      - "0.75"
    ports:
      - "30000:30000"
    environment:
      MINERU_MODEL_SOURCE: local
      CUDA_VISIBLE_DEVICES: "0"
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    shm_size: "32g"
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

  # GPU 1 — Secondary VLM OpenAI-compatible inference server
  mineru-openai-server-1:
    image: mineru:latest
    restart: always
    profiles: ["openai-server"]
    entrypoint: mineru-openai-server
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "30001"
      - --gpu-memory-utilization
      - "0.75"
    ports:
      - "30001:30001"
    environment:
      MINERU_MODEL_SOURCE: local
      CUDA_VISIBLE_DEVICES: "1"
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    shm_size: "32g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]

  # GPU 2 — FastAPI hybrid-auto-engine (VLM + pipeline, full accuracy)
  mineru-api-hybrid:
    image: mineru:latest
    restart: always
    profiles: ["api"]
    entrypoint: mineru-api
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    ports:
      - "8000:8000"
    environment:
      MINERU_MODEL_SOURCE: local
      CUDA_VISIBLE_DEVICES: "2"
      MINERU_API_MAX_CONCURRENT_REQUESTS: "8"
      MINERU_API_ENABLE_FASTAPI_DOCS: "false"
      MINERU_INTRA_OP_NUM_THREADS: "8"
      MINERU_INTER_OP_NUM_THREADS: "8"
      MINERU_PDF_RENDER_TIMEOUT: "600"
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    shm_size: "16g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]

  # GPU 3 — FastAPI pipeline-only (fastest, lowest VRAM, high throughput)
  mineru-api-pipeline:
    image: mineru:latest
    restart: always
    profiles: ["api"]
    entrypoint: mineru-api
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8001"
    ports:
      - "8001:8001"
    environment:
      MINERU_MODEL_SOURCE: local
      CUDA_VISIBLE_DEVICES: "3"
      MINERU_API_MAX_CONCURRENT_REQUESTS: "16"
      MINERU_API_ENABLE_FASTAPI_DOCS: "false"
      MINERU_INTRA_OP_NUM_THREADS: "8"
      MINERU_INTER_OP_NUM_THREADS: "8"
      MINERU_PDF_RENDER_TIMEOUT: "600"
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    shm_size: "8g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["3"]
              capabilities: [gpu]
```

### 3.4 Key Environment Variables Reference

| Variable | Value Used | Why |
|---|---|---|
| `MINERU_MODEL_SOURCE` | `local` | Models baked into image, no runtime download |
| `CUDA_VISIBLE_DEVICES` | `"0"` / `"1"` / `"2"` / `"3"` | GPU isolation — prevents VRAM conflicts |
| `--gpu-memory-utilization 0.75` | `0.75` | 18 GB allocated on 24 GB card, 6 GB KV cache headroom |
| `MINERU_API_MAX_CONCURRENT_REQUESTS` | `8` (hybrid), `16` (pipeline) | Semaphore guard against overload |
| `MINERU_INTRA_OP_NUM_THREADS` | `8` | Limit ONNX CPU threads to prevent contention on 16-core CPU |
| `MINERU_INTER_OP_NUM_THREADS` | `8` | Same as above |
| `MINERU_PDF_RENDER_TIMEOUT` | `600` | Handle large/complex PDFs without timeout |
| `ipc: host` | — | Required by PyTorch / vLLM shared-memory tensor IPC |
| `shm_size` | `32g` / `16g` / `8g` | Docker default (64 MB) is insufficient for VLM |

### 3.5 Start All Services

```bash
cd /opt/mineru

# Start VLM OpenAI servers (GPUs 0 and 1)
docker compose -f docker-compose.yml --profile openai-server up -d

# Start FastAPI servers (GPUs 2 and 3)
docker compose -f docker-compose.yml --profile api up -d

# Verify all containers are running
docker compose -f docker-compose.yml ps

# Check health
curl http://localhost:30000/health
curl http://localhost:30001/health
curl http://localhost:8000/docs   # will return 404 in production (docs disabled)
```

### 3.6 Nginx Reverse Proxy with Load Balancing (Optional)

For the dual VLM servers (ports 30000/30001), place Nginx upstream in front:

```nginx
upstream mineru_vlm {
    least_conn;
    server 127.0.0.1:30000;
    server 127.0.0.1:30001;
}

upstream mineru_api_hybrid {
    server 127.0.0.1:8000;
}

upstream mineru_api_pipeline {
    server 127.0.0.1:8001;
}

server {
    listen 443 ssl;
    server_name mineru.your-domain.internal;

    ssl_certificate     /etc/ssl/certs/mineru.crt;
    ssl_certificate_key /etc/ssl/private/mineru.key;

    # VLM endpoint — load-balanced across GPU 0 and GPU 1
    location /vlm/ {
        proxy_pass http://mineru_vlm/;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        client_max_body_size 512m;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Hybrid FastAPI endpoint
    location /api/hybrid/ {
        proxy_pass http://mineru_api_hybrid/;
        proxy_read_timeout 600s;
        client_max_body_size 512m;
        proxy_set_header Host $host;
    }

    # Pipeline FastAPI endpoint (fastest)
    location /api/pipeline/ {
        proxy_pass http://mineru_api_pipeline/;
        proxy_read_timeout 300s;
        client_max_body_size 512m;
        proxy_set_header Host $host;
    }
}
```

> **Security note:** Never expose MinerU services directly on public IP. Always use a reverse proxy with TLS, and restrict access to internal network CIDRs. The OpenAI server has no built-in auth — add API key validation at the Nginx or gateway layer.

---

## 4. End-User Client Usage

### 4.1 Via VLM HTTP Client (Lightest — CPU only on client)

```bash
# Install minimal client package (no GPU required on client machine)
pip install mineru

# Extract a single PDF via GPU 0 VLM server
mineru -p document.pdf -o ./output/ \
  -b vlm-http-client \
  -u http://SERVER_IP:30000

# Or via the Nginx load-balanced endpoint (round-robin between GPU 0 and GPU 1)
mineru -p document.pdf -o ./output/ \
  -b vlm-http-client \
  -u http://SERVER_IP/vlm
```

### 4.2 Via FastAPI REST Endpoint (Hybrid, highest accuracy)

```bash
# Upload PDF — returns JSON with markdown content
curl -X POST http://SERVER_IP:8000/file_parse \
  -F "files=@document.pdf" \
  -F "backend=hybrid-auto-engine" \
  -F "parse_method=auto" \
  -F "return_md=true" \
  -F "return_content_list=true" \
  -o result.json

# Upload PDF — returns ZIP archive with .md + images
curl -X POST http://SERVER_IP:8000/file_parse \
  -F "files=@document.pdf" \
  -F "backend=hybrid-auto-engine" \
  -F "response_format_zip=true" \
  -o results.zip
```

### 4.3 Via FastAPI REST Endpoint (Pipeline, fastest throughput)

```bash
# Pipeline backend — no VLM, fastest for standard PDFs
curl -X POST http://SERVER_IP:8001/file_parse \
  -F "files=@document.pdf" \
  -F "backend=pipeline" \
  -F "parse_method=auto" \
  -F "lang_list=en" \
  -F "return_md=true" \
  -o result.json
```

### 4.4 Via Hybrid HTTP Client (Client with pipeline, server for VLM)

```bash
# Install pipeline extras on client
pip install "mineru[pipeline]"

# Client does local OCR/layout, offloads VLM to server
mineru -p document.pdf -o ./output/ \
  -b hybrid-http-client \
  -u http://SERVER_IP:30000
```

### 4.5 Python SDK — Batch Processing

```python
import asyncio
from mineru.cli.common import aio_do_parse

async def batch_extract(pdf_paths: list[str], server_url: str) -> list[dict]:
    """Send multiple PDFs concurrently to the VLM server."""
    tasks = [
        aio_do_parse(
            output_dir=f"/tmp/output/{i}",
            pdf_file_names=[f"doc_{i}"],
            pdf_bytes_list=[open(p, "rb").read()],
            p_lang_list=["en"],
            backend="vlm-http-client",
            server_url=server_url,
        )
        for i, p in enumerate(pdf_paths)
    ]
    return await asyncio.gather(*tasks)

if __name__ == "__main__":
    results = asyncio.run(batch_extract(
        ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
        server_url="http://SERVER_IP:30000"
    ))
```

### 4.6 API Request/Response Reference

**POST `/file_parse`** key fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `files` | file(s) | required | PDF or image |
| `backend` | str | `hybrid-auto-engine` | `pipeline`, `vlm-auto-engine`, `vlm-http-client`, `hybrid-auto-engine`, `hybrid-http-client` |
| `parse_method` | str | `auto` | `auto`, `txt`, `ocr` |
| `lang_list` | list | `["ch"]` | OCR language hint |
| `formula_enable` | bool | `true` | Enable LaTeX formula extraction |
| `table_enable` | bool | `true` | Enable table extraction |
| `return_md` | bool | `true` | Markdown in JSON response |
| `return_content_list` | bool | `false` | Structured content list |
| `return_images` | bool | `false` | Base64 images in JSON |
| `response_format_zip` | bool | `false` | Return ZIP instead of JSON |
| `server_url` | str | `null` | Required for `*-http-client` backends |

**JSON Response** (when `response_format_zip=false`):
```json
{
  "backend": "hybrid-auto-engine",
  "version": "2.x.x",
  "results": {
    "document_name": {
      "md_content": "# Title\n...",
      "content_list": [...],
      "images": { "fig1.jpg": "data:image/jpeg;base64,..." }
    }
  }
}
```

---

## 5. GPU and Server Monitoring

### 5.1 Real-Time GPU Dashboard (nvidia-smi)

```bash
# One-shot snapshot
nvidia-smi

# Live refresh every 1 second — all 4 GPUs
watch -n 1 nvidia-smi

# Compact streaming metrics (utilisation + memory + power + temp)
nvidia-smi dmon -s pucvmet -d 1

# Per-process GPU usage
nvidia-smi pmon -d 1
```

### 5.2 Key Metrics to Watch

| Metric | Command | Warning Threshold |
|---|---|---|
| GPU Utilisation (%) | `nvidia-smi --query-gpu=utilization.gpu --format=csv` | < 60% → under-utilised; 100% sustained → bottleneck |
| VRAM Used (MiB) | `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` | > 22 GB / 24 GB → risk of OOM |
| GPU Temperature (°C) | `nvidia-smi --query-gpu=temperature.gpu --format=csv` | > 83°C sustained → throttling risk |
| Power Draw (W) | `nvidia-smi --query-gpu=power.draw --format=csv` | RTX 4090 TDP = 450 W; > 480 W → power limit |
| PCIe Bandwidth | `nvidia-smi dmon -s t` | — |

```bash
# One-liner: live CSV for all 4 GPUs
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,\
temperature.gpu,power.draw --format=csv,noheader,nounits -l 2
```

### 5.3 vLLM Prometheus Metrics

The `mineru-openai-server` (vLLM) exposes Prometheus metrics at `/metrics`:

```bash
# Check metrics from GPU 0 server
curl http://localhost:30000/metrics

# Key vLLM metrics to watch:
# vllm:num_requests_running       — active requests in flight
# vllm:gpu_cache_usage_perc       — KV cache utilisation (%)
# vllm:num_requests_waiting       — queued requests (backpressure indicator)
# vllm:request_success_total      — total successful requests
# vllm:e2e_request_latency_seconds — end-to-end request latency histogram
```

### 5.4 Prometheus + Grafana Stack (Production Monitoring)

`/opt/mineru/monitoring/docker-compose.monitoring.yml`:

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    restart: always
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: "changeme"

  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04
    restart: always
    runtime: nvidia
    environment:
      NVIDIA_VISIBLE_DEVICES: all
    ports:
      - "9400:9400"
    cap_add:
      - SYS_ADMIN

volumes:
  prometheus_data:
  grafana_data:
```

`/opt/mineru/monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  # NVIDIA GPU metrics via DCGM exporter
  - job_name: 'dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']

  # vLLM server metrics — GPU 0
  - job_name: 'vllm_gpu0'
    static_configs:
      - targets: ['host.docker.internal:30000']
    metrics_path: '/metrics'

  # vLLM server metrics — GPU 1
  - job_name: 'vllm_gpu1'
    static_configs:
      - targets: ['host.docker.internal:30001']
    metrics_path: '/metrics'
```

```bash
# Start monitoring stack
docker compose -f /opt/mineru/monitoring/docker-compose.monitoring.yml up -d

# Access Grafana at http://SERVER_IP:3000
# Import dashboard ID 12239 (DCGM NVIDIA GPU Metrics) from Grafana.com
```

**Key DCGM GPU metrics (available via Grafana)**:

| DCGM Metric | Description |
|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | GPU compute utilisation % |
| `DCGM_FI_DEV_FB_USED` | Framebuffer (VRAM) used in MiB |
| `DCGM_FI_DEV_FB_FREE` | Framebuffer free |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature (°C) |
| `DCGM_FI_DEV_POWER_USAGE` | Power draw (W) |
| `DCGM_FI_DEV_SM_CLOCK` | SM clock frequency (MHz) |
| `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` | PCIe transmit throughput |
| `DCGM_FI_DEV_PCIE_RX_THROUGHPUT` | PCIe receive throughput |

### 5.5 Container and Service Health

```bash
# Follow logs for all MinerU services
docker compose -f /opt/mineru/docker-compose.yml logs -f

# Follow logs for a specific service
docker logs -f mineru-openai-server-0

# Check health status of all containers
docker compose -f /opt/mineru/docker-compose.yml ps

# Manual health check endpoints
curl http://localhost:30000/health     # GPU 0 VLM server
curl http://localhost:30001/health     # GPU 1 VLM server
# FastAPI: GET /docs redirects → use /openapi.json or just check if port is open
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/file_parse  # 405 = running
```

### 5.6 System Resource Monitoring

```bash
# CPU and memory overview
htop
# or
vmstat -s -S M

# Disk I/O (model reads, output writes)
iostat -xz 1

# Network throughput on 10 GbE (ens6 or ethX — replace with actual interface)
iftop -i ens6
# or
nload ens6

# Check GPU memory leaks between requests
watch -n 5 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

---

## 6. Performance Tuning Summary for This Server

| Tuning Parameter | Recommended Value | Rationale |
|---|---|---|
| `--gpu-memory-utilization` | `0.75` | 18 GB on 24 GB card; 6 GB KV cache headroom |
| `--data-parallel-size` | 1 per GPU (isolated) | 4 independent workers > 1 job with 4-GPU DP for throughput |
| `MINERU_API_MAX_CONCURRENT_REQUESTS` | `8` (hybrid), `16` (pipeline) | Prevents RAM/VRAM exhaustion under burst load |
| `MINERU_INTRA_OP_NUM_THREADS` | `8` | 16-core CPU shared by 2 API containers on GPUs 2+3 |
| `MINERU_INTER_OP_NUM_THREADS` | `8` | Same — avoids thread over-subscription |
| `MINERU_PDF_RENDER_TIMEOUT` | `600` | Complex PDFs can take > 5 min to rasterise |
| `shm_size` (VLM containers) | `32g` | vLLM needs large shared mem; 256 GB RAM allows this easily |
| Batch size (auto, VRAM ≥ 16 GB) | `8` (auto-selected by MinerU) | RTX 4090 24 GB triggers max batch size automatically |
| Network binding | `0.0.0.0` behind Nginx | Nginx handles TLS, rate limiting, load balancing |

---

## 7. Docker vs. Bare-Metal Deployment: Pros & Cons for This Server

This server has characteristics that make the trade-offs less clear-cut than on a standard Linux box. The analysis below is specific to the hardware profile above.

### 7.1 Why This Server Changes the Calculus

| Hardware Factor | Impact on Docker | Impact on Bare-Metal |
|---|---|---|
| 4× RTX 4090, 24 GB VRAM each | `--gpus` flag or `device_ids` in Compose work reliably; NVIDIA Container Toolkit is mature | Direct `CUDA_VISIBLE_DEVICES` control, no driver version pinning |
| 256 GB CPU RAM | `shm_size: 32g` per container is trivial (8 containers × 32 GB = 256 GB still covered) | No overhead; PyTorch/vLLM can use the full pool natively |
| AMD Threadripper PRO 5955WX (16C/32T, NUMA) | Docker adds a thin namespacing layer but CPU pinning via `cpuset` is possible | Full NUMA-aware CPU pinning with `numactl`, best latency |
| 4 TB NVMe system disk | Docker layer storage + volume mounts add minor I/O overhead (~3–5%) for model reads | Direct NVMe access; mmap-friendly for model weights |
| 10 GbE NIC (Aquantia AQC113C) | Host-mode networking (`network_mode: host`) eliminates virtual bridge overhead | Native kernel networking stack, full 10 GbE at wire speed |

---

### 7.2 Docker Deployment

#### Pros

| # | Advantage | Relevance to This Server |
|---|---|---|
| 1 | **Reproducible environment** | vLLM, CUDA toolkit, Python, MinerU all pinned in a single image layer; eliminates "works on my machine" for the 4-GPU configuration |
| 2 | **Fast service restarts** | `docker restart mineru-openai-server-0` recovers a crashed GPU worker in seconds, without touching the others |
| 3 | **Isolated dependency trees** | VLM service (vLLM/lmdeploy) and pipeline service (ONNX Runtime/PaddleOCR) can have conflicting CUDA or Python package versions — Docker isolates them per container |
| 4 | **Model baking** | The Dockerfile `mineru-models-download` step bakes all ~20 GB of model weights into the image; containers start without a network download, critical for air-gapped or slow-link environments |
| 5 | **Straightforward GPU isolation** | `device_ids: ["0"]` in Compose + `CUDA_VISIBLE_DEVICES` inside the container gives clean GPU-to-service mapping without system-wide `LD_LIBRARY_PATH` juggling |
| 6 | **IPC and ulimit encapsulation** | `ipc: host`, `memlock: -1`, `stack: 67108864` are declared per service; no risk of a system-wide `ulimit` misconfiguration affecting all processes |
| 7 | **Rolling upgrades** | Pull a new `mineru:latest` image and restart one container at a time; 3 of 4 GPUs stay serving traffic during upgrades |
| 8 | **Monitoring sidecar pattern** | DCGM Exporter and Prometheus stack run as separate Compose services alongside MinerU, with no interference |

#### Cons

| # | Disadvantage | Severity on This Server |
|---|---|---|
| 1 | **Image build time** | ~30–60 min for the initial build (20 GB of models + CUDA layers). Amortised over long server lifetime, but painful for first bring-up. **Mitigation**: pre-pull the base `vllm/vllm-openai:v0.10.1.1` image separately. | Medium |
| 2 | **Storage overhead** | Docker image layers duplicate some CUDA libraries already on the host. With 4 TB disk this is not a capacity concern (~25 GB/image), but layer cache management needed. | Low |
| 3 | **vLLM PCIe / NVLink topology opacity** | Docker cannot expose PCIe topology or NVLink fabric to vLLM the same way bare-metal can. On this server, all 4 RTX 4090s are PCIe (no NVLink), so this gap is **irrelevant** — there is no NVLink topology to lose. | None |
| 4 | **SHM/IPC ceiling requires explicit config** | Forgetting `ipc: host` and `shm_size` causes cryptic vLLM crashes. Must be set correctly in Compose for every VLM container. | Medium (config risk) |
| 5 | **NUMA affinity loss** | By default Docker does not pin containers to NUMA nodes. Threadripper PRO 5955WX has a single NUMA domain (all cores on one die), so this is **irrelevant** for this specific CPU. | None |
| 6 | **Kernel driver version coupling** | The `vllm/vllm-openai:v0.10.1.1` base image's bundled CUDA runtime must match host driver ≥ 535. Upgrading host driver without rebuilding the image can break compatibility, though forward-compat usually holds. | Low–Medium |
| 7 | **No GPU MIG support** | This server uses full RTX 4090s (MIG not supported on consumer GPUs anyway), so not a factor here. | None |

---

### 7.3 Bare-Metal (No Docker) Deployment

#### Pros

| # | Advantage | Relevance to This Server |
|---|---|---|
| 1 | **Zero container overhead** | Direct process-to-GPU communication; no `runc` or cgroups layer. Measurable benefit only for microsecond-sensitive workloads — PDF extraction latency (seconds) makes this difference negligible in practice. | Low in practice |
| 2 | **Full ONNX / CUDA library control** | Can compile PaddleOCR or ONNX Runtime with AVX-512 / specific CUDA arch flags tuned for Ada Lovelace. Docker's pre-built wheels cover this well for standard use. | Low |
| 3 | **System-level process schedulers** | `numactl`, `taskset`, `cgroups v2` can pin each GPU worker to specific CPU cores for cache locality. Modest gain on 16C Threadripper with 32 GB/channel memory bandwidth. | Low–Medium |
| 4 | **Simpler GPU driver upgrades** | `apt upgrade nvidia-driver-xxx`, no image rebuild required. | Medium |
| 5 | **Native `nvidia-smi` / NVML access** | All monitoring tools (`nvtop`, `gpustat`, DCGM) work directly without a sidecar container. | Low (Docker sidecars achieve same) |
| 6 | **No Docker daemon as a single point of failure** | If `dockerd` crashes, all 4 services drop. On bare-metal, each service is an independent systemd unit. | Medium |

#### Cons

| # | Disadvantage | Severity on This Server |
|---|---|---|
| 1 | **Dependency conflicts** | vLLM requires specific PyTorch + CUDA versions. PaddleOCR for the pipeline backend may require a different CUDA/cuDNN build. Running both on the same Python environment without venv isolation routinely causes breakage. | High |
| 2 | **Manual environment management** | Must maintain separate `virtualenv`s or `conda` environments per service, set `CUDA_VISIBLE_DEVICES` in systemd unit files, and manage `LD_LIBRARY_PATH` carefully. Error-prone. | High |
| 3 | **No model build-time baking** | Models (~20 GB) must be explicitly downloaded and path-configured in `~/mineru.json` for each service user. Keeping 4 service instances pointed at the same model directory with correct permissions is fiddly. | Medium |
| 4 | **Harder rolling upgrades** | Upgrading MinerU forces stopping the service, running `pip install --upgrade`, and restarting — affecting that GPU immediately with no blue/green option unless scripted manually. | Medium |
| 5 | **Systemd unit proliferation** | 4 services × 2 (MinerU + Prometheus exporters) = 8+ systemd unit files to maintain, each with correct environment variables, restart policies, and log rotation. | Medium |
| 6 | **Reproducibility deficit** | Reinstalling the server or provisioning a second machine requires repeating all environment setup steps manually; no image digest to pin to. | High (ops risk) |

---

### 7.4 Verdict for This Server

| Criterion | Docker | Bare-Metal | Winner |
|---|---|---|---|
| Dependency isolation (vLLM vs. PaddleOCR) | Excellent | Hard to achieve | **Docker** |
| GPU utilisation efficiency (4× RTX 4090, PCIe, no NVLink) | Negligible loss | Marginal gain | Tie |
| NUMA / CPU topology (single NUMA domain on 5955WX) | No loss | No benefit | Tie |
| Operational reproducibility | Image digest pinned | Manual reinstall risk | **Docker** |
| Rolling upgrades (production uptime) | Per-container restart | Full service stop | **Docker** |
| Storage (4 TB disk) | ~25 GB/image, trivial | Slightly less | Tie |
| Dependency on Docker daemon | Single point of failure | N/A | **Bare-Metal** |
| First-time setup complexity | `docker build` once | Complex env setup | **Docker** |

**Recommendation: Use Docker for this server.**

The RTX 4090 is a PCIe consumer GPU with no NVLink, no MIG, and no multi-node topology — the scenarios where Docker meaningfully loses to bare-metal do not apply. The Threadripper PRO 5955WX has a single NUMA domain, also eliminating Docker's NUMA penalty. Meanwhile Docker's dependency isolation is critical here because the 4-GPU split layout runs two distinct technology stacks (vLLM + PaddleOCR) that cannot safely share a Python environment. The 256 GB RAM covers all `shm_size` needs with room to spare, and the 4 TB disk absorbs the image layer overhead trivially.

**The one genuine bare-metal advantage** — eliminating the Docker daemon as a SPOF — can be mitigated by configuring `dockerd` with `live-restore: true` in `/etc/docker/daemon.json`, which keeps containers running if the daemon crashes and restarts:

```json
{
  "live-restore": true,
  "default-runtime": "nvidia"
}
```

---

## 8. Docker Orchestration Options

For a **single-node, 4-GPU server** the orchestration choice ranges from plain Docker CLI to full Kubernetes. Each layer adds operational capability at the cost of complexity.

---

### 8.1 Orchestration Options Compared

| Tool | Scope | GPU Support | Complexity | Best Fit |
|---|---|---|---|---|
| **Docker CLI** | Single node | Manual env vars | Minimal | Dev / one-off experiments |
| **Docker Compose** | Single node | `device_ids` in Compose | Low | **Recommended for this server** |
| **Docker Swarm** | Multi-node | Limited (no `device_ids`) | Medium | Multi-machine scaling without k8s |
| **Kubernetes + GPU Operator** | Multi-node | Full (NVIDIA GPU Operator) | High | Large-scale fleet (10+ nodes) |
| **Nomad + NVIDIA plugin** | Multi-node | Good | Medium | HashiCorp shops |

---

### 8.2 Docker Compose (Recommended)

**Why it fits this single-node, 4-GPU server best:**

- `device_ids: ["0"]` maps cleanly to one GPU per service
- Profile system (`--profile openai-server`, `--profile api`) lets you activate only what you need
- `restart: always` covers automatic recovery from crashes
- `live-restore: true` in `daemon.json` keeps containers up during daemon restarts
- Zero learning curve beyond standard Docker knowledge
- Full Compose file is already documented in Section 3.3

**Key Compose operational commands:**

```bash
# Start all 4 GPU services
docker compose -f /opt/mineru/docker-compose.yml \
  --profile openai-server \
  --profile api \
  up -d

# Rolling restart of a single GPU worker (e.g., after mineru:latest rebuild)
docker compose -f /opt/mineru/docker-compose.yml \
  pull mineru-openai-server-0 && \
  docker compose restart mineru-openai-server-0

# Scale is not applicable here — each GPU is already isolated to one container.
# Instead rebuild the image and restart one container at a time for zero-downtime upgrades.

# View aggregated logs from all services
docker compose -f /opt/mineru/docker-compose.yml logs -f --tail=100

# Graceful stop without removing containers
docker compose -f /opt/mineru/docker-compose.yml stop

# Full teardown
docker compose -f /opt/mineru/docker-compose.yml down
```

**Compose limitations on this server:**

| Limitation | Impact | Mitigation |
|---|---|---|
| No built-in load balancing | Clients must know port 30000 vs 30001 | Nginx upstream (Section 3.6) |
| No automatic service discovery | Services addressed by static port | Acceptable for a fixed 4-service layout |
| No health-based traffic routing | A sick container still receives traffic at its port | Nginx `max_fails` + `fail_timeout` upstream directives |
| Single-node only | Cannot span to a second server natively | Promote to Swarm or k8s when adding nodes |

---

### 8.3 Docker Swarm

Swarm wraps Docker Compose semantics with multi-node clustering and basic service mesh.

**When to consider it:** You add a **second GPU server** and want both nodes to appear as one pool without adopting Kubernetes.

#### Setting Up Swarm on This Server (Manager Node)

```bash
# Initialise — bind to the 10 GbE interface for cluster traffic
docker swarm init --advertise-addr <10GbE_IP>

# Generate join token for worker nodes
docker swarm join-token worker
```

#### GPU Support Gap in Swarm

Docker Swarm **does not support `device_ids`** or the `deploy.resources.reservations.devices` GPU syntax from Compose v3.9+ in native Swarm mode. The workaround is to use generic resources:

```bash
# On each Swarm node, advertise GPU resources in /etc/docker/daemon.json
{
  "node-generic-resources": ["NVIDIA-GPU=0", "NVIDIA-GPU=1", "NVIDIA-GPU=2", "NVIDIA-GPU=3"]
}
```

```yaml
# In the Swarm stack file, reserve a GPU generically
services:
  mineru-openai-server:
    image: mineru:latest
    deploy:
      replicas: 4
      resources:
        reservations:
          generic_resources:
            - discrete_resource_spec:
                kind: "NVIDIA-GPU"
                value: 1
```

**Critical problem:** Swarm's generic resource model assigns *a* GPU but cannot pin *which* GPU (`CUDA_VISIBLE_DEVICES=0` vs `=3`). This means you lose the deterministic GPU-to-service layout that is central to Section 2's strategy. **Verdict: Swarm is viable only if you add a second machine and can tolerate non-deterministic GPU assignment.**

#### Swarm Pros (for future multi-node scaling)

- Built-in overlay networking between nodes
- `docker service update --image mineru:v2` for rolling updates across nodes
- `docker service scale mineru-openai-server=8` distributes replicas across nodes
- Swarm Raft provides manager HA with 3+ manager nodes

#### Swarm Cons (for this single server)

- Loses `device_ids` GPU pinning — major regression from Compose
- Requires at least 3 manager nodes for HA (overkill for one server)
- Adds overlay network latency for inter-service calls on the same host
- Stack file syntax differs from Compose; maintenance friction

---

### 8.4 Kubernetes with NVIDIA GPU Operator

**When to consider it:** You are building a **fleet of 3+ GPU servers** or need advanced features: automatic pod rescheduling, horizontal pod autoscaling, GPU time-slicing, or integration with a model registry.

#### NVIDIA GPU Operator (Single-Node k8s)

The NVIDIA GPU Operator automates CUDA driver, container runtime, device plugin, DCGM, and MIG configuration on each node.

```bash
# Install k3s (lightweight k8s) — suitable for a single powerful server
curl -sfL https://get.k3s.io | sh -

# Add NVIDIA GPU Operator via Helm
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false   # host driver already installed
```

#### MinerU GPU Pod Example (k8s)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mineru-openai-server-gpu0
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mineru-vlm
      gpu: "0"
  template:
    metadata:
      labels:
        app: mineru-vlm
        gpu: "0"
    spec:
      containers:
        - name: mineru-openai-server
          image: mineru:latest
          command: ["mineru-openai-server"]
          args: ["--host", "0.0.0.0", "--port", "30000", "--gpu-memory-utilization", "0.75"]
          ports:
            - containerPort: 30000
          env:
            - name: MINERU_MODEL_SOURCE
              value: local
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
          resources:
            limits:
              nvidia.com/gpu: "1"         # request exactly 1 GPU
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi              # replaces --shm-size 32g
---
apiVersion: v1
kind: Service
metadata:
  name: mineru-vlm-gpu0
spec:
  selector:
    app: mineru-vlm
    gpu: "0"
  ports:
    - port: 30000
      targetPort: 30000
  type: ClusterIP
```

#### GPU Pinning in k8s

To deterministically pin a pod to a specific physical GPU, use node labels + `nodeAffinity` or the NVIDIA Device Plugin's `NVIDIA_VISIBLE_DEVICES` env override:

```yaml
env:
  - name: NVIDIA_VISIBLE_DEVICES
    value: "GPU-<UUID>"        # get UUID from: nvidia-smi -L
```

Or label the node and use `nodeAffinity` if each physical GPU is exposed as a separate k8s node (MIG-like virtual node slicing).

#### k8s Pros

| Feature | Benefit |
|---|---|
| Automatic pod restart (`restartPolicy: Always`) | More robust than `restart: always` in Compose — kubelet monitors health |
| Horizontal Pod Autoscaler | Scale VLM replicas up/down based on request queue depth (vLLM Prometheus metrics → HPA custom metrics) |
| NVIDIA GPU Operator | Unified DCGM monitoring, MIG management, driver lifecycle via Operator pattern |
| Ingress controllers (Nginx Ingress, Traefik) | Native load balancing + TLS termination, replaces manual Nginx config |
| Rolling updates | `kubectl rollout restart deployment/mineru-openai-server-gpu0` with zero-downtime |
| Multi-node ready | Add a second server as a worker node — pods schedule automatically |
| Secrets management | `kubectl create secret` for API keys, TLS certs — cleaner than env files |

#### k8s Cons for This Server

| Limitation | Impact |
|---|---|
| Steep operational overhead | k8s control plane, etcd, kubelet, kubeproxy — ~4 processes per node just for orchestration |
| GPU device plugin complexity | NVIDIA device plugin must correctly expose 4× RTX 4090 as 4 allocatable resources |
| `shm` via `emptyDir` is less ergonomic | Must explicitly create the `emptyDir: {medium: Memory}` volume — easy to forget |
| `ipc: host` equivalent is `hostIPC: true` (pod-level) | Less fine-grained than Compose's per-service setting |
| Single-node k8s adds overhead with no resilience gain | etcd on one node is still a SPOF — no HA without 3 control-plane nodes |
| Image pull policy for large images | 20 GB `mineru:latest` must be cached locally; `imagePullPolicy: Never` needed for local registry |

---

### 8.5 Orchestration Decision Tree for This Server

```
Single server, 4× RTX 4090, production service?
│
├─ Just this one server, stable load
│   └─→ Docker Compose   ✓ (current recommendation, Section 3.3)
│
├─ Adding a 2nd GPU server, want minimal complexity
│   └─→ Docker Swarm   (accept non-deterministic GPU assignment, or pin via node labels)
│
├─ 3+ GPU servers, or need autoscaling / HA control plane
│   └─→ Kubernetes + NVIDIA GPU Operator
│         ├─ k3s (lightweight) if team is small
│         └─ RKE2 / kubeadm if enterprise compliance required
│
└─ Heavy MLOps (model versioning, A/B testing, canary VLM rollouts)
    └─→ Kubernetes + KServe or Ray Serve on top of NVIDIA GPU Operator
```

---

### 8.6 Summary Recommendation

For this specific server today, **Docker Compose is the right orchestration layer**. It provides deterministic 1-GPU-per-service mapping, `restart: always` crash recovery, profile-based service activation, and clean integration with the Nginx + Prometheus monitoring stack described in previous sections — all with minimal operational surface area.

Adopt **Docker Swarm** only when a second server is added and multi-node scheduling is needed. Adopt **Kubernetes** when the fleet grows to 3+ nodes or when autoscaling on vLLM queue depth becomes a requirement.

---

## 9. Quick-Start Checklist

- [ ] NVIDIA driver ≥ 535, CUDA ≥ 12.1 installed and verified via `nvidia-smi`
- [ ] Docker + nvidia-container-toolkit installed and configured
- [ ] MinerU Docker image built: `docker build -t mineru:latest -f Dockerfile .`
- [ ] `docker-compose.yml` placed at `/opt/mineru/docker-compose.yml`
- [ ] All 4 services started and health checks passing
- [ ] Nginx reverse proxy configured with TLS for production access
- [ ] DCGM exporter + Prometheus + Grafana stack running
- [ ] Grafana DCGM dashboard (ID 12239) imported and showing all 4 GPUs
- [ ] Client machines have `mineru` (base) or `mineru[pipeline]` installed
- [ ] End-to-end test: `curl http://SERVER_IP:30000/health` → `{"status":"ok"}`
