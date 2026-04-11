# MinerU Docker Compose Deployment Guide

## Server Hardware Profile

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen Threadripper PRO 5955WX (16 Cores / 32 Threads) |
| GPU | 4 × NVIDIA GeForce RTX 4090 (Ada Lovelace, Compute Capability 8.9, 24 GB VRAM each) |
| Total GPU VRAM | 96 GB |
| CPU RAM | 256 GB DDR4 ECC |
| System Disk | 4 TB NVMe |
| NIC 1 | Aquantia AQC113C — 10 GbE |
| NIC 2 | Intel I210 — 1 GbE |
| OS | Ubuntu 22.04.5 LTS |

> **GPU Architecture Note**: RTX 4090 is Ada Lovelace (CC 8.9).  
> Use the `vllm/vllm-openai:cu130-nightly-*` base image (Dockerfile global) which covers CC 8.0–9.0.

---

## Architecture Overview

```
                ┌─────────────────────────────────────────────┐
                │              Host: Ubuntu 22.04              │
                │                                             │
  S3 / Client ──►  mineru-nginx  (port 80 / 8080)            │
                │        │                                    │
                │   ┌────┴────────────────────────────┐       │
                │   │      Round-Robin Load Balancer   │       │
                │   └────┬───────┬───────┬────────┬───┘       │
                │        │       │       │        │            │
                │  api-0 │ api-1 │ api-2 │  api-3 │           │
                │  GPU 0 │ GPU 1 │ GPU 2 │  GPU 3 │           │
                │  p8000 │ p8001 │ p8002 │  p8003 │           │
                │        │       │       │        │            │
                │  vlm-0 │ vlm-1 │ vlm-2 │  vlm-3 │           │
                │  GPU 0 │ GPU 1 │ GPU 2 │  GPU 3 │           │
                │ p30000 │p30001 │p30002 │ p30003 │           │
                │                                             │
                │  mineru-gradio  (port 7860, GPU 0)          │
                └─────────────────────────────────────────────┘
```

**Services by profile:**

| Profile | Service | Container | Host Port | GPU |
|---------|---------|-----------|-----------|-----|
| `openai` | vLLM OpenAI server | `mineru-vlm-gpu0..3` | 30000–30003 | 0–3 |
| `api` | FastAPI PDF endpoint | `mineru-api-gpu0..3` | 8000–8003 | 0–3 |
| `nginx` | Nginx load balancer | `mineru-nginx` | 80, 8080 | — |
| `gradio` | Gradio web UI | `mineru-gradio` | 7860 | 0 |

---

## Step-by-Step Setup

### Step 1 — Install Prerequisites

```bash
# 1a. NVIDIA drivers (skip if already installed)
ubuntu-drivers autoinstall
reboot

# 1b. Verify GPUs visible
nvidia-smi

# 1c. Install Docker Engine
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# 1d. Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 1e. Verify GPU passthrough works
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 1f. Install Docker Compose plugin
sudo apt-get install -y docker-compose-plugin
docker compose version
```

---

### Step 2 — Build the MinerU Image

```bash
cd /path/to/MinerU

# Build from global Dockerfile (HuggingFace model source)
docker build \
  --no-cache \
  -f docker/global/Dockerfile \
  -t mineru:latest \
  .
```

> **Estimated build time**: ~25–40 min (model download from HuggingFace is the bottleneck).  
> The models are baked into the image layer (`mineru-models-download -s huggingface -m all`), so containers start instantly.

---

### Step 3 — Create the Directory Structure

```bash
mkdir -p /home/orbit_user/mineru/{output,nginx-conf,logs}
```

---

### Step 4 — Create the Docker Compose Files

Create the following files under `Docker_compose/`:

- [`docker-compose.yml`](#file-docker-composeyml) — All service definitions for 4×GPU
- [`nginx.conf`](#file-nginxconf) — Nginx load-balancer configuration
- [`docker-compose.scale.yml`](#file-docker-composescaleyml) — Override for peak-traffic scale-out
- [`.env`](#file-env) — Environment variable defaults

#### File: `docker-compose.yml`

```yaml
# MinerU 4×RTX4090 Docker Compose — Full deployment
# Usage:
#   Start vLLM servers:  docker compose --profile openai up -d
#   Start API servers:   docker compose --profile api --profile nginx up -d
#   Start Gradio UI:     docker compose --profile gradio up -d
#   All services:        docker compose --profile openai --profile api --profile nginx --profile gradio up -d

x-mineru-common: &mineru-common
  image: ${MINERU_IMAGE:-mineru:latest}
  restart: unless-stopped
  environment:
    MINERU_MODEL_SOURCE: local
    MINERU_PROCESSING_WINDOW_SIZE: ${PROCESSING_WINDOW_SIZE:-16}
    MINERU_API_MAX_CONCURRENT_REQUESTS: ${MAX_CONCURRENT_REQUESTS:-8}
    OMP_NUM_THREADS: "4"
  ulimits:
    memlock: -1
    stack: 67108864
  ipc: host
  shm_size: "32g"
  volumes:
    - ${MINERU_OUTPUT_DIR:-/home/orbit_user/mineru/output}:/output

# ─── vLLM OpenAI-compatible servers (one per GPU) ─────────────────────────────
services:
  mineru-vlm-gpu0:
    <<: *mineru-common
    container_name: mineru-vlm-gpu0
    profiles: ["openai"]
    ports:
      - "30000:30000"
    entrypoint: mineru-openai-server
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "30000"
      - --gpu-memory-utilization
      - "${GPU_MEM_UTIL:-0.85}"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s

  mineru-vlm-gpu1:
    <<: *mineru-common
    container_name: mineru-vlm-gpu1
    profiles: ["openai"]
    ports:
      - "30001:30000"
    entrypoint: mineru-openai-server
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "30000"
      - --gpu-memory-utilization
      - "${GPU_MEM_UTIL:-0.85}"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s

  mineru-vlm-gpu2:
    <<: *mineru-common
    container_name: mineru-vlm-gpu2
    profiles: ["openai"]
    ports:
      - "30002:30000"
    entrypoint: mineru-openai-server
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "30000"
      - --gpu-memory-utilization
      - "${GPU_MEM_UTIL:-0.85}"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s

  mineru-vlm-gpu3:
    <<: *mineru-common
    container_name: mineru-vlm-gpu3
    profiles: ["openai"]
    ports:
      - "30003:30000"
    entrypoint: mineru-openai-server
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "30000"
      - --gpu-memory-utilization
      - "${GPU_MEM_UTIL:-0.85}"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["3"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s

# ─── FastAPI servers (one per GPU, each calls its own vLLM server) ─────────────
  mineru-api-gpu0:
    <<: *mineru-common
    container_name: mineru-api-gpu0
    profiles: ["api"]
    ports:
      - "8000:8000"
    entrypoint: mineru-api
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    environment:
      MINERU_MODEL_SOURCE: local
      MINERU_VLM_SERVER_URL: "http://mineru-vlm-gpu0:30000"
      MINERU_PROCESSING_WINDOW_SIZE: "${PROCESSING_WINDOW_SIZE:-16}"
      MINERU_API_MAX_CONCURRENT_REQUESTS: "${MAX_CONCURRENT_REQUESTS:-8}"
      OMP_NUM_THREADS: "4"
    depends_on:
      mineru-vlm-gpu0:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

  mineru-api-gpu1:
    <<: *mineru-common
    container_name: mineru-api-gpu1
    profiles: ["api"]
    ports:
      - "8001:8000"
    entrypoint: mineru-api
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    environment:
      MINERU_MODEL_SOURCE: local
      MINERU_VLM_SERVER_URL: "http://mineru-vlm-gpu1:30000"
      MINERU_PROCESSING_WINDOW_SIZE: "${PROCESSING_WINDOW_SIZE:-16}"
      MINERU_API_MAX_CONCURRENT_REQUESTS: "${MAX_CONCURRENT_REQUESTS:-8}"
      OMP_NUM_THREADS: "4"
    depends_on:
      mineru-vlm-gpu1:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]

  mineru-api-gpu2:
    <<: *mineru-common
    container_name: mineru-api-gpu2
    profiles: ["api"]
    ports:
      - "8002:8000"
    entrypoint: mineru-api
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    environment:
      MINERU_MODEL_SOURCE: local
      MINERU_VLM_SERVER_URL: "http://mineru-vlm-gpu2:30000"
      MINERU_PROCESSING_WINDOW_SIZE: "${PROCESSING_WINDOW_SIZE:-16}"
      MINERU_API_MAX_CONCURRENT_REQUESTS: "${MAX_CONCURRENT_REQUESTS:-8}"
      OMP_NUM_THREADS: "4"
    depends_on:
      mineru-vlm-gpu2:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]

  mineru-api-gpu3:
    <<: *mineru-common
    container_name: mineru-api-gpu3
    profiles: ["api"]
    ports:
      - "8003:8000"
    entrypoint: mineru-api
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    environment:
      MINERU_MODEL_SOURCE: local
      MINERU_VLM_SERVER_URL: "http://mineru-vlm-gpu3:30000"
      MINERU_PROCESSING_WINDOW_SIZE: "${PROCESSING_WINDOW_SIZE:-16}"
      MINERU_API_MAX_CONCURRENT_REQUESTS: "${MAX_CONCURRENT_REQUESTS:-8}"
      OMP_NUM_THREADS: "4"
    depends_on:
      mineru-vlm-gpu3:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["3"]
              capabilities: [gpu]

# ─── Nginx load balancer ───────────────────────────────────────────────────────
  mineru-nginx:
    image: nginx:1.27-alpine
    container_name: mineru-nginx
    profiles: ["nginx"]
    restart: unless-stopped
    ports:
      - "80:80"      # vLLM OpenAI endpoint (port 80 → upstream 30000)
      - "8080:8080"  # FastAPI endpoint     (port 8080 → upstream 8000)
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - /home/orbit_user/mineru/logs/nginx:/var/log/nginx
    depends_on:
      - mineru-vlm-gpu0
      - mineru-vlm-gpu1
      - mineru-vlm-gpu2
      - mineru-vlm-gpu3
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost/health 2>/dev/null | grep -q 'ok' || nginx -t"]
      interval: 15s
      timeout: 5s
      retries: 3

# ─── Gradio web UI (GPU 0 only) ────────────────────────────────────────────────
  mineru-gradio:
    <<: *mineru-common
    container_name: mineru-gradio
    profiles: ["gradio"]
    ports:
      - "7860:7860"
    entrypoint: mineru-gradio
    command:
      - --server-name
      - "0.0.0.0"
      - --server-port
      - "7860"
      - --gpu-memory-utilization
      - "${GPU_MEM_UTIL:-0.85}"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:7860 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s
```

#### File: `nginx.conf`

```nginx
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    # ── Upstream: vLLM OpenAI servers (one per GPU) ──────────────────────────
    upstream vlm_backends {
        least_conn;
        server mineru-vlm-gpu0:30000 max_fails=3 fail_timeout=30s;
        server mineru-vlm-gpu1:30000 max_fails=3 fail_timeout=30s;
        server mineru-vlm-gpu2:30000 max_fails=3 fail_timeout=30s;
        server mineru-vlm-gpu3:30000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # ── Upstream: FastAPI PDF endpoints ──────────────────────────────────────
    upstream api_backends {
        least_conn;
        server mineru-api-gpu0:8000 max_fails=3 fail_timeout=30s;
        server mineru-api-gpu1:8000 max_fails=3 fail_timeout=30s;
        server mineru-api-gpu2:8000 max_fails=3 fail_timeout=30s;
        server mineru-api-gpu3:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # ── Common proxy settings ─────────────────────────────────────────────────
    proxy_http_version  1.1;
    proxy_set_header    Connection "";
    proxy_set_header    Host              $host;
    proxy_set_header    X-Real-IP         $remote_addr;
    proxy_set_header    X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_read_timeout  900s;   # large PDFs can take a while
    proxy_send_timeout  900s;
    client_max_body_size 512m;  # max PDF upload size

    access_log /var/log/nginx/access.log combined;
    error_log  /var/log/nginx/error.log warn;

    # ── Port 80: vLLM OpenAI API ──────────────────────────────────────────────
    server {
        listen 80;
        server_name _;

        location /health {
            return 200 'ok';
            add_header Content-Type text/plain;
        }

        location / {
            proxy_pass http://vlm_backends;
        }
    }

    # ── Port 8080: FastAPI PDF upload API ─────────────────────────────────────
    server {
        listen 8080;
        server_name _;

        location / {
            proxy_pass http://api_backends;
        }
    }
}
```

#### File: `docker-compose.scale.yml`

This override adds extra **pipeline-mode** workers for peak traffic when GPU capacity is saturated.
Apply with `-f docker-compose.yml -f docker-compose.scale.yml`.

```yaml
# Peak-traffic scale-out override
# Usage: docker compose -f docker-compose.yml -f docker-compose.scale.yml \
#          --profile openai --profile api --profile nginx up -d

services:
  # Increase nginx worker connections for burst traffic
  mineru-nginx:
    environment:
      NGINX_WORKER_PROCESSES: "4"

  # Additional pipeline-backend API replicas (CPU/OCR only, no GPU needed)
  mineru-api-pipeline-1:
    image: ${MINERU_IMAGE:-mineru:latest}
    container_name: mineru-api-pipeline-1
    profiles: ["scale"]
    restart: unless-stopped
    ports:
      - "8010:8000"
    entrypoint: mineru-api
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    environment:
      MINERU_MODEL_SOURCE: local
      OMP_NUM_THREADS: "4"
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    shm_size: "8g"
    volumes:
      - ${MINERU_OUTPUT_DIR:-/home/orbit_user/mineru/output}:/output

  mineru-api-pipeline-2:
    image: ${MINERU_IMAGE:-mineru:latest}
    container_name: mineru-api-pipeline-2
    profiles: ["scale"]
    restart: unless-stopped
    ports:
      - "8011:8000"
    entrypoint: mineru-api
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    environment:
      MINERU_MODEL_SOURCE: local
      OMP_NUM_THREADS: "4"
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    shm_size: "8g"
    volumes:
      - ${MINERU_OUTPUT_DIR:-/home/orbit_user/mineru/output}:/output
```

#### File: `.env`

```dotenv
# MinerU Docker Compose — environment defaults
MINERU_IMAGE=mineru:latest
MINERU_OUTPUT_DIR=/home/orbit_user/mineru/output
GPU_MEM_UTIL=0.85
PROCESSING_WINDOW_SIZE=16
MAX_CONCURRENT_REQUESTS=8
```

---

### Step 5 — Deploy

```bash
cd Docker_compose/

# ── Option A: all services at once ───────────────────────────────────────────
docker compose \
  --profile openai \
  --profile api \
  --profile nginx \
  --profile gradio \
  up -d

# ── Option B: vLLM + API + Nginx only (headless, for batch processing) ────────
docker compose \
  --profile openai \
  --profile api \
  --profile nginx \
  up -d

# ── Option C: gradio UI only (demo / testing on GPU 0) ───────────────────────
docker compose --profile gradio up -d
```

---

## Container Status — All Commands

### View All Container Status

```bash
# Compact summary (name, status, ports)
docker compose ps

# Full detail including health
docker compose ps --all --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# Raw docker ps with GPU context
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"

# Watch live (refresh every 2 s)
watch -n 2 'docker compose ps'
```

### Check All Container Health

```bash
# Health status only — all containers
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "healthy|unhealthy|starting"

# Inspect a specific container's health log
docker inspect --format='{{json .State.Health}}' mineru-vlm-gpu0 | python3 -m json.tool

# Loop over all mineru containers
for c in mineru-vlm-gpu{0..3} mineru-api-gpu{0..3} mineru-nginx mineru-gradio; do
  status=$(docker inspect --format='{{.State.Health.Status}}' "$c" 2>/dev/null || echo "N/A")
  echo "$c → $status"
done

# GPU utilization alongside container health
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free \
           --format=csv,noheader,nounits

# Combined GPU + container overview
docker stats --no-stream --format \
  "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
```

### Real-Time Log Tailing

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f mineru-vlm-gpu0

# Last 100 lines of all api services
docker compose logs --tail=100 mineru-api-gpu0 mineru-api-gpu1 \
                                mineru-api-gpu2 mineru-api-gpu3

# Nginx access log
docker exec mineru-nginx tail -f /var/log/nginx/access.log
```

---

## Auto-Scaling for Peak Traffic

This server does not run Kubernetes, so auto-scaling is achieved via a shell-based
monitoring loop that:

1. Reads GPU utilisation from `nvidia-smi`.
2. Starts/stops extra pipeline-backend replicas (CPU-only, no GPU) using
   `docker compose up / down`.
3. Hot-reloads the Nginx upstream config when replicas change.

### `autoscale.sh`

Create this file at `/home/orbit_user/mineru/autoscale.sh`:

```bash
#!/usr/bin/env bash
# autoscale.sh – scale MinerU pipeline workers based on GPU utilisation
set -euo pipefail

COMPOSE_DIR="/home/orbit_user/MinerU/Docker_compose"
LOG="/home/orbit_user/mineru/logs/autoscale.log"
SCALE_UP_THRESHOLD=80    # % GPU util → add pipeline workers
SCALE_DOWN_THRESHOLD=30  # % GPU util → remove extra workers
CHECK_INTERVAL=60        # seconds between checks

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

avg_gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    | awk '{s+=$1; n++} END {print int(s/n)}'
}

scale_up() {
  log "GPU util ${UTIL}% ≥ ${SCALE_UP_THRESHOLD}% — scaling UP pipeline workers"
  docker compose -f "${COMPOSE_DIR}/docker-compose.yml" \
                 -f "${COMPOSE_DIR}/docker-compose.scale.yml" \
                 --profile scale up -d --no-recreate
}

scale_down() {
  log "GPU util ${UTIL}% ≤ ${SCALE_DOWN_THRESHOLD}% — scaling DOWN pipeline workers"
  docker compose -f "${COMPOSE_DIR}/docker-compose.yml" \
                 -f "${COMPOSE_DIR}/docker-compose.scale.yml" \
                 --profile scale down 2>/dev/null || true
}

SCALED_UP=false

while true; do
  UTIL=$(avg_gpu_util)
  log "Average GPU util: ${UTIL}%"

  if [[ $UTIL -ge $SCALE_UP_THRESHOLD ]] && [[ "$SCALED_UP" == "false" ]]; then
    scale_up
    SCALED_UP=true
  elif [[ $UTIL -le $SCALE_DOWN_THRESHOLD ]] && [[ "$SCALED_UP" == "true" ]]; then
    scale_down
    SCALED_UP=false
  fi

  sleep "$CHECK_INTERVAL"
done
```

```bash
chmod +x /home/orbit_user/mineru/autoscale.sh

# Run as a background daemon via systemd (recommended)
sudo tee /etc/systemd/system/mineru-autoscale.service > /dev/null <<'EOF'
[Unit]
Description=MinerU Auto-Scale Monitor
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/home/orbit_user/mineru/autoscale.sh
Restart=always
RestartSec=10
User=orbit_user

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now mineru-autoscale.service
sudo systemctl status mineru-autoscale.service
```

### Manual Scale Commands

```bash
# Add scale-out workers immediately
docker compose \
  -f docker-compose.yml \
  -f docker-compose.scale.yml \
  --profile scale up -d

# Remove scale-out workers
docker compose \
  -f docker-compose.yml \
  -f docker-compose.scale.yml \
  --profile scale down

# Verify running containers after scaling
docker compose ps
```

---

## All YAML Files in This Deployment

| File | Purpose |
|------|---------|
| `Docker_compose/docker-compose.yml` | Primary service definitions (4×GPU vLLM, 4×FastAPI, Nginx, Gradio) |
| `Docker_compose/docker-compose.scale.yml` | Scale-out override (extra CPU pipeline workers for peak traffic) |
| `Docker_compose/nginx.conf` | Nginx load-balancer config |
| `Docker_compose/.env` | Default environment variables |
| `docker/compose.yaml` | Upstream single-GPU reference YAML (not used directly in this guide) |
| `docker/global/Dockerfile` | NVIDIA GPU image (HuggingFace model source) |
| `docker/china/Dockerfile` | NVIDIA GPU image (ModelScope mirror) |

---

## Full Docker Command Reference

```bash
# ── Build ─────────────────────────────────────────────────────────────────────
docker build -f docker/global/Dockerfile -t mineru:latest .
docker build -f docker/global/Dockerfile -t mineru:v2.7.0 --no-cache .

# ── Start / Stop / Restart ───────────────────────────────────────────────────
docker compose --profile openai --profile api --profile nginx up -d
docker compose --profile openai --profile api --profile nginx down
docker compose restart mineru-vlm-gpu0
docker compose stop mineru-api-gpu2
docker compose start mineru-api-gpu2

# ── Status & Health ───────────────────────────────────────────────────────────
docker compose ps
docker compose ps --format json | python3 -m json.tool
docker inspect --format='{{json .State.Health}}' mineru-vlm-gpu0 | python3 -m json.tool
docker stats --no-stream
nvidia-smi

# ── Logs ──────────────────────────────────────────────────────────────────────
docker compose logs -f
docker compose logs -f --tail=200 mineru-vlm-gpu0
docker exec mineru-nginx tail -f /var/log/nginx/access.log

# ── Exec / Debug ─────────────────────────────────────────────────────────────
docker exec -it mineru-vlm-gpu0 bash
docker exec mineru-vlm-gpu0 nvidia-smi
docker exec mineru-vlm-gpu0 env | grep MINERU

# ── Image Management ──────────────────────────────────────────────────────────
docker images | grep mineru
docker rmi mineru:latest
docker image prune -f

# ── Volume / Network ─────────────────────────────────────────────────────────
docker volume ls
docker network ls
docker network inspect docker_compose_default

# ── Scale-out (override file) ─────────────────────────────────────────────────
docker compose -f docker-compose.yml -f docker-compose.scale.yml \
  --profile scale up -d
docker compose -f docker-compose.yml -f docker-compose.scale.yml \
  --profile scale down

# ── Clean up everything (DESTRUCTIVE — stops all containers and removes images)
# docker compose down --rmi all --volumes --remove-orphans
```

---

## `batch_download_s3_final.py` — Compatibility with Docker Compose

### Current Behaviour (in-process mode)

The script calls `do_parse()` as a **Python function call inside the same process**:

```python
from mineru.cli.common import do_parse

do_parse(
    output_dir=MINERU_OUTPUT_DIR,
    pdf_file_names=pdf_names,
    pdf_bytes_list=pdf_bytes_list,
    p_lang_list=[LANG] * len(pdf_names),
    backend="vlm-auto-engine",        # ← runs vLLM in-process
    gpu_memory_utilization=GPU_MEM_UTIL,
    data_parallel_size=GPU_COUNT,
)
```

`backend="vlm-auto-engine"` resolves at runtime to either `vllm-engine` or
`lmdeploy-engine` (via `get_vlm_engine()`), and **loads the model directly into
the calling process via `ModelSingleton`**.

### Problem with Docker Compose

In a Docker Compose deployment:

- vLLM is already running inside `mineru-vlm-gpu{0..3}` containers.
- Running the script on the **host** (outside a container) will try to load
  another copy of the model into RAM/VRAM — **double the resource usage** and
  likely an OOM error on the host.
- Running the script **inside a container** that also runs a vLLM server is the
  same problem.

### Recommended Fix — Use the API Client Backend

Change `batch_download_s3_final.py` so that `do_parse` talks to the load-balanced
Nginx endpoint **instead of** loading the model in-process:

```python
# Before (in-process, loads vLLM locally):
backend = "vlm-auto-engine"
server_url = None

# After (client mode, routes to Docker Compose Nginx load balancer):
backend = "vlm-openai-client"
server_url = "http://localhost:80"   # Nginx → vlm_backends upstream
```

`do_parse` internally strips the `"vlm-"` prefix → `"openai-client"`.  
Because the backend string ends with `"client"`, the `server_url` is **not**
discarded (see `common.py`: `if not backend.endswith("client"): server_url = None`).
The request is forwarded to the running Docker Compose VLM servers.

### Minimal Change to `batch_download_s3_final.py`

```python
# Replace these two constants at the top of the file:
GPU_MEM_UTIL = 0.85          # no longer used when running as API client
GPU_COUNT = 2                # no longer used when running as API client

# Add:
MINERU_SERVER_URL = "http://localhost:80"   # Nginx load balancer

# In process_pdfs_in_memory(), change do_parse call to:
do_parse(
    output_dir=MINERU_OUTPUT_DIR,
    pdf_file_names=pdf_names,
    pdf_bytes_list=pdf_bytes_list,
    p_lang_list=[LANG] * len(pdf_names),
    backend="vlm-openai-client",    # ← client mode, no in-process model load
    parse_method="auto",
    server_url=MINERU_SERVER_URL,   # ← points at Nginx → vLLM containers
)
```

### Alternative — Run the Script Inside a Container

If you prefer **zero code changes**, run the batch script inside one of the
existing MinerU containers.  The container already has MinerU installed and the
right CUDA environment, and you can override the entrypoint:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e MINERU_MODEL_SOURCE=local \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
  -v /home/orbit_user/mineru/output:/home/orbit_user/mineru/output \
  -v /path/to/batch_download_s3_final.py:/workspace/batch_download_s3_final.py \
  --entrypoint python3 \
  mineru:latest \
  /workspace/batch_download_s3_final.py
```

In this mode `do_parse(backend="vlm-auto-engine")` works correctly because it
runs **inside** the container where the model is already cached at
`/root/.cache/modelscope` and the GPU is exclusively available.

### Summary

| Run mode | `do_parse` works as-is? | Notes |
|----------|------------------------|-------|
| Host machine, no Docker Compose | ✅ Yes | Loads model in-process; uses GPU directly |
| Host machine, Docker Compose running | ⚠️ Conflict | Host and container both try to claim GPU VRAM |
| Inside `docker run` container | ✅ Yes | No code change needed; use `--entrypoint python3` |
| Host machine calling Nginx API | ✅ Yes (with code change) | Change `backend` to `"vlm-openai-client"`, set `server_url` |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `unhealthy` on vLLM container | Model still loading (slow first start) | `start_period: 120s` in healthcheck; wait 2–3 min |
| `CUDA out of memory` | GPU_MEM_UTIL too high or dual usage (host + container) | Reduce `GPU_MEM_UTIL` to 0.8; stop host processes using GPU |
| `curl: connection refused` | Nginx started before vLLM was healthy | `depends_on: condition: service_healthy` ensures ordering |
| `shm_size` error on `docker compose up` | Old Docker version | Upgrade Docker Engine ≥ 24.0 |
| `nvidia-container-toolkit` not found | Toolkit not installed | Re-run Step 1d |
| PDF upload returns 503 | All API workers at max capacity | Increase `MAX_CONCURRENT_REQUESTS` in `.env` or add scale-out workers |

# Additional

What's in the guide
Hardware-tuned architecture — 4 independent vLLM servers (mineru-vlm-gpu0..3), each pinned to one RTX 4090 via device_ids, plus 4 FastAPI workers fronted by an Nginx round-robin load balancer. Uses the vllm/vllm-openai:cu130-nightly base image, which is correct for Ada Lovelace CC 8.9.

All YAML files defined inline:

docker-compose.yml — full 4-GPU service stack with x-mineru-common YAML anchors to avoid repetition
docker-compose.scale.yml — scale-out override adding CPU-only pipeline workers during peak load
nginx.conf — least_conn load balancing, 900 s proxy timeout for large PDFs, 512 MB upload limit
.env — tunable defaults (GPU_MEM_UTIL, MAX_CONCURRENT_REQUESTS, etc.)
Container monitoring commands — docker compose ps, docker inspect health JSON, per-container loop, docker stats, nvidia-smi.

Auto-scaling — shell daemon (autoscale.sh) that polls average GPU utilisation via nvidia-smi and starts/stops --profile scale workers; deployed as a systemd service.

do_parse compatibility analysis:

Running the script on the host while Docker Compose is live causes a VRAM conflict — both the host process and the containers would claim GPU memory.
Fix option 1 (minimal code change): change backend="vlm-openai-client" and server_url="http://localhost:80" so do_parse routes to the Nginx load balancer instead of loading the model in-process.
Fix option 2 (zero code change): run the batch script via docker run --entrypoint python3 mineru:latest /workspace/batch_download_s3_final.py, which keeps do_parse(backend="vlm-auto-engine") working correctly inside the container.

