# MinerU Deployment Strategies — Comparison & Best Practices

## Background: The Two-Tier Compute Problem

MinerU document processing has two fundamentally different workloads with mismatched resource demands:

| Tier | Workload | Bottleneck |
|---|---|---|
| **VLM inference** | Qwen2-VL 1.2B (`MinerU2.5-2509-1.2B`) — neural network over page images | GPU compute + VRAM bandwidth |
| **Pipeline post-processing** | OCR, layout parsing, formula/table detection, reading-order sorting | CPU + moderate GPU (CV models) |

Every strategy below is a different answer to the same question: **how do you distribute these two tiers across available hardware?**

---

## Strategy 0 — Hybrid Split Mode on a Single Server (from `4x_rtx4090_deploy.md`)

### Description

All four GPUs live in **one physical machine**. Roles are partitioned by GPU index:

```
GPU 0  →  LitServe pipeline worker A1 + A2  (hybrid-http-client backend)
GPU 1  →  LitServe pipeline worker B1 + B2  (hybrid-http-client backend)
GPU 2  →  vllm data-parallel shard 0        (VLM inference, port 30000)
GPU 3  →  vllm data-parallel shard 1        (VLM inference, port 30000)
```

- 2 pipeline workers × 2 GPUs = **4 concurrent PDF-to-Markdown jobs**
- vllm `data-parallel-size=2` gives two independent VLM inference streams
- An optional nginx reverse proxy serves as the public-facing entry point
- Systemd service dependency (`Requires=mineru-vlm.service`) prevents the pipeline from accepting requests before the VLM model finishes loading

### Why the rationale holds for this hardware

- The RTX 4090 has 24 GB VRAM. The 1.2B VLM model weights consume ~2.5 GB — a single card can hold the full model with 21.5 GB left for the KV cache. Tensor parallelism across two cards would only add NVLink/PCIe synchronisation overhead for no VRAM benefit: **data-parallel (one full copy per card) is strictly better here.**
- 16 CPU cores / 32 threads map cleanly onto 4 workers × 8 PDF render threads (`MINERU_PDF_RENDER_THREADS=8`), saturating the CPU at full load without contention.
- The 10 GbE NIC (AQC113C) provides ~1.25 GB/s ingress — sufficient for continuous high-frequency PDF uploads.
- Because everything runs on one machine, VLM calls from the pipeline workers travel over loopback (`127.0.0.1:30000`) with zero network latency and no serialisation overhead.

### Strengths

- **Lowest operational complexity** — one host to provision, monitor, and patch.
- **Minimum latency** — loopback communication between pipeline and VLM tiers.
- **Deterministic resource isolation** — `CUDA_VISIBLE_DEVICES` hard-pins each process to specific cards; no GPU contention between tiers.
- **Full throughput** — `workers_per_device=2` × 2 GPUs delivers 4 concurrent pipeline jobs while two VLM shards absorb page batches in parallel.
- **Production-grade resilience** — systemd dependency graph + `ExecStartPre` health probe ensures correct startup order.

### Weaknesses

- **Single point of failure.** The entire service is down if the machine is lost.
- **Vertical scaling ceiling.** Adding capacity requires a second server; horizontal scale-out is not built in.
- **Workload imbalance risk.** If the document set is predominantly native text (no image pages), GPU 2–3 sit underutilised while GPU 0–1 are the bottleneck. The guide notes this case: switch all 4 GPUs to pipeline workers with `parse_method=txt`.

### Best for

A single high-memory GPU workstation or on-prem server running a high-throughput but bounded document pipeline (research lab, legal department, internal data engineering).

---

## Strategy 1 — Centralized VLM, Distributed Pipeline

### Description

```
┌───────────────────────────────────────────────────────────────────┐
│  GPU Cluster (data center or cloud)                               │
│  mineru-openai-server  ──  VLM inference only  ──  port 30000    │
└────────────────────────────────┬──────────────────────────────────┘
                                 │  HTTP (internal network)
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
  Edge Device A           Edge Device B           Edge Device C
  mineru[pipeline]        mineru[pipeline]        mineru[pipeline]
  hybrid-http-client      hybrid-http-client      hybrid-http-client
  → server_url=...        → server_url=...        → server_url=...
```

Each edge client runs the full pipeline stack (layout detection, OCR, formula recognition, table parsing) locally on whatever hardware it has. The heavy VLM inference step is offloaded to the central GPU cluster over HTTP.

### Why this approach makes sense

The key insight is that **pipeline CV models are small** (~1–2 GB VRAM, ~200 MB each for layout, MFD, MFR, table, reading-order models). They run on a modest GPU or even CPU. **Only the VLM step (Qwen2-VL) demands high-end GPU VRAM.** Concentrating VLM capacity saves money: you get one large VLM server instead of buying a high-end GPU for every processing node.

`MINERU_OPENAI_BASE_URL` (or `--server-url` per request) is the single configuration knob that points a pipeline worker at the remote VLM server.

### Strengths

- **Cost efficiency at scale.** N edge workers share M VLM GPUs where M ≪ N if edge devices are CPU-only or have modest GPUs.
- **Horizontal pipeline scaling.** Add edge workers freely — they consume only CPU/bandwidth.
- **Heterogeneous hardware support.** Edge workers can be laptops, ARM servers, or small cloud instances. Only the VLM cluster needs high-VRAM GPUs.
- **Independent upgrade paths.** Upgrade the VLM cluster without touching edge workers, or deploy new VLM model versions (point URL, reload).

### Weaknesses

- **Network dependency is critical.** Every page image must round-trip to the VLM cluster. On a slow or unreliable network, this degrades latency and reliability.
- **VLM cluster is a single bottleneck.** If the cluster is saturated, all edge workers queue. Capacity planning shifts from per-machine to per-cluster.
- **Image serialisation overhead.** Pipeline workers must encode page images (JPEG/PNG) and HTTP-POST them to the VLM server. At high concurrency this adds measurable CPU load on edge devices.
- **Security surface.** The VLM server endpoint must be reachable from edge devices; TLS + auth are required in production (`Authorization: Bearer` header supported by `mineru-openai-server` via vllm's `--api-key` flag).
- **No meaningful fallback.** If the VLM cluster is down, `hybrid-http-client` has no local VLM fallback; requests fail until the server recovers. Strategy 0 avoids this by co-locating both tiers.

### Best for

Large enterprises with a mix of heterogeneous edge servers (branch offices, lab workstations) that need document processing but cannot justify a high-end GPU at every location. Ideal when the document workload is bursty — edge capacity scales freely while VLM capacity handles average load.

---

## Strategy 2 — Mixed Backend Services (Tiered Accuracy)

### Description

```
GPU 0  →  mineru-openai-server  (VLM only, port 30000)
GPU 1  →  mineru-api            (pipeline backend, port 8000)

Clients route based on accuracy requirement:
  ─ High accuracy (scanned PDFs, complex layouts) → port 30000 (VLM)
  ─ Standard accuracy (native PDFs, simple text)  → port 8000  (pipeline)
```

Two entirely different serving stacks run on the same machine, each owning one GPU. The routing decision sits in the calling application or an upstream API gateway.

### Why this approach exists

Not all documents need VLM. Native PDFs with embedded text layers can be parsed at high accuracy by the pipeline backend (pdfmium text extraction + CV detection) at a fraction of the compute cost. Running a dedicated VLM GPU for those documents is wasteful.

Strategy 2 exposes a **cost/accuracy trade-off** at the API level:
- `pipeline` backend on GPU 1 → fast, cheap, suitable for ≥90% of well-formed PDFs
- `vlm-http-client` against GPU 0 → accurate, slower, reserved for degraded/scanned/handwritten documents

### Strengths

- **Cost-effective for mixed workloads.** Cheap pipeline path handles the bulk; expensive VLM path handles the hard cases.
- **Simple hardware.** Two GPUs are enough; no need for 4× high-end cards beyond proof-of-concept.
- **Tiered SLA.** You can attach different pricing, quotas, or latency SLAs to each tier.

### Weaknesses

- **Client-side routing complexity.** The caller must decide which tier to use. Misclassification (sending a scanned PDF to the pipeline backend) silently produces low-quality output. Automating the routing decision requires a classifier or pre-scan step — reintroducing complexity at the application layer.
- **No GPU redundancy.** Each GPU has a distinct role. If GPU 0 (VLM) fails, all high-accuracy requests fail; pipeline requests are unaffected. If GPU 1 (pipeline) fails, the inverse. Contrast with Strategy 0 where both pipeline GPUs are interchangeable.
- **Thin throughput with 1 GPU per tier.** One VLM GPU handles only a single inference stream (no data-parallel). Under sustained load the VLM queue grows quickly.
- **Operational fragmentation.** Two services, two health checks, two monitoring dashboards, two restart policies.
- **VRAM efficiency loss.** With `workers_per_device=1` (implied by a single GPU per tier), you forfeit the pipeline throughput gains from Strategy 0's `workers_per_device=2`.

### Best for

Development environments, proof-of-concept deployments, or small teams that need to evaluate pipeline vs. VLM accuracy side-by-side without committing to a full multi-GPU setup.

---

## Strategy 3 — API Server as Gateway

### Description

```bash
# Central entry point — accepts HTTP file uploads
mineru-api --host 0.0.0.0 --port 8000

# Delegates VLM inference to a remote/local vllm cluster
export MINERU_OPENAI_BASE_URL=http://internal-vllm-cluster:30000
```

`mineru-api` (built on FastAPI) is deployed as the **sole public-facing service**. It receives file upload requests, runs the pipeline locally, and transparently forwards VLM page batches to whatever server `MINERU_OPENAI_BASE_URL` points to. The VLM endpoint can be local, on the same LAN, or a cloud inference provider.

### Why this approach exists

`mineru-api` provides a clean REST interface (`POST /file_parse`) with multipart form uploads, JSON responses, and built-in concurrency gating (`MINERU_API_MAX_CONCURRENT_REQUESTS`). Strategy 3 leans on this interface as the **canonical entry point** rather than treating it as an optional component (as in Strategy 0 Step 7).

The environment variable `MINERU_OPENAI_BASE_URL` makes the VLM target fully configurable without code changes — a good fit for infrastructure-as-code and 12-factor deployments.

### Relationship to Strategy 0

In `4x_rtx4090_deploy.md`, `mineru-api` appears in **Step 7** as *optional* — it is mentioned as a simpler alternative to the LitServe server when multi-GPU LitServe binding is not needed. Strategy 3 elevates this optional component to the primary architecture.

The key difference: Strategy 0 uses **LitServe** to manage multi-GPU worker dispatch (`accelerator="cuda"`, `devices=[0,1]`, `workers_per_device=2`), which gives finer-grained concurrency control. `mineru-api` does not natively manage per-GPU worker pools — it dispatches requests to whatever device MinerU's backend selects. This makes `mineru-api` simpler to operate but less controllable under sustained load.

### Strengths

- **Simplest deployment.** One command starts the full service. No LitServe server.py, no manual GPU pinning, no separate VLM start sequence.
- **Flexible VLM backend.** Point `MINERU_OPENAI_BASE_URL` at any OpenAI-compatible endpoint: local vllm, a cloud API, or a remote VLM cluster (Strategy 1's server).
- **Standard REST interface.** `POST /file_parse` with multipart uploads is straightforward for any HTTP client; no base64 encoding required (compare with LitServe's `POST /predict` + base64 body).
- **Good composability.** Works well in Docker Compose (`environment: MINERU_OPENAI_BASE_URL: http://mineru-vlm:30000`). The `4x_rtx4090_deploy.md` Docker Compose section uses exactly this pattern.

### Weaknesses

- **No native multi-GPU worker pool management.** `mineru-api` uses `MINERU_API_MAX_CONCURRENT_REQUESTS` to gate concurrency but does not explicitly distribute work across multiple pipeline GPUs the way LitServe does. On a 4-GPU server, GPU 0–1 may sit idle unless the caller or `CUDA_VISIBLE_DEVICES` explicitly sets device routing.
- **Hidden VLM dependency.** If `MINERU_OPENAI_BASE_URL` is unset or the remote server is down, requests silently fall back to local VLM (if installed) or fail with a connection error. The failure mode is less obvious than Strategy 0's systemd health probe.
- **Single-process bottleneck.** One `mineru-api` process handles all requests serially up to the concurrency gate. No equivalent of LitServe's per-GPU worker processes.
- **Output delivery model.** `mineru-api` returns results synchronously within the HTTP response (for small outputs) or via a filesystem path. For very large PDFs this can hit nginx/proxy timeout limits unless `proxy_read_timeout` is tuned (Strategy 0's nginx config addresses this explicitly at 900s).

### Best for

Teams that need a quick, standards-compliant HTTP endpoint for document processing and are comfortable pointing `MINERU_OPENAI_BASE_URL` at a managed or shared VLM cluster. Recommended for the Docker Compose deployment path (as shown in Step 10 of `4x_rtx4090_deploy.md`).

---

## Side-by-Side Comparison

| Dimension | Strategy 0 (Hybrid Split, 4× GPU) | Strategy 1 (Centralized VLM) | Strategy 2 (Mixed Backends) | Strategy 3 (API Gateway) |
|---|---|---|---|---|
| **Hardware requirement** | 1 × 4-GPU server | GPU cluster + N edge workers | 1 × 2-GPU server (minimum) | 1 server + any VLM endpoint |
| **VLM location** | Co-located (loopback) | Remote cluster | Co-located GPU 0 | Configurable via env var |
| **Pipeline location** | Co-located GPU 0–1 | Distributed (edge) | Co-located GPU 1 | Same process as gateway |
| **Max concurrent jobs** | 4 (2 workers × 2 GPUs) | Unlimited pipeline workers | 1 pipeline + 1 VLM | Limited by `MAX_CONCURRENT_REQUESTS` |
| **VLM throughput** | 2× (data-parallel) | Scales with cluster size | 1× (single GPU) | Depends on VLM endpoint |
| **Network latency (VLM calls)** | ~0 ms (loopback) | 10–100 ms (LAN/WAN) | ~0 ms (loopback) | Variable |
| **Operational complexity** | Medium (2 systemd units) | High (cluster + edge fleet) | Low (2 services) | Low (1 service) |
| **Horizontal scalability** | None (single host) | Excellent (edge unlimited) | None | Good (multiple API servers) |
| **GPU redundancy** | Partial (2× pipeline GPUs) | High (cluster redundancy) | None (each GPU unique) | Depends on VLM endpoint |
| **Docker Compose ready** | Yes (Step 10 of deploy guide) | Partial (VLM cluster separate) | Yes (simple) | Yes (canonical use case) |
| **Recommended for** | Single server, max throughput | Multi-site, heterogeneous HW | Dev/POC, mixed accuracy | Simple deployments, CI/CD |

---

## Best Practice Recommendations

### 1. Use Strategy 0 as your production baseline on a multi-GPU server

If you have 4 × RTX 4090 (or equivalent), the Hybrid Split Mode in `4x_rtx4090_deploy.md` is the right default. Data-parallel vllm on GPUs 2–3 + LitServe on GPUs 0–1 gives:
- Zero VLM network latency
- 4 concurrent pipeline jobs
- Full CPU utilisation via `workers_per_device=2` + render thread pinning
- Production-grade startup ordering via systemd

**Do not skip the `ExecStartPre` health probe.** The vllm model load takes 30–60 seconds; pipeline workers that start too early will send requests to a not-yet-ready VLM server and produce connection errors. The health probe loop (`until curl -sf http://127.0.0.1:30000/health`) is a low-effort guard against this.

### 2. Adopt Strategy 3 (API Gateway) for Docker and cloud-native deployments

When running in containers, the `mineru-api` + `MINERU_OPENAI_BASE_URL` combination is the most portable pattern. It integrates cleanly with Docker Compose service networking (`http://mineru-vlm:30000`), environment-variable-driven configuration, and Kubernetes ConfigMaps/Secrets. The `healthcheck` condition in Compose (`condition: service_healthy`) replaces the systemd `ExecStartPre` probe.

```yaml
# docker-compose.yml pattern (from 4x_rtx4090_deploy.md Step 10)
services:
  mineru-vlm:
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:30000/health || exit 1"]
      ...
  mineru-api:
    depends_on:
      mineru-vlm:
        condition: service_healthy
    environment:
      MINERU_OPENAI_BASE_URL: http://mineru-vlm:30000
```

### 3. Consider Strategy 1 when scaling pipeline workers beyond a single server

The `hybrid-http-client` backend + `MINERU_OPENAI_BASE_URL` (or `--server-url` per request) is the correct extension point for distributed scale-out. Once your VLM cluster is provisioned, adding pipeline workers is as simple as deploying `mineru[pipeline]` on additional machines and pointing them at the central VLM server URL. **No code changes are required** — only environment configuration.

Security note: protect the VLM server endpoint with TLS and an API key (`--api-key` in vllm, `Authorization: Bearer <token>` from clients) before exposing it across network boundaries.

### 4. Avoid Strategy 2 in production beyond prototyping

The absence of GPU redundancy and the client-side routing requirement make Strategy 2 fragile under real load. The one scenario where it remains useful is **benchmarking** — running both backends simultaneously on the same machine makes it easy to collect accuracy/speed metrics side-by-side for a representative document sample before committing to a production architecture.

### 5. Key tuning knobs that apply to all strategies

| Variable | Recommended value | Reason |
|---|---|---|
| `--gpu-memory-utilization` | `0.80` | Leaves CUDA context headroom (~4.8 GB per card) while maximising KV cache |
| `MINERU_VIRTUAL_VRAM_SIZE` | physical VRAM ÷ `workers_per_device` | Prevents batch-size logic from over-allocating when workers share a GPU |
| `MINERU_PDF_RENDER_THREADS` | CPU threads ÷ total pipeline workers | Saturates CPU without contention (4 workers × 8 threads = 32 = Threadripper PRO 5955WX) |
| `MINERU_API_MAX_CONCURRENT_REQUESTS` | 2× the number of pipeline workers | Gate externally; allow GPU queue to fill internally |
| `nginx proxy_read_timeout` | `900s` | 500-page scientific PDF with tables and formulas can exceed 5 minutes |

### 6. Strategy selection flowchart

```
Do you have a single multi-GPU machine?
  ├─ YES → Do you need maximum throughput?
  │          ├─ YES → Strategy 0 (Hybrid Split Mode)
  │          └─ NO  → Strategy 3 (API Gateway) for simplicity
  └─ NO → Do you have a central GPU cluster?
             ├─ YES → Strategy 1 (Centralized VLM) + edge pipeline workers
             └─ NO  → Strategy 3 with a managed/cloud VLM endpoint
                       (e.g. MINERU_OPENAI_BASE_URL=https://api.openai.com/v1
                        or any OpenAI-compatible provider)
```

---

## Summary

The `4x_rtx4090_deploy.md` guide describes **Strategy 0**, the highest-throughput single-server pattern. The three additional strategies extend or simplify it for different operational contexts:

- **Strategy 1** scales the pipeline horizontally at the cost of network dependency.
- **Strategy 2** exposes an accuracy trade-off at the expense of redundancy.
- **Strategy 3** simplifies operations at the expense of per-GPU worker control.

In all cases, the fundamental split — **VLM inference isolated from pipeline processing** — remains the correct architectural baseline. The choice of strategy determines only *where* that split happens (process boundary, machine boundary, or network boundary) and *how* it is managed (systemd, LitServe, Docker Compose, or environment variables).
