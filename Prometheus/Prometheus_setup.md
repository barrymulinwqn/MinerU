# Prometheus Monitoring Setup Guide

## Server Hardware Overview

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen Threadripper PRO 5955WX, 16 Cores (32 Threads) |
| GPU | NVIDIA GeForce RTX 4090 × 4 |
| CPU Memory | 256 GB DDR4 |
| GPU Memory | 24 GB × 4 = 96 GB Total |
| System Disk | 4 TB |
| Network (High Speed) | Aquantia AQC113C NBase-T 10 GbE |
| Network (Management) | Intel I210 Gigabit 1 GbE |
| OS | Ubuntu 22.04.5 LTS |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  GPU Server (This Machine)               │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │node_exporter│  │ nvidia_smi_  │  │  MinerU App   │  │
│  │  :9100      │  │  exporter    │  │  (optional)   │  │
│  │             │  │  :9835       │  │               │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                │                   │          │
│  ┌──────▼────────────────▼───────────────────▼───────┐  │
│  │              Prometheus  :9090                    │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │               Grafana  :3000                      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
          │
          │  Remote Access (Browser / Prometheus Client)
          ▼
   Remote Workstation / Laptop
```

---

## Part 1 — Install Prometheus

### 1.1 Create Dedicated User

```bash
sudo useradd --no-create-home --shell /bin/false prometheus
sudo useradd --no-create-home --shell /bin/false node_exporter
```

### 1.2 Create Directories

```bash
sudo mkdir -p /etc/prometheus /var/lib/prometheus
sudo chown prometheus:prometheus /etc/prometheus /var/lib/prometheus
```

### 1.3 Download and Install Prometheus

Check the latest release at https://github.com/prometheus/prometheus/releases and replace the version below if needed.

```bash
cd /tmp
PROM_VERSION="2.51.2"
wget https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz
tar xvf prometheus-${PROM_VERSION}.linux-amd64.tar.gz
cd prometheus-${PROM_VERSION}.linux-amd64

sudo cp prometheus /usr/local/bin/
sudo cp promtool  /usr/local/bin/
sudo chown prometheus:prometheus /usr/local/bin/prometheus /usr/local/bin/promtool

sudo cp -r consoles/          /etc/prometheus/
sudo cp -r console_libraries/ /etc/prometheus/
sudo chown -R prometheus:prometheus /etc/prometheus/consoles /etc/prometheus/console_libraries
```

### 1.4 Write Prometheus Configuration

```bash
sudo tee /etc/prometheus/prometheus.yml > /dev/null << 'EOF'
global:
  scrape_interval:     15s   # How often to scrape targets
  evaluation_interval: 15s   # How often to evaluate rules

alerting:
  alertmanagers:
    - static_configs:
        - targets: []         # Add alertmanager address here if needed

rule_files: []

scrape_configs:
  # Prometheus itself
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  # Node Exporter — CPU / Memory / Disk / Network
  - job_name: "node"
    static_configs:
      - targets: ["localhost:9100"]
        labels:
          instance: "gpu-server"
          hardware: "threadripper-pro-5955wx"

  # NVIDIA GPU Exporter
  - job_name: "nvidia_gpu"
    static_configs:
      - targets: ["localhost:9835"]
        labels:
          instance: "gpu-server"
          gpu_model: "rtx4090"
          gpu_count: "4"
EOF

sudo chown prometheus:prometheus /etc/prometheus/prometheus.yml
```

### 1.5 Create Prometheus systemd Service

```bash
sudo tee /etc/systemd/system/prometheus.service > /dev/null << 'EOF'
[Unit]
Description=Prometheus Monitoring
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/var/lib/prometheus/ \
    --storage.tsdb.retention.time=30d \
    --web.console.templates=/etc/prometheus/consoles \
    --web.console.libraries=/etc/prometheus/console_libraries \
    --web.listen-address=0.0.0.0:9090 \
    --web.enable-lifecycle
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
sudo systemctl status prometheus
```

---

## Part 2 — Install Node Exporter (CPU / Memory / Disk / Network)

Node Exporter collects host-level metrics: CPU usage per core, memory, disk I/O, network throughput per NIC, load average, etc.

### 2.1 Download and Install

```bash
cd /tmp
NE_VERSION="1.7.0"
wget https://github.com/prometheus/node_exporter/releases/download/v${NE_VERSION}/node_exporter-${NE_VERSION}.linux-amd64.tar.gz
tar xvf node_exporter-${NE_VERSION}.linux-amd64.tar.gz
sudo cp node_exporter-${NE_VERSION}.linux-amd64/node_exporter /usr/local/bin/
sudo chown node_exporter:node_exporter /usr/local/bin/node_exporter
```

### 2.2 Create systemd Service

```bash
sudo tee /etc/systemd/system/node_exporter.service > /dev/null << 'EOF'
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter \
    --collector.cpu \
    --collector.meminfo \
    --collector.diskstats \
    --collector.filesystem \
    --collector.netdev \
    --collector.loadavg \
    --collector.hwmon \
    --collector.thermal_zone \
    --web.listen-address=0.0.0.0:9100
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl start node_exporter
sudo systemctl status node_exporter
```

### 2.3 Verify Node Exporter

```bash
curl http://localhost:9100/metrics | grep -E "^node_cpu|^node_memory|^node_load"
```

---

## Part 3 — Install NVIDIA GPU Exporter

### 3.1 Prerequisites — Ensure nvidia-smi is Available

```bash
nvidia-smi
# Should list all 4x RTX 4090 cards
```

### 3.2 Install nvidia-smi-exporter (utkuozdemir/nvidia_gpu_exporter)

This exporter exposes per-GPU metrics via `nvidia-smi`.

```bash
cd /tmp
GPU_EXP_VERSION="1.2.1"
wget https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v${GPU_EXP_VERSION}/nvidia_gpu_exporter_${GPU_EXP_VERSION}_linux_x86_64.tar.gz
tar xvf nvidia_gpu_exporter_${GPU_EXP_VERSION}_linux_x86_64.tar.gz
sudo cp nvidia_gpu_exporter /usr/local/bin/
sudo chmod +x /usr/local/bin/nvidia_gpu_exporter
```

### 3.3 Create systemd Service

```bash
sudo tee /etc/systemd/system/nvidia_gpu_exporter.service > /dev/null << 'EOF'
[Unit]
Description=NVIDIA GPU Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=root
Type=simple
ExecStart=/usr/local/bin/nvidia_gpu_exporter \
    --web.listen-address=0.0.0.0:9835
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nvidia_gpu_exporter
sudo systemctl start nvidia_gpu_exporter
sudo systemctl status nvidia_gpu_exporter
```

### 3.4 Verify GPU Exporter

```bash
curl http://localhost:9835/metrics | grep -E "^nvidia_"
```

Expected metrics include:
- `nvidia_smi_utilization_gpu_ratio` — GPU core utilization per card
- `nvidia_smi_utilization_memory_ratio` — GPU memory bandwidth utilization
- `nvidia_smi_memory_used_bytes` — VRAM used
- `nvidia_smi_memory_total_bytes` — VRAM total (24 GB per card)
- `nvidia_smi_temperature_gpu` — GPU temperature in °C
- `nvidia_smi_power_draw_watts` — GPU power draw
- `nvidia_smi_fan_speed_ratio` — Fan speed

---

## Part 4 — Open Firewall Ports

Allow remote access to Prometheus and Grafana. Restrict source IPs to your trusted network (e.g., `192.168.1.0/24`).

```bash
# Allow Prometheus UI from trusted network only
sudo ufw allow from 192.168.1.0/24 to any port 9090 proto tcp comment "Prometheus"

# Allow Grafana from trusted network only
sudo ufw allow from 192.168.1.0/24 to any port 3000 proto tcp comment "Grafana"

# Exporters should NOT be exposed externally — they are localhost-only
# If you must expose node_exporter for remote scraping, restrict by IP:
# sudo ufw allow from 192.168.1.0/24 to any port 9100 proto tcp comment "node_exporter"
# sudo ufw allow from 192.168.1.0/24 to any port 9835 proto tcp comment "nvidia_gpu_exporter"

sudo ufw reload
sudo ufw status
```

---

## Part 5 — Install Grafana (Visualization Dashboard)

Grafana provides rich dashboards over Prometheus data.

### 5.1 Install Grafana via APT

```bash
sudo apt-get install -y apt-transport-https software-properties-common wget gnupg2

wget -q -O - https://apt.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list

sudo apt-get update
sudo apt-get install -y grafana
```

### 5.2 Start Grafana Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
sudo systemctl status grafana-server
```

### 5.3 Access Grafana

Open a browser and navigate to:

```
http://<SERVER_IP>:3000
```

- Default username: `admin`
- Default password: `admin`
- You will be prompted to change the password on first login.

### 5.4 Add Prometheus as Data Source

1. Go to **Configuration → Data Sources → Add data source**
2. Select **Prometheus**
3. Set URL to `http://localhost:9090`
4. Click **Save & Test**

### 5.5 Import Recommended Dashboards

| Dashboard | Grafana ID | Purpose |
|-----------|-----------|---------|
| Node Exporter Full | `1860` | CPU, Memory, Disk, Network |
| NVIDIA GPU Exporter | `14574` | 4× RTX 4090 per-GPU metrics |
| Prometheus Stats | `3662` | Prometheus self-monitoring |

To import: **Dashboards → Import → Enter ID → Load**

---

## Part 6 — Remote Access to Metrics

### 6.1 Access Prometheus Web UI Remotely

```
http://<SERVER_IP>:9090
```

Use the **Graph** tab to run instant PromQL queries, or use the **Targets** tab to verify all exporters are UP.

### 6.2 Access Raw Metrics Endpoint

```bash
# From a remote machine:
curl http://<SERVER_IP>:9090/api/v1/query?query=node_cpu_seconds_total
curl http://<SERVER_IP>:9090/api/v1/query?query=nvidia_smi_utilization_gpu_ratio
```

### 6.3 SSH Tunnel (Secure Remote Access Without Exposing Ports)

If you prefer not to open ports on the firewall, use an SSH tunnel:

```bash
# On your local machine — forward remote port 9090 to localhost:9090
ssh -L 9090:localhost:9090 -L 3000:localhost:3000 user@<SERVER_IP> -N

# Then open in local browser:
# http://localhost:9090  → Prometheus
# http://localhost:3000  → Grafana
```

---

## Part 7 — Python Prometheus Client

Use the official `prometheus_client` Python library to query metrics programmatically.

### 7.1 Install the Client

```bash
pip install prometheus-api-client requests
```

Or if using the MinerU virtual environment:

```bash
source /path/to/MinerU/.venv/bin/activate
pip install prometheus-api-client requests
```

### 7.2 Query Script — GPU, CPU, Memory Metrics

```python
#!/usr/bin/env python3
"""
query_metrics.py — Query GPU server metrics from Prometheus.
Server: AMD Threadripper PRO 5955WX + 4x RTX 4090, 256GB RAM, Ubuntu 22.04
"""

import time
from datetime import datetime
from prometheus_api_client import PrometheusConnect

PROMETHEUS_URL = "http://<SERVER_IP>:9090"  # Replace with actual IP


def get_prometheus_client() -> PrometheusConnect:
    return PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)


def query_current(prom: PrometheusConnect, query: str) -> list:
    return prom.custom_query(query=query)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def show_cpu_usage(prom: PrometheusConnect):
    print_section("CPU Usage — Threadripper PRO 5955WX (16C/32T)")

    # Overall CPU utilization across all cores
    result = query_current(
        prom,
        '100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)'
    )
    for r in result:
        val = float(r["value"][1])
        print(f"  Overall CPU Usage : {val:.2f}%")

    # Per-core utilization
    result = query_current(
        prom,
        '100 - (rate(node_cpu_seconds_total{mode="idle"}[1m]) * 100)'
    )
    print(f"\n  Per-core utilization (32 logical cores):")
    for r in result:
        cpu = r["metric"].get("cpu", "?")
        val = float(r["value"][1])
        bar = "#" * int(val / 5)
        print(f"    Core {cpu:>2} : {val:5.1f}%  [{bar:<20}]")

    # Load average
    result = query_current(prom, "node_load1")
    for r in result:
        print(f"\n  Load Average (1m)  : {float(r['value'][1]):.2f}")
    result = query_current(prom, "node_load5")
    for r in result:
        print(f"  Load Average (5m)  : {float(r['value'][1]):.2f}")
    result = query_current(prom, "node_load15")
    for r in result:
        print(f"  Load Average (15m) : {float(r['value'][1]):.2f}")


def show_memory_usage(prom: PrometheusConnect):
    print_section("Memory Usage — 256 GB DDR4")

    result = query_current(prom, "node_memory_MemTotal_bytes")
    total = float(result[0]["value"][1]) if result else 0

    result = query_current(prom, "node_memory_MemAvailable_bytes")
    avail = float(result[0]["value"][1]) if result else 0

    used = total - avail
    pct = (used / total * 100) if total > 0 else 0

    print(f"  Total    : {total / 1024**3:.1f} GB")
    print(f"  Used     : {used  / 1024**3:.1f} GB  ({pct:.1f}%)")
    print(f"  Available: {avail / 1024**3:.1f} GB")

    result = query_current(prom, "node_memory_Buffers_bytes")
    buf = float(result[0]["value"][1]) if result else 0
    result = query_current(prom, "node_memory_Cached_bytes")
    cached = float(result[0]["value"][1]) if result else 0
    print(f"  Buffers  : {buf    / 1024**3:.2f} GB")
    print(f"  Cached   : {cached / 1024**3:.2f} GB")


def show_gpu_usage(prom: PrometheusConnect):
    print_section("GPU Usage — 4× NVIDIA RTX 4090 (24 GB each)")

    # GPU utilization
    result = query_current(prom, "nvidia_smi_utilization_gpu_ratio")
    print("  GPU Core Utilization:")
    for r in result:
        idx = r["metric"].get("index", r["metric"].get("gpu", "?"))
        val = float(r["value"][1]) * 100
        bar = "#" * int(val / 5)
        print(f"    GPU {idx} : {val:5.1f}%  [{bar:<20}]")

    # GPU memory
    result_used  = query_current(prom, "nvidia_smi_memory_used_bytes")
    result_total = query_current(prom, "nvidia_smi_memory_total_bytes")
    print("\n  GPU VRAM Usage:")
    for r in result_used:
        idx   = r["metric"].get("index", r["metric"].get("gpu", "?"))
        used  = float(r["value"][1])
        total = next(
            (float(t["value"][1]) for t in result_total
             if t["metric"].get("index", t["metric"].get("gpu")) == idx),
            24 * 1024**3
        )
        pct = used / total * 100
        print(f"    GPU {idx} : {used/1024**3:5.1f} GB / {total/1024**3:.0f} GB  ({pct:.1f}%)")

    # GPU temperature
    result = query_current(prom, "nvidia_smi_temperature_gpu")
    print("\n  GPU Temperature:")
    for r in result:
        idx = r["metric"].get("index", r["metric"].get("gpu", "?"))
        val = float(r["value"][1])
        print(f"    GPU {idx} : {val:.0f} °C")

    # GPU power draw
    result = query_current(prom, "nvidia_smi_power_draw_watts")
    print("\n  GPU Power Draw:")
    for r in result:
        idx = r["metric"].get("index", r["metric"].get("gpu", "?"))
        val = float(r["value"][1])
        print(f"    GPU {idx} : {val:.1f} W")

    # Fan speed
    result = query_current(prom, "nvidia_smi_fan_speed_ratio")
    print("\n  GPU Fan Speed:")
    for r in result:
        idx = r["metric"].get("index", r["metric"].get("gpu", "?"))
        val = float(r["value"][1]) * 100
        print(f"    GPU {idx} : {val:.0f}%")


def show_disk_usage(prom: PrometheusConnect):
    print_section("Disk Usage — 4 TB System Disk")

    result = query_current(
        prom,
        'node_filesystem_size_bytes{mountpoint="/", fstype!="tmpfs"}'
    )
    for r in result:
        total = float(r["value"][1])

        avail_res = query_current(
            prom,
            'node_filesystem_avail_bytes{mountpoint="/", fstype!="tmpfs"}'
        )
        avail = float(avail_res[0]["value"][1]) if avail_res else 0
        used = total - avail
        pct = used / total * 100
        mp = r["metric"].get("mountpoint", "/")
        print(f"  Mount {mp}")
        print(f"    Total : {total/1024**4:.2f} TB")
        print(f"    Used  : {used /1024**3:.1f} GB  ({pct:.1f}%)")
        print(f"    Free  : {avail/1024**3:.1f} GB")

    # Disk I/O rates
    read_result = query_current(
        prom, 'rate(node_disk_read_bytes_total{device=~"sd.*|nvme.*"}[1m])'
    )
    write_result = query_current(
        prom, 'rate(node_disk_written_bytes_total{device=~"sd.*|nvme.*"}[1m])'
    )
    if read_result or write_result:
        print("\n  Disk I/O (1-min rate):")
        for r in read_result:
            dev = r["metric"].get("device", "?")
            val = float(r["value"][1])
            print(f"    {dev} Read  : {val/1024**2:.2f} MB/s")
        for r in write_result:
            dev = r["metric"].get("device", "?")
            val = float(r["value"][1])
            print(f"    {dev} Write : {val/1024**2:.2f} MB/s")


def show_network_usage(prom: PrometheusConnect):
    print_section("Network Usage — AQC113C 10GbE + Intel I210 1GbE")

    rx_result = query_current(
        prom, 'rate(node_network_receive_bytes_total{device!~"lo|docker.*|veth.*"}[1m])'
    )
    tx_result = query_current(
        prom, 'rate(node_network_transmit_bytes_total{device!~"lo|docker.*|veth.*"}[1m])'
    )
    print("  Network Throughput (1-min rate):")
    devices = set(r["metric"]["device"] for r in rx_result + tx_result)
    for dev in sorted(devices):
        rx = next((float(r["value"][1]) for r in rx_result if r["metric"]["device"] == dev), 0)
        tx = next((float(r["value"][1]) for r in tx_result if r["metric"]["device"] == dev), 0)
        print(f"    {dev:<12} RX: {rx/1024**2:7.2f} MB/s   TX: {tx/1024**2:7.2f} MB/s")


def show_peak_times(prom: PrometheusConnect):
    print_section("Peak Usage Analysis (last 24 hours)")

    # CPU peak
    result = query_current(
        prom,
        'max_over_time('
        '(100 - avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
        '[24h:5m])'
    )
    for r in result:
        print(f"  Peak CPU (24h)     : {float(r['value'][1]):.2f}%")

    # Memory peak
    result = query_current(
        prom,
        'max_over_time('
        '((node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)'
        '/ node_memory_MemTotal_bytes * 100)[24h:5m])'
    )
    for r in result:
        print(f"  Peak Mem (24h)     : {float(r['value'][1]):.2f}%")

    # GPU peak (all GPUs combined — max utilization)
    result = query_current(
        prom,
        'max_over_time(max(nvidia_smi_utilization_gpu_ratio)[24h:5m])'
    )
    for r in result:
        print(f"  Peak GPU (24h)     : {float(r['value'][1])*100:.2f}%")

    # GPU VRAM peak
    result = query_current(
        prom,
        'max_over_time(max(nvidia_smi_memory_used_bytes)[24h:5m])'
    )
    for r in result:
        print(f"  Peak VRAM/GPU (24h): {float(r['value'][1])/1024**3:.2f} GB")


def main():
    print(f"\nPrometheus Metrics Query — GPU Server")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: {PROMETHEUS_URL}")

    prom = get_prometheus_client()

    try:
        prom.check_prometheus_connection()
        print("  Connection: OK")
    except Exception as e:
        print(f"  Connection FAILED: {e}")
        return

    show_cpu_usage(prom)
    show_memory_usage(prom)
    show_gpu_usage(prom)
    show_disk_usage(prom)
    show_network_usage(prom)
    show_peak_times(prom)

    print("\n" + "="*60)
    print("  Done.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
```

### 7.3 Continuous Monitoring Script (Live Refresh)

```python
#!/usr/bin/env python3
"""
live_monitor.py — Continuously poll GPU server metrics every N seconds.
"""

import time
import os
from prometheus_api_client import PrometheusConnect

PROMETHEUS_URL = "http://<SERVER_IP>:9090"
REFRESH_SECONDS = 5


def clear():
    os.system("clear")


def fetch(prom, query):
    try:
        r = prom.custom_query(query=query)
        return r
    except Exception:
        return []


def main():
    prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

    while True:
        clear()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"GPU Server Live Monitor  [{now}]  (refresh: {REFRESH_SECONDS}s)\n")

        # CPU
        r = fetch(prom, '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[15s])) * 100)')
        cpu = float(r[0]["value"][1]) if r else 0.0
        print(f"  CPU Usage    : {cpu:5.1f}%")

        # Memory
        rt = fetch(prom, "node_memory_MemTotal_bytes")
        ra = fetch(prom, "node_memory_MemAvailable_bytes")
        if rt and ra:
            total = float(rt[0]["value"][1])
            avail = float(ra[0]["value"][1])
            used_pct = (total - avail) / total * 100
            print(f"  Mem Usage    : {used_pct:5.1f}%  ({(total-avail)/1024**3:.1f}/{total/1024**3:.0f} GB)")

        # Per-GPU
        print()
        gpu_util = fetch(prom, "nvidia_smi_utilization_gpu_ratio")
        gpu_mem_used  = fetch(prom, "nvidia_smi_memory_used_bytes")
        gpu_mem_total = fetch(prom, "nvidia_smi_memory_total_bytes")
        gpu_temp = fetch(prom, "nvidia_smi_temperature_gpu")
        gpu_pwr  = fetch(prom, "nvidia_smi_power_draw_watts")

        for r in sorted(gpu_util, key=lambda x: x["metric"].get("index", "0")):
            idx  = r["metric"].get("index", "?")
            util = float(r["value"][1]) * 100
            mem_u = next((float(m["value"][1])/1024**3 for m in gpu_mem_used
                          if m["metric"].get("index") == idx), 0)
            mem_t = next((float(m["value"][1])/1024**3 for m in gpu_mem_total
                          if m["metric"].get("index") == idx), 24)
            temp  = next((float(m["value"][1]) for m in gpu_temp
                          if m["metric"].get("index") == idx), 0)
            pwr   = next((float(m["value"][1]) for m in gpu_pwr
                          if m["metric"].get("index") == idx), 0)
            bar = "#" * int(util / 5)
            print(f"  GPU {idx}  util:{util:5.1f}%  [{bar:<20}]  "
                  f"VRAM:{mem_u:5.1f}/{mem_t:.0f}GB  {temp:.0f}°C  {pwr:.0f}W")

        print(f"\n  Next refresh in {REFRESH_SECONDS}s ...")
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
```

---

## Part 8 — Useful PromQL Queries Reference

Run these in the Prometheus UI at `http://<SERVER_IP>:9090/graph`.

### CPU

```promql
# Overall CPU usage %
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)

# Per-core usage %
100 - (rate(node_cpu_seconds_total{mode="idle"}[1m]) * 100)

# CPU iowait % (disk bottleneck indicator)
avg(rate(node_cpu_seconds_total{mode="iowait"}[1m])) * 100

# System load average
node_load1
node_load5
node_load15
```

### Memory

```promql
# Memory used %
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100

# Memory used in GB
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1024^3

# Swap usage %
(1 - node_memory_SwapFree_bytes / node_memory_SwapTotal_bytes) * 100
```

### GPU (4× RTX 4090)

```promql
# GPU utilization per card
nvidia_smi_utilization_gpu_ratio * 100

# GPU VRAM used per card (GB)
nvidia_smi_memory_used_bytes / 1024^3

# GPU VRAM used % per card
nvidia_smi_memory_used_bytes / nvidia_smi_memory_total_bytes * 100

# GPU temperature
nvidia_smi_temperature_gpu

# GPU power draw (W)
nvidia_smi_power_draw_watts

# Total power draw across all 4 GPUs
sum(nvidia_smi_power_draw_watts)

# Fan speed %
nvidia_smi_fan_speed_ratio * 100
```

### Disk

```promql
# Disk usage %
100 - (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} * 100)

# Disk read rate MB/s
rate(node_disk_read_bytes_total{device=~"sd.*|nvme.*"}[1m]) / 1024^2

# Disk write rate MB/s
rate(node_disk_written_bytes_total{device=~"sd.*|nvme.*"}[1m]) / 1024^2
```

### Network (AQC113C 10GbE + Intel I210 1GbE)

```promql
# Network receive rate MB/s per NIC
rate(node_network_receive_bytes_total{device!~"lo|docker.*"}[1m]) / 1024^2

# Network transmit rate MB/s per NIC
rate(node_network_transmit_bytes_total{device!~"lo|docker.*"}[1m]) / 1024^2

# Total receive across all physical NICs
sum(rate(node_network_receive_bytes_total{device!~"lo|docker.*|veth.*"}[1m])) / 1024^2
```

### Peak Detection

```promql
# Which hour of day had the highest CPU usage (last 7 days)
max_over_time(
  (100 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
  [7d:1h]
)

# Max GPU VRAM used in the last 24 hours
max_over_time(max(nvidia_smi_memory_used_bytes)[24h:5m]) / 1024^3
```

---

## Part 9 — Alerting Rules (Optional)

Create alert rules so Prometheus notifies you when resources are critically high.

```bash
sudo tee /etc/prometheus/alert_rules.yml > /dev/null << 'EOF'
groups:
  - name: gpu_server_alerts
    rules:

      - alert: HighCPUUsage
        expr: 100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on GPU server"
          description: "CPU usage > 90% for 5 minutes (current: {{ $value | printf \"%.1f\" }}%)"

      - alert: HighMemoryUsage
        expr: (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on GPU server"
          description: "Memory usage > 85% (current: {{ $value | printf \"%.1f\" }}%)"

      - alert: GPUHighUtilization
        expr: nvidia_smi_utilization_gpu_ratio * 100 > 95
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "GPU {{ $labels.index }} sustained high utilization"
          description: "GPU utilization > 95% for 15 min (current: {{ $value | printf \"%.1f\" }}%)"

      - alert: GPUHighTemperature
        expr: nvidia_smi_temperature_gpu > 83
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU {{ $labels.index }} overheating"
          description: "GPU temperature > 83°C (current: {{ $value }}°C)"

      - alert: GPUVRAMAlmostFull
        expr: nvidia_smi_memory_used_bytes / nvidia_smi_memory_total_bytes * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.index }} VRAM almost full"
          description: "GPU VRAM > 90% used (current: {{ $value | printf \"%.1f\" }}%)"

      - alert: DiskAlmostFull
        expr: 100 - (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "System disk almost full"
          description: "Disk usage > 80% (current: {{ $value | printf \"%.1f\" }}%)"
EOF

sudo chown prometheus:prometheus /etc/prometheus/alert_rules.yml
```

Then reference this file in `prometheus.yml`:

```bash
sudo sed -i 's|rule_files: \[\]|rule_files:\n  - "/etc/prometheus/alert_rules.yml"|' /etc/prometheus/prometheus.yml
sudo systemctl reload prometheus
```

---

## Part 10 — Verify Full Stack

```bash
# Check all services
sudo systemctl status prometheus node_exporter nvidia_gpu_exporter grafana-server

# Confirm all Prometheus scrape targets are UP
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool | grep -E '"health"|"job"'

# Quick sanity check — GPU count should be 4
curl -s "http://localhost:9090/api/v1/query?query=count(nvidia_smi_utilization_gpu_ratio)" \
  | python3 -m json.tool
```

Expected output:

```json
{ "data": { "result": [{ "value": [<ts>, "4"] }] } }
```

---

## Service Port Summary

| Service | Port | Access |
|---------|------|--------|
| Prometheus | 9090 | Remote (firewall-filtered) |
| Grafana | 3000 | Remote (firewall-filtered) |
| Node Exporter | 9100 | Localhost only |
| NVIDIA GPU Exporter | 9835 | Localhost only |
