# MinerU — Customized Models Guide

This document explains every model used in MinerU's PDF extraction pipeline, and provides step-by-step instructions for switching to different models or configurations.

---

## 1. Architecture Overview: Three Backends

MinerU supports three extraction backends. The backend is selected with the `-b` / `--backend` CLI flag (default: `hybrid-auto-engine`).

| Backend | Description |
|---|---|
| `pipeline` | Traditional multi-model pipeline: separate models for layout, OCR, formulas, and tables |
| `vlm-<engine>` | Single Vision-Language Model (VLM) handles layout, OCR, formulas, and tables end-to-end |
| `hybrid-<engine>` | VLM handles layout/block classification; pipeline models handle OCR and inline formula recognition |

**Key source files:**
- Backend dispatch: [mineru/cli/common.py](mineru/cli/common.py)
- Pipeline model definitions: [mineru/backend/pipeline/model_list.py](mineru/backend/pipeline/model_list.py), [mineru/backend/pipeline/model_init.py](mineru/backend/pipeline/model_init.py)
- VLM backend: [mineru/backend/vlm/vlm_analyze.py](mineru/backend/vlm/vlm_analyze.py)
- Hybrid backend: [mineru/backend/hybrid/hybrid_analyze.py](mineru/backend/hybrid/hybrid_analyze.py)
- Model path constants: [mineru/utils/enum_class.py](mineru/utils/enum_class.py) (`ModelPath` class)
- Model download/resolution: [mineru/utils/models_download_utils.py](mineru/utils/models_download_utils.py)
- Config reader: [mineru/utils/config_reader.py](mineru/utils/config_reader.py)
- VLM engine selection: [mineru/utils/engine_utils.py](mineru/utils/engine_utils.py)

---

## 2. Pipeline Backend — All Models

All pipeline model weights are bundled in the `opendatalab/PDF-Extract-Kit-1.0` repository (HuggingFace) / `OpenDataLab/PDF-Extract-Kit-1.0` (ModelScope).

### 2.1 Layout Detection

| Property | Value |
|---|---|
| **Model** | DocLayout-YOLO |
| **Purpose** | Detects page regions: title, text, image body, table body, equations, footnotes, etc. |
| **Weight path** | `models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt` |
| **Class** | `DocLayoutYOLOModel` ([mineru/model/layout/doclayoutyolo.py](mineru/model/layout/doclayoutyolo.py)) |
| **Config key** | `AtomicModel.Layout` |

### 2.2 Math Formula Detection (MFD)

| Property | Value |
|---|---|
| **Model** | YOLOv8 (fine-tuned) |
| **Purpose** | Detects bounding boxes of inline and display (interline) math formulas |
| **Weight path** | `models/MFD/YOLO/yolo_v8_ft.pt` |
| **Class** | `YOLOv8MFDModel` ([mineru/model/mfd/yolo_v8.py](mineru/model/mfd/yolo_v8.py)) |
| **Config key** | `AtomicModel.MFD` |
| **Toggle** | `MINERU_FORMULA_ENABLE=true/false` (default `true`) or `-f / --formula true/false` |

### 2.3 Math Formula Recognition (MFR)

**Two alternatives** — switched by the `MINERU_FORMULA_CH_SUPPORT` environment variable:

| Env var value | Model | Weight path | Notes |
|---|---|---|---|
| `false` (default) | **UniMERNet Small** | `models/MFR/unimernet_hf_small_2503` | General-purpose, fastest |
| `true` | **PP-FormulaNet-Plus-M** | `models/MFR/pp_formulanet_plus_m` | Adds Chinese formula character support |

**Class files:** [mineru/model/mfr/](mineru/model/mfr/)

### 2.4 OCR

| Property | Value |
|---|---|
| **Model** | PaddleOCR (PyTorch port) |
| **Purpose** | Text detection and recognition for all page regions |
| **Weight path** | `models/OCR/paddleocr_torch` |
| **Class** | `PytorchPaddleOCR` ([mineru/model/ocr/pytorch_paddle.py](mineru/model/ocr/pytorch_paddle.py)) |
| **Config key** | `AtomicModel.OCR` |
| **Language** | Set with `-l / --lang` (default: `ch`) |

Supported language values: `ch`, `ch_server`, `ch_lite`, `en`, `korean`, `japan`, `chinese_cht`, `ta`, `te`, `ka`, `th`, `el`, `latin`, `arabic`, `east_slavic`, `cyrillic`, `devanagari`

### 2.5 Table Recognition (Two Models)

Tables are first classified as wired or wireless, then routed to the appropriate model:

| Type | Model | Weight path | Class |
|---|---|---|---|
| **Wired table** | UNet Structure | `models/TabRec/UnetStructure/unet.onnx` | `UnetTableModel` |
| **Wireless table** | SLANet+ (RapidTable) | `models/TabRec/SlanetPlus/slanet-plus.onnx` | `RapidTableModel` |

**Config key:** `AtomicModel.WiredTable` / `AtomicModel.WirelessTable`  
**Toggle:** `MINERU_TABLE_ENABLE=true/false` (default `true`) or `-t / --table true/false`

### 2.6 Table Classification

| Property | Value |
|---|---|
| **Model** | PP-LCNet_x1_0 (PaddleCLS) |
| **Purpose** | Classifies each detected table region as wired or wireless |
| **Weight path** | `models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx` |
| **Config key** | `AtomicModel.TableCls` |

### 2.7 Page Orientation Classification

| Property | Value |
|---|---|
| **Model** | PP-LCNet_x1_0_doc_ori (PaddleCLS) |
| **Purpose** | Detects page rotation (0°/90°/180°/270°) before running table OCR |
| **Weight path** | `models/OriCls/paddle_orientation_classification/PP-LCNet_x1_0_doc_ori.onnx` |
| **Config key** | `AtomicModel.ImgOrientationCls` |

### 2.8 Reading Order

| Property | Value |
|---|---|
| **Model** | LayoutReader |
| **Purpose** | Orders detected blocks into natural reading sequence |
| **Weight path** | `models/ReadingOrder/layout_reader` |

---

## 3. VLM Backend — MinerU2.5

The VLM backend replaces the entire pipeline with a single Vision-Language Model.

| Property | Value |
|---|---|
| **Model** | `MinerU2.5-2509-1.2B` (based on Qwen2-VL) |
| **HuggingFace repo** | `opendatalab/MinerU2.5-2509-1.2B` |
| **ModelScope repo** | `OpenDataLab/MinerU2.5-2509-1.2B` |
| **Architecture** | `Qwen2VLForConditionalGeneration` |
| **Purpose** | End-to-end PDF understanding: text, tables (HTML), equations (LaTeX), images, code, page structure — all in one model pass |

### 3.1 VLM Inference Engines

The `<engine>` part of `vlm-<engine>` controls how the VLM is loaded. Auto-selection logic is in [mineru/utils/engine_utils.py](mineru/utils/engine_utils.py):

| CLI backend value | Engine | Platform / Condition |
|---|---|---|
| `vlm-auto-engine` | Auto-detected (see below) | All platforms |
| `vlm-transformers` | HuggingFace Transformers | All platforms (slowest, most portable) |
| `vlm-vllm-engine` | vLLM (sync) | Linux, requires `vllm` installed |
| `vlm-vllm-async-engine` | vLLM (async) | Linux, requires `vllm` installed |
| `vlm-lmdeploy-engine` | LMDeploy | Linux / Windows / Ascend NPU |
| `vlm-mlx-engine` | MLX-VLM | macOS Apple Silicon + macOS ≥ 13.5 |
| `vlm-http-client` | Remote OpenAI-compatible server | Any (requires `-u <server_url>`) |

**Auto-detection order per platform:**

| Platform | 1st choice | 2nd choice | Fallback |
|---|---|---|---|
| Linux | `vllm-engine` | `lmdeploy-engine` | `transformers` |
| macOS | `mlx-engine` (if Apple Silicon + macOS ≥ 13.5) | — | `transformers` |
| Windows | `lmdeploy-engine` | — | `transformers` |

---

## 4. Hybrid Backend

The hybrid backend combines the VLM (for layout/block classification) with pipeline models (for OCR and inline formula detection/recognition). It is the **default** backend (`hybrid-auto-engine`).

The same engine sub-options apply as the VLM backend:

| CLI value | Meaning |
|---|---|
| `hybrid-auto-engine` | Use auto-detected engine (default) |
| `hybrid-transformers` | Force HuggingFace Transformers for the VLM part |
| `hybrid-vllm-engine` | Use vLLM for the VLM part (Linux) |
| `hybrid-lmdeploy-engine` | Use LMDeploy for the VLM part |
| `hybrid-mlx-engine` | Use MLX for the VLM part (macOS) |
| `hybrid-http-client` | Use a remote VLM server |

In hybrid mode, OCR and MFD/MFR (if formula enabled) still use the pipeline models (PaddleOCR + YOLOv8 + UniMERNet/PP-FormulaNet).

---

## 5. Configuration Reference

### 5.1 Config File: `~/mineru.json`

MinerU reads a JSON config file from `~/mineru.json` by default. The path can be overridden.

**Template:** [mineru.template.json](mineru.template.json)

```json
{
    "bucket_info": {
        "bucket-name-1": ["ak", "sk", "endpoint"]
    },
    "latex-delimiter-config": {
        "display": { "left": "$$", "right": "$$" },
        "inline":  { "left": "$",  "right": "$"  }
    },
    "llm-aided-config": {
        "title_aided": {
            "api_key": "<your api key>",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3-next-80b-a3b-instruct",
            "enable_thinking": false,
            "enable": false
        }
    },
    "models-dir": {
        "pipeline": "",
        "vlm": ""
    },
    "config_version": "1.3.1"
}
```

Key fields:

| Field | Purpose |
|---|---|
| `models-dir.pipeline` | Absolute local path to pipeline models root (used when `MINERU_MODEL_SOURCE=local`) |
| `models-dir.vlm` | Absolute local path to VLM weights root (used when `MINERU_MODEL_SOURCE=local`) |
| `latex-delimiter-config` | LaTeX delimiters for inline (`$...$`) and display (`$$...$$`) math |
| `llm-aided-config.title_aided` | Optional LLM post-processing for title detection (disabled by default) |

### 5.2 Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `MINERU_TOOLS_CONFIG_JSON` | Config file absolute path, or filename relative to `~/` | `mineru.json` |
| `MINERU_MODEL_SOURCE` | Where to load models: `huggingface`, `modelscope`, `local` | `huggingface` |
| `MINERU_DEVICE_MODE` | Compute device: `cpu`, `cuda`, `cuda:0`, `mps`, `npu`, `gcu`, `musa`, `mlu`, `sdaa` | auto-detected |
| `MINERU_VIRTUAL_VRAM_SIZE` | Max GPU VRAM (GB) to use per process | auto |
| `MINERU_FORMULA_ENABLE` | Enable formula detection/recognition (`true`/`false`) | `true` |
| `MINERU_TABLE_ENABLE` | Enable table recognition (`true`/`false`) | `true` |
| `MINERU_FORMULA_CH_SUPPORT` | Switch MFR model: `false`=UniMERNet, `true`=PP-FormulaNet | `false` |
| `MINERU_PDF_RENDER_TIMEOUT` | Timeout in seconds for loading PDF pages | `300` |
| `MINERU_PDF_RENDER_THREADS` | Threads used for PDF rendering | `4` |
| `MINERU_MIN_BATCH_INFERENCE_SIZE` | Pages per inference batch (pipeline only) | `384` |
| `MINERU_VLM_FORMULA_ENABLE` | Enable formula in VLM/hybrid mode | set by `--formula` |
| `MINERU_VLM_TABLE_ENABLE` | Enable table in VLM/hybrid mode | set by `--table` |
| `MINERU_LMDEPLOY_DEVICE` | LMDeploy device type: `cuda`, `ascend`, `maca`, `camb` | `cuda` |
| `MINERU_LMDEPLOY_BACKEND` | LMDeploy backend: `pytorch`, `turbomind` | auto-selected |
| `MINERU_LOG_LEVEL` | Log verbosity | `INFO` |

### 5.3 CLI Flags

```bash
mineru -p <pdf_file> -o <output_dir> [options]

  -b, --backend     Extraction backend (default: hybrid-auto-engine)
                    pipeline | vlm-auto-engine | vlm-http-client |
                    vlm-transformers | vlm-vllm-engine | vlm-lmdeploy-engine |
                    vlm-mlx-engine | hybrid-auto-engine | hybrid-http-client |
                    hybrid-transformers | hybrid-vllm-engine | hybrid-lmdeploy-engine
  -m, --method      Page handling method: auto | txt | ocr  (pipeline/hybrid only; default: auto)
  -l, --lang        OCR language (pipeline/hybrid only; default: ch)
  -f, --formula     Enable formula recognition: true | false  (default: true)
  -t, --table       Enable table recognition: true | false  (default: true)
  -d, --device      Compute device: cpu | cuda | cuda:0 | mps | npu | ...
  --vram            Max VRAM in GB (pipeline only)
  --source          Model source: huggingface | modelscope | local
  -u, --url         Remote VLM server URL (for http-client backends)
  -s, --start       Start page (0-based, inclusive)
  -e, --end         End page (0-based, inclusive)

  # VLM / hybrid extra options:
  --gpu_memory_utilization  float  (vLLM only)
  --batch_size              int    (transformers only)
  --max_concurrency         int    (http-client only)
  --lmdeploy_backend        pytorch | turbomind
  --lmdeploy_device         cuda | ascend | maca | camb
```

---

## 6. How to Switch Models

### 6.1 Switch the Overall Extraction Backend

```bash
# Use the traditional multi-model pipeline (best for CPU-only or memory-constrained setups)
mineru -p doc.pdf -o output -b pipeline

# Use the VLM end-to-end (best accuracy, GPU recommended)
mineru -p doc.pdf -o output -b vlm-auto-engine

# Use the hybrid backend (default — VLM layout + pipeline OCR)
mineru -p doc.pdf -o output -b hybrid-auto-engine

# Use a remote VLM API server (OpenAI-compatible)
mineru -p doc.pdf -o output -b vlm-http-client -u http://localhost:8000
```

### 6.2 Switch the VLM Inference Engine

```bash
# Force HuggingFace Transformers (portable, slowest)
mineru -p doc.pdf -o output -b vlm-transformers

# Force vLLM (Linux, fastest on multi-GPU)
mineru -p doc.pdf -o output -b vlm-vllm-engine

# Force LMDeploy (Linux/Windows/Ascend)
mineru -p doc.pdf -o output -b vlm-lmdeploy-engine

# Force MLX (macOS Apple Silicon only)
mineru -p doc.pdf -o output -b vlm-mlx-engine
```

Or set the environment variable to avoid typing the flag every time:
```bash
export MINERU_DEVICE_MODE=cuda  # or mps, cpu, npu, etc.
```

### 6.3 Switch the Formula Recognition Model (MFR)

By default, UniMERNet Small is used for formula recognition. To switch to PP-FormulaNet-Plus-M (adds Chinese formula support):

```bash
# Via environment variable
export MINERU_FORMULA_CH_SUPPORT=true
mineru -p doc.pdf -o output
```

To revert to UniMERNet:
```bash
export MINERU_FORMULA_CH_SUPPORT=false  # or unset the variable
```

**Note:** This only affects the `pipeline` and `hybrid` backends where MFR is used separately.

### 6.4 Use Local Model Weights (Offline Mode)

To avoid downloading models every run, or to use your own fine-tuned weights:

**Step 1:** Edit `~/mineru.json`:
```json
{
    "models-dir": {
        "pipeline": "/path/to/your/PDF-Extract-Kit-1.0",
        "vlm":      "/path/to/your/MinerU2.5-2509-1.2B"
    }
}
```

**Step 2:** Set the model source to `local`:
```bash
export MINERU_MODEL_SOURCE=local
mineru -p doc.pdf -o output
```

Or pass it inline:
```bash
mineru -p doc.pdf -o output --source local
```

The expected directory structure under the pipeline root is:
```
PDF-Extract-Kit-1.0/
  models/
    Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt
    MFD/YOLO/yolo_v8_ft.pt
    MFR/unimernet_hf_small_2503/
    MFR/pp_formulanet_plus_m/
    OCR/paddleocr_torch/
    ReadingOrder/layout_reader/
    TabRec/SlanetPlus/slanet-plus.onnx
    TabRec/UnetStructure/unet.onnx
    TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx
    OriCls/paddle_orientation_classification/PP-LCNet_x1_0_doc_ori.onnx
```

The VLM root should be a directory containing a standard HuggingFace model (config, tokenizer, weights).

### 6.5 Switch to ModelScope as the Download Source

If HuggingFace is inaccessible (e.g., in mainland China), use ModelScope:

```bash
# CLI flag
mineru -p doc.pdf -o output --source modelscope

# Or environment variable (persists across runs)
export MINERU_MODEL_SOURCE=modelscope
```

Pre-download all models with:
```bash
mineru-models-download -s modelscope -m all    # downloads both pipeline and vlm
mineru-models-download -s modelscope -m pipeline
mineru-models-download -s modelscope -m vlm
```

### 6.6 Disable Formula or Table Recognition

To skip formula processing (faster for text-heavy documents):
```bash
mineru -p doc.pdf -o output -f false
# Or:
export MINERU_FORMULA_ENABLE=false
```

To skip table recognition:
```bash
mineru -p doc.pdf -o output -t false
# Or:
export MINERU_TABLE_ENABLE=false
```

### 6.7 Change the OCR Language

The `pipeline` and `hybrid` backends support multi-language OCR:
```bash
mineru -p doc.pdf -o output -l en      # English
mineru -p doc.pdf -o output -l korean  # Korean
mineru -p doc.pdf -o output -l japan   # Japanese
mineru -p doc.pdf -o output -l arabic  # Arabic
```

### 6.8 Use an LMDeploy-Specific Backend or Device

```bash
# Ascend NPU with PyTorch backend
mineru -p doc.pdf -o output -b vlm-lmdeploy-engine \
    --lmdeploy_device ascend --lmdeploy_backend pytorch

# Windows CUDA with TurboMind backend
mineru -p doc.pdf -o output -b vlm-lmdeploy-engine \
    --lmdeploy_device cuda --lmdeploy_backend turbomind
```

Or use environment variables:
```bash
export MINERU_LMDEPLOY_DEVICE=ascend
export MINERU_LMDEPLOY_BACKEND=pytorch
```

### 6.9 Enable LLM-Aided Title Detection

For improved title classification using an external LLM API, edit `~/mineru.json`:

```json
{
    "llm-aided-config": {
        "title_aided": {
            "api_key": "<your-api-key>",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3-next-80b-a3b-instruct",
            "enable_thinking": false,
            "enable": true
        }
    }
}
```

Set `"enable": true` to activate. Any OpenAI-compatible API is supported by changing `base_url` and `model`.

### 6.10 Use a Custom Config File Path

If you want to keep `~/mineru.json` clean and use a different config:

```bash
export MINERU_TOOLS_CONFIG_JSON=/path/to/my_custom_config.json
# or use a filename relative to ~/
export MINERU_TOOLS_CONFIG_JSON=myminerucfg.json
```

---

## 7. Model Source Code Locations

| Component | Source File |
|---|---|
| Layout (DocLayout-YOLO) | [mineru/model/layout/](mineru/model/layout/) |
| MFD (YOLOv8) | [mineru/model/mfd/yolo_v8.py](mineru/model/mfd/yolo_v8.py) |
| MFR (UniMERNet / PP-FormulaNet) | [mineru/model/mfr/](mineru/model/mfr/) |
| OCR (PaddleOCR PyTorch) | [mineru/model/ocr/pytorch_paddle.py](mineru/model/ocr/pytorch_paddle.py) |
| Table recognition (UNet / SLANet+) | [mineru/model/table/](mineru/model/table/) |
| Table classification (PP-LCNet) | [mineru/model/table/](mineru/model/table/) |
| Orientation classification | [mineru/model/ori_cls/](mineru/model/ori_cls/) |
| Reading order (LayoutReader) | [mineru/model/reading_order/](mineru/model/reading_order/) |
| VLM model wrappers | [mineru/model/vlm/](mineru/model/vlm/) |
| Model path constants | [mineru/utils/enum_class.py](mineru/utils/enum_class.py) (`ModelPath` class) |
| Model download logic | [mineru/utils/models_download_utils.py](mineru/utils/models_download_utils.py) |
| Config reading | [mineru/utils/config_reader.py](mineru/utils/config_reader.py) |
| VLM engine selection | [mineru/utils/engine_utils.py](mineru/utils/engine_utils.py) |

---

## 8. Quick Reference — All Configuration Points

| What to change | How |
|---|---|
| Switch backend (pipeline/vlm/hybrid) | `-b` CLI flag or default behavior |
| Switch VLM inference engine | `-b vlm-<engine>` or `-b hybrid-<engine>` |
| Switch formula recognition model | `MINERU_FORMULA_CH_SUPPORT=true/false` |
| Use local model weights | `MINERU_MODEL_SOURCE=local` + `models-dir` in `~/mineru.json` |
| Use ModelScope instead of HuggingFace | `MINERU_MODEL_SOURCE=modelscope` or `--source modelscope` |
| Disable formula processing | `MINERU_FORMULA_ENABLE=false` or `-f false` |
| Disable table processing | `MINERU_TABLE_ENABLE=false` or `-t false` |
| Change OCR language | `-l <lang>` |
| Force compute device | `MINERU_DEVICE_MODE=<device>` or `-d <device>` |
| Use remote VLM server | `-b vlm-http-client -u <url>` |
| LMDeploy device/backend | `--lmdeploy_device` / `--lmdeploy_backend` |
| Enable LLM-aided title detection | Set `enable: true` in `llm-aided-config` in `~/mineru.json` |
| Custom config file path | `MINERU_TOOLS_CONFIG_JSON=<path>` |
| LaTeX math delimiters | `latex-delimiter-config` in `~/mineru.json` |
