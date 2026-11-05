# ⚡ Flash1017-Engine (`flash1017-gx10`) — Standalone GB10 Turbo Engine

**`flash1017-gx10`** is a standalone, 100% closed-source, hardware-locked (`NVIDIA Grace Blackwell GB10 / sm_121`) inference engine and `vLLM` / `llama-server`-compatible HTTP/SSE server for **[`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4)** (`1T`-class MoE, `404 GB` on disk, `40` layers, `384` routed experts/layer + `33 GB` N-Gram Engram + `32`-layer Vision ViT) running on a **single ASUS Ascent GX10 / NVIDIA DGX Spark (`128 GB Unified Memory`)**.

---

## 📊 Real-World Benchmarks on ASUS Ascent GX10 (`Single 128 GB GB10`, Batch Size = 1)

Wait-free, real-world single-user performance measured directly on **ASUS Ascent GX10 (`NVIDIA GB10`, `sm_121`, `128 GB` LPDDR5x Unified Memory)** against the official `404 GB` [`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) checkpoint:

| Workload / Memory Mode | Cold Model Open (`mmap`) | Prompt Prefill (TTFT) | Generation Speed (Decode) | Active RAM Used | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **🚀 `--mode 55gb` + `--mtp adaptive`** *(Recommended Fast Mode)* | `0.85 s` | `~1.1 – 1.8 s` *(short chat)* | **`16.95 tok/s`** (`59.0 ms/tok`) | **`~62 GB`** (`54.9 GB` experts + `7 GB` core) | `2,960` hot `Tier A` experts pinned in Unified Memory + 3-layer MTP (`100%` greedy match with `K=0`). |
| **⚙️ `--mode 55gb` + `--mtp off`** *(Standard `K=0` Baseline)* | `0.85 s` | `~1.1 – 1.8 s` *(short chat)* | **`14.31 tok/s`** (`69.9 ms/tok`) | **`~62 GB`** | Pure single-token autoregressive decoding without speculative MTP (`69.9 ms/token`). |
| **⚡ `--mode 7gb` + `--mtp adaptive`** *(Low-RAM Default)* | `0.88 s` | `~1.1 – 1.7 s` *(short chat)* | **`13.06 – 14.60 tok/s`** (`68 – 76 ms/tok`) | **`~7.2 GB`** (+ OS Page Cache) | Zero-copy `mmap` on-demand expert routing. Leaves **`> 110 GB` RAM free** for other workloads. |
| **💿 `--mode tierc` (All `384/384` Experts)** *(Full `404 GB` NVMe)* | `0.88 s` | `~10.6 s` *(cold NVMe)* | **`2.57 tok/s`** (`389 ms/tok`) | **`~62 GB`** + NVMe stream | Activates all `15,360` experts (`384/layer`) + `33 GB` Engram (`0` skips). Bottlenecked by NVMe SSD read bandwidth (`404 GB > 128 GB` RAM). |
| **👁️ Multimodal Vision (`--image`)** *(32-Layer ViT + Aligner)* | `0.88 s` | `15.76 s` (`1,296` image patches) | **`8.85 tok/s`** (`113 ms/tok`) | **`~9.5 GB`** | Full `756×756` image encoding (`32`-layer `deepseek_v41_vision` ViT + projector) + text generation. |

### Built-In Speculative MTP Benchmark (`./bin/flash1017-gx10-linux-arm64 bench -m ./DeepSeek-V4.1-Flash-1017-NVFP4 --mode 55gb`)

```text
Mode              Tok/s    ms/tok   Steps   Acc%     Tok/Step   Match K0
----------------------------------------------------------------------------
K=0 (Off)         14.31    69.9     64      0.0    % 1.00       ✅
K=1               16.17    61.8     43      50.0   % 1.50       ✅
K=2               14.79    67.6     36      39.3   % 1.79       ✅
K=3               13.95    71.7     32      33.3   % 2.00       ✅
Adaptive (1..3)   16.95    59.0     42      47.4   % 1.53       ✅
```

---

## 🔒 Hardware Lock & Closed-Source Binary Distribution

This repository is distributed **exclusively as stripped native ARM64 binaries** with zero source code or unstripped symbols:
1. **`bin/flash1017-gx10-linux-arm64`** (`1.13 MB`, `strip = "symbols"`, `lto = "thin"`, `panic = "abort"`) — Standalone CLI, interactive REPL, and OpenAI / `vLLM` / `llama-server` HTTP/SSE server (`aarch64-unknown-linux-gnu`).
2. **`bin/addon_gx10_rust.so`** (`4.7 MB`) — `sm_121` Blackwell PTX V3 Warp-per-Row + Hybrid-Sync + FlashAttn + `NVFP4` / `1.58-bit` / `MXFP8` / `BF16` CUDA compute kernels and cryptographic GB10 hardware verification (`unlock_gb10_layer`).
3. **`bin/merkaba_tier_map.bin`** (`15 KB`, `MKTMAP01`) — `SplitMix64`-encrypted 40-layer `Tier A / Tier B / Tier C` anatomical expert routing map for `DeepSeek-V4.1-Flash-1017-NVFP4`.
4. **`bin/engram_token_map.bin`** (`505 KB`) — Canonical token projection table for the `33 GB` `O(1)` Zero-Copy `MXFP4` N-Gram Engram conditional memory (Layer 1 & Layer 14).

---

## 🚀 Quick Start on ASUS Ascent GX10 / DGX Spark (`GB10`)

### 1. Download NVIDIA's Official `DeepSeek-V4.1-Flash-1017-NVFP4` from Hugging Face
The engine runs **directly against the official [`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) checkpoint**:

```bash
huggingface-cli download nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --local-dir ./DeepSeek-V4.1-Flash-1017-NVFP4
```

- **Zero manual file copying required:** `flash1017-gx10` automatically locates [`bin/addon_gx10_rust.so`](bin/addon_gx10_rust.so), [`bin/merkaba_tier_map.bin`](bin/merkaba_tier_map.bin), and [`bin/engram_token_map.bin`](bin/engram_token_map.bin) directly from the `bin/` folder.
- On the first launch against a freshly downloaded `DeepSeek-V4.1-Flash-1017-NVFP4` directory, the engine automatically builds its accelerated `NVFP4` / `C2F` turbo sidecar files (`merkaba_backbone_q4.bin`, `merkaba_head_q8_c2f.bin`, `merkaba_mtp0..2_q4.bin`) inside the model directory. On every subsequent run, the entire 48-shard model opens via Zero-Copy `mmap` in **`< 1 second` (`~880 ms`)**!

---

### 2. Direct CLI Execution (`llama-cli`-compatible)

```bash
./bin/flash1017-gx10-linux-arm64 \
  -m ./DeepSeek-V4.1-Flash-1017-NVFP4 \
  --mode 7gb \
  --mtp adaptive \
  -p "Explain why NVIDIA Grace Blackwell is fast in two short sentences." \
  -n 256
```

### 3. Memory Profiles (`--mode 7gb | 55gb | tierc`)
Run the exact same `DeepSeek-V4.1-Flash-1017-NVFP4` directory in three selectable memory modes depending on how much of your `128 GB Unified Memory` you want to allocate:

| Memory Profile | CLI Flag / REPL | Active RAM | Description |
| :--- | :--- | :--- | :--- |
| **⚡ 7 GB Turbo** *(default)* | `--mode 7gb` / `/7gb` | `~7 GB` (+ OS Page Cache) | Routes the high-energy `Tier A + Tier B` expert core using [`bin/merkaba_tier_map.bin`](bin/merkaba_tier_map.bin) via on-demand Zero-Copy `mmap`. Leaves `> 110 GB` of RAM free for other workloads. |
| **🚀 55 GB RAM Preload** | `--mode 55gb` / `/55gb` | `~55 GB` locked in RAM | Preloads and pins `2,960` hot `Tier A` experts (`54.88 GiB`) in Unified Memory (`0 ms` SSD page faults, maximum `tok/s`). |
| **💿 Full 384-Expert (`Tier C`)** | `--mode tierc` / `/tierc` | `~55 GB` + NVMe Zero-Copy | Preloads `55 GB` in RAM and activates **all `15,360` MoE experts (`384/layer`)** + `33 GB` N-Gram Engram directly from `DeepSeek-V4.1-Flash-1017-NVFP4` (`0` skips). |

---

### 4. Multimodal Vision (`-i` / `--image` — 32-Layer `deepseek_v41_vision` ViT + Aligner)

```bash
./bin/flash1017-gx10-linux-arm64 \
  -m ./DeepSeek-V4.1-Flash-1017-NVFP4 \
  --image /path/to/image.png \
  -p "Describe this image in detail: what objects, colors, and materials do you see?" \
  -n 128
```

---

### 5. Interactive Turbo REPL (`chat`)

```bash
./bin/flash1017-gx10-linux-arm64 chat \
  -m ./DeepSeek-V4.1-Flash-1017-NVFP4 \
  --mode 7gb \
  --mtp adaptive
```

Available live REPL commands:
- `/7gb` (or `/experts ondemand`) — Switch live to `7 GB Turbo` mode.
- `/55gb` (or `/experts preload`) — Switch live to `+55 GB in RAM` mode.
- `/tierc` (or `/experts tierc`) — Switch live to `All 384 Experts (Tier A + B + C)` mode.
- `/bench` — Run the built-in 3-layer speculative MTP benchmark (`K=0`, `K=1..4`, `Adaptive`).
- `/quit` (or `/exit`) — Exit the REPL.

---

## 🌐 OpenAI / `vLLM` / `llama-server` Drop-In HTTP/SSE Server (`serve`)

Start the HTTP server (default port `8000`, `0.0.0.0`):

```bash
./bin/flash1017-gx10-linux-arm64 serve \
  -m ./DeepSeek-V4.1-Flash-1017-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --mode 7gb \
  --mtp adaptive
```

### Supported Endpoints
| Endpoint | Description |
| :--- | :--- |
| `POST /v1/chat/completions` | OpenAI Chat Completions (supports both `stream: false` and SSE `stream: true`, `enable_thinking`, `logprobs`, `top_logprobs`, `stop`, and multimodal Vision `image_url` with local file paths or `data:image/...;base64,...`) |
| `POST /v1/completions` | OpenAI Text Completions (`stream: false` & SSE `stream: true`, `logprobs`, `stop` — compatible with `lm-evaluation-harness`) |
| `GET /v1/models` | Lists `DeepSeek-V4.1-Flash-1017-NVFP4` with `max_model_len: 1048576` (`1M` IN-context) and `max_output_tokens: 8192` |
| `GET /health` & `GET /props` | Health check and context/memory telemetry (`vLLM` & `llama-server`-compatible) |
| `POST /v1/engine/mode` | Switch memory profile live via JSON (`{"mode": "7gb" \| "55gb" \| "tierc"}`) |

---

## 📏 Context Window (`1M` IN / `8K` OUT) & Sampling Flags

`flash1017-gx10` supports the full official `DeepSeek-V4.1-Flash-1017` long-context architecture (36 SWA layers + 4 global MLA source layers `[2, 8, 14, 20]` with `compress_ratio = 2` and YaRN RoPE `factor = 16`):
- **IN-Context (`1,048,576` tokens / `1M` default):**
  - Configured via `-c <tokens>`, `--ctx-size <tokens>`, `--max-model-len <tokens>`, or `--max-input-tokens <tokens>` (and per request in JSON via `max_context_tokens` / `max_model_len` / `n_ctx`).
  - The KV cache starts pre-allocated for `4,096` tokens and grows dynamically with automatic `cuMemHostRegister` re-registration up to `1,048,576` tokens (`~5 KB/token` long-context footprint).
- **OUT-Context (`8,192` tokens default):**
  - Configured via `-n <tokens>`, `--n-predict <tokens>`, `--max-tokens <tokens>`, or `--max-output-tokens <tokens>` (and per request in JSON via `max_tokens` / `max_completion_tokens` / `n_predict`).

### Complete CLI Flag Reference
- `-m <dir>` / `--model <dir>` — Path to `DeepSeek-V4.1-Flash-1017-NVFP4`
- `--mode <7gb|55gb|tierc>` / `--memory-mode <...>` — Memory profile
- `--mtp <adaptive|off|1|2|3|4>` — Speculative 3-layer Multi-Token Prediction (`mtp.0..2`)
- `--head <c2f|q8|bf16>` — LM head mode (`c2f` = `CoarseToFineConceptHead` default)
- `-c <usize>` / `--ctx-size <usize>` / `--max-model-len <usize>` — Max IN-context (`1048576` default)
- `-n <usize>` / `--max-tokens <usize>` / `--max-output-tokens <usize>` — Max OUT-context (`8192` default)
- `--temp <f32>` (`temperature`, default `0.0`)
- `--top-p <f32>` (`top_p`, default `1.0`)
- `--top-k <usize>` (`top_k`, default `0`)
- `--min-p <f32>` (`min_p`, default `0.0`)
- `--repeat-penalty <f32>` & `--repeat-last-n <usize>` (`repetition_penalty`)
- `--frequency-penalty <f32>` (`frequency_penalty`)
- `--presence-penalty <f32>` (`presence_penalty`)
- `-s <u64>` / `--seed <u64>` (`seed`, deterministic SplitMix64)
- `--stop <str>` (`stop`, repeatable)
- `--ignore-eos` (`ignore_eos`)
- `--think` / `--no-think` (`enable_thinking`)
- `--engram` / `--no-engram` (enable/disable `33 GB` N-Gram Engram conditional memory; enabled by default)
- `--logprobs` & `--top-logprobs <0..20>`
- `-i <path>` / `--image <path_or_data_url>` (repeatable)
- `--api-key <key>` (optional Bearer API key for `serve`)
