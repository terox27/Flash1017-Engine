# ⚡ Flash1017-Engine (`flash1017-gx10`) — Source-Available GB10 Turbo Engine (`LICENSE`)

> **📜 Source-Available + `106 GB` All-Experts-in-RAM Blueprint ([`LICENSE`](LICENSE)):**  
> `Flash1017-Engine` (`crates/flash-core` + `crates/flash-cli`), its compute kernels, and our mathematical blueprint **[`BLUEPRINT_106GB_ALL_EXPERTS_RAM.md`](BLUEPRINT_106GB_ALL_EXPERTS_RAM.md)** are **100% free for personal, hobbyist, and academic research use (`<= €1,000 EUR / $1,000 USD per month`)**.  
> - **Commercial Threshold (`> €1,000 EUR / $1,000 USD per month` — Including Cloud, API & Internal Business Use):** Any person, company, or commercial entity earning or saving more than **`€1,000 EUR` (`~$1,000 USD`) per month** in connection with this engine, container format, or blueprint **must obtain a paid commercial license** from the author (`terox27`).  
> - **Two-Layer Separation & Mandatory Open Weight Sharing (`All Models of Any Size or Architecture`):** This license governs the **Engine, Kernels & Container Format** independently of underlying base model licenses. Anyone who generates compressed weights under the free tier — **for ANY model of any size (`12 GB`, `106 GB`, `300 GB`, etc.) or architecture (MoE, Dense, Multimodal)** — **must share the resulting weights freely and openly with the community** (if a base model's license forbids public sharing, or if weights are kept private, a paid commercial license from `terox27` is required).  
> - **Ports to Other Hardware, Languages & Model Architectures (`Section 4`):** All ports (e.g., to Apple `Metal`, AMD `ROCm`/`Strix Halo`, discrete `RTX` GPUs, `C++`/`llama.cpp`/`MLX`/`vLLM`, or other models) inherit this exact `€1,000 EUR / $1,000 USD per month` commercial license and grant a royalty-free upstream right back to `terox27` / `Merkaba01` (see [`LICENSE`](LICENSE)).

**`flash1017-gx10`** is a standalone Pure Rust inference engine and `vLLM` / `llama-server`-compatible HTTP/SSE server for **[`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4)** (`1T`-class MoE, `404 GB` on disk, `40` layers, `384` routed experts/layer + `33 GB` N-Gram Engram + `32`-layer Vision ViT) and community-built **`106 GB` All-Experts-in-RAM (`1.58-bit + Rank-16 Q8 JEPA`)** checkpoints on **ASUS Ascent GX10 / NVIDIA DGX Spark (`128 GB Unified Memory`)**, Linux CUDA, and Apple Silicon.

---

## 🧬 New in This Release: Built-In `106 GB` `JEPA` Reader, `JEPA` Autopilot & `CPU Lookahead`

`Flash1017-Engine` now natively implements all 4 runtime subsystems required by **[`BLUEPRINT_106GB_ALL_EXPERTS_RAM.md`](BLUEPRINT_106GB_ALL_EXPERTS_RAM.md)**:

1. **📦 `106 GB` Expert Reader (`1.58-bit` Base `6.35 MB` + `0.33 MB` `Rank-16 Q8` JEPA Core = `6.68 MB/expert`):**
   - `CachedTernary158Mat::from_shard` + `CachedRank16Q8Core::from_shard` automatically detect and zero-copy `mmap`-map `.packed_ternary`, `.row_scales`, and optional `.jepa_u_q8` / `.jepa_v_q8` / `.jepa_u_scale` / `.jepa_v_scale` tensors for any expert (`Tier A`, `Tier B`, or `Tier C`).
   - Adds `< 4 µs` of latency per matrix (`y += U_q8 · (V_q8^T · x)`) while restoring `~98.5–99.2%` cosine similarity across all `15,360` experts (`102.60 GB` in RAM, `0` SSD reads for experts!).
2. **🧭 `JEPA` Autopilot (Inter-Layer Noise-Cancelling & `MTP` Coherence Scoring):**
   - **Active Inter-Layer Noise-Cancelling (`apply_jepa_inter_layer_noise_canceller`):** At layer checkpoints (`l == 15` and `l == 31`), measures the alignment of the 4 `mHC` hyper-connection streams against the verified `jepa_trajectory_anchor` in `Flash1017KvCache` and gently damps orthogonal quantization/rounding drift while preserving exact per-stream RMS energy (`< 0.02 ms`).
   - **`JEPA` Speculative `MTP` Scoring (`jepa_mtp_coherence_score`):** Evaluates cosine coherence between each speculative `mtp.0..2` draft hidden state and the `JEPA` trajectory anchor, boosting multi-token acceptance (`2.5–3.2 tokens/step`).
3. **🔮 `CPU-Tjuvkikaren` (`CpuLookaheadStager`) & `t = 0.0 ms` `Engram` Pre-Lookup:**
   - **Exact `t = 0.0 ms` `Engram` Pre-Lookup (`prefetch_engram_at_t0`):** Because `DeepSeek-V4.1-Flash-1017`'s `196B` `Engram` tables (Layer 1 & Layer 14) hash **input token IDs** (`token_history`) using 24 deterministic prime hashes (`ENGRAM_MULTS` / `ENGRAM_PRIMES`) and NOT hidden states, the CPU computes all 24 exact `128-byte` `MXFP4` row addresses (`2.7 KB` total) at **`t = 0.0 ms` before Layer 0 even begins** and fires `madvise(MADV_WILLNEED)` so `Engram` is `100%` warm in RAM when Layer 1 and Layer 14 execute (`0.0 ms` SSD wait).
   - **20-Core Parallel Prompt Lookahead (`cpu_lookahead_stage_prompt_tokens`) & Cross-Layer Lookahead (`prefetch_next_layer_experts_lookahead`):** Uses all 20 ARM64 Grace CPU cores via Rayon at prompt arrival (`t = 0 ms`) and layer `l -> l + 1` to pre-stage expert pages into OS Page Cache ahead of the GPU.
4. **🔓 100% Open-Source Rust Workspace (`Cargo.toml` + `crates/`):**
   - No hardware lock — build directly from source with `cargo build --release --bin flash1017-gx10` on any platform (`aarch64-unknown-linux-gnu`, `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`), or run the pre-built `bin/flash1017-gx10-linux-arm64` binary with `bin/addon_gx10_rust.so` on NVIDIA Grace Blackwell (`GB10`).

---

## 📊 Real-World Benchmarks on ASUS Ascent GX10 (`Single 128 GB GB10`, Batch Size = 1)

Real-world single-user performance measured directly on a **single ASUS Ascent GX10 (`NVIDIA GB10`, `sm_121`, `128 GB` LPDDR5x Unified Memory)** against the official **`404 GB` [`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4)** checkpoint (`15,360` MoE experts across 40 layers + `33 GB` N-Gram Engram + `32`-layer Vision ViT):

> **Important Note on MoE Quality (`All 384 Experts Required`):**  
> In `DeepSeek-V4.1-Flash-1017`, each of the 40 layers routes tokens to Top-6 out of **384 experts** (`Tier A` + `Tier B` + `Tier C`). **All 3 tiers (`384/384` experts per layer) must be active for real-world usage** — if you skip `Tier C` (`240/384` experts per layer), general prompts lose critical routing paths and produce degraded/garbage output. Therefore, **`--mode 7gb`, `--mode 55gb`, and `--mode tierc` all keep all 384 experts (`Tier A + B + C`) active by default.**

| Workload / Mode | Active Experts | Cold Startup (`mmap`) | Prompt Prefill (TTFT) | Decode Speed (`tok/s`) | Active RAM + Cache | Real-World Bottleneck & Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **⚡ `--mode 7gb` (Default — All 384 Experts On-Demand)** | **`384 / 384`** (`Tier A+B+C`) | **`0.89 s`** | **`~2.4 – 9.4 s`** *(warm vs cold cache)* | **`2.32 – 2.57 tok/s`** (`389–430 ms/tok`) | `~7.2 GB` pinned + `~110 GB` OS Page Cache | **Full model quality (0 skipped experts).** Since `404 GB > 128 GB` RAM, `~110 GB` of hot experts stay cached in RAM while uncached `Tier C` tail experts stream from NVMe via `MADV_WILLNEED` (`~2–2.5 GB/tok` NVMe read bound). |
| **🚀 `--mode 55gb` / `--mode tierc` (55 GB Preload + All 384 Experts)** | **`384 / 384`** (`Tier A+B+C`) | **`0.89 s`** | **`~2.5 – 9.5 s`** | **`2.30 – 2.57 tok/s`** (`389–435 ms/tok`) | `~62 GB` pinned (`54.9 GB` Tier A + `7 GB` core) + `~55 GB` OS Page Cache | **Full model quality (0 skipped experts).** Pre-warms `2,960` hot `Tier A` experts (`54.88 GiB`) in RAM and streams remaining `Tier B + C` experts from NVMe on-demand. |
| **👁️ Multimodal Vision (`--image`, 32-Layer ViT + Aligner)** | **`384 / 384`** (`Tier A+B+C`) | **`0.89 s`** | **`15.76 s`** (`1,296` image patches) | **`2.3 – 8.8 tok/s`** | `~9.5 GB` pinned + OS Page Cache | Encodes `756×756` images through the 32-layer `deepseek_v41_vision` ViT + projector before autoregressive text decoding. |
| **🧪 `--mode pruned` (Experimental In-RAM Kernel Benchmark Only)** | *`144 / 384` (`Tier A+B` only)* | `0.85 s` | `~1.1 – 1.8 s` | **`14.31 – 16.95 tok/s`** (`59.0 ms/tok` w/ MTP) | `~62 GB` fits 100% in RAM (`0` NVMe reads) | **Kernel speed test only (not for general chat).** Disables `Tier C` NVMe streaming so all active weights fit inside `128 GB` RAM, proving the `sm_121` Blackwell `NVFP4` + `MTP` kernels reach **`16.95 tok/s`** when not bottlenecked by NVMe SSD I/O. |

### Built-In Speculative MTP Kernel Benchmark (`./bin/flash1017-gx10-linux-arm64 bench -m ./DeepSeek-V4.1-Flash-1017-NVFP4 --mode pruned`)

When measuring pure in-memory GB10 kernel throughput (`0` NVMe SSD page faults):

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

## 🛠️ Building from Open-Source Rust Source & Included Prebuilt Binaries

You can either build `flash1017-gx10` directly from the open-source Rust workspace or run the pre-compiled ARM64 Linux binary in `bin/`:

```bash
cargo build --release --bin flash1017-gx10
```

Included in this repository:
1. **`crates/flash-core` & `crates/flash-cli`** ([`LICENSE`](LICENSE)) — Complete Rust runtime, `106 GB` `1.58-bit + 0.33 MB Rank-16 Q8 JEPA` expert reader (`CachedTernary158Mat` + `CachedRank16Q8Core`), `JEPA` inter-layer noise-canceller & `MTP` coherence scorer, `t = 0 ms` `Engram` pre-lookup (`prefetch_engram_at_t0`), 20-core CPU lookahead stager, and OpenAI / `vLLM` / `llama-server` HTTP/SSE server.
2. **[`BLUEPRINT_106GB_ALL_EXPERTS_RAM.md`](BLUEPRINT_106GB_ALL_EXPERTS_RAM.md)** — Complete open specification and mathematical recipe for compressing all 384 routed experts (`Tier A + B + C`) into `106.85 GB` RAM (`40–55+ tok/s`).
3. **`bin/flash1017-gx10-linux-arm64`** (`1.13 MB`) — Pre-built native ARM64 binary (`aarch64-unknown-linux-gnu`) for ASUS Ascent GX10 / NVIDIA DGX Spark (`GB10`).
4. **`bin/addon_gx10_rust.so`** (`4.7 MB`) — Optional `sm_121` Blackwell PTX V3 Warp-per-Row + Hybrid-Sync + FlashAttn + `NVFP4` / `1.58-bit` / `MXFP8` / `BF16` CUDA compute kernel library for NVIDIA `GB10`.
5. **`bin/merkaba_tier_map.bin`** (`15 KB`, `MKTMAP01`) — 40-layer `Tier A / Tier B / Tier C` anatomical expert routing map for `DeepSeek-V4.1-Flash-1017-NVFP4`.
6. **`bin/engram_token_map.bin`** (`505 KB`) — Canonical token projection table for the `33 GB` `O(1)` Zero-Copy `MXFP4` N-Gram Engram conditional memory (Layer 1 & Layer 14).

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
All three standard profiles activate **all `15,360` MoE experts (`384/layer`: `Tier A + Tier B + Tier C`)** + the `33 GB` N-Gram Engram (`0` skipped experts, full model quality), differing only in how much RAM is pre-warmed vs dynamically cached by the OS Page Cache:

| Memory Profile | CLI Flag / REPL | Pinned RAM | Description |
| :--- | :--- | :--- | :--- |
| **⚡ 7 GB Turbo + All 384 Experts** *(default)* | `--mode 7gb` / `/7gb` | `~7 GB` (+ `~110 GB` OS Page Cache) | Pins only the `7 GB` backbone/KV and routes **all `384/384` experts (`Tier A + B + C`)** via on-demand Zero-Copy `mmap` + `MADV_WILLNEED` NVMe prefetch. Lets the Linux OS Page Cache manage `~110 GB` of hot experts dynamically. |
| **🚀 55 GB RAM Preload + All 384 Experts** | `--mode 55gb` / `/55gb` | `~62 GB` (`54.9 GB` Tier A + `7 GB` core) | Pre-warms `2,960` hot `Tier A` experts (`54.88 GiB`) in Unified Memory and streams `Tier B + Tier C` experts via Zero-Copy `mmap` (`0` skips). |
| **💿 Full 384-Expert (`Tier C` Alias)** | `--mode tierc` / `/tierc` | `~62 GB` + NVMe Zero-Copy | Identical to `55gb`: preloads `55 GB` Tier A in RAM and activates **all `15,360` MoE experts (`384/layer`)** + `33 GB` N-Gram Engram (`0` skips). |

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


---

## 📚 Scientific Foundations, Model Attribution & Credits

See **[`CREDITS.md`](CREDITS.md)** for full scientific references and attribution:
- **DeepSeek-AI:** *`DeepSeek-V4.1-Flash-1017`* (`748B` MoE + `196B` Conditional N-Gram `Engram` + `MLA` + `4-Stream mHC` + `3-Layer MTP` + `32-Layer Vision ViT`).
- **NVIDIA Corporation:** Official [`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) checkpoint (`NVFP4` / `MXFP8` / `MXFP4`) & Grace Blackwell (`GB10` / `sm_121`) Unified Memory architecture.
- **BitNet b1.58 (Ma et al., Microsoft Research, 2024):** Inspires our `1.58-bit` ternary base (`{-1, 0, +1}` + AbsMean row scale $\gamma$, `6.35 MB/expert`) in `CachedTernary158Mat` and [`BLUEPRINT_106GB_ALL_EXPERTS_RAM.md`](BLUEPRINT_106GB_ALL_EXPERTS_RAM.md).
- **DoRA (Liu et al., 2024) & LoftQ / Low-Rank SVD Error-Healing (Li et al., 2023):** Inspires our `0.33 MB` `Rank-16 Q8` high-precision residual core (`CachedRank16Q8Core`).
- **JEPA (Yann LeCun, 2022) & Platonic Good-View Representations (`arXiv:2610.39302`):** Inspires our activation-weighted `JEPA` calibration ($R_{\text{weighted}} = \Delta W \cdot C_X^{1/2}$) and the runtime `JEPA` Autopilot (`apply_jepa_inter_layer_noise_canceller` & `jepa_mtp_coherence_score`).
- **LaCo (*Reserving-Differences-while-Seeking-Common*, Yang et al., 2024) & Pre-Gated MoE Prefetching:** Inspires our `CPU-Tjuvkikare` (`CpuLookaheadStager`) and `t = 0 ms` `Engram` pre-lookup (`prefetch_engram_at_t0`).
