# ⚡ Flash1017-Engine (`flash1017-gx10`) — Standalone GB10 Turbo Engine

**`flash1017-gx10`** är en fristående, 100 % closed-source och hårdvarulåst (`NVIDIA Grace Blackwell GB10 / sm_121`) inferensmotor och `vLLM` / `llama-server`-kompatibel server för **`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`** på **ASUS Ascent GX10 / DGX Spark (`128 GB Unified Memory`)**.

---

## 🔒 Hårdvarulås & Closed-Source Distribution

Paketet distribueras **endast som strippade maskinkodsbinärer** utan källkod eller symboler:
1. **`bin/flash1017-gx10-linux-arm64`** (`1.13 MB`, `strip = "symbols"`, `lto = "thin"`, `panic = "abort"`) — Fristående CLI-, REPL- och OpenAI/vLLM HTTP-serverbinär (`aarch64-unknown-linux-gnu`).
2. **`bin/addon_gx10_rust.so`** (`4.7 MB`) — `sm_121` Blackwell PTX V3 Warp-per-Row + Hybrid-Sync + FlashAttn + `NVFP4` / `1.58-bit` / `MXFP8` / `BF16` CUDA-kärnor och kryptografiskt GB10-hårdvarulås (`unlock_gb10_layer`).
3. **`bin/merkaba_tier_map.bin`** (`15 KB`, `MKTMAP01`) — `SplitMix64`-krypterad 40-lagers `Tier A / Tier B / Tier C` anatomisk expertkarta för `DeepSeek-V4.1-Flash-1017-NVFP4`.
4. **`bin/engram_token_map.bin`** (`505 KB`) — Kanonisk token-projektionskarta för `33 GB` `O(1)` Zero-Copy `MXFP4` N-Gram Engram-minnet (Lager 1 & 14).

---

## 🚀 Snabbstart på ASUS Ascent GX10 / DGX Spark (`GB10`)

### 1. Ladda ner NVIDIA:s `DeepSeek-V4.1-Flash-1017-NVFP4` från Hugging Face
Motorn kör **direkt mot den officiella modellen [`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4)**:

```bash
huggingface-cli download nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --local-dir ./DeepSeek-V4.1-Flash-1017-NVFP4
```

- Du behöver **inte** kopiera in några extra filer i modellmappen: `flash1017-gx10` hittar automatiskt [`bin/addon_gx10_rust.so`](bin/addon_gx10_rust.so), [`bin/merkaba_tier_map.bin`](bin/merkaba_tier_map.bin) och [`bin/engram_token_map.bin`](bin/engram_token_map.bin) direkt i `bin/`-mappen.
- Första gången du startar mot en ny `DeepSeek-V4.1-Flash-1017-NVFP4`-mapp genererar motorn automatiskt sina accelererade `NVFP4` / `C2F` turbo-sidecar-filer (`merkaba_backbone_q4.bin`, `merkaba_head_q8_c2f.bin`, `merkaba_mtp0..2_q4.bin`) i modellmappen. Vid alla efterföljande starter öppnas hela modellen via Zero-Copy `mmap` på **`< 1 sekund` (`~880 ms`)**!

---

### 2. Kör direkt från kommandoraden (`llama-cli`-kompatibel)

```bash
./bin/flash1017-gx10-linux-arm64 \
  -m ./DeepSeek-V4.1-Flash-1017-NVFP4 \
  --mode 7gb \
  --mtp adaptive \
  -p "Explain why NVIDIA Grace Blackwell is fast in two short sentences." \
  -n 256
```

### 3. Välj Minnesprofil (`--mode 7gb | 55gb | tierc`)
Du kan köra samma `DeepSeek-V4.1-Flash-1017-NVFP4`-mapp i tre olika minneslägen beroende på hur mycket av dina `128 GB Unified Memory` du vill reservera:

| Minnesläge | Flagga / REPL | Aktivt RAM | Beskrivning |
| :--- | :--- | :--- | :--- |
| **⚡ 7 GB Turbo** *(standard)* | `--mode 7gb` / `/7gb` | `~7 GB` (+ OS Page Cache) | Använder vår krypterade `Tier A + Tier B`-expertkarta (`bin/merkaba_tier_map.bin`) via Zero-Copy `mmap` on-demand. Lämnar `> 110 GB` RAM fritt för andra uppgifter. |
| **🚀 55 GB RAM Preload** | `--mode 55gb` / `/55gb` | `~55 GB` låst i RAM | Förladdar och låser `2 960` heta `Tier A`-experter (`54.88 GiB`) i Unified Memory (`0 ms` page-faults, högsta `tok/s`). |
| **💿 Full 384-Expert (`Tier C`)** | `--mode tierc` / `/tierc` | `~55 GB` + NVMe Zero-Copy | Förladdar `55 GB` i RAM och aktiverar **samtliga `15 360` MoE-experter (`384/lager`)** + `33 GB` N-Gram Engram direkt ur `DeepSeek-V4.1-Flash-1017-NVFP4` (`0` skips). |

---

### 4. Multimodal Vision (`-i` / `--image` — 32-lagers `deepseek_v41_vision` ViT + Aligner)

```bash
./bin/flash1017-gx10-linux-arm64 \
  -m ./DeepSeek-V4.1-Flash-1017-NVFP4 \
  --image /sökväg/till/bild.png \
  -p "Describe this image in detail: what objects, colors, and materials do you see?" \
  -n 128
```

---

### 5. Interaktiv Turbo-REPL (`chat`)

```bash
./bin/flash1017-gx10-linux-arm64 chat \
  -m ./DeepSeek-V4.1-Flash-1017-NVFP4 \
  --mode 7gb \
  --mtp adaptive
```

Tillgängliga REPL-kommandon under körning:
- `/7gb` (eller `/experts ondemand`) — Växla live till `7 GB Turbo`-läget.
- `/55gb` (eller `/experts preload`) — Växla live till `+55 GB i RAM`-läget.
- `/tierc` (eller `/experts tierc`) — Växla live till `Alla 384 experter (Tier A + B + C)`-läget.
- `/bench` — Kör inbyggt MTP-benchmark (`K=0`, `K=1..4`, `Adaptive`).
- `/quit` (eller `/exit`) — Avsluta REPL.

---

## 🌐 OpenAI / `vLLM` / `llama-server` Drop-In HTTP/SSE-Server (`serve`)

Starta servern (standardport `8000`, `0.0.0.0`):

```bash
./bin/flash1017-gx10-linux-arm64 serve \
  -m ./DeepSeek-V4.1-Flash-1017-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --mode 7gb \
  --mtp adaptive
```

### Endpoints som stöds
| Endpoint | Beskrivning |
| :--- | :--- |
| `POST /v1/chat/completions` | OpenAI Chat Completions (stödjer både `stream: false` och SSE `stream: true`, `enable_thinking`, `logprobs`, `top_logprobs`, `stop`, samt multimodal Vision `image_url` med lokal filsökväg eller `data:image/...;base64,...`) |
| `POST /v1/completions` | OpenAI Text Completions (`stream: false` & SSE `stream: true`, `logprobs`, `stop` — kompatibel med `lm-evaluation-harness`) |
| `GET /v1/models` | Returnerar `DeepSeek-V4.1-Flash-1017-NVFP4` med `max_model_len: 1048576` (`1M` IN-context) och `max_output_tokens: 8192` |
| `GET /health` & `GET /props` | Hälsokontroll och kontext-/minnesstatus (`vLLM` & `llama-server`-kompatibel) |
| `POST /v1/engine/mode` | Växla minnesprofil live under drift via JSON (`{"mode": "7gb" \| "55gb" \| "tierc"}`) |

---

## 📏 Context-Fönster (`1M` IN / `8K` UT) & Samplingsflaggor

Motorn stödjer `DeepSeek-V4.1-Flash-1017`:s fulla officiella kontextarkitektur (36 SWA-lager + 4 globala MLA-källager `[2, 8, 14, 20]` med `compress_ratio = 2` och YaRN RoPE `factor = 16`):
- **IN-Context (`1 048 576` tokens / `1M` som standard):**
  - Styrs via `-c <tokens>`, `--ctx-size <tokens>`, `--max-model-len <tokens>` eller `--max-input-tokens <tokens>` (samt per anrop i JSON via `max_context_tokens` / `max_model_len` / `n_ctx`).
  - KV-cachen startar förallokerad för `4 096` tokens och växer dynamiskt med automatisk `cuMemHostRegister`-omregistrering upp till `1 048 576` tokens (`~5 KB/token` långtidsminne).
- **UT-Context (`8 192` tokens som standard):**
  - Styrs via `-n <tokens>`, `--n-predict <tokens>`, `--max-tokens <tokens>` eller `--max-output-tokens <tokens>` (samt per anrop i JSON via `max_tokens` / `max_completion_tokens` / `n_predict`).

### Kompletta CLI-flaggor
- `-m <dir>` / `--model <dir>` — Sökväg till `DeepSeek-V4.1-Flash-1017-NVFP4`
- `--mode <7gb|55gb|tierc>` / `--memory-mode <...>` — Minnesprofil
- `--mtp <adaptive|off|1|2|3|4>` — Spekulativ 3-lagers Multi-Token Prediction (`mtp.0..2`)
- `--head <c2f|q8|bf16>` — LM-head-läge (`c2f` = `CoarseToFineConceptHead` som standard)
- `-c <usize>` / `--ctx-size <usize>` / `--max-model-len <usize>` — Max IN-context (`1048576` standard)
- `-n <usize>` / `--max-tokens <usize>` / `--max-output-tokens <usize>` — Max UT-context (`8192` standard)
- `--temp <f32>` (`temperature`, standard `0.0`)
- `--top-p <f32>` (`top_p`, standard `1.0`)
- `--top-k <usize>` (`top_k`, standard `0`)
- `--min-p <f32>` (`min_p`, standard `0.0`)
- `--repeat-penalty <f32>` & `--repeat-last-n <usize>` (`repetition_penalty`)
- `--frequency-penalty <f32>` (`frequency_penalty`)
- `--presence-penalty <f32>` (`presence_penalty`)
- `-s <u64>` / `--seed <u64>` (`seed`, deterministisk SplitMix64)
- `--stop <str>` (`stop`, kan anges flera gånger)
- `--ignore-eos` (`ignore_eos`)
- `--think` / `--no-think` (`enable_thinking`)
- `--engram` / `--no-engram` (aktivera/inaktivera `33 GB` N-Gram Engram-minne; aktivt som standard)
- `--logprobs` & `--top-logprobs <0..20>`
- `-i <path>` / `--image <path_or_data_url>` (kan anges flera gånger)
- `--api-key <key>` (valfri Bearer API-nyckel för `serve`)
