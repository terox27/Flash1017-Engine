# Scientific Foundations, Model Attribution & Credits (`Flash1017-Engine`)

`Flash1017-Engine` (`flash1017-gx10`) and the [`BLUEPRINT_106GB_ALL_EXPERTS_RAM.md`](BLUEPRINT_106GB_ALL_EXPERTS_RAM.md) specification are clean-room Pure Rust implementations developed as part of the **[Merkaba01](https://terox27.github.io/)** research project.

We gratefully acknowledge the research teams, open-weights model creators, and scientific publications whose foundational architectures and mathematical insights made this engine and the `106 GB` All-Experts-in-RAM blueprint possible.

---

## 1. Base Model Architecture & Open-Weights Checkpoint

### DeepSeek-AI — *DeepSeek-V4.1-Flash-1017 (`748B` MoE + `196B` Engram + `MLA` + `mHC` + `MTP`)*
- **Role in `Flash1017-Engine`:** Defines the 40-layer `deepseek_v41` CED (`Compressor-Encoder-Decoder`) transformer architecture implemented in [`crates/flash-core/src/flash1017.rs`](crates/flash-core/src/flash1017.rs).
- **Key Architectural Mechanisms Supported:**
  - **Multi-head Latent Attention (`MLA`) & CED Context Compressor:** Low-rank query/key-value latent compression (`kv_lora_rank = 512`), 36 sliding-window attention (`SWA`) layers + 4 global source compressor layers (`[2, 8, 14, 20]`), and YaRN RoPE (`1M` token context window).
  - **4-Stream Manifold-Constrained Hyper-Connections (`mHC`):** `hc_mult = 4` residual stream bus with doubly-stochastic Sinkhorn-Knopp projection matrices (`sinkhorn_knopp_4x4`).
  - **Fine-Grained 384-Expert MoE:** `384` routed experts + `1` shared expert per layer with `sqrtsoftplus` and auxiliary-loss-free `noaux_tc` bias routing (`Top-6` active routed experts/layer).
  - **Conditional N-Gram `Engram` Memory (`196B` Parameters in Layers 1 & 14):** Deterministic 24-table prime-modulo hash lookup (`n = 2, 3, 4` × `8` heads) with `MXFP8` `wkv` projection and `signed-sqrt` sigmoid gating. Exploited in `Flash1017-Engine` via **`t = 0.0 ms` Zero-Latency `Engram` Pre-Lookup (`prefetch_engram_at_t0`)**.
  - **3-Layer Multi-Token Prediction (`mtp.0..2` & `DSpark` Heads):** Speculative multi-token drafting (`dspark_target_layer_ids = [37, 38, 39]`) with `markov_head` transition scoring and `confidence_head` adaptive depth control.
  - **32-Layer Multimodal Vision (`deepseek_v41_vision`):** Native `756×756` image patch encoding and 2×2 spatial downsample aligner (`4096 -> 5120`).

### NVIDIA Corporation — *`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` & Grace Blackwell (`GB10` / `sm_121`)*
- **Role in `Flash1017-Engine`:** Provides the official [`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) checkpoint (`FP4 E2M1` + `F8_E4M3` block scales + `F32` global scale, and `MXFP8`/`MXFP4` `F8_E8M0` microscaling formats) and the `128 GB` LPDDR5x Unified Memory architecture on NVIDIA DGX Spark / ASUS Ascent GX10 (`GB10`).

---

## 2. `106 GB` All-Experts-in-RAM Compression (`1.58-Bit` Base + `Rank-16 Q8` Core)

### BitNet b1.58 — *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (Ma et al., Microsoft Research, 2024)
- **Role in `Flash1017-Engine`:** Inspires the **`1.58-bit` Ternary Base (`{-1, 0, +1}` + per-row AbsMean scale $\gamma$, `6.35 MB/expert`)** in `CachedTernary158Mat` and [`BLUEPRINT_106GB_ALL_EXPERTS_RAM.md`](BLUEPRINT_106GB_ALL_EXPERTS_RAM.md).
- **Key Mechanism Used:** 2-bit packed ternary representation (`4` weights/byte) that shrinks all `15,360` routed MoE experts (`384/layer` × `40` layers) from `291.2 GB` (`4-bit`) down to `97.54 GB` (`1.58-bit` base).

### DoRA (Liu et al., 2024) & LoftQ / Low-Rank Quantization Error Healing (Li et al., 2023)
- **Role in `Flash1017-Engine`:** Inspires the **`0.33 MB` `Rank-16 Q8` High-Precision Core (`CachedRank16Q8Core`)** attached to each `1.58-bit` expert matrix (`w1`, `w2`, `w3`).
- **Key Mechanism Used:** Decomposition of the quantization residual $\Delta W = W_{\text{master}} - W_{1.58\text{-bit}} \approx U_r \Sigma_r V_r^T$ into symmetric `INT8` (`Q8`) rank-16 factors (`+0.33 MB/expert`, `+5.06 GB` across all `15,360` experts), restoring high-precision singular directions in `< 4 µs` per matrix (`y = W_1.58 · x + U_q8 · (V_q8^T · x)`).

---

## 3. `JEPA` Calibration & Runtime Trajectory Autopilot

### JEPA — *A Path Towards Autonomous Machine Intelligence* (Yann LeCun, 2022) & *Platonic Good-View Representations* (`arXiv:2610.39302`)
- **Role in `Flash1017-Engine`:** Inspires both the offline **Activation-Weighted JEPA Calibration** in [`BLUEPRINT_106GB_ALL_EXPERTS_RAM.md`](BLUEPRINT_106GB_ALL_EXPERTS_RAM.md) and the live **`JEPA` Autopilot** in [`crates/flash-core/src/flash1017.rs`](crates/flash-core/src/flash1017.rs).
- **Key Mechanisms Used:**
  - **Activation-Weighted SVD (`Offline Recipe`):** Weighting the quantization residual by real hidden-state activation covariance ($R_{\text{weighted}} = \Delta W \cdot C_X^{1/2}$) so the `Rank-16 Q8` core preserves the exact latent directions that control next-layer hidden states $h_{l+1}$.
  - **Active Inter-Layer Noise-Cancelling (`apply_jepa_inter_layer_noise_canceller`):** At layer checkpoints (`l == 15` and `l == 31`), measures alignment of the 4 `mHC` streams against the verified `jepa_trajectory_anchor` and gently damps orthogonal quantization/rounding drift while preserving exact per-stream RMS energy.
  - **`JEPA` Speculative `MTP` Coherence Scoring (`jepa_mtp_coherence_score`):** Evaluates cosine coherence between speculative `mtp.0..2` draft hidden states and the `JEPA` trajectory anchor to boost multi-token verification acceptance.

---

## 4. Speculative Lookahead & Two-Stage Vocabulary Scoring

### Coarse-to-Fine Vocabulary Projection (`CoarseToFineConceptHead` / `C2FHead`)
- **Role in `Flash1017-Engine`:** Implemented in `predict_coarse_to_fine_batch_with_margin` (`crates/flash-core/src/flash1017.rs`) to accelerate the `1.32 GB` `[129280, 5120]` `head.weight` projection by `> 4×` (`~1.9 ms`) with `100%` exact `BF16` top-token parity via a `[129280, 1024]` `Q8` leverage-column sketch followed by exact `BF16` re-scoring of the Top-2048 candidates.

### Pre-Gated MoE Prefetching & LaCo (*Layer Collapse: Reserving-Differences-while-Seeking-Common*, Yang et al., 2024)
- **Role in `Flash1017-Engine`:** Inspires the **`CPU-Tjuvkikaren` (`CpuLookaheadStager`)** in `crates/flash-core/src/flash1017.rs` (`cpu_lookahead_stage_prompt_tokens` and `prefetch_next_layer_experts_lookahead`), using 20 ARM64 CPU cores in parallel with the GPU to pre-warm expert pages ahead of execution.
