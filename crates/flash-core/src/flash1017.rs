//! # `flash1017` — DeepSeek V4.1 Flash-1017 NVFP4 Shard & Expert Inspector
//!
//! Provides zero-copy memory-mapped structural inspection, expert verification,
//! and Frobenius energy calculation for the 40-layer CED DeepSeek Flash-1017 NVFP4 model.
//!
//! Implements a resilient zero-copy `.safetensors` header parser that natively supports
//! modern Blackwell data types including `F8_E8M0`, `F8_E4M3`, and `NVFP4`.

use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Standard FP4 E2M1 decompression lookup table (IEEE / NVIDIA FP4 format).
/// Index: 4-bit nibble (0..15). Value: f32 representation.
pub const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Architecture parameters for DeepSeek V4.1 Flash-1017 NVFP4.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flash1017Config {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
}

impl Default for Flash1017Config {
    fn default() -> Self {
        Self {
            num_layers: 40,
            hidden_size: 5120,
            num_experts: 384,
            top_k: 6,
            intermediate_size: 2304,
        }
    }
}

impl Flash1017Config {
    /// Returns true if the layer is a protected anchor layer that must NEVER be pruned.
    /// Layer 0: Causal Encoder input anchor.
    /// Layer 1: Engram n-gram vocabulary lookup table 1 (98.3B params).
    /// Layer 14: Engram semantic entity lookup table 2 (98.3B params).
    /// Layer 20: Causal Encoder -> Decoder bridge.
    /// Layer 39: Final output decoder anchor layer.
    #[inline]
    pub fn is_layer_protected(layer_idx: usize) -> bool {
        matches!(layer_idx, 0 | 1 | 14 | 20 | 39)
    }

    /// Returns true if this layer is one of the two official Engram lookup tables.
    #[inline]
    pub fn is_engram_layer(layer_idx: usize) -> bool {
        matches!(layer_idx, 1 | 14)
    }
}

/// Structural summary of an inspected Flash-1017 shard file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flash1017LayerSummary {
    pub shard_filename: String,
    pub layer_idx: usize,
    pub is_protected: bool,
    pub has_engram: bool,
    pub detected_experts: usize,
    pub total_moe_bytes: u64,
    pub has_attention: bool,
    pub has_attn_sink: bool,
    pub intermediate_size: usize,
    pub hidden_size: usize,
}

/// Tensor entry metadata parsed from `.safetensors` header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardTensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Zero-copy view into a single MoE expert's NVFP4 weight tensors.
pub struct Flash1017ExpertView<'a> {
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub w1_data: &'a [u8],
    pub w2_data: &'a [u8],
    pub w3_data: &'a [u8],
    pub w1_shape: Vec<usize>,
    pub w2_shape: Vec<usize>,
    pub w3_shape: Vec<usize>,
    pub total_bytes: usize,
}

/// Native zero-copy inspector for Flash-1017 `.safetensors` shards.
/// Tolerates modern Blackwell `F8_E8M0` and `F8_E4M3` metadata without crashing.
pub struct Flash1017ShardInspector {
    mmap: Mmap,
    pub payload_offset: usize,
    pub tensors: HashMap<String, ShardTensorMeta>,
    pub shard_filename: String,
}

impl Flash1017ShardInspector {
    /// Loads and memory-maps a Flash-1017 `.safetensors` shard without heap allocations for weights.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let filename = path_ref
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown.safetensors".to_string());

        let file = File::open(path_ref).map_err(|e| format!("Failed to open file: {}", e))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| format!("Failed to mmap file: {}", e))?;

        if mmap.len() < 8 {
            return Err("File too small to be a valid safetensors archive".to_string());
        }

        // Read 8-byte little-endian header length
        let header_len_bytes: [u8; 8] = mmap[0..8]
            .try_into()
            .map_err(|_| "Failed to read header length".to_string())?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        if mmap.len() < 8 + header_len {
            return Err("File truncated: shorter than specified header length".to_string());
        }

        let header_str = std::str::from_utf8(&mmap[8..8 + header_len])
            .map_err(|e| format!("Invalid UTF-8 in safetensors header: {}", e))?;

        let raw_map: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| format!("Failed to parse safetensors JSON header: {}", e))?;

        let mut tensors = HashMap::new();
        for (k, v) in raw_map {
            if k == "__metadata__" {
                continue;
            }
            let dtype = v.get("dtype")
                .and_then(|d| d.as_str())
                .unwrap_or("UNKNOWN")
                .to_string();

            let shape = v.get("shape")
                .and_then(|s| s.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
                .unwrap_or_default();

            let offsets = v.get("data_offsets")
                .and_then(|o| o.as_array())
                .and_then(|arr| {
                    if arr.len() == 2 {
                        let start = arr[0].as_u64()? as usize;
                        let end = arr[1].as_u64()? as usize;
                        Some((start, end))
                    } else {
                        None
                    }
                })
                .unwrap_or((0, 0));

            tensors.insert(k, ShardTensorMeta {
                dtype,
                shape,
                data_offsets: offsets,
            });
        }

        let payload_offset = 8 + header_len;

        let is_engram_shard = filename.contains("00047-of-00048")
            || filename.contains("00048-of-00048")
            || tensors.keys().any(|k| k.contains(".engram.embed."));
        if !is_engram_shard && mmap.len() >= 65536 {
            crate::hardware_addon::register_host_mmap_bounds(mmap.as_ptr(), mmap.len());
        }

        // Linux NVLink-C2C / Unified Memory optimization (Fas F1.2):
        // - Engram lookup shards (shard 47 & shard 48, 97.3 GiB) get MADV_RANDOM + MADV_NOHUGEPAGE so
        //   O(1) n-gram lookups read < 4 KB directly without promoting 2 MB pages or evicting MoE weights.
        // - Active compute shards (MoE, MLA Attention, head.weight, mtp.0) >= 2 MB get MADV_HUGEPAGE
        //   (512x fewer NVLink-C2C GMMU/SMMU ATS page translations!).
        #[cfg(target_os = "linux")]
        unsafe {
            const MADV_RANDOM: i32 = 1;
            const MADV_HUGEPAGE: i32 = 14;
            const MADV_NOHUGEPAGE: i32 = 15;
            extern "C" {
                fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            if is_engram_shard {
                let _ = madvise(mmap.as_ptr() as *mut std::ffi::c_void, mmap.len(), MADV_RANDOM);
                let _ = madvise(
                    mmap.as_ptr() as *mut std::ffi::c_void,
                    mmap.len(),
                    MADV_NOHUGEPAGE,
                );
            } else if mmap.len() >= 2 * 1024 * 1024 {
                let _ = madvise(
                    mmap.as_ptr() as *mut std::ffi::c_void,
                    mmap.len(),
                    MADV_HUGEPAGE,
                );
            }
        }

        Ok(Self {
            mmap,
            payload_offset,
            tensors,
            shard_filename: filename,
        })
    }

    /// Gets a borrowed slice directly from the memory-mapped file for a given tensor.
    pub fn get_tensor_data(&self, name: &str) -> Result<(&ShardTensorMeta, &[u8]), String> {
        let meta = self.tensors.get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in shard '{}'", name, self.shard_filename))?;

        let abs_start = self.payload_offset + meta.data_offsets.0;
        let abs_end = self.payload_offset + meta.data_offsets.1;

        if abs_end > self.mmap.len() || abs_start > abs_end {
            return Err(format!("Data offsets out of bounds for tensor '{}'", name));
        }

        Ok((meta, &self.mmap[abs_start..abs_end]))
    }

    /// Scans the shard and summarizes its layer and expert composition.
    pub fn summarize(&self) -> Result<Flash1017LayerSummary, String> {
        // 1. Detect the layer index from tensor names (e.g. "layers.5.attn.q_proj.weight")
        let mut detected_layer: Option<usize> = None;
        let mut has_attention = false;
        let mut has_attn_sink = false;

        for name in self.tensors.keys() {
            if name.starts_with("layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 2 {
                    if let Ok(l) = parts[1].parse::<usize>() {
                        detected_layer = Some(l);
                    }
                }
                if name.contains(".attn.") {
                    has_attention = true;
                }
                if name.contains("attn_sink") {
                    has_attn_sink = true;
                }
            }
        }

        let layer_idx = detected_layer.ok_or_else(|| {
            format!(
                "No transformer layer found in shard '{}'. (Could be embeddings or vision tower)",
                self.shard_filename
            )
        })?;

        // 2. Count detected experts and calculate MoE byte size
        let mut expert_indices = std::collections::BTreeSet::new();
        let mut total_moe_bytes: u64 = 0;

        let expert_prefix = format!("layers.{}.ffn.experts.", layer_idx);
        for (name, meta) in &self.tensors {
            if name.starts_with(&expert_prefix) {
                let remainder = &name[expert_prefix.len()..];
                if let Some(dot_pos) = remainder.find('.') {
                    if let Ok(exp_idx) = remainder[..dot_pos].parse::<usize>() {
                        expert_indices.insert(exp_idx);
                    }
                }
                let len = meta.data_offsets.1.saturating_sub(meta.data_offsets.0) as u64;
                total_moe_bytes += len;
            }
        }

        let is_protected = Flash1017Config::is_layer_protected(layer_idx);
        let has_engram = Flash1017Config::is_engram_layer(layer_idx);

        Ok(Flash1017LayerSummary {
            shard_filename: self.shard_filename.clone(),
            layer_idx,
            is_protected,
            has_engram,
            detected_experts: expert_indices.len(),
            total_moe_bytes,
            has_attention,
            has_attn_sink,
            intermediate_size: 2304,
            hidden_size: 5120,
        })
    }

    /// Returns a zero-copy borrowed view into the weights of a specific expert in this shard.
    pub fn get_expert_view(&self, layer_idx: usize, expert_idx: usize) -> Result<Flash1017ExpertView<'_>, String> {
        let w1_name = format!("layers.{}.ffn.experts.{}.w1.weight", layer_idx, expert_idx);
        let w2_name = format!("layers.{}.ffn.experts.{}.w2.weight", layer_idx, expert_idx);
        let w3_name = format!("layers.{}.ffn.experts.{}.w3.weight", layer_idx, expert_idx);

        let (meta1, slice1) = self.get_tensor_data(&w1_name)?;
        let (meta2, slice2) = self.get_tensor_data(&w2_name)?;
        let (meta3, slice3) = self.get_tensor_data(&w3_name)?;

        let total_bytes = slice1.len() + slice2.len() + slice3.len();

        Ok(Flash1017ExpertView {
            layer_idx,
            expert_idx,
            w1_data: slice1,
            w2_data: slice2,
            w3_data: slice3,
            w1_shape: meta1.shape.clone(),
            w2_shape: meta2.shape.clone(),
            w3_shape: meta3.shape.clone(),
            total_bytes,
        })
    }

    /// Computes the Frobenius energy norm of an expert by unpacking NVFP4 nibbles.
    /// This runs in < 2 ms and measures the structural weight mass of the expert.
    pub fn compute_expert_frobenius_energy(&self, layer_idx: usize, expert_idx: usize) -> Result<f32, String> {
        let view = self.get_expert_view(layer_idx, expert_idx)?;

        let mut sum_sq: f64 = 0.0;

        for slice in [view.w1_data, view.w2_data, view.w3_data] {
            for &byte in slice {
                let low_nibble = (byte & 0x0F) as usize;
                let high_nibble = ((byte >> 4) & 0x0F) as usize;

                let v_low = FP4_E2M1_LUT[low_nibble] as f64;
                let v_high = FP4_E2M1_LUT[high_nibble] as f64;

                sum_sq += v_low * v_low + v_high * v_high;
            }
        }

        Ok((sum_sq.sqrt()) as f32)
    }

    /// Scans all detected experts in parallel using Rayon and returns a ranked energy report.
    pub fn scan_all_experts_energy(&self, layer_idx: usize, num_experts: usize) -> Result<LayerEnergyReport, String> {
        use rayon::prelude::*;

        let mut profiles: Vec<ExpertEnergyProfile> = (0..num_experts)
            .into_par_iter()
            .map(|exp_idx| {
                let energy = self.compute_expert_frobenius_energy(layer_idx, exp_idx).unwrap_or(0.0);
                ExpertEnergyProfile {
                    expert_idx: exp_idx,
                    frobenius_energy: energy,
                }
            })
            .collect();

        // Sort descending: highest energy first
        profiles.sort_by(|a, b| b.frobenius_energy.partial_cmp(&a.frobenius_energy).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f32 = profiles.iter().map(|p| p.frobenius_energy).sum();
        let mean = if !profiles.is_empty() { sum / profiles.len() as f32 } else { 0.0 };
        let min = profiles.last().map(|p| p.frobenius_energy).unwrap_or(0.0);
        let max = profiles.first().map(|p| p.frobenius_energy).unwrap_or(0.0);

        Ok(LayerEnergyReport {
            layer_idx,
            total_experts: profiles.len(),
            mean_energy: mean,
            min_energy: min,
            max_energy: max,
            energy_spread: max - min,
            sorted_profiles: profiles,
        })
    }

    /// Executes zero-copy `NVFP4` (`group_size = 16`, `U8` E2M1 nibbles + `F8_E4M3` `weight_scale` + `F32` `weight_scale_2`)
    /// matrix-vector multiplication for `<prefix>.weight` (e.g. `layers.5.ffn.experts.0.w1`).
    /// Automatically dispatches to `HardwareAddonVTable::v3_matvec_nvfp4` on Blackwell GPU / Grace CPU when loaded.
    pub fn matvec_nvfp4(&self, prefix: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        use rayon::prelude::*;
        let (w_meta, w_bytes) = self.get_tensor_data(&format!("{}.weight", prefix))?;
        let (_sc_meta, sc_bytes) = self.get_tensor_data(&format!("{}.weight_scale", prefix))?;
        let (_sc2_meta, sc2_bytes) = self.get_tensor_data(&format!("{}.weight_scale_2", prefix))?;
        if w_meta.shape.len() != 2 || sc2_bytes.len() < 4 {
            return Err(format!("Invalid NVFP4 tensor layout for '{}'", prefix));
        }
        let rows = w_meta.shape[0];
        let packed_cols = w_meta.shape[1];
        let cols = packed_cols * 2;
        if x.len() != cols {
            return Err(format!(
                "NVFP4 input length mismatch on '{}': expected {}, got {}",
                prefix,
                cols,
                x.len()
            ));
        }
        let global_scale = f32::from_le_bytes([sc2_bytes[0], sc2_bytes[1], sc2_bytes[2], sc2_bytes[3]]);

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mv) = vt.v3_matvec_nvfp4() {
                let mut out = vec![0.0_f32; rows];
                unsafe {
                    mv(
                        rows,
                        cols,
                        w_bytes.as_ptr(),
                        sc_bytes.as_ptr(),
                        global_scale,
                        x.as_ptr(),
                        out.as_mut_ptr(),
                    );
                }
                return Ok(out);
            }
        }

        let groups_per_row = cols / 16;
        let out: Vec<f32> = w_bytes
            .par_chunks_exact(packed_cols)
            .zip(sc_bytes.par_chunks_exact(groups_per_row))
            .with_min_len(32)
            .map(|(w_row, sc_row)| {
                let mut row_acc = 0.0_f32;
                for g in 0..groups_per_row {
                    let g_sc = f8_e4m3_to_f32(sc_row[g]);
                    let b_base = g * 8;
                    let x_base = g * 16;
                    let mut g_sum = 0.0_f32;
                    for k in 0..8 {
                        let b = w_row[b_base + k];
                        let v0 = FP4_E2M1_LUT[(b & 0x0F) as usize];
                        let v1 = FP4_E2M1_LUT[(b >> 4) as usize];
                        g_sum += v0 * x[x_base + k * 2] + v1 * x[x_base + k * 2 + 1];
                    }
                    row_acc += g_sum * g_sc;
                }
                row_acc * global_scale
            })
            .collect();
        Ok(out)
    }

    /// Executes batched zero-copy `NVFP4` GEMM (`[seq_len, cols] -> [seq_len, rows]`) for `<prefix>.weight`.
    pub fn matbatch_nvfp4(
        &self,
        prefix: &str,
        x_batch_flat: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, String> {
        let (w_meta, w_bytes) = self.get_tensor_data(&format!("{}.weight", prefix))?;
        let (_sc_meta, sc_bytes) = self.get_tensor_data(&format!("{}.weight_scale", prefix))?;
        let (_sc2_meta, sc2_bytes) = self.get_tensor_data(&format!("{}.weight_scale_2", prefix))?;
        let rows = w_meta.shape[0];
        let cols = w_meta.shape[1] * 2;
        if x_batch_flat.len() != seq_len * cols {
            return Err(format!("NVFP4 batch input size mismatch on '{}'", prefix));
        }
        let global_scale = f32::from_le_bytes([sc2_bytes[0], sc2_bytes[1], sc2_bytes[2], sc2_bytes[3]]);

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mb) = vt.v3_matbatch_nvfp4() {
                let mut out = vec![0.0_f32; seq_len * rows];
                unsafe {
                    mb(
                        rows,
                        cols,
                        seq_len,
                        w_bytes.as_ptr(),
                        sc_bytes.as_ptr(),
                        global_scale,
                        x_batch_flat.as_ptr(),
                        out.as_mut_ptr(),
                    );
                }
                return Ok(out);
            }
        }

        let mut out = Vec::with_capacity(seq_len * rows);
        for t in 0..seq_len {
            let v = self.matvec_nvfp4(prefix, &x_batch_flat[t * cols..(t + 1) * cols])?;
            out.extend_from_slice(&v);
        }
        Ok(out)
    }

    /// Executes zero-copy `1.58-bit Ternary` (`<prefix>.packed_ternary` + `<prefix>.row_scales`) matrix-vector multiplication.
    pub fn matvec_ternary_158(&self, prefix: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        use rayon::prelude::*;
        let (p_meta, p_bytes) = self.get_tensor_data(&format!("{}.packed_ternary", prefix))?;
        let (_s_meta, s_bytes) = self.get_tensor_data(&format!("{}.row_scales", prefix))?;
        let rows = p_meta.shape[0];
        let packed_cols = p_meta.shape[1];
        let cols = packed_cols * 4;
        if x.len() != cols || s_bytes.len() != rows * 4 {
            return Err(format!("Ternary 1.58-bit dimension mismatch on '{}'", prefix));
        }

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mv) = vt.v3_matvec_ternary_158() {
                if (s_bytes.as_ptr() as usize) % 4 == 0 {
                    let mut out = vec![0.0_f32; rows];
                    unsafe {
                        mv(
                            rows,
                            cols,
                            p_bytes.as_ptr(),
                            s_bytes.as_ptr() as *const f32,
                            x.as_ptr(),
                            out.as_mut_ptr(),
                        );
                    }
                    return Ok(out);
                }
            }
        }

        const LUT: [f32; 4] = [-1.0, 0.0, 1.0, 0.0];
        let out: Vec<f32> = p_bytes
            .par_chunks_exact(packed_cols)
            .zip(s_bytes.par_chunks_exact(4))
            .with_min_len(32)
            .map(|(row_b, sc_b)| {
                let sc = f32::from_le_bytes([sc_b[0], sc_b[1], sc_b[2], sc_b[3]]);
                let mut acc = 0.0_f32;
                for (i, &b) in row_b.iter().enumerate() {
                    if b == 0x55 {
                        continue;
                    }
                    let base = i * 4;
                    acc += LUT[(b & 3) as usize] * x[base]
                        + LUT[((b >> 2) & 3) as usize] * x[base + 1]
                        + LUT[((b >> 4) & 3) as usize] * x[base + 2]
                        + LUT[((b >> 6) & 3) as usize] * x[base + 3];
                }
                acc * sc
            })
            .collect();
        Ok(out)
    }

    /// Executes batched zero-copy `1.58-bit Ternary` GEMM (`[seq_len, cols] -> [seq_len, rows]`) for `<prefix>`.
    pub fn matbatch_ternary_158(
        &self,
        prefix: &str,
        x_batch_flat: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, String> {
        let (p_meta, p_bytes) = self.get_tensor_data(&format!("{}.packed_ternary", prefix))?;
        let (_s_meta, s_bytes) = self.get_tensor_data(&format!("{}.row_scales", prefix))?;
        let rows = p_meta.shape[0];
        let cols = p_meta.shape[1] * 4;
        if x_batch_flat.len() != seq_len * cols || s_bytes.len() != rows * 4 {
            return Err(format!("Ternary 1.58-bit batch mismatch on '{}'", prefix));
        }

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mb) = vt.v3_matbatch_ternary_158() {
                if (s_bytes.as_ptr() as usize) % 4 == 0 {
                    let mut out = vec![0.0_f32; seq_len * rows];
                    unsafe {
                        mb(
                            rows,
                            cols,
                            seq_len,
                            p_bytes.as_ptr(),
                            s_bytes.as_ptr() as *const f32,
                            x_batch_flat.as_ptr(),
                            out.as_mut_ptr(),
                        );
                    }
                    return Ok(out);
                }
            }
        }

        let mut out = Vec::with_capacity(seq_len * rows);
        for t in 0..seq_len {
            let v = self.matvec_ternary_158(prefix, &x_batch_flat[t * cols..(t + 1) * cols])?;
            out.extend_from_slice(&v);
        }
        Ok(out)
    }

    /// Executes zero-copy `MXFP8` (`<prefix>.weight` in `F8_E4M3` + `<prefix>.scale` in `F8_E8M0` `32x32` blocks)
    /// matrix-vector multiplication (used by Flash-1017 MLA Attention, Shared Experts, and Engram `wkv`).
    pub fn matvec_mxfp8(&self, prefix: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        use rayon::prelude::*;
        let (w_meta, w_bytes) = self.get_tensor_data(&format!("{}.weight", prefix))?;
        let (_s_meta, s_bytes) = self.get_tensor_data(&format!("{}.scale", prefix))?;
        let rows = w_meta.shape[0];
        let cols = w_meta.shape[1];
        if x.len() != cols {
            return Err(format!("MXFP8 input mismatch on '{}'", prefix));
        }

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mv) = vt.v3_matvec_mxfp8() {
                let mut out = vec![0.0_f32; rows];
                unsafe {
                    mv(
                        rows,
                        cols,
                        w_bytes.as_ptr(),
                        s_bytes.as_ptr(),
                        x.as_ptr(),
                        out.as_mut_ptr(),
                    );
                }
                return Ok(out);
            }
        }

        let scale_cols = (cols + 31) / 32;
        let out: Vec<f32> = (0..rows)
            .into_par_iter()
            .with_min_len(32)
            .map(|r| {
                let w_row = &w_bytes[r * cols..(r + 1) * cols];
                let sc_row_off = (r / 32) * scale_cols;
                let mut acc = 0.0_f32;
                for bc in 0..scale_cols {
                    let sc = f8_e8m0_to_f32(s_bytes[sc_row_off + bc]);
                    let c_start = bc * 32;
                    let c_end = (c_start + 32).min(cols);
                    let mut b_sum = 0.0_f32;
                    for c in c_start..c_end {
                        b_sum += f8_e4m3_to_f32(w_row[c]) * x[c];
                    }
                    acc += b_sum * sc;
                }
                acc
            })
            .collect();
        Ok(out)
    }

    /// Executes batched zero-copy `MXFP8` GEMM (`[seq_len, cols] -> [seq_len, rows]`) for `<prefix>`.
    pub fn matbatch_mxfp8(
        &self,
        prefix: &str,
        x_batch_flat: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, String> {
        let (w_meta, w_bytes) = self.get_tensor_data(&format!("{}.weight", prefix))?;
        let (_s_meta, s_bytes) = self.get_tensor_data(&format!("{}.scale", prefix))?;
        let rows = w_meta.shape[0];
        let cols = w_meta.shape[1];
        if x_batch_flat.len() != seq_len * cols {
            return Err(format!("MXFP8 batch mismatch on '{}'", prefix));
        }

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mb) = vt.v3_matbatch_mxfp8() {
                let mut out = vec![0.0_f32; seq_len * rows];
                unsafe {
                    mb(
                        rows,
                        cols,
                        seq_len,
                        w_bytes.as_ptr(),
                        s_bytes.as_ptr(),
                        x_batch_flat.as_ptr(),
                        out.as_mut_ptr(),
                    );
                }
                return Ok(out);
            }
        }

        let mut out = Vec::with_capacity(seq_len * rows);
        for t in 0..seq_len {
            let v = self.matvec_mxfp8(prefix, &x_batch_flat[t * cols..(t + 1) * cols])?;
            out.extend_from_slice(&v);
        }
        Ok(out)
    }

    /// Executes grouped `MXFP8` matrix-vector multiplication (`o_groups = 8` on `layers.{l}.attn.wo_a`,
    /// where `weight` is `[num_groups * out_per_group, in_per_group]` and `x` is `[num_groups * in_per_group]`).
    pub fn matvec_mxfp8_grouped(
        &self,
        prefix: &str,
        x: &[f32],
        num_groups: usize,
    ) -> Result<Vec<f32>, String> {
        let (w_meta, w_bytes) = self.get_tensor_data(&format!("{}.weight", prefix))?;
        let (_s_meta, s_bytes) = self.get_tensor_data(&format!("{}.scale", prefix))?;
        if w_meta.shape.len() != 2 || num_groups == 0 {
            return Err(format!("Invalid grouped MXFP8 tensor shape for '{}'", prefix));
        }
        let total_rows = w_meta.shape[0];
        let cols_per_group = w_meta.shape[1];
        if total_rows % num_groups != 0 || x.len() != num_groups * cols_per_group {
            return Err(format!(
                "Grouped MXFP8 mismatch on '{}': shape={:?}, groups={}, x.len()={}",
                prefix,
                w_meta.shape,
                num_groups,
                x.len()
            ));
        }
        let rows_per_group = total_rows / num_groups;
        let scale_rows_per_group = rows_per_group / 32;
        let scale_cols = (cols_per_group + 31) / 32;
        let w_group_bytes = rows_per_group * cols_per_group;
        let s_group_bytes = scale_rows_per_group * scale_cols;

        let mut out = vec![0.0_f32; total_rows];
        if let Some(gmv) = crate::hardware_addon::active_mxfp8_grouped_matvec() {
            let ok = unsafe {
                gmv(
                    total_rows,
                    cols_per_group,
                    num_groups,
                    w_bytes.as_ptr(),
                    s_bytes.as_ptr(),
                    x.as_ptr(),
                    out.as_mut_ptr(),
                )
            };
            if ok {
                return Ok(out);
            }
        }
        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mv) = vt.v3_matvec_mxfp8() {
                for g in 0..num_groups {
                    let w_ptr = unsafe { w_bytes.as_ptr().add(g * w_group_bytes) };
                    let s_ptr = unsafe { s_bytes.as_ptr().add(g * s_group_bytes) };
                    let x_ptr = unsafe { x.as_ptr().add(g * cols_per_group) };
                    let out_ptr = unsafe { out.as_mut_ptr().add(g * rows_per_group) };
                    unsafe {
                        mv(rows_per_group, cols_per_group, w_ptr, s_ptr, x_ptr, out_ptr);
                    }
                }
                return Ok(out);
            }
        }

        for g in 0..num_groups {
            let x_g = &x[g * cols_per_group..(g + 1) * cols_per_group];
            for r in 0..rows_per_group {
                let global_r = g * rows_per_group + r;
                let w_row = &w_bytes[global_r * cols_per_group..(global_r + 1) * cols_per_group];
                let sc_row_off = (global_r / 32) * scale_cols;
                let mut acc = 0.0_f32;
                for bc in 0..scale_cols {
                    let sc = f8_e8m0_to_f32(s_bytes[sc_row_off + bc]);
                    let c_start = bc * 32;
                    let c_end = (c_start + 32).min(cols_per_group);
                    let mut b_sum = 0.0_f32;
                    for c in c_start..c_end {
                        b_sum += f8_e4m3_to_f32(w_row[c]) * x_g[c];
                    }
                    acc += b_sum * sc;
                }
                out[global_r] = acc;
            }
        }
        Ok(out)
    }

    /// Executes zero-copy `BF16` matrix-vector multiplication for `tensor_name` (e.g. `head.weight` or `layers.5.ffn.gate.weight`).
    /// Dispatches to `HardwareAddonVTable::v2_matvec_bf16` on Blackwell GPU when loaded.
    pub fn matvec_bf16(&self, tensor_name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        use rayon::prelude::*;
        let (meta, raw_bytes) = self.get_tensor_data(tensor_name)?;
        if meta.shape.len() != 2 {
            return Err(format!("BF16 tensor '{}' is not 2D: {:?}", tensor_name, meta.shape));
        }
        let rows = meta.shape[0];
        let cols = meta.shape[1];
        if x.len() != cols || raw_bytes.len() < rows * cols * 2 {
            return Err(format!(
                "BF16 matvec dimension mismatch on '{}': expected {}, got {}",
                tensor_name,
                cols,
                x.len()
            ));
        }

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mv_bf16) = vt.v2_matvec_bf16() {
                if (raw_bytes.as_ptr() as usize) % 2 == 0 {
                    let mut out = vec![0.0_f32; rows];
                    unsafe {
                        mv_bf16(
                            rows,
                            cols,
                            raw_bytes.as_ptr() as *const u16,
                            x.as_ptr(),
                            out.as_mut_ptr(),
                        );
                    }
                    return Ok(out);
                }
            }
        }

        let row_bytes = cols * 2;
        let out: Vec<f32> = raw_bytes
            .par_chunks_exact(row_bytes)
            .with_min_len(64)
            .map(|r_slice| {
                let mut sum = 0.0_f32;
                for c in 0..cols {
                    let bits = u16::from_le_bytes([r_slice[c * 2], r_slice[c * 2 + 1]]);
                    let w = f32::from_bits((bits as u32) << 16);
                    sum += w * x[c];
                }
                sum
            })
            .collect();
        Ok(out)
    }

    /// `F4.2` & `F7.C2`: Evaluates the exact `BF16` dot product for a sparse subset of rows `row_indices`,
    /// returning `(best_row_id, best_logit, second_best_logit)` so `Adaptive MTP` obtains the exact top-1 vs top-2 margin in `0.0 ms` extra time.
    pub fn score_bf16_rows_top2(
        &self,
        tensor_name: &str,
        x: &[f32],
        row_indices: &[u32],
    ) -> Result<(u32, f32, f32), String> {
        use rayon::prelude::*;
        let (meta, raw_bytes) = self.get_tensor_data(tensor_name)?;
        if meta.shape.len() != 2 {
            return Err(format!("BF16 tensor '{}' is not 2D", tensor_name));
        }
        let rows = meta.shape[0];
        let cols = meta.shape[1];
        if x.len() != cols || raw_bytes.len() < rows * cols * 2 {
            return Err(format!("BF16 row-score mismatch on '{}'", tensor_name));
        }
        let row_bytes = cols * 2;
        let is_u16_aligned = (raw_bytes.as_ptr() as usize) % 2 == 0;
        let raw_ptr_usize = raw_bytes.as_ptr() as usize;
        let (best_id, best_val, second_val) = row_indices
            .par_iter()
            .with_min_len(32)
            .map(|&r_u32| {
                let r = (r_u32 as usize).min(rows - 1);
                let mut sum = 0.0_f32;
                if is_u16_aligned {
                    let u16_row = unsafe {
                        std::slice::from_raw_parts(
                            (raw_ptr_usize + r * row_bytes) as *const u16,
                            cols,
                        )
                    };
                    let chunks8 = cols / 8;
                    let mut a0 = 0.0_f32;
                    let mut a1 = 0.0_f32;
                    let mut a2 = 0.0_f32;
                    let mut a3 = 0.0_f32;
                    let mut a4 = 0.0_f32;
                    let mut a5 = 0.0_f32;
                    let mut a6 = 0.0_f32;
                    let mut a7 = 0.0_f32;
                    for c8 in 0..chunks8 {
                        let base = c8 * 8;
                        a0 += f32::from_bits((u16_row[base] as u32) << 16) * x[base];
                        a1 += f32::from_bits((u16_row[base + 1] as u32) << 16) * x[base + 1];
                        a2 += f32::from_bits((u16_row[base + 2] as u32) << 16) * x[base + 2];
                        a3 += f32::from_bits((u16_row[base + 3] as u32) << 16) * x[base + 3];
                        a4 += f32::from_bits((u16_row[base + 4] as u32) << 16) * x[base + 4];
                        a5 += f32::from_bits((u16_row[base + 5] as u32) << 16) * x[base + 5];
                        a6 += f32::from_bits((u16_row[base + 6] as u32) << 16) * x[base + 6];
                        a7 += f32::from_bits((u16_row[base + 7] as u32) << 16) * x[base + 7];
                    }
                    sum = (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
                    for c in (chunks8 * 8)..cols {
                        sum += f32::from_bits((u16_row[c] as u32) << 16) * x[c];
                    }
                } else {
                    let r_slice = &raw_bytes[r * row_bytes..(r + 1) * row_bytes];
                    for c in 0..cols {
                        let bits = u16::from_le_bytes([r_slice[c * 2], r_slice[c * 2 + 1]]);
                        let w = f32::from_bits((bits as u32) << 16);
                        sum += w * x[c];
                    }
                }
                (r_u32, sum, f32::NEG_INFINITY)
            })
            .reduce(
                || (0u32, f32::NEG_INFINITY, f32::NEG_INFINITY),
                |a, b| {
                    if b.1 > a.1 {
                        (b.0, b.1, a.1.max(b.2))
                    } else {
                        (a.0, a.1, b.1.max(a.2))
                    }
                },
            );
        Ok((best_id, best_val, second_val))
    }

    /// `F4.2`: Evaluates the exact `BF16` dot product for a sparse subset of rows `row_indices` (e.g. Top-2048 candidates in `head.weight`),
    /// reading only `row_indices.len() * cols * 2` bytes (`20.97 MB` for 2048 rows instead of `1 324 MB` for all 129 280 rows)
    /// and returning `(best_row_id, best_logit)`.
    pub fn score_bf16_rows(
        &self,
        tensor_name: &str,
        x: &[f32],
        row_indices: &[u32],
    ) -> Result<(u32, f32), String> {
        let (best_id, best_val, _) = self.score_bf16_rows_top2(tensor_name, x, row_indices)?;
        Ok((best_id, best_val))
    }

    /// `F4.3`: Executes zero-copy `MXFP4` (`<prefix>.weight` packed `FP4_E2M1` nibbles + `<prefix>.scale` `F8_E8M0` block-32 scales)
    /// matrix-vector multiplication (used by `mtp.0.ffn.experts.{0..127}` in `model-00044-of-00048.safetensors`).
    pub fn matvec_mxfp4(&self, prefix: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        use rayon::prelude::*;
        let (w_meta, w_bytes) = self.get_tensor_data(&format!("{}.weight", prefix))?;
        let (_s_meta, s_bytes) = self.get_tensor_data(&format!("{}.scale", prefix))?;
        let rows = w_meta.shape[0];
        let packed_cols = w_meta.shape[1];
        let cols = packed_cols * 2;
        let scale_cols = cols / 32;
        if x.len() != cols || s_bytes.len() < rows * scale_cols {
            return Err(format!(
                "MXFP4 dimension mismatch on '{}': expected cols={}, got x.len()={}",
                prefix,
                cols,
                x.len()
            ));
        }

        if let Some(mv_mxfp4) = crate::hardware_addon::active_mxfp4_matvec() {
            let mut out = vec![0.0_f32; rows];
            let ok = unsafe {
                mv_mxfp4(
                    rows,
                    cols,
                    w_bytes.as_ptr(),
                    s_bytes.as_ptr(),
                    x.as_ptr(),
                    out.as_mut_ptr(),
                )
            };
            if ok {
                return Ok(out);
            }
        }

        let out: Vec<f32> = w_bytes
            .par_chunks_exact(packed_cols)
            .zip(s_bytes.par_chunks_exact(scale_cols))
            .with_min_len(32)
            .map(|(w_row, s_row)| {
                let mut acc = 0.0_f32;
                for g in 0..scale_cols {
                    let sc = f8_e8m0_to_f32(s_row[g]);
                    let b_off = g * 16;
                    let x_off = g * 32;
                    let mut b_sum = 0.0_f32;
                    for k in 0..16 {
                        let b = w_row[b_off + k];
                        let w0 = FP4_E2M1_LUT[(b & 0x0F) as usize];
                        let w1 = FP4_E2M1_LUT[(b >> 4) as usize];
                        b_sum += w0 * x[x_off + k * 2] + w1 * x[x_off + k * 2 + 1];
                    }
                    acc += b_sum * sc;
                }
                acc
            })
            .collect();
        Ok(out)
    }

    /// Extracts a 1D or 2D `BF16` tensor as `Vec<f32>`.
    pub fn extract_bf16_vec(&self, tensor_name: &str) -> Result<Vec<f32>, String> {
        let (_meta, raw) = self.get_tensor_data(tensor_name)?;
        let count = raw.len() / 2;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
            out.push(f32::from_bits((bits as u32) << 16));
        }
        Ok(out)
    }

    /// Extracts a 1D or 2D `F32` tensor as `Vec<f32>`.
    pub fn extract_f32_vec(&self, tensor_name: &str) -> Result<Vec<f32>, String> {
        let (_meta, raw) = self.get_tensor_data(tensor_name)?;
        let count = raw.len() / 4;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let b = i * 4;
            out.push(f32::from_le_bytes([raw[b], raw[b + 1], raw[b + 2], raw[b + 3]]));
        }
        Ok(out)
    }

    /// Extracts a 1D `I64` index tensor (such as `merkaba.tier_a_indices`) as `Vec<usize>`.
    pub fn extract_i64_usize_vec(&self, tensor_name: &str) -> Result<Vec<usize>, String> {
        let (_meta, raw) = self.get_tensor_data(tensor_name)?;
        let count = raw.len() / 8;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let b = i * 8;
            let v = i64::from_le_bytes([
                raw[b],
                raw[b + 1],
                raw[b + 2],
                raw[b + 3],
                raw[b + 4],
                raw[b + 5],
                raw[b + 6],
                raw[b + 7],
            ]);
            if v >= 0 {
                out.push(v as usize);
            }
        }
        Ok(out)
    }
}

/// Bit-exact OCP `F8_E4M3` (`1` sign, `4` exp bias 7, `3` mantissa) to IEEE-754 `f32` conversion.
#[inline(always)]
pub fn f8_e4m3_to_f32(b: u8) -> f32 {
    let sign = ((b as u32) & 0x80) << 24;
    let em = (b as u32) & 0x7F;
    if em == 0x7F {
        return f32::NAN;
    }
    if em < 8 {
        let sub = (em as f32) * (1.0_f32 / 512.0_f32);
        f32::from_bits(sign | sub.to_bits())
    } else {
        f32::from_bits(sign | ((em + 960) << 20))
    }
}

/// Bit-exact OCP `F8_E8M0` (`2^(b - 127)`) block scale to IEEE-754 `f32` conversion.
#[inline(always)]
pub fn f8_e8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        f32::NAN
    } else if b == 0 {
        f32::from_bits(0x0040_0000)
    } else {
        f32::from_bits((b as u32) << 23)
    }
}

/// Profile for an individual expert's weight energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertEnergyProfile {
    pub expert_idx: usize,
    pub frobenius_energy: f32,
}

/// Complete layer energy distribution and ranking across all experts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerEnergyReport {
    pub layer_idx: usize,
    pub total_experts: usize,
    pub mean_energy: f32,
    pub min_energy: f32,
    pub max_energy: f32,
    pub energy_spread: f32,
    pub sorted_profiles: Vec<ExpertEnergyProfile>,
}

/// The 32 canonical top-level knowledge domains in `databases/` (Hybrid Skeleton).
/// Format: `(domain_folder_name, display_title)`.
pub const CORE_DOMAINS: &[(&str, &str)] = &[
    // 1–20: Core Academic, Scientific & Engineering Domains
    ("fysik", "Physics / Klassisk & Modern Fysik (Mekanik, Termodynamik, Elektromagnetism & Kvantfysik)"),
    ("kemi", "Chemistry / Kemi (Periodiska systemet, Stökiometri, Organisk & Oorganisk Kemi)"),
    ("biologi", "Biology / Biologi & Livsvetenskap (Cellbiologi, Genetik, Ekologi & Fysiologi)"),
    ("matematik", "Mathematics / Matematik (Algebra, Analys, Geometri, Statistik & Diskret Matematik)"),
    ("medicin", "Medicine & Health / Medicin, Anatomi, Patologi & Klinisk Behandling"),
    ("teknik", "Engineering / Ingenjörskonst, Maskinteknik & Systemarkitektur"),
    ("programmering", "Programming & CS / Programmering, Algoritmer, Datastrukturer & Mjukvaruarkitektur"),
    ("elektronik", "Electronics / Elektronik, Halvledare, Kretsteori & Inbyggda System"),
    ("historia", "History / Världs- & Nordisk Historia (Epoker, Händelser, Personer & Kronologi)"),
    ("geografi", "Geography / Geografi, Länder, Städer, Demografi & Naturgeografi"),
    ("astronomi", "Astronomy & Astrophysics / Astronomi, Solsystemet, Stjärnor & Kosmologi"),
    ("geologi", "Geology & Earth Science / Geologi, Mineralogi & Geovetenskap"),
    ("ekonomi", "Economics & Finance / Nationalekonomi, Företagsekonomi & Finans"),
    ("juridik", "Law / Juridik, Rättsvetenskap, Avtalsrätt & Förvaltningsrätt"),
    ("psykologi", "Psychology & Neuroscience / Psykologi, Kognitionsvetenskap & Neurovetenskap"),
    ("filosofi", "Philosophy & Logic / Filosofi, Etik, Vetenskapsteori & Logik"),
    ("sprak", "Languages & Linguistics / Språkvetenskap, Grammatik, Etymologi & Översättning"),
    ("arkitektur", "Architecture & Civil Engineering / Arkitektur, Byggteknik & Stadsplanering"),
    ("materialvetenskap", "Materials Science / Materialvetenskap, Metallurgi, Polymerer & Keramer"),
    ("energi", "Energy Systems / Energiteknik, Kraftsystem, Kärnteknik & Förnybar Energi"),
    // 21–32: Practical, Applied, Cultural, Industrial & Specialized Domains
    ("farmakologi", "Pharmacology & Toxicology / Farmakologi, Läkemedelskemi, Dosering & Toxikologi"),
    ("lantbruk_och_skog", "Agriculture & Forestry / Jordbruk, Skogsbruk, Växtodling, Djurhållning & Permakultur"),
    ("mat_och_kokskonst", "Food Science & Culinary Arts / Livsmedelskemi, Näringslära, Recept & Fermentering"),
    ("musik_och_ljudteknik", "Music & Audio Engineering / Musikteori, Akustik, Instrument & Ljudproduktion/DSP"),
    ("konst_och_kultur", "Arts, Literature & Culture / Konsthistoria, Litteratur, Film, Fotografi & Design"),
    ("fordon_och_transport", "Vehicles & Transport / Bilmekanik, Flygteknik/Aerodynamik, Sjöfart & Järnväg"),
    ("cybersakerhet_och_natverk", "Cybersecurity & Networking / Kryptering, Nätverksprotokoll, Linux & Systemadministration"),
    ("miljo_och_klimat", "Environment & Climate / Meteorologi, Klimatvetenskap, Hydrologi & Hållbarhet"),
    ("hantverk_och_produktion", "Crafts & Manufacturing / Svetsning, CNC, Träbearbetning, VVS & Elinstallation"),
    ("idrott_och_fysiologi", "Sports & Exercise Physiology / Träningslära, Biomekanik, Ergonomi & Rehabilitering"),
    ("rymden_och_satelliter", "Aerospace & Spaceflight / Raketteknik, Satelliter, Omloppsmekanik & Rymdfart"),
    ("samhalle_och_statsvetenskap", "Political Science & Society / Statsvetenskap, Offentlig Förvaltning, Sociologi & Diplomati"),
];

/// Ensures that `<db_root>/<domain>/domain_profile.json` exists; creates a default profile if missing.
pub fn ensure_domain_profile(db_root: &Path, domain: &str) -> Result<std::path::PathBuf, String> {
    let dom_clean = domain.trim().to_lowercase();
    let dom_dir = db_root.join(&dom_clean);
    std::fs::create_dir_all(&dom_dir)
        .map_err(|e| format!("Failed to create domain dir {}: {}", dom_dir.display(), e))?;

    let profile_path = dom_dir.join("domain_profile.json");
    if !profile_path.exists() {
        let display_title = CORE_DOMAINS
            .iter()
            .find(|(d, _)| *d == dom_clean)
            .map(|(_, t)| (*t).to_string())
            .unwrap_or_else(|| format!("Knowledge Domain: {}", dom_clean));

        let profile_json = serde_json::json!({
            "domain_name": dom_clean,
            "display_title": display_title,
            "requires_external_app": false,
            "process_names": [],
            "app_missing_message": "",
            "mcp_tcp_addr": null,
            "mcp_port_closed_message": "",
            "mcp_server_name": null,
            "mcp_command": null,
            "mcp_args": [],
            "priority_search_paths": ["."],
            "practice_refs_dir": format!("/Volumes/NET/ai-projects/practice_refs/{}", dom_clean),
            "default_spec_name": format!("{}.spec", dom_clean),
            "vision_prompt_hint": format!("Analyze this {} diagram or image and extract key technical entities and formulas.", dom_clean)
        });

        let json_str = serde_json::to_string_pretty(&profile_json)
            .map_err(|e| format!("Failed to serialize domain_profile.json: {}", e))?;
        std::fs::write(&profile_path, json_str.as_bytes())
            .map_err(|e| format!("Failed to write {}: {}", profile_path.display(), e))?;
    }

    Ok(profile_path)
}

/// Initializes all 20 canonical top-level domain folders and their `domain_profile.json` files in `db_root`.
pub fn init_core_domain_skeleton(db_root: &Path) -> Result<usize, String> {
    let mut count = 0;
    for (dom, _) in CORE_DOMAINS {
        ensure_domain_profile(db_root, dom)?;
        count += 1;
    }
    Ok(count)
}

/// Primary languages: 1. English (`fact_en`), 2. Swedish (`fact_sv`), 3. Additional languages (`translations`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CrystallizedKnowledgeCard {
    pub id: String,
    pub title: String,
    pub domain: String,
    pub category: String,
    pub key_triggers: Vec<String>,
    /// Primary / fallback fact or `<tool_call>` recipe (defaults to `fact_en` if not explicitly set).
    #[serde(default)]
    pub fact_or_recipe: String,
    /// #1 Priority: Verified English fact statement.
    #[serde(default)]
    pub fact_en: String,
    /// #2 Priority: Verified Swedish fact statement.
    #[serde(default)]
    pub fact_sv: String,
    /// Optional translations map for additional languages (`"de"`, `"fr"`, `"es"`, `"no"`, `"da"`, `"fi"`, etc.).
    #[serde(default)]
    pub translations: HashMap<String, String>,
    #[serde(default)]
    pub components: Vec<String>,
    #[serde(default)]
    pub notes: String,
}

impl CrystallizedKnowledgeCard {
    /// Populates fallback fields so `fact_or_recipe`, `fact_en`, and `fact_sv` stay consistent.
    pub fn normalize_fields(&mut self) {
        if self.fact_or_recipe.trim().is_empty() {
            if !self.fact_en.trim().is_empty() {
                self.fact_or_recipe = self.fact_en.clone();
            } else if !self.fact_sv.trim().is_empty() {
                self.fact_or_recipe = self.fact_sv.clone();
            }
        }
        if self.fact_en.trim().is_empty() && !self.fact_or_recipe.trim().is_empty() {
            if let Some(en) = self.translations.get("en") {
                self.fact_en = en.clone();
            }
        }
        if self.fact_sv.trim().is_empty() {
            if let Some(sv) = self.translations.get("sv") {
                self.fact_sv = sv.clone();
            }
        }
    }

    /// Strictly validates that all mandatory fields are present, non-empty, and safe for filesystem paths.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("Knowledge card 'id' cannot be empty".to_string());
        }
        if !self
            .id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "Knowledge card 'id' ('{}') must only contain ASCII alphanumeric, '_', or '-'",
                self.id
            ));
        }
        if self.title.trim().is_empty() {
            return Err("Knowledge card 'title' cannot be empty".to_string());
        }
        if self.domain.trim().is_empty() || self.domain.contains("..") || self.domain.contains('/') {
            return Err(format!("Invalid 'domain': '{}'", self.domain));
        }
        if self.category.trim().is_empty() || self.category.contains("..") {
            return Err(format!("Invalid 'category': '{}'", self.category));
        }
        if self.key_triggers.is_empty() || self.key_triggers.iter().all(|k| k.trim().is_empty()) {
            return Err("Knowledge card must have at least one non-empty 'key_triggers' entry".to_string());
        }
        let has_valid_fact = self.fact_or_recipe.trim().len() >= 10
            || self.fact_en.trim().len() >= 10
            || self.fact_sv.trim().len() >= 10;
        if !has_valid_fact {
            return Err("Knowledge card fact ('fact_en' / 'fact_sv' / 'fact_or_recipe') is too short (< 10 chars)".to_string());
        }
        Ok(())
    }

    /// Atomically saves the validated card to `<db_root>/<domain>/<category>/<id>.json`
    /// and automatically ensures `<db_root>/<domain>/domain_profile.json` exists.
    pub fn save_to_databases(&self, db_root: &Path) -> Result<std::path::PathBuf, String> {
        let mut normalized = self.clone();
        normalized.normalize_fields();
        normalized.validate()?;

        ensure_domain_profile(db_root, &normalized.domain)?;

        let target_dir = db_root.join(&normalized.domain).join(&normalized.category);
        std::fs::create_dir_all(&target_dir)
            .map_err(|e| format!("Failed to create directory {}: {}", target_dir.display(), e))?;

        let final_path = target_dir.join(format!("{}.json", normalized.id));
        let tmp_path = target_dir.join(format!(".{}.json.tmp", normalized.id));

        let json_str = serde_json::to_string_pretty(&normalized)
            .map_err(|e| format!("Failed to serialize card to JSON: {}", e))?;

        std::fs::write(&tmp_path, json_str.as_bytes())
            .map_err(|e| format!("Failed to write temp file {}: {}", tmp_path.display(), e))?;

        std::fs::rename(&tmp_path, &final_path)
            .map_err(|e| format!("Failed to atomically rename {} to {}: {}", tmp_path.display(), final_path.display(), e))?;

        Ok(final_path)
    }

    /// Parses one or more `CrystallizedKnowledgeCard` items from raw LLM output
    /// (automatically strips `<think>...</think>` blocks and markdown code fences).
    pub fn parse_from_llm_output(raw: &str) -> Result<Vec<Self>, String> {
        // 1. Strip <think>...</think> if present
        let after_think = if let Some(end_idx) = raw.find("</think>") {
            &raw[end_idx + "</think>".len()..]
        } else {
            raw
        };

        let trimmed = after_think.trim();

        // 2. Extract JSON payload from ```json ... ``` or ``` ... ``` fences if present
        let json_slice = if let Some(fence_start) = trimmed.find("```") {
            let after_fence = &trimmed[fence_start + 3..];
            let content_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
            let inner = &after_fence[content_start..];
            if let Some(fence_end) = inner.rfind("```") {
                inner[..fence_end].trim()
            } else {
                inner.trim()
            }
        } else if let Some(arr_start) = trimmed.find('[') {
            if let Some(obj_start) = trimmed.find('{') {
                if arr_start < obj_start {
                    let end = trimmed.rfind(']').map(|i| i + 1).unwrap_or(trimmed.len());
                    &trimmed[arr_start..end]
                } else {
                    let end = trimmed.rfind('}').map(|i| i + 1).unwrap_or(trimmed.len());
                    &trimmed[obj_start..end]
                }
            } else {
                trimmed
            }
        } else if let Some(obj_start) = trimmed.find('{') {
            let end = trimmed.rfind('}').map(|i| i + 1).unwrap_or(trimmed.len());
            &trimmed[obj_start..end]
        } else {
            trimmed
        };

        // Try parsing as Vec<CrystallizedKnowledgeCard> first, then single CrystallizedKnowledgeCard
        if json_slice.trim_start().starts_with('[') {
            let parse_res: Result<Vec<Self>, _> = serde_json::from_str(json_slice);
            let mut cards = match parse_res {
                Ok(c) => c,
                Err(err) => {
                    // Resilient fallback: If array was cut off, try closing at last complete '}'
                    if let Some(last_brace) = json_slice.rfind('}') {
                        let recovered = format!("{}]", &json_slice[..=last_brace]);
                        serde_json::from_str(&recovered).map_err(|_| {
                            format!("Failed to parse JSON card array: {}", err)
                        })?
                    } else {
                        return Err(format!("Failed to parse JSON card array: {}", err));
                    }
                }
            };
            for c in &mut cards {
                c.normalize_fields();
                c.validate()?;
            }
            Ok(cards)
        } else {
            let mut card: Self = serde_json::from_str(json_slice)
                .map_err(|e| format!("Failed to parse JSON card object: {}", e))?;
            card.normalize_fields();
            card.validate()?;
            Ok(vec![card])
        }
    }
}

/// Transactional checkpoint state for the Collaborative Distillation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CrystallizationCheckpoint {
    pub total_cards_saved: usize,
    pub completed_topics: Vec<String>,
    pub domain_card_counts: HashMap<String, usize>,
    pub pruned_experts_per_layer: HashMap<usize, Vec<usize>>,
    pub last_updated_iso: String,
}

impl CrystallizationCheckpoint {
    /// Loads the checkpoint from disk, or returns a fresh default if not yet created.
    pub fn load_or_default(path: &Path) -> Self {
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(cp) = serde_json::from_slice::<Self>(&bytes) {
                return cp;
            }
        }
        Self::default()
    }

    /// Atomically saves the checkpoint to disk using a temporary file and rename.
    pub fn save_atomic(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create checkpoint parent dir: {}", e))?;
        }
        let tmp_path = path.with_extension("json.tmp");
        let json_str = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize checkpoint: {}", e))?;
        std::fs::write(&tmp_path, json_str.as_bytes())
            .map_err(|e| format!("Failed to write checkpoint tmp file: {}", e))?;
        std::fs::rename(&tmp_path, path)
            .map_err(|e| format!("Failed to rename checkpoint tmp file: {}", e))?;
        Ok(())
    }

    /// Records a newly saved knowledge card into the checkpoint statistics.
    pub fn record_saved_card(&mut self, card: &CrystallizedKnowledgeCard, topic_key: Option<&str>) {
        self.total_cards_saved += 1;
        *self.domain_card_counts.entry(card.domain.clone()).or_insert(0) += 1;
        if let Some(t) = topic_key {
            if !self.completed_topics.iter().any(|x| x == t) {
                self.completed_topics.push(t.to_string());
            }
        }
    }
}

/// Pure-Rust HTTP/1.1 client for communicating with the Qwen Teacher model on ASUS GX10 (`:30001`).
#[derive(Debug, Clone)]
pub struct Gx10TeacherClient {
    pub host: String,
    pub port: u16,
    pub api_key: String,
    pub model_name: String,
}

impl Default for Gx10TeacherClient {
    fn default() -> Self {
        Self {
            host: std::env::var("MERKABA_GX10_HOST").unwrap_or_else(|_| "192.168.10.32".to_string()),
            port: std::env::var("MERKABA_GX10_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(30001),
            api_key: std::env::var("MERKABA_GX10_API_KEY")
                .unwrap_or_else(|_| "dgByDfydZc3loyzM99TqrMkwSe3mOcX".to_string()),
            model_name: std::env::var("MERKABA_GX10_MODEL")
                .unwrap_or_else(|_| "qwen3.8-flash-next".to_string()),
        }
    }
}

impl Gx10TeacherClient {
    /// Requests the GX10 Teacher model to synthesize validated Merkaba JSON knowledge cards for a given domain/topic.
    pub fn generate_knowledge_cards(
        &self,
        domain: &str,
        category: &str,
        topic_prompt: &str,
        num_cards: usize,
    ) -> Result<Vec<CrystallizedKnowledgeCard>, String> {
        use std::io::{Read, Write};
        use std::net::{TcpStream, ToSocketAddrs};
        use std::time::Duration;

        let system_prompt = "You are the Professor and Knowledge Editor in Merkaba01. Your task is to generate 100% verified, \
            encyclopedic knowledge cards in strict JSON format (a JSON array `[...]`). \
            Respond ONLY with valid JSON without markdown commentary outside the block. \
            Each object in the array MUST include these fields (with English as #1 primary language and Swedish as #2 secondary language):\n\
            - \"id\": unique filename-safe id (only a-z, 0-9, _), e.g. \"fysik_newtons_second_law\"\n\
            - \"title\": concise clear title (e.g. \"Newton's Second Law of Motion / Newtons andra rörelselag\")\n\
            - \"domain\": top-level domain name (lowercase, e.g. \"fysik\", \"kemi\", \"programmering\")\n\
            - \"category\": subcategory folder name (lowercase, e.g. \"01_mekanik\")\n\
            - \"key_triggers\": array with at least 4 search keywords/phrases in lowercase covering BOTH English (#1) and Swedish (#2)\n\
            - \"fact_en\": (#1 Primary Language - English) concise, fact-dense technical description in English with exact numbers, SI units, and formulas (at least 80 characters)\n\
            - \"fact_sv\": (#2 Secondary Language - Swedish) concise, fact-dense technical description in Swedish with exact numbers, SI units, and formulas (at least 80 characters)\n\
            - \"fact_or_recipe\": set equal to \"fact_en\" (for backward compatibility)\n\
            - \"translations\": optional map of additional language codes (e.g. {\"de\": \"...\", \"es\": \"...\"}) or empty object `{}`\n\
            - \"components\": array of key parameters/quantities/components (can be empty)\n\
            - \"notes\": brief technical note or boundary condition";

        let user_prompt = format!(
            "/no_think\nGenerate exactly {} knowledge cards for:\nDomain: \"{}\"\nCategory: \"{}\"\nTopic: {}\n\nReturn ONLY a JSON array `[...]` with both `fact_en` (English #1) and `fact_sv` (Swedish #2).",
            num_cards, domain, category, topic_prompt
        );

        let max_tokens = (num_cards * 2048).clamp(4096, 16384);
        let payload = serde_json::json!({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2
        });

        let body_bytes = serde_json::to_vec(&payload)
            .map_err(|e| format!("Failed to serialize request JSON: {}", e))?;

        let addr_str = format!("{}:{}", self.host, self.port);
        let socket_addr = addr_str
            .to_socket_addrs()
            .map_err(|e| format!("Failed to resolve {}: {}", addr_str, e))?
            .next()
            .ok_or_else(|| format!("No socket address found for {}", addr_str))?;

        let mut stream = TcpStream::connect_timeout(&socket_addr, Duration::from_secs(10))
            .map_err(|e| format!("Failed to connect to GX10 at {}: {}", addr_str, e))?;
        let _ = stream.set_read_timeout(Some(Duration::from_secs(180)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));

        let http_req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: {}:{}\r\n\
             Authorization: Bearer {}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n",
            self.host,
            self.port,
            self.api_key,
            body_bytes.len()
        );

        stream
            .write_all(http_req.as_bytes())
            .and_then(|_| stream.write_all(&body_bytes))
            .and_then(|_| stream.flush())
            .map_err(|e| format!("Failed to send HTTP request to GX10: {}", e))?;

        let mut raw_resp = Vec::new();
        stream
            .read_to_end(&mut raw_resp)
            .map_err(|e| format!("Failed to read HTTP response from GX10: {}", e))?;

        let resp_str = String::from_utf8_lossy(&raw_resp);
        let header_end = resp_str
            .find("\r\n\r\n")
            .ok_or_else(|| "Malformed HTTP response from GX10 (missing header separator)".to_string())?;

        let headers = &resp_str[..header_end];
        let raw_body = &resp_str[header_end + 4..];

        if !headers.contains("200 OK") {
            return Err(format!(
                "GX10 returned non-200 HTTP status:\n{}\nBody: {}",
                headers.lines().next().unwrap_or(""),
                raw_body
            ));
        }

        // Decode chunked transfer-encoding if applicable
        let decoded_body = if headers.to_ascii_lowercase().contains("transfer-encoding: chunked") {
            decode_http_chunked_body(raw_body)
        } else {
            raw_body.to_string()
        };

        let resp_json: serde_json::Value = serde_json::from_str(decoded_body.trim())
            .map_err(|e| format!("Failed to parse GX10 JSON response: {} (Body: {})", e, decoded_body))?;

        let msg_obj = resp_json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c0| c0.get("message"))
            .ok_or_else(|| format!("Missing choices[0].message in GX10 response: {}", decoded_body))?;

        let content = msg_obj
            .get("content")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let reasoning = msg_obj
            .get("reasoning_content")
            .and_then(|s| s.as_str())
            .unwrap_or("");

        let combined = if !content.trim().is_empty() {
            content
        } else {
            reasoning
        };

        CrystallizedKnowledgeCard::parse_from_llm_output(combined)
    }
}

/// Decodes an HTTP/1.1 `Transfer-Encoding: chunked` body into a contiguous string.
fn decode_http_chunked_body(input: &str) -> String {
    let mut out = String::new();
    let mut rest = input;
    while let Some(pos) = rest.find("\r\n") {
        let len_hex = rest[..pos].trim();
        let Ok(chunk_len) = usize::from_str_radix(len_hex, 16) else {
            break;
        };
        if chunk_len == 0 {
            break;
        }
        let chunk_start = pos + 2;
        if rest.len() < chunk_start + chunk_len {
            out.push_str(&rest[chunk_start..]);
            break;
        }
        out.push_str(&rest[chunk_start..chunk_start + chunk_len]);
        rest = &rest[chunk_start + chunk_len..];
        if rest.starts_with("\r\n") {
            rest = &rest[2..];
        }
    }
    if out.is_empty() {
        input.to_string()
    } else {
        out
    }
}

// ============================================================================
// FAS F3 (F3.1–F3.5): Clean-Room Rust `Flash1017Runtime` (40-Layer CED Engine)
// ============================================================================

/// Special token IDs for `DeepSeek-V4.1-Flash-1017` (`vocab_size = 129280`).
pub const FLASH1017_BOS_TOKEN_ID: u32 = 0;
pub const FLASH1017_EOS_TOKEN_ID: u32 = 1;
pub const FLASH1017_PAD_TOKEN_ID: u32 = 2;
pub const FLASH1017_SYSTEM_TOKEN_ID: u32 = 128799;
pub const FLASH1017_USER_TOKEN_ID: u32 = 128803;
pub const FLASH1017_ASSISTANT_TOKEN_ID: u32 = 128804;
pub const FLASH1017_EOT_TOKEN_ID: u32 = 128805;
pub const FLASH1017_THINK_START_TOKEN_ID: u32 = 128821;
pub const FLASH1017_THINK_END_TOKEN_ID: u32 = 128822;
pub const FLASH1017_NEWLINE_TOKEN_ID: u32 = 201;

/// Builds the canonical GPT-2 / HuggingFace `ByteLevel` byte-to-unicode tables.
fn build_byte_level_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::with_capacity(256);
    for b in b'!'..=b'~' {
        bs.push(b);
    }
    for b in 0xA1u8..=0xACu8 {
        bs.push(b);
    }
    for b in 0xAEu8..=0xFFu8 {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    for (&b, &c_u32) in bs.iter().zip(cs.iter()) {
        if let Some(ch) = char::from_u32(c_u32) {
            byte_to_char[b as usize] = ch;
            char_to_byte.insert(ch, b);
        }
    }
    (byte_to_char, char_to_byte)
}

/// `F3.1`: Clean-Room Rust `ByteLevel` BPE Tokenizer for `DeepSeek-V4.1-Flash-1017` (`vocab_size = 129,280`).
pub struct Flash1017Tokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub added_tokens: Vec<(String, u32)>,
    pub bpe_ranks: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

impl Flash1017Tokenizer {
    /// Loads `tokenizer.json` from a Flash-1017 model directory or direct file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let tok_path = if path_ref.is_dir() {
            path_ref.join("tokenizer.json")
        } else {
            path_ref.to_path_buf()
        };
        let file = File::open(&tok_path)
            .map_err(|e| format!("Failed to open {}: {}", tok_path.display(), e))?;
        let root: serde_json::Value = serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| format!("Failed to parse {}: {}", tok_path.display(), e))?;

        let vocab_obj = root
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| "Missing model.vocab in tokenizer.json".to_string())?;

        let mut vocab = HashMap::with_capacity(vocab_obj.len() + 1500);
        let mut id_to_token = HashMap::with_capacity(vocab_obj.len() + 1500);
        for (k, v) in vocab_obj {
            if let Some(id) = v.as_u64() {
                let id_u32 = id as u32;
                vocab.insert(k.clone(), id_u32);
                id_to_token.insert(id_u32, k.clone());
            }
        }

        let mut added_tokens = Vec::new();
        if let Some(arr) = root.get("added_tokens").and_then(|a| a.as_array()) {
            for item in arr {
                if let (Some(content), Some(id)) = (
                    item.get("content").and_then(|c| c.as_str()),
                    item.get("id").and_then(|i| i.as_u64()),
                ) {
                    let id_u32 = id as u32;
                    vocab.insert(content.to_string(), id_u32);
                    id_to_token.insert(id_u32, content.to_string());
                    if !content.contains("place_holder") && !content.contains("place▁holder") {
                        added_tokens.push((content.to_string(), id_u32));
                    }
                }
            }
        }
        added_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut bpe_ranks = HashMap::new();
        if let Some(merges) = root
            .get("model")
            .and_then(|m| m.get("merges"))
            .and_then(|m| m.as_array())
        {
            for (rank, entry) in merges.iter().enumerate() {
                if let Some(s) = entry.as_str() {
                    if let Some((a, b)) = s.split_once(' ') {
                        bpe_ranks.insert((a.to_string(), b.to_string()), rank);
                    }
                } else if let Some(arr) = entry.as_array() {
                    if arr.len() == 2 {
                        if let (Some(a), Some(b)) = (arr[0].as_str(), arr[1].as_str()) {
                            bpe_ranks.insert((a.to_string(), b.to_string()), rank);
                        }
                    }
                }
            }
        }

        let (byte_to_char, char_to_byte) = build_byte_level_tables();
        Ok(Self {
            vocab,
            id_to_token,
            added_tokens,
            bpe_ranks,
            byte_to_char,
            char_to_byte,
        })
    }

    fn bpe_word(&self, mapped_word: &str) -> Vec<u32> {
        if let Some(&id) = self.vocab.get(mapped_word) {
            return vec![id];
        }
        let mut parts: Vec<String> = mapped_word.chars().map(|c| c.to_string()).collect();
        while parts.len() > 1 {
            let mut best_pair: Option<(usize, usize)> = None;
            for i in 0..(parts.len() - 1) {
                if let Some(&rank) = self.bpe_ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best_pair.map_or(true, |(_, best_r)| rank < best_r) {
                        best_pair = Some((i, rank));
                    }
                }
            }
            let Some((idx, _)) = best_pair else {
                break;
            };
            let merged = format!("{}{}", parts[idx], parts[idx + 1]);
            parts[idx] = merged;
            parts.remove(idx + 1);
        }
        let mut out = Vec::with_capacity(parts.len());
        for p in parts {
            if let Some(&id) = self.vocab.get(&p) {
                out.push(id);
            }
        }
        out
    }

    fn encode_plain_segment(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        // Split into GPT-style pre-tokenized chunks: optional leading space + word/digits/punctuation
        let mut chunks: Vec<String> = Vec::new();
        let mut cur = String::new();
        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\n' || ch == '\r' {
                if !cur.is_empty() {
                    chunks.push(std::mem::take(&mut cur));
                }
                cur.push(ch);
                while let Some(&nc) = chars.peek() {
                    if nc == '\n' || nc == '\r' {
                        cur.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                chunks.push(std::mem::take(&mut cur));
            } else if ch == ' ' {
                if !cur.is_empty() {
                    chunks.push(std::mem::take(&mut cur));
                }
                cur.push(ch);
                if let Some(&nc) = chars.peek() {
                    if nc.is_alphabetic() {
                        while let Some(& wc) = chars.peek() {
                            if wc.is_alphabetic() {
                                cur.push(chars.next().unwrap());
                            } else {
                                break;
                            }
                        }
                    }
                }
                chunks.push(std::mem::take(&mut cur));
            } else if ch.is_alphabetic() {
                cur.push(ch);
                while let Some(&wc) = chars.peek() {
                    if wc.is_alphabetic() {
                        cur.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                chunks.push(std::mem::take(&mut cur));
            } else if ch.is_ascii_digit() {
                cur.push(ch);
                chunks.push(std::mem::take(&mut cur));
            } else {
                cur.push(ch);
                chunks.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }

        let mut ids = Vec::new();
        for chunk in chunks {
            let mapped: String = chunk
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize])
                .collect();
            ids.extend(self.bpe_word(&mapped));
        }
        ids
    }

    /// Encodes a string into Flash-1017 token IDs (`u32`).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut earliest: Option<(usize, &str, u32)> = None;
            for (pat, id) in &self.added_tokens {
                if let Some(pos) = rest.find(pat.as_str()) {
                    if earliest.map_or(true, |(best_pos, _, _)| pos < best_pos) {
                        earliest = Some((pos, pat.as_str(), *id));
                    }
                }
            }
            if let Some((pos, pat, id)) = earliest {
                if pos > 0 {
                    out.extend(self.encode_plain_segment(&rest[..pos]));
                }
                out.push(id);
                rest = &rest[pos + pat.len()..];
            } else {
                out.extend(self.encode_plain_segment(rest));
                break;
            }
        }
        out
    }

    /// Formats a chat turn for `DeepSeek-V4.1-Flash-1017`.
    pub fn format_chat_prompt(&self, user_prompt: &str, enable_thinking: bool) -> Vec<u32> {
        let mut tokens = vec![FLASH1017_BOS_TOKEN_ID, FLASH1017_USER_TOKEN_ID];
        tokens.extend(self.encode(user_prompt));
        tokens.push(FLASH1017_ASSISTANT_TOKEN_ID);
        if enable_thinking {
            tokens.push(FLASH1017_THINK_START_TOKEN_ID);
            tokens.push(FLASH1017_NEWLINE_TOKEN_ID);
        } else {
            tokens.push(FLASH1017_THINK_START_TOKEN_ID);
            tokens.push(FLASH1017_NEWLINE_TOKEN_ID);
            tokens.push(FLASH1017_THINK_END_TOKEN_ID);
            tokens.push(FLASH1017_NEWLINE_TOKEN_ID);
        }
        tokens
    }

    /// Decodes a single token ID into its readable UTF-8 string.
    pub fn decode_token(&self, id: u32) -> String {
        self.decode(&[id])
    }

    /// Decodes a slice of Flash-1017 token IDs back into a UTF-8 string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::with_capacity(tokens.len() * 4);
        let mut out = String::new();
        for &id in tokens {
            let Some(tok_str) = self.id_to_token.get(&id) else {
                continue;
            };
            if id >= 128000 || id <= 2 {
                if !bytes.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&bytes));
                    bytes.clear();
                }
                out.push_str(tok_str);
            } else {
                for ch in tok_str.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        bytes.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        if !bytes.is_empty() {
            out.push_str(&String::from_utf8_lossy(&bytes));
        }
        out
    }
}

/// `F7.A1`: Zero-overhead pre-cached `MXFP8` tensor descriptor (`0` `format!` / `HashMap` lookups per token).
#[derive(Clone, Copy)]
pub struct CachedMxfp8Mat {
    pub w_ptr: *const u8,
    pub w_len: usize,
    pub s_ptr: *const u8,
    pub s_len: usize,
    pub rows: usize,
    pub cols: usize,
}
unsafe impl Send for CachedMxfp8Mat {}
unsafe impl Sync for CachedMxfp8Mat {}

impl CachedMxfp8Mat {
    pub fn from_shard(shard: &Flash1017ShardInspector, prefix: &str) -> Result<Self, String> {
        let (w_meta, w_bytes) = shard.get_tensor_data(&format!("{}.weight", prefix))?;
        let (_s_meta, s_bytes) = shard.get_tensor_data(&format!("{}.scale", prefix))?;
        Ok(Self {
            w_ptr: w_bytes.as_ptr(),
            w_len: w_bytes.len(),
            s_ptr: s_bytes.as_ptr(),
            s_len: s_bytes.len(),
            rows: w_meta.shape[0],
            cols: w_meta.shape[1],
        })
    }

    #[inline]
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let mut out = vec![0.0_f32; self.rows];
        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mv) = vt.v3_matvec_mxfp8() {
                unsafe {
                    mv(
                        self.rows,
                        self.cols,
                        self.w_ptr,
                        self.s_ptr,
                        x.as_ptr(),
                        out.as_mut_ptr(),
                    );
                }
                return out;
            }
        }
        let w_bytes = unsafe { std::slice::from_raw_parts(self.w_ptr, self.w_len) };
        let s_bytes = unsafe { std::slice::from_raw_parts(self.s_ptr, self.s_len) };
        let scale_cols = (self.cols + 31) / 32;
        (0..self.rows)
            .into_par_iter()
            .with_min_len(32)
            .map(|r| {
                let w_row = &w_bytes[r * self.cols..(r + 1) * self.cols];
                let sc_row_off = (r / 32) * scale_cols;
                let mut acc = 0.0_f32;
                for bc in 0..scale_cols {
                    let sc = f8_e8m0_to_f32(s_bytes[sc_row_off + bc]);
                    let c_start = bc * 32;
                    let c_end = (c_start + 32).min(self.cols);
                    let mut b_sum = 0.0_f32;
                    for c in c_start..c_end {
                        b_sum += f8_e4m3_to_f32(w_row[c]) * x[c];
                    }
                    acc += b_sum * sc;
                }
                acc
            })
            .collect()
    }

    #[inline]
    pub fn matvec_grouped(&self, x: &[f32], num_groups: usize) -> Vec<f32> {
        let total_rows = self.rows;
        let cols_per_group = self.cols;
        let mut out = vec![0.0_f32; total_rows];
        if let Some(gmv) = crate::hardware_addon::active_mxfp8_grouped_matvec() {
            let ok = unsafe {
                gmv(
                    total_rows,
                    cols_per_group,
                    num_groups,
                    self.w_ptr,
                    self.s_ptr,
                    x.as_ptr(),
                    out.as_mut_ptr(),
                )
            };
            if ok {
                return out;
            }
        }
        let rows_per_group = total_rows / num_groups;
        let scale_rows_per_group = rows_per_group / 32;
        let scale_cols = (cols_per_group + 31) / 32;
        let w_group_bytes = rows_per_group * cols_per_group;
        let s_group_bytes = scale_rows_per_group * scale_cols;
        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mv) = vt.v3_matvec_mxfp8() {
                for g in 0..num_groups {
                    unsafe {
                        mv(
                            rows_per_group,
                            cols_per_group,
                            self.w_ptr.add(g * w_group_bytes),
                            self.s_ptr.add(g * s_group_bytes),
                            x.as_ptr().add(g * cols_per_group),
                            out.as_mut_ptr().add(g * rows_per_group),
                        );
                    }
                }
                return out;
            }
        }
        let w_bytes = unsafe { std::slice::from_raw_parts(self.w_ptr, self.w_len) };
        let s_bytes = unsafe { std::slice::from_raw_parts(self.s_ptr, self.s_len) };
        for g in 0..num_groups {
            let x_g = &x[g * cols_per_group..(g + 1) * cols_per_group];
            for r in 0..rows_per_group {
                let global_r = g * rows_per_group + r;
                let w_row = &w_bytes[global_r * cols_per_group..(global_r + 1) * cols_per_group];
                let sc_row_off = (global_r / 32) * scale_cols;
                let mut acc = 0.0_f32;
                for bc in 0..scale_cols {
                    let sc = f8_e8m0_to_f32(s_bytes[sc_row_off + bc]);
                    let c_start = bc * 32;
                    let c_end = (c_start + 32).min(cols_per_group);
                    let mut b_sum = 0.0_f32;
                    for c in c_start..c_end {
                        b_sum += f8_e4m3_to_f32(w_row[c]) * x_g[c];
                    }
                    acc += b_sum * sc;
                }
                out[global_r] = acc;
            }
        }
        out
    }

    /// `F7.C1`: Batched zero-copy `MXFP8` GEMM (`[seq_len, cols] -> [seq_len, rows]`) reading weights once per batch.
    #[inline]
    pub fn matbatch(&self, x_batch: &[f32], seq_len: usize) -> Vec<f32> {
        if seq_len == 1 {
            return self.matvec(x_batch);
        }
        if self.cols <= 5120 {
            if let Some(vt) = crate::hardware_addon::active_vtable() {
                if let Some(mb) = vt.v3_matbatch_mxfp8() {
                    let mut out = vec![0.0_f32; seq_len * self.rows];
                    unsafe {
                        mb(
                            self.rows,
                            self.cols,
                            seq_len,
                            self.w_ptr,
                            self.s_ptr,
                            x_batch.as_ptr(),
                            out.as_mut_ptr(),
                        );
                    }
                    return out;
                }
            }
        }
        let mut out = Vec::with_capacity(seq_len * self.rows);
        for t in 0..seq_len {
            let v = self.matvec(&x_batch[t * self.cols..(t + 1) * self.cols]);
            out.extend_from_slice(&v);
        }
        out
    }

    /// `F7.C1`: Batched grouped `MXFP8` GEMM (`o_groups = 8` on `wo_a`) reading each group's weights once across `seq_len` tokens.
    #[inline]
    pub fn matbatch_grouped(&self, x_batch: &[f32], seq_len: usize, num_groups: usize) -> Vec<f32> {
        if seq_len == 1 {
            return self.matvec_grouped(x_batch, num_groups);
        }
        let total_rows = self.rows;
        let cols_per_group = self.cols;
        let in_per_tok = num_groups * cols_per_group;
        let rows_per_group = total_rows / num_groups;
        let scale_rows_per_group = rows_per_group / 32;
        let scale_cols = (cols_per_group + 31) / 32;
        let w_group_bytes = rows_per_group * cols_per_group;
        let s_group_bytes = scale_rows_per_group * scale_cols;

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mb) = vt.v3_matbatch_mxfp8() {
                let mut out = vec![0.0_f32; seq_len * total_rows];
                let mut x_g_batch = vec![0.0_f32; seq_len * cols_per_group];
                let mut out_g_batch = vec![0.0_f32; seq_len * rows_per_group];
                for g in 0..num_groups {
                    for t in 0..seq_len {
                        let src_off = t * in_per_tok + g * cols_per_group;
                        x_g_batch[t * cols_per_group..(t + 1) * cols_per_group]
                            .copy_from_slice(&x_batch[src_off..src_off + cols_per_group]);
                    }
                    unsafe {
                        mb(
                            rows_per_group,
                            cols_per_group,
                            seq_len,
                            self.w_ptr.add(g * w_group_bytes),
                            self.s_ptr.add(g * s_group_bytes),
                            x_g_batch.as_ptr(),
                            out_g_batch.as_mut_ptr(),
                        );
                    }
                    for t in 0..seq_len {
                        let dst_off = t * total_rows + g * rows_per_group;
                        out[dst_off..dst_off + rows_per_group]
                            .copy_from_slice(&out_g_batch[t * rows_per_group..(t + 1) * rows_per_group]);
                    }
                }
                return out;
            }
        }
        let mut out = Vec::with_capacity(seq_len * total_rows);
        for t in 0..seq_len {
            let v = self.matvec_grouped(&x_batch[t * in_per_tok..(t + 1) * in_per_tok], num_groups);
            out.extend_from_slice(&v);
        }
        out
    }
}

/// `F7.A1` & `F8.1`: Zero-overhead pre-cached `BF16` tensor descriptor with guaranteed 2-byte (`*const u16`) alignment.
#[derive(Clone)]
pub struct CachedBf16Mat {
    pub w_ptr: *const u8,
    pub w_len: usize,
    pub rows: usize,
    pub cols: usize,
    _aligned_u16: Option<std::sync::Arc<[u16]>>,
}
unsafe impl Send for CachedBf16Mat {}
unsafe impl Sync for CachedBf16Mat {}

impl CachedBf16Mat {
    pub fn from_shard(shard: &Flash1017ShardInspector, name: &str) -> Result<Self, String> {
        let (meta, raw) = shard.get_tensor_data(name)?;
        let (w_ptr, _aligned_u16) = if (raw.as_ptr() as usize) % 2 == 0 {
            (raw.as_ptr(), None)
        } else {
            let n_u16 = raw.len() / 2;
            let mut aligned = Vec::with_capacity(n_u16);
            for i in 0..n_u16 {
                aligned.push(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
            }
            let arc: std::sync::Arc<[u16]> = aligned.into();
            crate::hardware_addon::register_host_range(arc.as_ptr() as *const u8, raw.len());
            (arc.as_ptr() as *const u8, Some(arc))
        };
        Ok(Self {
            w_ptr,
            w_len: raw.len(),
            rows: meta.shape[0],
            cols: meta.shape[1],
            _aligned_u16,
        })
    }

    #[inline]
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mv_bf16) = vt.v2_matvec_bf16() {
                if (self.w_ptr as usize) % 2 == 0 {
                    let mut out = vec![0.0_f32; self.rows];
                    unsafe {
                        mv_bf16(
                            self.rows,
                            self.cols,
                            self.w_ptr as *const u16,
                            x.as_ptr(),
                            out.as_mut_ptr(),
                        );
                    }
                    return out;
                }
            }
        }
        let raw_bytes = unsafe { std::slice::from_raw_parts(self.w_ptr, self.w_len) };
        let row_bytes = self.cols * 2;
        raw_bytes
            .par_chunks_exact(row_bytes)
            .with_min_len(64)
            .map(|r_slice| {
                let mut sum = 0.0_f32;
                for c in 0..self.cols {
                    let bits = u16::from_le_bytes([r_slice[c * 2], r_slice[c * 2 + 1]]);
                    let w = f32::from_bits((bits as u32) << 16);
                    sum += w * x[c];
                }
                sum
            })
            .collect()
    }

    /// `F7.C1`: Batched zero-copy `BF16` GEMM (`[seq_len, cols] -> [seq_len, rows]`).
    #[inline]
    pub fn matbatch(&self, x_batch: &[f32], seq_len: usize) -> Vec<f32> {
        if seq_len == 0 {
            return Vec::new();
        }
        if seq_len == 1 && self.cols % 32 == 0 {
            return self.matvec(x_batch);
        }
        if seq_len <= 8 && self.cols % 32 == 0 {
            if let Some(vt) = crate::hardware_addon::active_vtable() {
                if let Some(mb_bf16) = vt.v3_matbatch_bf16() {
                    if (self.w_ptr as usize) % 2 == 0 {
                        let mut out = vec![0.0_f32; seq_len * self.rows];
                        unsafe {
                            mb_bf16(
                                self.rows,
                                self.cols,
                                seq_len,
                                self.w_ptr as *const u16,
                                x_batch.as_ptr(),
                                out.as_mut_ptr(),
                            );
                        }
                        return out;
                    }
                }
            }
        }
        self.matbatch_cpu(x_batch, seq_len)
    }

    /// High-throughput 20-core Rayon CPU `BF16` GEMM (`[seq_len, cols] -> [seq_len, rows]`)
    /// that unpacks each `BF16` weight row once into L1 cache and computes `flash1017_dot_f32` across all `seq_len` tokens.
    pub fn matbatch_cpu(&self, x_batch: &[f32], seq_len: usize) -> Vec<f32> {
        use rayon::prelude::*;
        let raw_bytes = unsafe { std::slice::from_raw_parts(self.w_ptr, self.w_len) };
        let cols = self.cols;
        let rows = self.rows;
        let row_bytes = cols * 2;
        // Compute [rows, seq_len] in parallel across rows (unpacking each BF16 row once), then transpose to [seq_len, rows]
        let row_major: Vec<f32> = raw_bytes
            .par_chunks_exact(row_bytes)
            .with_min_len(16)
            .flat_map_iter(|r_slice| {
                let mut w_row = vec![0.0_f32; cols];
                for c in 0..cols {
                    let bits = u16::from_le_bytes([r_slice[c * 2], r_slice[c * 2 + 1]]);
                    w_row[c] = f32::from_bits((bits as u32) << 16);
                }
                (0..seq_len).map(move |t| {
                    flash1017_dot_f32(&w_row, &x_batch[t * cols..(t + 1) * cols])
                })
            })
            .collect();
        let mut out = vec![0.0_f32; seq_len * rows];
        for r in 0..rows {
            let src = &row_major[r * seq_len..(r + 1) * seq_len];
            for t in 0..seq_len {
                out[t * rows + r] = src[t];
            }
        }
        out
    }
}

/// `F7.A1`: Zero-overhead pre-cached `NVFP4` tensor descriptor for Tier A experts.
#[derive(Clone, Copy)]
pub struct CachedNvfp4Mat {
    pub w_ptr: *const u8,
    pub w_len: usize,
    pub s_ptr: *const u8,
    pub s_len: usize,
    pub global_scale: f32,
    pub rows: usize,
    pub cols: usize,
}
unsafe impl Send for CachedNvfp4Mat {}
unsafe impl Sync for CachedNvfp4Mat {}

impl CachedNvfp4Mat {
    pub fn from_shard(shard: &Flash1017ShardInspector, prefix: &str) -> Result<Self, String> {
        let (w_meta, w_bytes) = shard.get_tensor_data(&format!("{}.weight", prefix))?;
        let (_sc_meta, sc_bytes) = shard.get_tensor_data(&format!("{}.weight_scale", prefix))?;
        let (_sc2_meta, sc2_bytes) = shard.get_tensor_data(&format!("{}.weight_scale_2", prefix))?;
        let global_scale = f32::from_le_bytes([sc2_bytes[0], sc2_bytes[1], sc2_bytes[2], sc2_bytes[3]]);
        Ok(Self {
            w_ptr: w_bytes.as_ptr(),
            w_len: w_bytes.len(),
            s_ptr: sc_bytes.as_ptr(),
            s_len: sc_bytes.len(),
            global_scale,
            rows: w_meta.shape[0],
            cols: w_meta.shape[1] * 2,
        })
    }

    #[inline]
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mv) = vt.v3_matvec_nvfp4() {
                let mut out = vec![0.0_f32; self.rows];
                unsafe {
                    mv(
                        self.rows,
                        self.cols,
                        self.w_ptr,
                        self.s_ptr,
                        self.global_scale,
                        x.as_ptr(),
                        out.as_mut_ptr(),
                    );
                }
                return out;
            }
        }
        let w_bytes = unsafe { std::slice::from_raw_parts(self.w_ptr, self.w_len) };
        let sc_bytes = unsafe { std::slice::from_raw_parts(self.s_ptr, self.s_len) };
        let packed_cols = self.cols / 2;
        let groups_per_row = self.cols / 16;
        let g_sc2 = self.global_scale;
        w_bytes
            .par_chunks_exact(packed_cols)
            .zip(sc_bytes.par_chunks_exact(groups_per_row))
            .with_min_len(32)
            .map(|(w_row, sc_row)| {
                let mut row_acc = 0.0_f32;
                for g in 0..groups_per_row {
                    let g_sc = f8_e4m3_to_f32(sc_row[g]);
                    let b_base = g * 8;
                    let x_base = g * 16;
                    let mut g_sum = 0.0_f32;
                    for k in 0..8 {
                        let b = w_row[b_base + k];
                        let v0 = FP4_E2M1_LUT[(b & 0x0F) as usize];
                        let v1 = FP4_E2M1_LUT[(b >> 4) as usize];
                        g_sum += v0 * x[x_base + k * 2] + v1 * x[x_base + k * 2 + 1];
                    }
                    row_acc += g_sum * g_sc;
                }
                row_acc * g_sc2
            })
            .collect()
    }

    /// `F7.C1`: Batched zero-copy `NVFP4` GEMM (`[seq_len, cols] -> [seq_len, rows]`).
    #[inline]
    pub fn matbatch(&self, x_batch: &[f32], seq_len: usize) -> Vec<f32> {
        if seq_len == 1 {
            return self.matvec(x_batch);
        }
        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mb) = vt.v3_matbatch_nvfp4() {
                let mut out = vec![0.0_f32; seq_len * self.rows];
                unsafe {
                    mb(
                        self.rows,
                        self.cols,
                        seq_len,
                        self.w_ptr,
                        self.s_ptr,
                        self.global_scale,
                        x_batch.as_ptr(),
                        out.as_mut_ptr(),
                    );
                }
                return out;
            }
        }
        let mut out = Vec::with_capacity(seq_len * self.rows);
        for t in 0..seq_len {
            let v = self.matvec(&x_batch[t * self.cols..(t + 1) * self.cols]);
            out.extend_from_slice(&v);
        }
        out
    }
}

/// `106 GB Blueprint`: Optional `0.33 MB` (`Rank-16 Q8_0`) JEPA-calibrated high-precision outlier core ($B \in \mathbb{R}^{\text{rows} \times r}, A \in \mathbb{R}^{r \times \text{cols}}$)
/// paired with a `1.58-Bit Ternary` (`6.35 MB`) residual base matrix (`6.68 MB` total per expert across all 384 experts).
#[derive(Clone)]
pub struct CachedRank16Q8Core {
    pub b_ptr: *const i8,
    pub b_len: usize,
    pub b_scales: std::sync::Arc<[f32]>,
    pub a_ptr: *const i8,
    pub a_len: usize,
    pub a_scales: std::sync::Arc<[f32]>,
    pub rank: usize,
    pub rows: usize,
    pub cols: usize,
}
unsafe impl Send for CachedRank16Q8Core {}
unsafe impl Sync for CachedRank16Q8Core {}

impl CachedRank16Q8Core {
    pub fn from_shard(
        shard: &Flash1017ShardInspector,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Option<Self> {
        let (b_meta, b_bytes) = shard.get_tensor_data(&format!("{}.core_b_q8", prefix)).ok()?;
        let (a_meta, a_bytes) = shard.get_tensor_data(&format!("{}.core_a_q8", prefix)).ok()?;
        let b_sc = shard.extract_f32_vec(&format!("{}.core_b_scale", prefix)).ok()?;
        let a_sc = shard.extract_f32_vec(&format!("{}.core_a_scale", prefix)).ok()?;
        let rank = if b_meta.shape.len() == 2 {
            b_meta.shape[1]
        } else if a_meta.shape.len() == 2 {
            a_meta.shape[0]
        } else {
            16
        };
        if rank == 0 || b_bytes.len() < rows * rank || a_bytes.len() < rank * cols {
            return None;
        }
        Some(Self {
            b_ptr: b_bytes.as_ptr() as *const i8,
            b_len: b_bytes.len(),
            b_scales: b_sc.into(),
            a_ptr: a_bytes.as_ptr() as *const i8,
            a_len: a_bytes.len(),
            a_scales: a_sc.into(),
            rank,
            rows,
            cols,
        })
    }

    /// Evaluates $\mathbf{y} \mathrel{+}= \text{scale} \cdot B (A \mathbf{x})$ in `< 4 µs` using unrolled `Q8` dot products.
    #[inline]
    pub fn apply_add(&self, x: &[f32], y: &mut [f32], scale: f32) {
        if x.len() < self.cols || y.len() < self.rows || self.rank == 0 {
            return;
        }
        let a_slice = unsafe { std::slice::from_raw_parts(self.a_ptr, self.a_len) };
        let b_slice = unsafe { std::slice::from_raw_parts(self.b_ptr, self.b_len) };
        let mut z = [0.0_f32; 32];
        let r_lim = self.rank.min(32);
        for r in 0..r_lim {
            let a_row = &a_slice[r * self.cols..(r + 1) * self.cols];
            let a_sc = self.a_scales.get(r).copied().unwrap_or(1.0);
            let b_sc = self.b_scales.get(r).copied().unwrap_or(1.0);
            let mut dot = 0.0_f32;
            for c in 0..self.cols {
                dot += (a_row[c] as f32) * x[c];
            }
            z[r] = dot * a_sc * b_sc * scale;
        }
        for row in 0..self.rows {
            let b_row = &b_slice[row * self.rank..row * self.rank + r_lim];
            let mut acc = 0.0_f32;
            for r in 0..r_lim {
                acc += (b_row[r] as f32) * z[r];
            }
            y[row] += acc;
        }
    }
}

/// `F7.A1` & `F8.1`: Zero-overhead pre-cached `1.58-Bit Ternary` tensor descriptor for Tier B/C experts
/// (plus optional `0.33 MB` `Rank-16 Q8` JEPA core for the `106 GB` all-experts-in-RAM format)
/// with guaranteed 4-byte (`*const f32`) alignment on `s_ptr` (`row_scales`) so GPU MoE streams never abort.
#[derive(Clone)]
pub struct CachedTernary158Mat {
    pub p_ptr: *const u8,
    pub p_len: usize,
    pub s_ptr: *const u8,
    pub s_len: usize,
    pub bs_ptr: *const u8,
    pub bs_len: usize,
    pub rows: usize,
    pub cols: usize,
    pub jepa_core: Option<CachedRank16Q8Core>,
    _aligned_scales: Option<std::sync::Arc<[f32]>>,
}
unsafe impl Send for CachedTernary158Mat {}
unsafe impl Sync for CachedTernary158Mat {}

impl CachedTernary158Mat {
    pub fn from_shard(shard: &Flash1017ShardInspector, prefix: &str) -> Result<Self, String> {
        let (p_meta, p_bytes) = shard.get_tensor_data(&format!("{}.packed_ternary", prefix))?;
        let (_s_meta, s_bytes) = shard.get_tensor_data(&format!("{}.row_scales", prefix))?;
        let (bs_ptr, bs_len) = match shard.get_tensor_data(&format!("{}.block_scales", prefix)) {
            Ok((_bm, b_slice)) => (b_slice.as_ptr(), b_slice.len()),
            Err(_) => (std::ptr::null(), 0),
        };
        let (s_ptr, _aligned_scales) = if (s_bytes.as_ptr() as usize) % 4 == 0 {
            (s_bytes.as_ptr(), None)
        } else {
            let n_f32 = s_bytes.len() / 4;
            let mut aligned = Vec::with_capacity(n_f32);
            for i in 0..n_f32 {
                let b = i * 4;
                aligned.push(f32::from_le_bytes([
                    s_bytes[b],
                    s_bytes[b + 1],
                    s_bytes[b + 2],
                    s_bytes[b + 3],
                ]));
            }
            let arc: std::sync::Arc<[f32]> = aligned.into();
            crate::hardware_addon::register_host_range(arc.as_ptr() as *const u8, s_bytes.len());
            (arc.as_ptr() as *const u8, Some(arc))
        };
        let rows = p_meta.shape[0];
        let cols = p_meta.shape[1] * 4;
        let jepa_core = CachedRank16Q8Core::from_shard(shard, prefix, rows, cols);
        Ok(Self {
            p_ptr: p_bytes.as_ptr(),
            p_len: p_bytes.len(),
            s_ptr,
            s_len: s_bytes.len(),
            bs_ptr,
            bs_len,
            rows,
            cols,
            jepa_core,
            _aligned_scales,
        })
    }

    #[inline]
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        if self.bs_len == 0 {
            if let Some(vt) = crate::hardware_addon::active_vtable() {
                if let Some(mv) = vt.v3_matvec_ternary_158() {
                    if (self.s_ptr as usize) % 4 == 0 {
                        let mut out = vec![0.0_f32; self.rows];
                        unsafe {
                            mv(
                                self.rows,
                                self.cols,
                                self.p_ptr,
                                self.s_ptr as *const f32,
                                x.as_ptr(),
                                out.as_mut_ptr(),
                            );
                        }
                        if let Some(ref core) = self.jepa_core {
                            core.apply_add(x, &mut out, 1.0);
                        }
                        return out;
                    }
                }
            }
        }
        let p_bytes = unsafe { std::slice::from_raw_parts(self.p_ptr, self.p_len) };
        let s_bytes = unsafe { std::slice::from_raw_parts(self.s_ptr, self.s_len) };
        let packed_cols = self.cols / 4;
        const LUT: [f32; 4] = [-1.0, 0.0, 1.0, 0.0];
        let mut out: Vec<f32> = if self.bs_len >= self.rows * (self.cols / 64) && !self.bs_ptr.is_null() {
            let bs_bytes = unsafe { std::slice::from_raw_parts(self.bs_ptr, self.bs_len) };
            let blocks_per_row = self.cols / 64;
            p_bytes
                .par_chunks_exact(packed_cols)
                .zip(s_bytes.par_chunks_exact(4))
                .zip(bs_bytes.par_chunks_exact(blocks_per_row))
                .with_min_len(32)
                .map(|((row_b, sc_b), bs_row)| {
                    let row_sc = f32::from_le_bytes([sc_b[0], sc_b[1], sc_b[2], sc_b[3]]);
                    let mut row_acc = 0.0_f32;
                    for blk in 0..blocks_per_row {
                        let u8_sc = bs_row[blk];
                        if u8_sc == 0 {
                            continue;
                        }
                        let p_off = blk * 16;
                        let mut b_acc = 0.0_f32;
                        for k in 0..16 {
                            let b = row_b[p_off + k];
                            if b == 0x55 {
                                continue;
                            }
                            let base = (p_off + k) * 4;
                            b_acc += LUT[(b & 3) as usize] * x[base]
                                + LUT[((b >> 2) & 3) as usize] * x[base + 1]
                                + LUT[((b >> 4) & 3) as usize] * x[base + 2]
                                + LUT[((b >> 6) & 3) as usize] * x[base + 3];
                        }
                        row_acc += b_acc * (u8_sc as f32);
                    }
                    row_acc * row_sc
                })
                .collect()
        } else {
            p_bytes
                .par_chunks_exact(packed_cols)
                .zip(s_bytes.par_chunks_exact(4))
                .with_min_len(32)
                .map(|(row_b, sc_b)| {
                    let sc = f32::from_le_bytes([sc_b[0], sc_b[1], sc_b[2], sc_b[3]]);
                    let mut acc = 0.0_f32;
                    for (i, &b) in row_b.iter().enumerate() {
                        if b == 0x55 {
                            continue;
                        }
                        let base = i * 4;
                        acc += LUT[(b & 3) as usize] * x[base]
                            + LUT[((b >> 2) & 3) as usize] * x[base + 1]
                            + LUT[((b >> 4) & 3) as usize] * x[base + 2]
                            + LUT[((b >> 6) & 3) as usize] * x[base + 3];
                    }
                    acc * sc
                })
                .collect()
        };
        if let Some(ref core) = self.jepa_core {
            core.apply_add(x, &mut out, 1.0);
        }
        out
    }

    /// `F7.C1`: Batched zero-copy `1.58-Bit Ternary` GEMM (`[seq_len, cols] -> [seq_len, rows]`).
    #[inline]
    pub fn matbatch(&self, x_batch: &[f32], seq_len: usize) -> Vec<f32> {
        if seq_len == 1 {
            return self.matvec(x_batch);
        }
        if self.bs_len == 0 {
            if let Some(vt) = crate::hardware_addon::active_vtable() {
                if let Some(mb) = vt.v3_matbatch_ternary_158() {
                    if (self.s_ptr as usize) % 4 == 0 {
                        let mut out = vec![0.0_f32; seq_len * self.rows];
                        unsafe {
                            mb(
                                self.rows,
                                self.cols,
                                seq_len,
                                self.p_ptr,
                                self.s_ptr as *const f32,
                                x_batch.as_ptr(),
                                out.as_mut_ptr(),
                            );
                        }
                        if let Some(ref core) = self.jepa_core {
                            for t in 0..seq_len {
                                core.apply_add(
                                    &x_batch[t * self.cols..(t + 1) * self.cols],
                                    &mut out[t * self.rows..(t + 1) * self.rows],
                                    1.0,
                                );
                            }
                        }
                        return out;
                    }
                }
            }
        }
        let mut out = Vec::with_capacity(seq_len * self.rows);
        for t in 0..seq_len {
            let v = self.matvec(&x_batch[t * self.cols..(t + 1) * self.cols]);
            out.extend_from_slice(&v);
        }
        out
    }
}

/// `F7.A1` & `F7.A3`: Zero-overhead pre-cached `MXFP4` tensor descriptor for `mtp.0` experts.
#[derive(Clone, Copy)]
pub struct CachedMxfp4Mat {
    pub w_ptr: *const u8,
    pub w_len: usize,
    pub s_ptr: *const u8,
    pub s_len: usize,
    pub rows: usize,
    pub cols: usize,
}
unsafe impl Send for CachedMxfp4Mat {}
unsafe impl Sync for CachedMxfp4Mat {}

impl CachedMxfp4Mat {
    pub fn from_shard(shard: &Flash1017ShardInspector, prefix: &str) -> Result<Self, String> {
        let (w_meta, w_bytes) = shard.get_tensor_data(&format!("{}.weight", prefix))?;
        let (_s_meta, s_bytes) = shard.get_tensor_data(&format!("{}.scale", prefix))?;
        Ok(Self {
            w_ptr: w_bytes.as_ptr(),
            w_len: w_bytes.len(),
            s_ptr: s_bytes.as_ptr(),
            s_len: s_bytes.len(),
            rows: w_meta.shape[0],
            cols: w_meta.shape[1] * 2,
        })
    }

    #[inline]
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        if let Some(mv_mxfp4) = crate::hardware_addon::active_mxfp4_matvec() {
            let mut out = vec![0.0_f32; self.rows];
            let ok = unsafe {
                mv_mxfp4(
                    self.rows,
                    self.cols,
                    self.w_ptr,
                    self.s_ptr,
                    x.as_ptr(),
                    out.as_mut_ptr(),
                )
            };
            if ok {
                return out;
            }
        }
        let w_bytes = unsafe { std::slice::from_raw_parts(self.w_ptr, self.w_len) };
        let s_bytes = unsafe { std::slice::from_raw_parts(self.s_ptr, self.s_len) };
        let packed_cols = self.cols / 2;
        let scale_cols = self.cols / 32;
        w_bytes
            .par_chunks_exact(packed_cols)
            .zip(s_bytes.par_chunks_exact(scale_cols))
            .with_min_len(32)
            .map(|(w_row, s_row)| {
                let mut acc = 0.0_f32;
                for g in 0..scale_cols {
                    let sc = f8_e8m0_to_f32(s_row[g]);
                    let b_off = g * 16;
                    let x_off = g * 32;
                    let mut b_sum = 0.0_f32;
                    for k in 0..16 {
                        let b = w_row[b_off + k];
                        let w0 = FP4_E2M1_LUT[(b & 0x0F) as usize];
                        let w1 = FP4_E2M1_LUT[(b >> 4) as usize];
                        b_sum += w0 * x[x_off + k * 2] + w1 * x[x_off + k * 2 + 1];
                    }
                    acc += b_sum * sc;
                }
                acc
            })
            .collect()
    }
}

/// `F9.1`: Pre-cached 4-bit `NVFP4` (`group_size = 16`, `global_scale = 1.0 / 2048.0`) backbone descriptors
/// for one CED layer (`wq_b`, `wo_a`, `wo_b`, `shared_w1`, `shared_w2`, `shared_w3`), backed by Zero-Copy `merkaba_backbone_q4.bin`.
#[derive(Clone, Copy)]
pub struct CachedQ4LayerBackbone {
    pub wq_b: CachedNvfp4Mat,
    pub wo_a: CachedNvfp4Mat,
    pub wo_b: CachedNvfp4Mat,
    pub shared_w1: CachedNvfp4Mat,
    pub shared_w2: CachedNvfp4Mat,
    pub shared_w3: CachedNvfp4Mat,
}

const MKBQ4_MAGIC: &[u8; 8] = b"MKBQ4\x01\x00\x00";
const MKBQ4_HEADER_SIZE: usize = 4096;
const MKBQ4_LAYER_BYTES: usize = 85_966_848;
const MKBQ4_GLOBAL_SCALE: f32 = 1.0_f32 / 2048.0_f32;

#[inline(always)]
fn quantize_fp4_e2m1_pos_mag(a: f32) -> (u8, f32) {
    if a < 0.25 {
        (0, 0.0)
    } else if a < 0.75 {
        (1, 0.5)
    } else if a < 1.25 {
        (2, 1.0)
    } else if a < 1.75 {
        (3, 1.5)
    } else if a < 2.50 {
        (4, 2.0)
    } else if a < 3.50 {
        (5, 3.0)
    } else if a < 5.00 {
        (6, 4.0)
    } else {
        (7, 6.0)
    }
}

/// `F9.1`: Quantizes an `MXFP8` matrix (`[rows, cols]`) into `4-bit NVFP4` (`group_size = 16`, `F8_E4M3` group scale,
/// `global_scale = 1.0 / 2048.0`) using a 3-candidate MMSE scale search across all CPU cores via Rayon.
fn quantize_mxfp8_mat_to_nvfp4(
    mat: &CachedMxfp8Mat,
    f8_pos_lut: &[f32; 127],
) -> (Vec<u8>, Vec<u8>) {
    use rayon::prelude::*;
    let rows = mat.rows;
    let cols = mat.cols;
    let packed_cols = cols / 2;
    let groups_per_row = cols / 16;
    let scale_cols_mxfp8 = cols / 32;
    let w_bytes = unsafe { std::slice::from_raw_parts(mat.w_ptr, mat.w_len) };
    let s_bytes = unsafe { std::slice::from_raw_parts(mat.s_ptr, mat.s_len) };

    let mut fp4_out = vec![0u8; rows * packed_cols];
    let mut sc_out = vec![0u8; rows * groups_per_row];

    fp4_out
        .par_chunks_exact_mut(packed_cols)
        .zip(sc_out.par_chunks_exact_mut(groups_per_row))
        .enumerate()
        .with_min_len(64)
        .for_each(|(r, (fp4_row, sc_row))| {
            let w_row = &w_bytes[r * cols..(r + 1) * cols];
            let s_row_off = (r / 32) * scale_cols_mxfp8;
            for g in 0..groups_per_row {
                let e8m0_sc = f8_e8m0_to_f32(s_bytes[s_row_off + (g >> 1)]) * 2048.0_f32;
                let c_base = g * 16;
                let mut u = [0.0_f32; 16];
                let mut max_abs = 0.0_f32;
                for k in 0..16 {
                    let v = f8_e4m3_to_f32(w_row[c_base + k]) * e8m0_sc;
                    u[k] = v;
                    let a = v.abs();
                    if a > max_abs {
                        max_abs = a;
                    }
                }
                if max_abs <= 1e-12 {
                    sc_row[g] = 0;
                    for k in 0..8 {
                        fp4_row[g * 8 + k] = 0;
                    }
                    continue;
                }
                let s_target = max_abs * (1.0_f32 / 6.0_f32);
                let idx = f8_pos_lut
                    .partition_point(|&v| v < s_target)
                    .clamp(1, 126);
                let cands = [idx.saturating_sub(1).max(1), idx, (idx + 1).min(126)];
                let mut best_b = idx as u8;
                let mut best_err = f32::INFINITY;
                for &b_cand in &cands {
                    let sc = f8_pos_lut[b_cand];
                    let inv_sc = 1.0_f32 / sc;
                    let mut err = 0.0_f32;
                    for k in 0..16 {
                        let (_, q_mag) = quantize_fp4_e2m1_pos_mag(u[k].abs() * inv_sc);
                        let q_val = if u[k] < 0.0 { -q_mag * sc } else { q_mag * sc };
                        let d = q_val - u[k];
                        err += d * d;
                    }
                    if err < best_err {
                        best_err = err;
                        best_b = b_cand as u8;
                    }
                }
                sc_row[g] = best_b;
                let best_inv_sc = 1.0_f32 / f8_pos_lut[best_b as usize];
                for k in 0..8 {
                    let v0 = u[k * 2];
                    let v1 = u[k * 2 + 1];
                    let (m0, _) = quantize_fp4_e2m1_pos_mag(v0.abs() * best_inv_sc);
                    let (m1, _) = quantize_fp4_e2m1_pos_mag(v1.abs() * best_inv_sc);
                    let n0 = if v0 < 0.0 && m0 > 0 { m0 | 8 } else { m0 };
                    let n1 = if v1 < 0.0 && m1 > 0 { m1 | 8 } else { m1 };
                    fp4_row[g * 8 + k] = n0 | (n1 << 4);
                }
            }
        });

    (fp4_out, sc_out)
}

fn load_or_build_q4_backbone_sidecar(
    model_dir: &Path,
    layer_meta: &mut [Flash1017LayerMeta],
) -> Result<Option<Mmap>, String> {
    if std::env::var("MERKABA_NO_BACKBONE_Q4").is_ok() || layer_meta.len() != 40 {
        return Ok(None);
    }
    let cache_path = model_dir.join("merkaba_backbone_q4.bin");
    let expected_len = MKBQ4_HEADER_SIZE + 40 * MKBQ4_LAYER_BYTES;

    let need_build = match std::fs::metadata(&cache_path) {
        Ok(m) => (m.len() as usize) != expected_len,
        Err(_) => true,
    };

    if need_build {
        use std::io::{BufWriter, Write};
        let t_build = std::time::Instant::now();
        eprintln!(
            "  ⚡ [F9.1] Bygger 4-bit NVFP4 ryggrad ({}, {:.2} GiB från MXFP8)...",
            cache_path.display(),
            (expected_len as f64) / (1024.0 * 1024.0 * 1024.0)
        );
        let mut f8_pos_lut = [0.0_f32; 127];
        for b in 0..127usize {
            f8_pos_lut[b] = f8_e4m3_to_f32(b as u8);
        }
        let tmp_path = cache_path.with_extension("bin.tmp");
        let file = File::create(&tmp_path)
            .map_err(|e| format!("Failed to create {}: {}", tmp_path.display(), e))?;
        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

        let mut header = vec![0u8; MKBQ4_HEADER_SIZE];
        header[0..8].copy_from_slice(MKBQ4_MAGIC);
        header[8..16].copy_from_slice(&(40u64).to_le_bytes());
        header[16..20].copy_from_slice(&MKBQ4_GLOBAL_SCALE.to_le_bytes());
        writer.write_all(&header).map_err(|e| e.to_string())?;

        for l in 0..40 {
            let lm = &layer_meta[l];
            for mat in [
                &lm.wq_b,
                &lm.wo_a,
                &lm.wo_b,
                &lm.shared_w1,
                &lm.shared_w3,
                &lm.shared_w2,
            ] {
                let (fp4, sc) = quantize_mxfp8_mat_to_nvfp4(mat, &f8_pos_lut);
                writer.write_all(&fp4).map_err(|e| e.to_string())?;
                writer.write_all(&sc).map_err(|e| e.to_string())?;
            }
        }
        writer.flush().map_err(|e| e.to_string())?;
        drop(writer);
        std::fs::rename(&tmp_path, &cache_path)
            .map_err(|e| format!("Failed to rename {}: {}", tmp_path.display(), e))?;
        eprintln!(
            "  ✅ [F9.1] 4-bit NVFP4 ryggrad klar på {:.2} s (sparat 2.91 GB minnestrafik/token)!",
            t_build.elapsed().as_secs_f64()
        );
    }

    let file = File::open(&cache_path)
        .map_err(|e| format!("Failed to open {}: {}", cache_path.display(), e))?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| format!("Failed to mmap {}: {}", cache_path.display(), e))?
    };
    if mmap.len() != expected_len || &mmap[0..8] != MKBQ4_MAGIC {
        return Err("Invalid merkaba_backbone_q4.bin header or size".to_string());
    }
    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" {
            fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
        }
        let _ = madvise(
            mmap.as_ptr() as *mut std::ffi::c_void,
            mmap.len(),
            14, // MADV_HUGEPAGE
        );
    }
    let base_ptr = mmap.as_ptr();
    crate::hardware_addon::register_host_mmap_bounds(base_ptr, mmap.len());
    // Register in 512 MB chunks so the entire 3.20 GiB sidecar is zero-copy pinned upfront
    let chunk_size = 512 * 1024 * 1024usize;
    let mut off = 0usize;
    while off < mmap.len() {
        let len = (mmap.len() - off).min(chunk_size);
        unsafe {
            crate::hardware_addon::register_host_range(base_ptr.add(off), len);
        }
        off += len;
    }

    for l in 0..40 {
        let l_base = unsafe { base_ptr.add(MKBQ4_HEADER_SIZE + l * MKBQ4_LAYER_BYTES) };
        let mk_mat = |fp4_off: usize,
                      fp4_len: usize,
                      sc_off: usize,
                      sc_len: usize,
                      rows: usize,
                      cols: usize| unsafe {
            CachedNvfp4Mat {
                w_ptr: l_base.add(fp4_off),
                w_len: fp4_len,
                s_ptr: l_base.add(sc_off),
                s_len: sc_len,
                global_scale: MKBQ4_GLOBAL_SCALE,
                rows,
                cols,
            }
        };
        layer_meta[l].q4_backbone = Some(CachedQ4LayerBackbone {
            wq_b: mk_mat(0, 20_971_520, 20_971_520, 2_621_440, 32768, 1280),
            wo_a: mk_mat(23_592_960, 16_777_216, 40_370_176, 2_097_152, 8192, 4096),
            wo_b: mk_mat(42_467_328, 20_971_520, 63_438_848, 2_621_440, 5120, 8192),
            shared_w1: mk_mat(66_060_288, 5_898_240, 71_958_528, 737_280, 2304, 5120),
            shared_w3: mk_mat(72_695_808, 5_898_240, 78_594_048, 737_280, 2304, 5120),
            shared_w2: mk_mat(79_331_328, 5_898_240, 85_229_568, 737_280, 5120, 2304),
        });
    }

    Ok(Some(mmap))
}

/// `F9.2`: Loads or builds the `82.0 MiB` (`85_970_944` bytes) Zero-Copy `4-Bit NVFP4` sidecar
/// (`merkaba_mtp0_q4.bin` / `merkaba_mtp1_q4.bin` / `merkaba_mtp2_q4.bin`) for a speculative `mtp` (`dspark`) sub-layer (`wq_b`, `wo_a`, `wo_b`, `shared_w1/w3/w2`).
pub fn load_or_build_mtp_named_q4_sidecar(
    model_dir: &Path,
    sidecar_filename: &str,
    shard: &Flash1017ShardInspector,
    wq_b: &CachedMxfp8Mat,
    wo_a: &CachedMxfp8Mat,
    wo_b: &CachedMxfp8Mat,
    shared_w1: &CachedMxfp8Mat,
    shared_w3: &CachedMxfp8Mat,
    shared_w2: &CachedMxfp8Mat,
) -> Result<(Mmap, CachedQ4LayerBackbone), String> {
    const MKBQ4_MTP_MAGIC: &[u8; 8] = b"MKQ4MTP1";
    let mut sidecar_path = model_dir.join(sidecar_filename);
    let tmp_fallback_path = std::path::PathBuf::from("/tmp").join(sidecar_filename);
    if !sidecar_path.exists() && tmp_fallback_path.exists() {
        sidecar_path = tmp_fallback_path.clone();
    }
    let expected_size = MKBQ4_HEADER_SIZE + MKBQ4_LAYER_BYTES;

    let mut need_build = true;
    if sidecar_path.exists() {
        if let Ok(md) = std::fs::metadata(&sidecar_path) {
            if md.len() as usize == expected_size {
                if let Ok(mut f) = std::fs::File::open(&sidecar_path) {
                    use std::io::Read;
                    let mut magic = [0u8; 8];
                    if f.read_exact(&mut magic).is_ok() && &magic == MKBQ4_MTP_MAGIC {
                        need_build = false;
                    }
                }
            }
        }
    }

    if need_build {
        use std::io::{BufWriter, Write};
        let _ = shard;
        let mut f8_pos_lut = [0.0_f32; 127];
        for b in 0..127usize {
            f8_pos_lut[b] = f8_e4m3_to_f32(b as u8);
        }
        let mut tmp_path = sidecar_path.with_extension("bin.tmp");
        let file = match File::create(&tmp_path) {
            Ok(f) => f,
            Err(_) => {
                sidecar_path = tmp_fallback_path;
                tmp_path = sidecar_path.with_extension("bin.tmp");
                File::create(&tmp_path)
                    .map_err(|e| format!("Failed to create {}: {}", tmp_path.display(), e))?
            }
        };
        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);
        let mut header = vec![0u8; MKBQ4_HEADER_SIZE];
        header[0..8].copy_from_slice(MKBQ4_MTP_MAGIC);
        header[8..16].copy_from_slice(&(1u64).to_le_bytes());
        header[16..20].copy_from_slice(&MKBQ4_GLOBAL_SCALE.to_le_bytes());
        writer.write_all(&header).map_err(|e| e.to_string())?;
        for mat in [wq_b, wo_a, wo_b, shared_w1, shared_w3, shared_w2] {
            let (fp4, sc) = quantize_mxfp8_mat_to_nvfp4(mat, &f8_pos_lut);
            writer.write_all(&fp4).map_err(|e| e.to_string())?;
            writer.write_all(&sc).map_err(|e| e.to_string())?;
        }
        writer.flush().map_err(|e| e.to_string())?;
        drop(writer);
        std::fs::rename(&tmp_path, &sidecar_path)
            .map_err(|e| format!("Failed to rename {}: {}", tmp_path.display(), e))?;
    }

    let f_ro = std::fs::File::open(&sidecar_path)
        .map_err(|e| format!("Failed to open {}: {}", sidecar_path.display(), e))?;
    let mmap = unsafe {
        Mmap::map(&f_ro).map_err(|e| format!("Failed to mmap {}: {}", sidecar_path.display(), e))?
    };
    crate::hardware_addon::register_host_mmap_bounds(mmap.as_ptr(), mmap.len());
    crate::hardware_addon::register_host_range(mmap.as_ptr(), mmap.len());
    let l_base = unsafe { mmap.as_ptr().add(MKBQ4_HEADER_SIZE) };
    let mk_mat = |fp4_off: usize,
                  fp4_len: usize,
                  sc_off: usize,
                  sc_len: usize,
                  rows: usize,
                  cols: usize| unsafe {
        CachedNvfp4Mat {
            w_ptr: l_base.add(fp4_off),
            w_len: fp4_len,
            s_ptr: l_base.add(sc_off),
            s_len: sc_len,
            global_scale: MKBQ4_GLOBAL_SCALE,
            rows,
            cols,
        }
    };
    let q4 = CachedQ4LayerBackbone {
        wq_b: mk_mat(0, 20_971_520, 20_971_520, 2_621_440, 32768, 1280),
        wo_a: mk_mat(23_592_960, 16_777_216, 40_370_176, 2_097_152, 8192, 4096),
        wo_b: mk_mat(42_467_328, 20_971_520, 63_438_848, 2_621_440, 5120, 8192),
        shared_w1: mk_mat(66_060_288, 5_898_240, 71_958_528, 737_280, 2304, 5120),
        shared_w3: mk_mat(72_695_808, 5_898_240, 78_594_048, 737_280, 2304, 5120),
        shared_w2: mk_mat(79_331_328, 5_898_240, 85_229_568, 737_280, 5120, 2304),
    };
    Ok((mmap, q4))
}

pub fn load_or_build_mtp0_q4_sidecar(
    model_dir: &Path,
    shard: &Flash1017ShardInspector,
    wq_b: &CachedMxfp8Mat,
    wo_a: &CachedMxfp8Mat,
    wo_b: &CachedMxfp8Mat,
    shared_w1: &CachedMxfp8Mat,
    shared_w3: &CachedMxfp8Mat,
    shared_w2: &CachedMxfp8Mat,
) -> Result<(Mmap, CachedQ4LayerBackbone), String> {
    load_or_build_mtp_named_q4_sidecar(
        model_dir,
        "merkaba_mtp0_q4.bin",
        shard,
        wq_b,
        wo_a,
        wo_b,
        shared_w1,
        shared_w3,
        shared_w2,
    )
}

/// Preloaded norms, mHC hyper-connection parameters, pre-cached tensor pointers (`F7.A1`),
/// and surgical Tier A/B/C routing sets for one CED layer (`0..39`).
pub struct Flash1017LayerMeta {
    pub layer_idx: usize,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub attn_sink: Vec<f32>,
    pub compressor_norm: Option<Vec<f32>>,
    pub hc_attn_base: [f32; 24],
    pub hc_attn_scale: [f32; 3],
    pub hc_ffn_base: [f32; 24],
    pub hc_ffn_scale: [f32; 3],
    pub hc_attn_fn_ptr: *const f32,
    pub hc_ffn_fn_ptr: *const f32,
    pub _hc_aligned: Option<(Vec<f32>, Vec<f32>)>,
    pub gate_bias: Vec<f32>,
    pub gate_bias_vl: Vec<f32>,
    pub tier_a_set: Vec<bool>,
    pub tier_b_set: Vec<bool>,
    pub tier_a_list: Vec<usize>,
    pub tier_b_list: Vec<usize>,
    /// Calibrated scale factor for Tier B 1.58-bit ternary experts (`mean(weight_scale) * weight_scale_2` ~ `0.01065`).
    pub tier_b_proj_scale: f32,
    pub comp_wkv: Option<CachedBf16Mat>,
    pub comp_wgate: Option<CachedBf16Mat>,
    pub wq_a: CachedMxfp8Mat,
    pub wq_b: CachedMxfp8Mat,
    pub wkv: CachedMxfp8Mat,
    pub wo_a: CachedMxfp8Mat,
    pub wo_b: CachedMxfp8Mat,
    pub ffn_gate: CachedBf16Mat,
    pub shared_w1: CachedMxfp8Mat,
    pub shared_w2: CachedMxfp8Mat,
    pub shared_w3: CachedMxfp8Mat,
    pub q4_backbone: Option<CachedQ4LayerBackbone>,
    pub tier_a_experts: Vec<Option<(CachedNvfp4Mat, CachedNvfp4Mat, CachedNvfp4Mat)>>,
    pub tier_b_experts: Vec<Option<(CachedTernary158Mat, CachedTernary158Mat, CachedTernary158Mat)>>,
}
unsafe impl Send for Flash1017LayerMeta {}
unsafe impl Sync for Flash1017LayerMeta {}

/// `F3.3`: Multi-head Latent Attention (`MLA`) KV-Cache across all 40 CED layers.
/// Each layer stores only `1` shared latent KV head of `512` floats per token (`2 KB / token / layer` = `80 KB / token` total!).
#[derive(Clone)]
pub struct Flash1017KvCache {
    /// Per-layer shared latent KV cache `[40][seq_len * 512]` (post-RoPE on dims `448..512`, `k = v = kv_t`).
    pub k_latent: Vec<Vec<f32>>,
    /// Legacy alias kept for compatibility.
    pub v_latent: Vec<Vec<f32>>,
    /// Compressed global KV cache from `kv_source_layer_ids = [2, 8, 14, 20]` (`slot 0..3`).
    pub compressed_k: Vec<Vec<f32>>,
    pub compressed_v: Vec<Vec<f32>>,
    /// Pending `even`-position `(kv, score)` buffers for `compress_ratio = 2` source layers `[2, 8, 14]`.
    pub pending_c2_kv: [Vec<f32>; 3],
    pub pending_c2_score: [Vec<f32>; 3],
    /// `F4.3`: Shared latent KV cache `[seq_len * 512]` for the speculative `mtp.0` (`dspark`) layer.
    pub mtp_k_latent: Vec<f32>,
    /// `F9.2`: Latent KV caches `[seq_len * 512]` for `mtp.1` (`model-00045`) and `mtp.2` (`model-00046`).
    pub mtp12_k_latent: [Vec<f32>; 2],
    /// `F4.3`: Most recent combined `5120`-dim hidden states from `dspark_target_layer_ids = [37, 38, 39]`.
    pub dspark_target_h: [Vec<f32>; 3],
    /// `F7.C2`: Verified Layer-39 & `mtp.0` trajectory anchors for high-acceptance multi-token drafting.
    pub mtp_verified_states: Vec<Flash1017VerifiedMtpState>,
    /// `JEPA Autopilot`: Exponentially smoothed `5120`-dim semantic trajectory anchor across verified tokens.
    pub jepa_trajectory_anchor: Vec<f32>,
    /// Recent token IDs for `F3.4` Engram N-Gram (`1..4`-gram) conditional memory lookup.
    pub token_history: Vec<u32>,
    pub seq_len: usize,
}

/// `F7.C2` & `F9.2`: Verified Layer-39 & `mtp.0` state entry used to anchor `mtp.0` (`dspark` layer 0)
/// onto the target model's Layer-39 (`head.weight`) manifold, including the verified continuation token (`next_token_id`).
#[derive(Clone)]
pub struct Flash1017VerifiedMtpState {
    pub seq_pos: usize,
    pub token_id: u32,
    pub prev_token_id: u32,
    pub prev_h39: Vec<f32>,
    pub mtp_raw: Vec<f32>,
    pub target_h39: Vec<f32>,
    pub next_token_id: u32,
    pub next_logit: f32,
    pub next_margin: f32,
}

impl Flash1017KvCache {
    pub fn new() -> Self {
        let cap_floats = 4096 * 512; // Pre-allocate 4096 tokens * 512 floats (8 MB/layer) for zero-reallocation Zero-Copy GPU MLA
        let mut k_latent = Vec::with_capacity(40);
        for _ in 0..40 {
            let v: Vec<f32> = Vec::with_capacity(cap_floats);
            crate::hardware_addon::register_host_range(
                v.as_ptr() as *const u8,
                cap_floats * std::mem::size_of::<f32>(),
            );
            k_latent.push(v);
        }
        let mut compressed_k = Vec::with_capacity(4);
        for _ in 0..4 {
            let v: Vec<f32> = Vec::with_capacity(cap_floats);
            crate::hardware_addon::register_host_range(
                v.as_ptr() as *const u8,
                cap_floats * std::mem::size_of::<f32>(),
            );
            compressed_k.push(v);
        }
        let mtp_k_latent: Vec<f32> = Vec::with_capacity(cap_floats);
        crate::hardware_addon::register_host_range(
            mtp_k_latent.as_ptr() as *const u8,
            cap_floats * std::mem::size_of::<f32>(),
        );
        let mtp1_k: Vec<f32> = Vec::with_capacity(cap_floats);
        let mtp2_k: Vec<f32> = Vec::with_capacity(cap_floats);
        crate::hardware_addon::register_host_range(
            mtp1_k.as_ptr() as *const u8,
            cap_floats * std::mem::size_of::<f32>(),
        );
        crate::hardware_addon::register_host_range(
            mtp2_k.as_ptr() as *const u8,
            cap_floats * std::mem::size_of::<f32>(),
        );
        Self {
            k_latent,
            v_latent: vec![Vec::new(); 40],
            compressed_k,
            compressed_v: vec![Vec::new(); 4],
            pending_c2_kv: [Vec::new(), Vec::new(), Vec::new()],
            pending_c2_score: [Vec::new(), Vec::new(), Vec::new()],
            mtp_k_latent,
            mtp12_k_latent: [mtp1_k, mtp2_k],
            dspark_target_h: [vec![0.0_f32; 5120], vec![0.0_f32; 5120], vec![0.0_f32; 5120]],
            mtp_verified_states: Vec::with_capacity(512),
            jepa_trajectory_anchor: Vec::new(),
            token_history: Vec::with_capacity(4096),
            seq_len: 0,
        }
    }

    /// Updates the `5120`-dim JEPA semantic trajectory anchor with an exponential moving average (`EMA`)
    /// of verified Layer-39 normalized hidden states (`< 5 µs`).
    #[inline]
    pub fn update_jepa_trajectory(&mut self, final_normed: &[f32]) {
        if final_normed.len() != 5120 {
            return;
        }
        if self.jepa_trajectory_anchor.len() != 5120 {
            self.jepa_trajectory_anchor = final_normed.to_vec();
        } else {
            for i in 0..5120 {
                self.jepa_trajectory_anchor[i] =
                    0.82_f32 * self.jepa_trajectory_anchor[i] + 0.18_f32 * final_normed[i];
            }
        }
    }

    #[inline]
    fn ensure_vec_host_registered(v: &mut Vec<f32>, needed_floats: usize) {
        if needed_floats > v.capacity() {
            let new_tokens = (needed_floats / 512)
                .max(4096)
                .next_power_of_two()
                .min(1_048_576);
            let target_cap = (new_tokens * 512).max(needed_floats);
            let cur_len = v.len();
            if target_cap > cur_len {
                v.reserve(target_cap - cur_len);
            }
            crate::hardware_addon::register_host_range(
                v.as_ptr() as *const u8,
                v.capacity() * std::mem::size_of::<f32>(),
            );
        }
    }

    /// Ensures that all MLA KV buffers have sufficient capacity for `add_tokens` additional tokens
    /// and re-registers with CUDA Zero-Copy (`cuMemHostRegister`) whenever a buffer grows up to `1,048,576` (`1M`) tokens.
    pub fn ensure_capacity_for_additional_tokens(&mut self, add_tokens: usize) {
        let target_tokens = self.seq_len + add_tokens + 16;
        let needed_floats = target_tokens * 512;
        for l in 0..40 {
            Self::ensure_vec_host_registered(&mut self.k_latent[l], needed_floats);
        }
        for slot in 0..4 {
            Self::ensure_vec_host_registered(&mut self.compressed_k[slot], needed_floats);
        }
        Self::ensure_vec_host_registered(&mut self.mtp_k_latent, needed_floats);
        for m_idx in 0..2 {
            Self::ensure_vec_host_registered(&mut self.mtp12_k_latent[m_idx], needed_floats);
        }
    }

    /// Clones the KV cache while preserving the pre-allocated capacity and CUDA host registration.
    pub fn clone_preallocated(&self) -> Self {
        let mut cloned = self.clone();
        let cap_floats = (self.seq_len + 512).max(4096) * 512;
        for l in 0..40 {
            let cur_len = cloned.k_latent[l].len();
            if cloned.k_latent[l].capacity() < cap_floats {
                cloned.k_latent[l].reserve(cap_floats - cur_len);
            }
            crate::hardware_addon::register_host_range(
                cloned.k_latent[l].as_ptr() as *const u8,
                cloned.k_latent[l].capacity() * std::mem::size_of::<f32>(),
            );
        }
        for slot in 0..4 {
            let cur_len = cloned.compressed_k[slot].len();
            if cloned.compressed_k[slot].capacity() < cap_floats {
                cloned.compressed_k[slot].reserve(cap_floats - cur_len);
            }
            crate::hardware_addon::register_host_range(
                cloned.compressed_k[slot].as_ptr() as *const u8,
                cloned.compressed_k[slot].capacity() * std::mem::size_of::<f32>(),
            );
        }
        let cur_mtp_len = cloned.mtp_k_latent.len();
        if cloned.mtp_k_latent.capacity() < cap_floats {
            cloned.mtp_k_latent.reserve(cap_floats - cur_mtp_len);
        }
        crate::hardware_addon::register_host_range(
            cloned.mtp_k_latent.as_ptr() as *const u8,
            cloned.mtp_k_latent.capacity() * std::mem::size_of::<f32>(),
        );
        for m_idx in 0..2 {
            let cur_m_len = cloned.mtp12_k_latent[m_idx].len();
            if cloned.mtp12_k_latent[m_idx].capacity() < cap_floats {
                cloned.mtp12_k_latent[m_idx].reserve(cap_floats - cur_m_len);
            }
            crate::hardware_addon::register_host_range(
                cloned.mtp12_k_latent[m_idx].as_ptr() as *const u8,
                cloned.mtp12_k_latent[m_idx].capacity() * std::mem::size_of::<f32>(),
            );
        }
        cloned
    }

    #[inline]
    pub fn sync_mtp12_to_mtp0(&mut self, target_floats: usize) {
        for k_lat in &mut self.mtp12_k_latent {
            if k_lat.len() > target_floats {
                k_lat.truncate(target_floats);
            } else if k_lat.len() < target_floats {
                while k_lat.len() + 512 <= target_floats && k_lat.len() >= 512 {
                    let start = k_lat.len() - 512;
                    k_lat.extend_from_within(start..);
                }
                if k_lat.len() < target_floats {
                    k_lat.resize(target_floats, 0.0);
                }
            }
        }
    }

    /// Truncates the KV cache to `target_len` tokens (used by speculative rollback or `/undo`).
    pub fn truncate(&mut self, target_len: usize) {
        if target_len >= self.seq_len {
            return;
        }
        for l in 0..40 {
            self.k_latent[l].truncate(target_len * 512);
            self.v_latent[l].truncate(target_len * 512);
        }
        for slot in 0..3 {
            self.compressed_k[slot].truncate((target_len / 2) * 512);
            self.compressed_v[slot].truncate((target_len / 2) * 512);
        }
        self.compressed_k[3].truncate(target_len * 512);
        self.compressed_v[3].truncate(target_len * 512);
        let mtp_keep = target_len.saturating_sub(1) * 512;
        self.mtp_k_latent.truncate(mtp_keep);
        self.mtp12_k_latent[0].truncate(mtp_keep);
        self.mtp12_k_latent[1].truncate(mtp_keep);
        self.mtp_verified_states.retain(|s| s.seq_pos < target_len);
        self.token_history.truncate(target_len);
        self.seq_len = target_len;
    }

    /// `F7.C1`: Restores the exact KV-cache, compressor-cache, pending pair-pooling buffers, and `dspark_target_h`
    /// state at `state` (used when rolling back rejected speculative `mtp.0` draft tokens).
    pub fn restore_position_state(&mut self, state: &Flash1017PositionState, mtp_k_floats: usize) {
        for l in 0..40 {
            self.k_latent[l].truncate(state.seq_len * 512);
            self.v_latent[l].truncate(state.seq_len * 512);
        }
        for slot in 0..4 {
            self.compressed_k[slot].truncate(state.compressed_k_lens[slot]);
            self.compressed_v[slot].truncate(state.compressed_k_lens[slot]);
        }
        self.pending_c2_kv = state.pending_c2_kv.clone();
        self.pending_c2_score = state.pending_c2_score.clone();
        self.dspark_target_h = state.dspark_target_h.clone();
        self.mtp_k_latent.truncate(mtp_k_floats);
        self.mtp12_k_latent[0].truncate(mtp_k_floats);
        self.mtp12_k_latent[1].truncate(mtp_k_floats);
        self.mtp_verified_states
            .retain(|s| s.seq_pos < state.seq_len);
        self.token_history.truncate(state.seq_len);
        self.seq_len = state.seq_len;
    }
}

/// `F7.C1`: Exact per-token state snapshot captured during 40-layer `matbatch` verification (`seq_len = 2..5`).
#[derive(Clone)]
pub struct Flash1017PositionState {
    pub seq_len: usize,
    pub compressed_k_lens: [usize; 4],
    pub pending_c2_kv: [Vec<f32>; 3],
    pub pending_c2_score: [Vec<f32>; 3],
    pub dspark_target_h: [Vec<f32>; 3],
    pub final_normed: Vec<f32>,
}

/// `F7.C2`: Speculative Multi-Token Prediction (`mtp.0`) depth mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flash1017MtpMode {
    /// Standard 1-token-at-a-time autoregressive decoding (`K = 0`).
    Off,
    /// Fixed speculative draft depth `K` (`1..=4`), verified in parallel via `1 + K` 40-layer `matbatch`.
    Fixed(usize),
    /// Self-regulating adaptive depth (`K = 1..3`) gated by `mtp.0` top-1 vs top-2 logit margin.
    Adaptive,
}

impl Flash1017MtpMode {
    pub fn parse_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "0" | "off" | "none" | "false" => Self::Off,
            "1" | "k1" => Self::Fixed(1),
            "2" | "k2" => Self::Fixed(2),
            "3" | "k3" => Self::Fixed(3),
            "4" | "k4" => Self::Fixed(4),
            "auto" | "adaptive" | "adapt" => Self::Adaptive,
            _ => Self::Adaptive,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Off => "K=0 (Off)",
            Self::Fixed(1) => "K=1",
            Self::Fixed(2) => "K=2",
            Self::Fixed(3) => "K=3",
            Self::Fixed(4) => "K=4",
            Self::Fixed(_) => "K=Custom",
            Self::Adaptive => "Adaptive (1..3)",
        }
    }

    pub fn max_draft_tokens(&self) -> usize {
        match self {
            Self::Off => 0,
            Self::Fixed(k) => (*k).min(4),
            Self::Adaptive => 3,
        }
    }
}

/// `F7.C3`: Telemetry and duel row for speculative `mtp.0` multi-token decoding.
#[derive(Debug, Clone, Default)]
pub struct Flash1017MtpDuelRow {
    pub mode_label: String,
    pub generated_ids: Vec<u32>,
    pub decoded_text: String,
    pub prefill_ms: f64,
    pub total_decode_ms: f64,
    pub total_draft_ms: f64,
    pub total_verify_ms: f64,
    pub total_head_ms: f64,
    pub verify_steps: usize,
    pub total_drafted_tokens: usize,
    pub total_accepted_drafts: usize,
    pub acceptance_rate_pct: f64,
    pub mean_tokens_per_step: f64,
    pub effective_ms_per_tok: f64,
    pub effective_tok_per_s: f64,
    pub speedup_vs_k0: f64,
    pub exact_match_k0: bool,
    pub tier_a_hits: usize,
    pub tier_b_hits: usize,
    pub tier_c_skips: usize,
    pub engram_lookups: usize,
}

impl Default for Flash1017KvCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Output head evaluation mode for `Flash1017Runtime` (`Fas F4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flash1017HeadMode {
    /// Baseline zero-copy `BF16` `[129280, 5120]` (`1.324 GB`, `~9.16 ms` on Blackwell GPU).
    Bf16,
    /// `F4.1`: Coalesced symmetric `Q8` `[129280, 5120]` (`662 MB`, `> 0.9999` cosine similarity).
    Q8,
    /// `F4.2`: Two-Stage `CoarseToFineConceptHead` (`C2FHead`: `[129280, 1024]` `Q8` GPU sketch + Top-2048 exact `BF16` re-scoring).
    CoarseToFine,
}

/// Side-by-side benchmark report for `Fas F4` (`F4.1` `Q8` Head, `F4.2` `CoarseToFineConceptHead`, and `F4.3` `mtp.0`).
#[derive(Debug, Clone, Default)]
pub struct Flash1017F4BenchReport {
    pub bf16_ms: f64,
    pub bf16_token: u32,
    pub bf16_logit: f32,
    pub q8_ms: f64,
    pub q8_token: u32,
    pub q8_logit: f32,
    pub q8_cosine_sim: f32,
    pub c2f_stage_a_ms: f64,
    pub c2f_stage_b_ms: f64,
    pub c2f_total_ms: f64,
    pub c2f_token: u32,
    pub c2f_logit: f32,
    pub mtp_draft_ms: f64,
    pub mtp_draft_token: u32,
    pub mtp_draft_logit: f32,
    pub matbatch_verify_2tok_ms: f64,
    pub seq_verify_2tok_ms: f64,
}

const MKC2F_MAGIC: &[u8; 8] = b"MKC2F\x02\x00\x00";
const MKC2F_HEADER_SIZE: usize = 128;

#[inline]
fn align_up_128(v: usize) -> usize {
    (v + 127) & !127
}

/// `F4.1` & `F4.2`: Clean-Room `CoarseToFineConceptHead` (`C2FHead`) + Coalesced `Q8` `head.weight` (`[129280, 5120]`),
/// backed by a persistent Zero-Copy `mmap` file (`merkaba_head_q8_c2f.bin`) with Linux `MADV_HUGEPAGE`.
pub struct CoarseToFineConceptHead {
    mmap: Mmap,
    pub vocab_rows: usize,
    pub hidden_cols: usize,
    pub sketch_cols: usize,
    q8_weights_offset: usize,
    q8_scales_offset: usize,
    sketch_q8_offset: usize,
    sketch_scales_offset: usize,
    pub sketch_col_indices: Vec<u16>,
}

/// Clean-room type alias (`GEMINI.md` Rule 5.2).
pub type C2FHead = CoarseToFineConceptHead;

impl CoarseToFineConceptHead {
    /// Opens `<model_dir>/merkaba_head_q8_c2f.bin` via Zero-Copy `mmap` in `< 1 ms` if present,
    /// or builds it in parallel across all CPU cores from `head_shard` (`model-00043-of-00048.safetensors`) on first run.
    pub fn load_or_build<P: AsRef<Path>>(
        model_dir: P,
        head_shard: &Flash1017ShardInspector,
        final_norm: &[f32],
    ) -> Result<Self, String> {
        let cache_path = model_dir.as_ref().join("merkaba_head_q8_c2f.bin");
        if cache_path.exists() {
            if let Ok(loaded) = Self::open_mmap(&cache_path) {
                return Ok(loaded);
            }
        }
        Self::build_and_save(&cache_path, head_shard, final_norm)?;
        Self::open_mmap(&cache_path)
    }

    fn open_mmap(path: &Path) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| format!("Failed to mmap {}: {}", path.display(), e))?
        };
        #[cfg(target_os = "linux")]
        unsafe {
            extern "C" {
                fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            let _ = madvise(
                mmap.as_ptr() as *mut std::ffi::c_void,
                mmap.len(),
                14, // MADV_HUGEPAGE
            );
        }
        crate::hardware_addon::register_host_mmap_bounds(mmap.as_ptr(), mmap.len());
        crate::hardware_addon::register_host_range(mmap.as_ptr(), mmap.len());
        if mmap.len() < MKC2F_HEADER_SIZE || &mmap[0..8] != MKC2F_MAGIC {
            return Err("Invalid MKC2F header".to_string());
        }
        let read_u64 = |off: usize| -> usize {
            u64::from_le_bytes([
                mmap[off],
                mmap[off + 1],
                mmap[off + 2],
                mmap[off + 3],
                mmap[off + 4],
                mmap[off + 5],
                mmap[off + 6],
                mmap[off + 7],
            ]) as usize
        };
        let vocab_rows = read_u64(8);
        let hidden_cols = read_u64(16);
        let sketch_cols = read_u64(24);
        let q8_weights_offset = read_u64(32);
        let q8_scales_offset = read_u64(40);
        let sketch_cols_idx_offset = read_u64(48);
        let sketch_q8_offset = read_u64(56);
        let sketch_scales_offset = read_u64(64);

        let mut sketch_col_indices = Vec::with_capacity(sketch_cols);
        for i in 0..sketch_cols {
            let b = sketch_cols_idx_offset + i * 2;
            sketch_col_indices.push(u16::from_le_bytes([mmap[b], mmap[b + 1]]));
        }

        Ok(Self {
            mmap,
            vocab_rows,
            hidden_cols,
            sketch_cols,
            q8_weights_offset,
            q8_scales_offset,
            sketch_q8_offset,
            sketch_scales_offset,
            sketch_col_indices,
        })
    }

    fn build_and_save(
        cache_path: &Path,
        head_shard: &Flash1017ShardInspector,
        final_norm: &[f32],
    ) -> Result<(), String> {
        use rayon::prelude::*;
        use std::io::{BufWriter, Write};

        let (meta, raw_bytes) = head_shard.get_tensor_data("head.weight")?;
        let vocab_rows = meta.shape[0];
        let hidden_cols = meta.shape[1];
        let sketch_cols = 1024usize;
        let row_bytes = hidden_cols * 2;

        // 1. Estimate column leverage `col_rms[c] * |final_norm[c]|` across a 512-row stride sample
        let mut col_energy = vec![0.0_f32; hidden_cols];
        let step = (vocab_rows / 512).max(1);
        for r in (0..vocab_rows).step_by(step) {
            let r_slice = &raw_bytes[r * row_bytes..(r + 1) * row_bytes];
            for c in 0..hidden_cols {
                let bits = u16::from_le_bytes([r_slice[c * 2], r_slice[c * 2 + 1]]);
                let w = f32::from_bits((bits as u32) << 16);
                let nw = final_norm.get(c).copied().unwrap_or(1.0);
                let prod = w * nw;
                col_energy[c] += prod * prod;
            }
        }
        let mut col_order: Vec<(usize, f32)> = col_energy.into_iter().enumerate().collect();
        col_order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let half_sketch = sketch_cols / 2;
        let mut selected = vec![false; hidden_cols];
        let mut sketch_col_indices: Vec<u16> = Vec::with_capacity(sketch_cols);
        for &(c, _) in col_order.iter().take(half_sketch) {
            selected[c] = true;
            sketch_col_indices.push(c as u16);
        }
        let remaining: Vec<usize> = (0..hidden_cols).filter(|&c| !selected[c]).collect();
        let stride = (remaining.len() / half_sketch.max(1)).max(1);
        for idx in (0..remaining.len()).step_by(stride) {
            if sketch_col_indices.len() >= sketch_cols {
                break;
            }
            sketch_col_indices.push(remaining[idx] as u16);
        }
        sketch_col_indices.sort_unstable();

        // 2. Quantize all 129 280 rows to symmetric Q8 ([129280, 5120]) AND 1024-col Q8 sketch ([129280, 1024]) in parallel
        let quantized_rows: Vec<(Vec<u8>, f32, Vec<u8>, f32)> = raw_bytes
            .par_chunks_exact(row_bytes)
            .with_min_len(64)
            .map(|r_slice| {
                let mut f32_row = vec![0.0_f32; hidden_cols];
                let mut max_abs = 0.0_f32;
                for c in 0..hidden_cols {
                    let bits = u16::from_le_bytes([r_slice[c * 2], r_slice[c * 2 + 1]]);
                    let w = f32::from_bits((bits as u32) << 16);
                    f32_row[c] = w;
                    let a = w.abs();
                    if a > max_abs {
                        max_abs = a;
                    }
                }
                let scale = if max_abs > 1e-12 { max_abs / 127.0_f32 } else { 1.0_f32 };
                let inv_scale = if max_abs > 1e-12 { 127.0_f32 / max_abs } else { 0.0_f32 };
                let mut q8_row = vec![0u8; hidden_cols];
                for c in 0..hidden_cols {
                    let q = (f32_row[c] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    q8_row[c] = q as u8;
                }

                let mut sk_max_abs = 0.0_f32;
                for &c_u16 in &sketch_col_indices {
                    let a = f32_row[c_u16 as usize].abs();
                    if a > sk_max_abs {
                        sk_max_abs = a;
                    }
                }
                let sk_scale = if sk_max_abs > 1e-12 {
                    sk_max_abs / 127.0_f32
                } else {
                    1.0_f32
                };
                let sk_inv = if sk_max_abs > 1e-12 {
                    127.0_f32 / sk_max_abs
                } else {
                    0.0_f32
                };
                let mut sk_row = vec![0u8; sketch_cols];
                for (i, &c_u16) in sketch_col_indices.iter().enumerate() {
                    let q = (f32_row[c_u16 as usize] * sk_inv).round().clamp(-127.0, 127.0) as i8;
                    sk_row[i] = q as u8;
                }

                (q8_row, scale, sk_row, sk_scale)
            })
            .collect();

        // 3. Compute 128-byte aligned file layout
        let q8_weights_offset = MKC2F_HEADER_SIZE;
        let q8_weights_len = vocab_rows * hidden_cols;
        let q8_scales_offset = align_up_128(q8_weights_offset + q8_weights_len);
        let q8_scales_len = vocab_rows * 4;
        let sketch_cols_idx_offset = align_up_128(q8_scales_offset + q8_scales_len);
        let sketch_cols_idx_len = sketch_cols * 2;
        let sketch_q8_offset = align_up_128(sketch_cols_idx_offset + sketch_cols_idx_len);
        let sketch_q8_len = vocab_rows * sketch_cols;
        let sketch_scales_offset = align_up_128(sketch_q8_offset + sketch_q8_len);
        let sketch_scales_len = vocab_rows * 4;
        let total_file_len = align_up_128(sketch_scales_offset + sketch_scales_len);

        let tmp_path = cache_path.with_extension("bin.tmp");
        let file = File::create(&tmp_path)
            .map_err(|e| format!("Failed to create {}: {}", tmp_path.display(), e))?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

        let mut header = [0u8; MKC2F_HEADER_SIZE];
        header[0..8].copy_from_slice(MKC2F_MAGIC);
        header[8..16].copy_from_slice(&(vocab_rows as u64).to_le_bytes());
        header[16..24].copy_from_slice(&(hidden_cols as u64).to_le_bytes());
        header[24..32].copy_from_slice(&(sketch_cols as u64).to_le_bytes());
        header[32..40].copy_from_slice(&(q8_weights_offset as u64).to_le_bytes());
        header[40..48].copy_from_slice(&(q8_scales_offset as u64).to_le_bytes());
        header[48..56].copy_from_slice(&(sketch_cols_idx_offset as u64).to_le_bytes());
        header[56..64].copy_from_slice(&(sketch_q8_offset as u64).to_le_bytes());
        header[64..72].copy_from_slice(&(sketch_scales_offset as u64).to_le_bytes());
        writer.write_all(&header).map_err(|e| e.to_string())?;

        // Write q8_weights
        for (q8_row, _, _, _) in &quantized_rows {
            writer.write_all(q8_row).map_err(|e| e.to_string())?;
        }
        let pad1 = q8_scales_offset - (q8_weights_offset + q8_weights_len);
        if pad1 > 0 {
            writer.write_all(&vec![0u8; pad1]).map_err(|e| e.to_string())?;
        }

        // Write q8_scales
        for (_, sc, _, _) in &quantized_rows {
            writer.write_all(&sc.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        let pad2 = sketch_cols_idx_offset - (q8_scales_offset + q8_scales_len);
        if pad2 > 0 {
            writer.write_all(&vec![0u8; pad2]).map_err(|e| e.to_string())?;
        }

        // Write sketch_col_indices
        for &c_u16 in &sketch_col_indices {
            writer.write_all(&c_u16.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        let pad3 = sketch_q8_offset - (sketch_cols_idx_offset + sketch_cols_idx_len);
        if pad3 > 0 {
            writer.write_all(&vec![0u8; pad3]).map_err(|e| e.to_string())?;
        }

        // Write sketch_q8
        for (_, _, sk_row, _) in &quantized_rows {
            writer.write_all(sk_row).map_err(|e| e.to_string())?;
        }
        let pad4 = sketch_scales_offset - (sketch_q8_offset + sketch_q8_len);
        if pad4 > 0 {
            writer.write_all(&vec![0u8; pad4]).map_err(|e| e.to_string())?;
        }

        // Write sketch_scales
        for (_, _, _, sk_sc) in &quantized_rows {
            writer.write_all(&sk_sc.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        let pad5 = total_file_len - (sketch_scales_offset + sketch_scales_len);
        if pad5 > 0 {
            writer.write_all(&vec![0u8; pad5]).map_err(|e| e.to_string())?;
        }
        writer.flush().map_err(|e| e.to_string())?;
        drop(writer);
        std::fs::rename(&tmp_path, cache_path)
            .map_err(|e| format!("Failed to rename {}: {}", tmp_path.display(), e))?;
        Ok(())
    }

    /// `F4.1`: Evaluates the coalesced symmetric `Q8` `head.weight` (`[129280, 5120]`, `662 MB`) on the Blackwell GPU
    /// via `HardwareAddonVTable::v1_matvec_q8` (or parallel CPU fallback) and returns all `129 280` logits.
    pub fn matvec_q8(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        use rayon::prelude::*;
        if x.len() != self.hidden_cols {
            return Err(format!(
                "Q8 head input mismatch: expected {}, got {}",
                self.hidden_cols,
                x.len()
            ));
        }
        let w_ptr = unsafe { self.mmap.as_ptr().add(self.q8_weights_offset) as *const i8 };
        let s_ptr = unsafe { self.mmap.as_ptr().add(self.q8_scales_offset) as *const f32 };

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(matvec_fn) = vt.matvec_q8 {
                let mut out = vec![0.0_f32; self.vocab_rows];
                unsafe {
                    matvec_fn(
                        self.vocab_rows,
                        self.hidden_cols,
                        w_ptr,
                        s_ptr,
                        x.as_ptr(),
                        out.as_mut_ptr(),
                    );
                }
                return Ok(out);
            }
        }

        let w_slice = &self.mmap[self.q8_weights_offset..self.q8_weights_offset + self.vocab_rows * self.hidden_cols];
        let s_slice = &self.mmap[self.q8_scales_offset..self.q8_scales_offset + self.vocab_rows * 4];
        let out: Vec<f32> = w_slice
            .par_chunks_exact(self.hidden_cols)
            .zip(s_slice.par_chunks_exact(4))
            .with_min_len(64)
            .map(|(r_w, r_s)| {
                let sc = f32::from_le_bytes([r_s[0], r_s[1], r_s[2], r_s[3]]);
                let mut sum = 0.0_f32;
                for c in 0..self.hidden_cols {
                    sum += (r_w[c] as i8 as f32) * x[c];
                }
                sum * sc
            })
            .collect();
        Ok(out)
    }

    /// `F4.2`: Executes Two-Stage `CoarseToFineConceptHead` (`C2FHead`):
    /// - **Stage A (Coarse GPU Sketch):** Evaluates the `[129280, 1024]` `Q8` sketch (`132.4 MB` = 10x smaller than `1.324 GB` `BF16`)
    ///   on the Blackwell GPU via `v1_matvec_q8` and selects the **Top-2048** candidate token IDs in $O(V)$.
    /// - **Stage B (Exact `BF16` Re-Scoring):** Computes the exact 5120-dim `BF16` dot product from `head_shard`
    ///   only on those `2048` candidate rows (`20.97 MB` instead of `1 324 MB`), returning `(best_token_id, best_bf16_logit, stage_a_ms, stage_b_ms)`.
    pub fn predict_coarse_to_fine(
        &self,
        head_shard: &Flash1017ShardInspector,
        x: &[f32],
        topk_candidates: usize,
    ) -> Result<(u32, f32, f64, f64), String> {
        use rayon::prelude::*;
        let t_a = std::time::Instant::now();
        let mut x_sketch = Vec::with_capacity(self.sketch_cols);
        for &c_u16 in &self.sketch_col_indices {
            x_sketch.push(x[c_u16 as usize]);
        }

        let sk_w_ptr = unsafe { self.mmap.as_ptr().add(self.sketch_q8_offset) as *const i8 };
        let sk_s_ptr = unsafe { self.mmap.as_ptr().add(self.sketch_scales_offset) as *const f32 };
        let mut coarse_scores = vec![0.0_f32; self.vocab_rows];

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(matvec_fn) = vt.matvec_q8 {
                unsafe {
                    matvec_fn(
                        self.vocab_rows,
                        self.sketch_cols,
                        sk_w_ptr,
                        sk_s_ptr,
                        x_sketch.as_ptr(),
                        coarse_scores.as_mut_ptr(),
                    );
                }
            } else {
                let w_slice = &self.mmap[self.sketch_q8_offset..self.sketch_q8_offset + self.vocab_rows * self.sketch_cols];
                let s_slice = &self.mmap[self.sketch_scales_offset..self.sketch_scales_offset + self.vocab_rows * 4];
                coarse_scores = w_slice
                    .par_chunks_exact(self.sketch_cols)
                    .zip(s_slice.par_chunks_exact(4))
                    .with_min_len(64)
                    .map(|(r_w, r_s)| {
                        let sc = f32::from_le_bytes([r_s[0], r_s[1], r_s[2], r_s[3]]);
                        let mut sum = 0.0_f32;
                        for c in 0..self.sketch_cols {
                            sum += (r_w[c] as i8 as f32) * x_sketch[c];
                        }
                        sum * sc
                    })
                    .collect();
            }
        } else {
            let w_slice = &self.mmap[self.sketch_q8_offset..self.sketch_q8_offset + self.vocab_rows * self.sketch_cols];
            let s_slice = &self.mmap[self.sketch_scales_offset..self.sketch_scales_offset + self.vocab_rows * 4];
            coarse_scores = w_slice
                .par_chunks_exact(self.sketch_cols)
                .zip(s_slice.par_chunks_exact(4))
                .with_min_len(64)
                .map(|(r_w, r_s)| {
                    let sc = f32::from_le_bytes([r_s[0], r_s[1], r_s[2], r_s[3]]);
                    let mut sum = 0.0_f32;
                    for c in 0..self.sketch_cols {
                        sum += (r_w[c] as i8 as f32) * x_sketch[c];
                    }
                    sum * sc
                })
                .collect();
        }

        let k = topk_candidates.clamp(512, self.vocab_rows);
        let core_anchor_count = 512usize.min(k / 2);
        let sketch_pick = k - core_anchor_count;
        let mut indexed: Vec<(u32, f32)> = coarse_scores
            .into_iter()
            .enumerate()
            .skip(core_anchor_count)
            .map(|(i, s)| (i as u32, s))
            .collect();
        if sketch_pick < indexed.len() && sketch_pick > 0 {
            indexed.select_nth_unstable_by(sketch_pick - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(sketch_pick);
        }
        let mut candidate_ids: Vec<u32> = Vec::with_capacity(k);
        for id in 0..(core_anchor_count as u32) {
            candidate_ids.push(id);
        }
        candidate_ids.extend(indexed.into_iter().map(|(id, _)| id));
        let stage_a_ms = t_a.elapsed().as_secs_f64() * 1000.0;

        let t_b = std::time::Instant::now();
        let (best_id, best_val, second_val) =
            head_shard.score_bf16_rows_top2("head.weight", x, &candidate_ids)?;
        let stage_b_ms = t_b.elapsed().as_secs_f64() * 1000.0;
        let _margin = (best_val - second_val).max(0.0);

        Ok((best_id, best_val, stage_a_ms, stage_b_ms))
    }

    /// `F4.2` & `F7.C2`: Same as `predict_coarse_to_fine`, also returning `margin = (best_logit - second_best_logit)`
    /// in `0.0 ms` extra time for `Adaptive MTP (1..3)`.
    pub fn predict_coarse_to_fine_with_margin(
        &self,
        head_shard: &Flash1017ShardInspector,
        x: &[f32],
        topk_candidates: usize,
    ) -> Result<(u32, f32, f32, f64, f64), String> {
        use rayon::prelude::*;
        let t_a = std::time::Instant::now();
        let mut x_sketch = Vec::with_capacity(self.sketch_cols);
        for &c_u16 in &self.sketch_col_indices {
            x_sketch.push(x[c_u16 as usize]);
        }

        let sk_w_ptr = unsafe { self.mmap.as_ptr().add(self.sketch_q8_offset) as *const i8 };
        let sk_s_ptr = unsafe { self.mmap.as_ptr().add(self.sketch_scales_offset) as *const f32 };
        let mut coarse_scores = vec![0.0_f32; self.vocab_rows];

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(matvec_fn) = vt.matvec_q8 {
                unsafe {
                    matvec_fn(
                        self.vocab_rows,
                        self.sketch_cols,
                        sk_w_ptr,
                        sk_s_ptr,
                        x_sketch.as_ptr(),
                        coarse_scores.as_mut_ptr(),
                    );
                }
            } else {
                let w_slice = &self.mmap[self.sketch_q8_offset
                    ..self.sketch_q8_offset + self.vocab_rows * self.sketch_cols];
                let s_slice = &self.mmap
                    [self.sketch_scales_offset..self.sketch_scales_offset + self.vocab_rows * 4];
                coarse_scores = w_slice
                    .par_chunks_exact(self.sketch_cols)
                    .zip(s_slice.par_chunks_exact(4))
                    .with_min_len(64)
                    .map(|(r_w, r_s)| {
                        let sc = f32::from_le_bytes([r_s[0], r_s[1], r_s[2], r_s[3]]);
                        let mut sum = 0.0_f32;
                        for c in 0..self.sketch_cols {
                            sum += (r_w[c] as i8 as f32) * x_sketch[c];
                        }
                        sum * sc
                    })
                    .collect();
            }
        } else {
            let w_slice = &self.mmap
                [self.sketch_q8_offset..self.sketch_q8_offset + self.vocab_rows * self.sketch_cols];
            let s_slice = &self.mmap
                [self.sketch_scales_offset..self.sketch_scales_offset + self.vocab_rows * 4];
            coarse_scores = w_slice
                .par_chunks_exact(self.sketch_cols)
                .zip(s_slice.par_chunks_exact(4))
                .with_min_len(64)
                .map(|(r_w, r_s)| {
                    let sc = f32::from_le_bytes([r_s[0], r_s[1], r_s[2], r_s[3]]);
                    let mut sum = 0.0_f32;
                    for c in 0..self.sketch_cols {
                        sum += (r_w[c] as i8 as f32) * x_sketch[c];
                    }
                    sum * sc
                })
                .collect();
        }

        let k = topk_candidates.clamp(512, self.vocab_rows);
        let core_anchor_count = 512usize.min(k / 2);
        let sketch_pick = k - core_anchor_count;
        let mut indexed: Vec<(u32, f32)> = coarse_scores
            .into_iter()
            .enumerate()
            .skip(core_anchor_count)
            .map(|(i, s)| (i as u32, s))
            .collect();
        if sketch_pick < indexed.len() && sketch_pick > 0 {
            indexed.select_nth_unstable_by(sketch_pick - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(sketch_pick);
        }
        let mut candidate_ids: Vec<u32> = Vec::with_capacity(k);
        for id in 0..(core_anchor_count as u32) {
            candidate_ids.push(id);
        }
        candidate_ids.extend(indexed.into_iter().map(|(id, _)| id));
        let stage_a_ms = t_a.elapsed().as_secs_f64() * 1000.0;

        let t_b = std::time::Instant::now();
        let (best_id, best_val, second_val) =
            head_shard.score_bf16_rows_top2("head.weight", x, &candidate_ids)?;
        let stage_b_ms = t_b.elapsed().as_secs_f64() * 1000.0;
        let margin = if second_val.is_finite() {
            (best_val - second_val).max(0.0)
        } else {
            0.0
        };

        Ok((best_id, best_val, margin, stage_a_ms, stage_b_ms))
    }

    /// Stage A candidate selector for speculative `DSpark` drafting (`topk_candidates` + `extra_ids`).
    pub fn select_candidate_ids(
        &self,
        x: &[f32],
        topk_candidates: usize,
        extra_ids: &[u32],
    ) -> Vec<u32> {
        use rayon::prelude::*;
        let mut x_sketch = Vec::with_capacity(self.sketch_cols);
        for &c_u16 in &self.sketch_col_indices {
            x_sketch.push(x[c_u16 as usize]);
        }

        let sk_w_ptr = unsafe { self.mmap.as_ptr().add(self.sketch_q8_offset) as *const i8 };
        let sk_s_ptr = unsafe { self.mmap.as_ptr().add(self.sketch_scales_offset) as *const f32 };
        let mut coarse_scores = vec![0.0_f32; self.vocab_rows];

        let mut gpu_done = false;
        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(matvec_fn) = vt.matvec_q8 {
                unsafe {
                    matvec_fn(
                        self.vocab_rows,
                        self.sketch_cols,
                        sk_w_ptr,
                        sk_s_ptr,
                        x_sketch.as_ptr(),
                        coarse_scores.as_mut_ptr(),
                    );
                }
                gpu_done = true;
            }
        }
        if !gpu_done {
            let w_slice = &self.mmap
                [self.sketch_q8_offset..self.sketch_q8_offset + self.vocab_rows * self.sketch_cols];
            let s_slice = &self.mmap
                [self.sketch_scales_offset..self.sketch_scales_offset + self.vocab_rows * 4];
            coarse_scores = w_slice
                .par_chunks_exact(self.sketch_cols)
                .zip(s_slice.par_chunks_exact(4))
                .with_min_len(64)
                .map(|(r_w, r_s)| {
                    let sc = f32::from_le_bytes([r_s[0], r_s[1], r_s[2], r_s[3]]);
                    let mut sum = 0.0_f32;
                    for c in 0..self.sketch_cols {
                        sum += (r_w[c] as i8 as f32) * x_sketch[c];
                    }
                    sum * sc
                })
                .collect();
        }

        let k = topk_candidates.clamp(512, self.vocab_rows);
        let core_anchor_count = 512usize.min(k / 2);
        let sketch_pick = k - core_anchor_count;
        let tail = &coarse_scores[core_anchor_count..];
        let mut indexed: Vec<(u32, f32)> = tail
            .iter()
            .enumerate()
            .map(|(i, &s)| ((core_anchor_count + i) as u32, s))
            .collect();
        if sketch_pick < indexed.len() && sketch_pick > 0 {
            indexed.select_nth_unstable_by(sketch_pick - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(sketch_pick);
        }
        let mut candidate_ids: Vec<u32> = Vec::with_capacity(k + extra_ids.len());
        for id in 0..(core_anchor_count as u32) {
            candidate_ids.push(id);
        }
        candidate_ids.extend(indexed.into_iter().map(|(id, _)| id));
        for &ext in extra_ids {
            if (ext as usize) < self.vocab_rows && !candidate_ids.contains(&ext) {
                candidate_ids.push(ext);
            }
        }
        candidate_ids
    }

    /// `F8.3`: Batched Two-Stage `CoarseToFineConceptHead` (`C2FHead`) for `1 + K` verified tokens:
    /// - **Stage A:** Evaluates the `[129280, 1024]` `Q8` sketch (`132.4 MB`) for all `b_len` tokens in **1 single `v2_matbatch_q8` GPU call**
    ///   (reading the `132.4 MB` sketch matrix only ONCE from Unified Memory instead of `b_len` times!).
    /// - **Stage B:** Selects the Top-`topk_candidates` rows per token and computes the exact 8-way unrolled `BF16` dot products.
    pub fn predict_coarse_to_fine_batch_with_margin(
        &self,
        head_shard: &Flash1017ShardInspector,
        x_batch: &[&[f32]],
        topk_candidates: usize,
    ) -> Result<Vec<(u32, f32, f32)>, String> {
        let b_len = x_batch.len();
        if b_len == 0 {
            return Ok(Vec::new());
        }
        if b_len == 1 {
            let (id, val, margin, _, _) =
                self.predict_coarse_to_fine_with_margin(head_shard, x_batch[0], topk_candidates)?;
            return Ok(vec![(id, val, margin)]);
        }

        let mut x_sketch_batch = Vec::with_capacity(b_len * self.sketch_cols);
        for &x in x_batch {
            for &c_u16 in &self.sketch_col_indices {
                x_sketch_batch.push(x[c_u16 as usize]);
            }
        }

        let sk_w_ptr = unsafe { self.mmap.as_ptr().add(self.sketch_q8_offset) as *const i8 };
        let sk_s_ptr = unsafe { self.mmap.as_ptr().add(self.sketch_scales_offset) as *const f32 };
        let mut coarse_scores_batch = vec![0.0_f32; b_len * self.vocab_rows];
        let mut batched_ok = false;

        if let Some(vt) = crate::hardware_addon::active_vtable() {
            if let Some(mb_q8) = vt.v2_matbatch_q8() {
                unsafe {
                    mb_q8(
                        self.vocab_rows,
                        self.sketch_cols,
                        b_len,
                        sk_w_ptr,
                        sk_s_ptr,
                        x_sketch_batch.as_ptr(),
                        coarse_scores_batch.as_mut_ptr(),
                    );
                }
                batched_ok = true;
            }
        }

        if !batched_ok {
            let mut results = Vec::with_capacity(b_len);
            for &x in x_batch {
                let (id, val, margin, _, _) =
                    self.predict_coarse_to_fine_with_margin(head_shard, x, topk_candidates)?;
                results.push((id, val, margin));
            }
            return Ok(results);
        }

        let k = topk_candidates.clamp(512, self.vocab_rows);
        let core_anchor_count = 512usize.min(k / 2);
        let sketch_pick = k - core_anchor_count;
        let mut results = Vec::with_capacity(b_len);
        let mut indexed: Vec<(u32, f32)> = Vec::with_capacity(sketch_pick * 3);
        let mut sample_buf: Vec<f32> = Vec::with_capacity(4096);

        for (t, &x) in x_batch.iter().enumerate() {
            let coarse_scores =
                &coarse_scores_batch[t * self.vocab_rows..(t + 1) * self.vocab_rows];
            let tail = &coarse_scores[core_anchor_count..];
            indexed.clear();

            if sketch_pick > 0 && tail.len() > sketch_pick * 4 {
                sample_buf.clear();
                for &s in tail.iter().step_by(32) {
                    sample_buf.push(s);
                }
                let target_sample_idx = ((sketch_pick * 2) / 32).min(sample_buf.len().saturating_sub(1));
                sample_buf.select_nth_unstable_by(target_sample_idx, |a, b| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
                let threshold = sample_buf[target_sample_idx];
                for (i, &s) in tail.iter().enumerate() {
                    if s >= threshold {
                        indexed.push(((core_anchor_count + i) as u32, s));
                    }
                }
            }
            if indexed.len() < sketch_pick {
                indexed.clear();
                indexed.extend(
                    tail.iter()
                        .enumerate()
                        .map(|(i, &s)| ((core_anchor_count + i) as u32, s)),
                );
            }
            if sketch_pick < indexed.len() && sketch_pick > 0 {
                indexed.select_nth_unstable_by(sketch_pick - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                indexed.truncate(sketch_pick);
            }
            let mut candidate_ids: Vec<u32> = Vec::with_capacity(k);
            for id in 0..(core_anchor_count as u32) {
                candidate_ids.push(id);
            }
            candidate_ids.extend(indexed.iter().map(|&(id, _)| id));

            let (best_id, best_val, second_val) =
                head_shard.score_bf16_rows_top2("head.weight", x, &candidate_ids)?;
            let margin = if second_val.is_finite() {
                (best_val - second_val).max(0.0)
            } else {
                0.0
            };
            results.push((best_id, best_val, margin));
        }

        Ok(results)
    }
}

/// `F9.2`: Additional MTP transformer sub-layer (`mtp.1` in `model-00045` or `mtp.2` in `model-00046`)
/// for the complete 3-layer `DSpark` (`mtp.0 -> mtp.1 -> mtp.2`) speculative predictor.
pub struct Flash1017MtpSubLayer {
    pub layer_idx: usize,
    pub shard: Flash1017ShardInspector,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub attn_sink: Vec<f32>,
    pub hc_attn_base: [f32; 24],
    pub hc_attn_scale: [f32; 3],
    pub hc_ffn_base: [f32; 24],
    pub hc_ffn_scale: [f32; 3],
    pub hc_attn_fn_ptr: *const f32,
    pub hc_ffn_fn_ptr: *const f32,
    pub _hc_aligned: Option<(Vec<f32>, Vec<f32>)>,
    pub gate_bias: Vec<f32>,
    pub wq_a: CachedMxfp8Mat,
    pub wq_b: CachedMxfp8Mat,
    pub wkv: CachedMxfp8Mat,
    pub wo_a: CachedMxfp8Mat,
    pub wo_b: CachedMxfp8Mat,
    pub ffn_gate: CachedBf16Mat,
    pub shared_w1: CachedMxfp8Mat,
    pub shared_w2: CachedMxfp8Mat,
    pub shared_w3: CachedMxfp8Mat,
    pub q4_backbone_mmap: Option<Mmap>,
    pub q4_backbone: Option<CachedQ4LayerBackbone>,
    pub mxfp4_experts: Vec<(CachedMxfp4Mat, CachedMxfp4Mat, CachedMxfp4Mat)>,
}
unsafe impl Send for Flash1017MtpSubLayer {}
unsafe impl Sync for Flash1017MtpSubLayer {}

impl Flash1017MtpSubLayer {
    pub fn open(model_dir_ref: &Path, layer_idx: usize, shard_filename: &str) -> Result<Self, String> {
        let p_shard = model_dir_ref.join(shard_filename);
        let shard = Flash1017ShardInspector::open(&p_shard)?;
        let pfx = format!("mtp.{}", layer_idx);
        let attn_norm = shard.extract_bf16_vec(&format!("{}.attn_norm.weight", pfx))?;
        let ffn_norm = shard.extract_bf16_vec(&format!("{}.ffn_norm.weight", pfx))?;
        let q_norm = shard.extract_bf16_vec(&format!("{}.attn.q_norm.weight", pfx))?;
        let kv_norm = shard.extract_bf16_vec(&format!("{}.attn.kv_norm.weight", pfx))?;
        let attn_sink = shard.extract_f32_vec(&format!("{}.attn.attn_sink", pfx))?;
        let gate_bias = shard.extract_f32_vec(&format!("{}.ffn.gate.bias", pfx))?;

        let hc_ab_vec = shard.extract_f32_vec(&format!("{}.hc_attn_base", pfx))?;
        let hc_as_vec = shard.extract_f32_vec(&format!("{}.hc_attn_scale", pfx))?;
        let hc_fb_vec = shard.extract_f32_vec(&format!("{}.hc_ffn_base", pfx))?;
        let hc_fs_vec = shard.extract_f32_vec(&format!("{}.hc_ffn_scale", pfx))?;

        let mut hc_attn_base = [0.0_f32; 24];
        hc_attn_base.copy_from_slice(&hc_ab_vec[..24]);
        let mut hc_attn_scale = [0.0_f32; 3];
        hc_attn_scale.copy_from_slice(&hc_as_vec[..3]);
        let mut hc_ffn_base = [0.0_f32; 24];
        hc_ffn_base.copy_from_slice(&hc_fb_vec[..24]);
        let mut hc_ffn_scale = [0.0_f32; 3];
        hc_ffn_scale.copy_from_slice(&hc_fs_vec[..3]);

        let (_, hc_a_bytes) = shard.get_tensor_data(&format!("{}.hc_attn_fn", pfx))?;
        let (_, hc_f_bytes) = shard.get_tensor_data(&format!("{}.hc_ffn_fn", pfx))?;
        let (hc_attn_fn_ptr, hc_ffn_fn_ptr, _hc_aligned) =
            if (hc_a_bytes.as_ptr() as usize) % 4 == 0 && (hc_f_bytes.as_ptr() as usize) % 4 == 0 {
                crate::hardware_addon::register_host_range(hc_a_bytes.as_ptr(), hc_a_bytes.len());
                crate::hardware_addon::register_host_range(hc_f_bytes.as_ptr(), hc_f_bytes.len());
                (
                    hc_a_bytes.as_ptr() as *const f32,
                    hc_f_bytes.as_ptr() as *const f32,
                    None,
                )
            } else {
                let na = hc_a_bytes.len() / 4;
                let nf = hc_f_bytes.len() / 4;
                let mut va = Vec::with_capacity(na);
                for i in 0..na {
                    let b = i * 4;
                    va.push(f32::from_le_bytes([
                        hc_a_bytes[b],
                        hc_a_bytes[b + 1],
                        hc_a_bytes[b + 2],
                        hc_a_bytes[b + 3],
                    ]));
                }
                let mut vf = Vec::with_capacity(nf);
                for i in 0..nf {
                    let b = i * 4;
                    vf.push(f32::from_le_bytes([
                        hc_f_bytes[b],
                        hc_f_bytes[b + 1],
                        hc_f_bytes[b + 2],
                        hc_f_bytes[b + 3],
                    ]));
                }
                crate::hardware_addon::register_host_range(
                    va.as_ptr() as *const u8,
                    hc_a_bytes.len(),
                );
                crate::hardware_addon::register_host_range(
                    vf.as_ptr() as *const u8,
                    hc_f_bytes.len(),
                );
                let pa = va.as_ptr();
                let pf = vf.as_ptr();
                (pa, pf, Some((va, vf)))
            };

        let wq_a = CachedMxfp8Mat::from_shard(&shard, &format!("{}.attn.wq_a", pfx))?;
        let wq_b = CachedMxfp8Mat::from_shard(&shard, &format!("{}.attn.wq_b", pfx))?;
        let wkv = CachedMxfp8Mat::from_shard(&shard, &format!("{}.attn.wkv", pfx))?;
        let wo_a = CachedMxfp8Mat::from_shard(&shard, &format!("{}.attn.wo_a", pfx))?;
        let wo_b = CachedMxfp8Mat::from_shard(&shard, &format!("{}.attn.wo_b", pfx))?;
        let ffn_gate = CachedBf16Mat::from_shard(&shard, &format!("{}.ffn.gate.weight", pfx))?;
        let shared_w1 = CachedMxfp8Mat::from_shard(&shard, &format!("{}.ffn.shared_experts.w1", pfx))?;
        let shared_w2 = CachedMxfp8Mat::from_shard(&shard, &format!("{}.ffn.shared_experts.w2", pfx))?;
        let shared_w3 = CachedMxfp8Mat::from_shard(&shard, &format!("{}.ffn.shared_experts.w3", pfx))?;

        let sidecar_name = format!("merkaba_mtp{}_q4.bin", layer_idx);
        let (q4_backbone_mmap, q4_backbone) = match load_or_build_mtp_named_q4_sidecar(
            model_dir_ref,
            &sidecar_name,
            &shard,
            &wq_b,
            &wo_a,
            &wo_b,
            &shared_w1,
            &shared_w3,
            &shared_w2,
        ) {
            Ok((m, q4)) => (Some(m), Some(q4)),
            Err(_) => (None, None),
        };
        crate::hardware_addon::register_host_mmap_bounds(shard.mmap.as_ptr(), shard.mmap.len());
        let reg_mxfp8 = |m: &CachedMxfp8Mat| {
            crate::hardware_addon::register_host_range(m.w_ptr, m.w_len);
            crate::hardware_addon::register_host_range(m.s_ptr, m.s_len);
        };
        reg_mxfp8(&wq_a);
        reg_mxfp8(&wkv);
        if q4_backbone.is_none() {
            reg_mxfp8(&wq_b);
            reg_mxfp8(&wo_a);
            reg_mxfp8(&wo_b);
            reg_mxfp8(&shared_w1);
            reg_mxfp8(&shared_w2);
            reg_mxfp8(&shared_w3);
        }
        crate::hardware_addon::register_host_range(ffn_gate.w_ptr as *const u8, ffn_gate.w_len);

        let mut mxfp4_experts = Vec::with_capacity(128);
        for e in 0..128 {
            let w1 = CachedMxfp4Mat::from_shard(&shard, &format!("{}.ffn.experts.{}.w1", pfx, e))?;
            let w2 = CachedMxfp4Mat::from_shard(&shard, &format!("{}.ffn.experts.{}.w2", pfx, e))?;
            let w3 = CachedMxfp4Mat::from_shard(&shard, &format!("{}.ffn.experts.{}.w3", pfx, e))?;
            for m in [&w1, &w2, &w3] {
                crate::hardware_addon::register_host_range(m.w_ptr, m.w_len);
                crate::hardware_addon::register_host_range(m.s_ptr, m.s_len);
            }
            mxfp4_experts.push((w1, w2, w3));
        }

        Ok(Self {
            layer_idx,
            shard,
            attn_norm,
            ffn_norm,
            q_norm,
            kv_norm,
            attn_sink,
            hc_attn_base,
            hc_attn_scale,
            hc_ffn_base,
            hc_ffn_scale,
            hc_attn_fn_ptr,
            hc_ffn_fn_ptr,
            _hc_aligned,
            gate_bias,
            wq_a,
            wq_b,
            wkv,
            wo_a,
            wo_b,
            ffn_gate,
            shared_w1,
            shared_w2,
            shared_w3,
            q4_backbone_mmap,
            q4_backbone,
            mxfp4_experts,
        })
    }
}

/// `F4.3` & `F9.2`: Speculative Multi-Token Prediction (`mtp.0 -> mtp.1 -> mtp.2` / `dspark` + `mtp.2` Markov & Confidence heads) Layer
/// from `model-00044`, `model-00045`, and `model-00046` (or `merkaba_dspark_heads.safetensors`).
pub struct Flash1017MtpLayer {
    pub shard: Flash1017ShardInspector,
    pub main_norm: Vec<f32>,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub attn_sink: Vec<f32>,
    pub hc_attn_base: [f32; 24],
    pub hc_attn_scale: [f32; 3],
    pub hc_ffn_base: [f32; 24],
    pub hc_ffn_scale: [f32; 3],
    pub hc_attn_fn_ptr: *const f32,
    pub hc_ffn_fn_ptr: *const f32,
    pub _hc_aligned: Option<(Vec<f32>, Vec<f32>)>,
    pub gate_bias: Vec<f32>,
    pub main_proj: CachedMxfp8Mat,
    pub wq_a: CachedMxfp8Mat,
    pub wq_b: CachedMxfp8Mat,
    pub wkv: CachedMxfp8Mat,
    pub wo_a: CachedMxfp8Mat,
    pub wo_b: CachedMxfp8Mat,
    pub ffn_gate: CachedBf16Mat,
    pub shared_w1: CachedMxfp8Mat,
    pub shared_w2: CachedMxfp8Mat,
    pub shared_w3: CachedMxfp8Mat,
    pub q4_backbone_mmap: Option<Mmap>,
    pub q4_backbone: Option<CachedQ4LayerBackbone>,
    pub dspark_heads_shard: Option<Flash1017ShardInspector>,
    pub mtp2_norm: Option<Vec<f32>>,
    pub confidence_proj: Option<Vec<f32>>,
    pub markov_embed: Option<CachedBf16Mat>,
    pub markov_head: Option<CachedBf16Mat>,
    pub mxfp4_experts: Vec<(CachedMxfp4Mat, CachedMxfp4Mat, CachedMxfp4Mat)>,
    pub extra_sub_layers: Vec<Flash1017MtpSubLayer>,
}
unsafe impl Send for Flash1017MtpLayer {}
unsafe impl Sync for Flash1017MtpLayer {}

impl Flash1017MtpLayer {
    #[inline]
    pub fn num_active_sub_layers(&self) -> usize {
        1 + self.extra_sub_layers.len()
    }

    /// Opens `model-00044-of-00048.safetensors` (`2.5 GB`), `model-00045` (`mtp.1`), `model-00046` (`mtp.2`),
    /// and 4-bit NVFP4 sidecars (`merkaba_mtp0..2_q4.bin`) via Zero-Copy `mmap`.
    pub fn open<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let model_dir_ref = model_dir.as_ref();
        let p44 = model_dir_ref.join("model-00044-of-00048.safetensors");
        let shard = Flash1017ShardInspector::open(&p44)?;
        let main_norm = shard.extract_bf16_vec("mtp.0.main_norm.weight")?;
        let attn_norm = shard.extract_bf16_vec("mtp.0.attn_norm.weight")?;
        let ffn_norm = shard.extract_bf16_vec("mtp.0.ffn_norm.weight")?;
        let q_norm = shard.extract_bf16_vec("mtp.0.attn.q_norm.weight")?;
        let kv_norm = shard.extract_bf16_vec("mtp.0.attn.kv_norm.weight")?;
        let attn_sink = shard.extract_f32_vec("mtp.0.attn.attn_sink")?;
        let gate_bias = shard.extract_f32_vec("mtp.0.ffn.gate.bias")?;

        let hc_ab_vec = shard.extract_f32_vec("mtp.0.hc_attn_base")?;
        let hc_as_vec = shard.extract_f32_vec("mtp.0.hc_attn_scale")?;
        let hc_fb_vec = shard.extract_f32_vec("mtp.0.hc_ffn_base")?;
        let hc_fs_vec = shard.extract_f32_vec("mtp.0.hc_ffn_scale")?;

        let mut hc_attn_base = [0.0_f32; 24];
        hc_attn_base.copy_from_slice(&hc_ab_vec[..24]);
        let mut hc_attn_scale = [0.0_f32; 3];
        hc_attn_scale.copy_from_slice(&hc_as_vec[..3]);
        let mut hc_ffn_base = [0.0_f32; 24];
        hc_ffn_base.copy_from_slice(&hc_fb_vec[..24]);
        let mut hc_ffn_scale = [0.0_f32; 3];
        hc_ffn_scale.copy_from_slice(&hc_fs_vec[..3]);

        let (_, hc_a_bytes) = shard.get_tensor_data("mtp.0.hc_attn_fn")?;
        let (_, hc_f_bytes) = shard.get_tensor_data("mtp.0.hc_ffn_fn")?;
        let (hc_attn_fn_ptr, hc_ffn_fn_ptr, _hc_aligned) =
            if (hc_a_bytes.as_ptr() as usize) % 4 == 0 && (hc_f_bytes.as_ptr() as usize) % 4 == 0 {
                crate::hardware_addon::register_host_range(hc_a_bytes.as_ptr(), hc_a_bytes.len());
                crate::hardware_addon::register_host_range(hc_f_bytes.as_ptr(), hc_f_bytes.len());
                (
                    hc_a_bytes.as_ptr() as *const f32,
                    hc_f_bytes.as_ptr() as *const f32,
                    None,
                )
            } else {
                let na = hc_a_bytes.len() / 4;
                let nf = hc_f_bytes.len() / 4;
                let mut va = Vec::with_capacity(na);
                for i in 0..na {
                    let b = i * 4;
                    va.push(f32::from_le_bytes([
                        hc_a_bytes[b],
                        hc_a_bytes[b + 1],
                        hc_a_bytes[b + 2],
                        hc_a_bytes[b + 3],
                    ]));
                }
                let mut vf = Vec::with_capacity(nf);
                for i in 0..nf {
                    let b = i * 4;
                    vf.push(f32::from_le_bytes([
                        hc_f_bytes[b],
                        hc_f_bytes[b + 1],
                        hc_f_bytes[b + 2],
                        hc_f_bytes[b + 3],
                    ]));
                }
                crate::hardware_addon::register_host_range(
                    va.as_ptr() as *const u8,
                    hc_a_bytes.len(),
                );
                crate::hardware_addon::register_host_range(
                    vf.as_ptr() as *const u8,
                    hc_f_bytes.len(),
                );
                let pa = va.as_ptr();
                let pf = vf.as_ptr();
                (pa, pf, Some((va, vf)))
            };

        let main_proj = CachedMxfp8Mat::from_shard(&shard, "mtp.0.main_proj")?;
        let wq_a = CachedMxfp8Mat::from_shard(&shard, "mtp.0.attn.wq_a")?;
        let wq_b = CachedMxfp8Mat::from_shard(&shard, "mtp.0.attn.wq_b")?;
        let wkv = CachedMxfp8Mat::from_shard(&shard, "mtp.0.attn.wkv")?;
        let wo_a = CachedMxfp8Mat::from_shard(&shard, "mtp.0.attn.wo_a")?;
        let wo_b = CachedMxfp8Mat::from_shard(&shard, "mtp.0.attn.wo_b")?;
        let ffn_gate = CachedBf16Mat::from_shard(&shard, "mtp.0.ffn.gate.weight")?;
        let shared_w1 = CachedMxfp8Mat::from_shard(&shard, "mtp.0.ffn.shared_experts.w1")?;
        let shared_w2 = CachedMxfp8Mat::from_shard(&shard, "mtp.0.ffn.shared_experts.w2")?;
        let shared_w3 = CachedMxfp8Mat::from_shard(&shard, "mtp.0.ffn.shared_experts.w3")?;

        let (q4_backbone_mmap, q4_backbone) = match load_or_build_mtp0_q4_sidecar(
            model_dir_ref,
            &shard,
            &wq_b,
            &wo_a,
            &wo_b,
            &shared_w1,
            &shared_w3,
            &shared_w2,
        ) {
            Ok((m, q4)) => (Some(m), Some(q4)),
            Err(_) => (None, None),
        };
        crate::hardware_addon::register_host_mmap_bounds(shard.mmap.as_ptr(), shard.mmap.len());
        let reg_mxfp8 = |m: &CachedMxfp8Mat| {
            crate::hardware_addon::register_host_range(m.w_ptr, m.w_len);
            crate::hardware_addon::register_host_range(m.s_ptr, m.s_len);
        };
        reg_mxfp8(&main_proj);
        reg_mxfp8(&wq_a);
        reg_mxfp8(&wkv);
        if q4_backbone.is_none() {
            reg_mxfp8(&wq_b);
            reg_mxfp8(&wo_a);
            reg_mxfp8(&wo_b);
            reg_mxfp8(&shared_w1);
            reg_mxfp8(&shared_w2);
            reg_mxfp8(&shared_w3);
        }
        crate::hardware_addon::register_host_range(ffn_gate.w_ptr as *const u8, ffn_gate.w_len);

        let mut extra_sub_layers = Vec::with_capacity(2);
        let p_45 = model_dir_ref.join("model-00045-of-00048.safetensors");
        let p_46 = model_dir_ref.join("model-00046-of-00048.safetensors");
        if p_45.exists() && p_46.exists() {
            if let (Ok(sub1), Ok(sub2)) = (
                Flash1017MtpSubLayer::open(model_dir_ref, 1, "model-00045-of-00048.safetensors"),
                Flash1017MtpSubLayer::open(model_dir_ref, 2, "model-00046-of-00048.safetensors"),
            ) {
                extra_sub_layers.push(sub1);
                extra_sub_layers.push(sub2);
            }
        }

        let mut dspark_heads_shard = None;
        let mut mtp2_norm = None;
        let mut confidence_proj = None;
        let mut markov_embed = None;
        let mut markov_head = None;
        if extra_sub_layers.len() == 2 {
            let ds = &extra_sub_layers[1].shard;
            mtp2_norm = ds.extract_bf16_vec("mtp.2.norm.weight").ok();
            confidence_proj = ds.extract_bf16_vec("mtp.2.confidence_head.proj.weight").ok();
            markov_embed = CachedBf16Mat::from_shard(ds, "mtp.2.markov_head.embed.weight").ok();
            markov_head = CachedBf16Mat::from_shard(ds, "mtp.2.markov_head.head.weight").ok();
        } else {
            let dspark_path = {
                let p_side = model_dir_ref.join("merkaba_dspark_heads.safetensors");
                if p_side.exists() {
                    Some(p_side)
                } else if p_46.exists() {
                    Some(p_46)
                } else {
                    None
                }
            };
            if let Some(dp) = dspark_path {
                if let Ok(ds) = Flash1017ShardInspector::open(&dp) {
                    mtp2_norm = ds.extract_bf16_vec("mtp.2.norm.weight").ok();
                    confidence_proj = ds.extract_bf16_vec("mtp.2.confidence_head.proj.weight").ok();
                    markov_embed = CachedBf16Mat::from_shard(&ds, "mtp.2.markov_head.embed.weight").ok();
                    markov_head = CachedBf16Mat::from_shard(&ds, "mtp.2.markov_head.head.weight").ok();
                    dspark_heads_shard = Some(ds);
                }
            }
        }

        let mut mxfp4_experts = Vec::with_capacity(128);
        for e in 0..128 {
            let w1 = CachedMxfp4Mat::from_shard(&shard, &format!("mtp.0.ffn.experts.{}.w1", e))?;
            let w2 = CachedMxfp4Mat::from_shard(&shard, &format!("mtp.0.ffn.experts.{}.w2", e))?;
            let w3 = CachedMxfp4Mat::from_shard(&shard, &format!("mtp.0.ffn.experts.{}.w3", e))?;
            for m in [&w1, &w2, &w3] {
                crate::hardware_addon::register_host_range(m.w_ptr, m.w_len);
                crate::hardware_addon::register_host_range(m.s_ptr, m.s_len);
            }
            mxfp4_experts.push((w1, w2, w3));
        }

        Ok(Self {
            shard,
            main_norm,
            attn_norm,
            ffn_norm,
            q_norm,
            kv_norm,
            attn_sink,
            hc_attn_base,
            hc_attn_scale,
            hc_ffn_base,
            hc_ffn_scale,
            hc_attn_fn_ptr,
            hc_ffn_fn_ptr,
            _hc_aligned,
            gate_bias,
            main_proj,
            wq_a,
            wq_b,
            wkv,
            wo_a,
            wo_b,
            ffn_gate,
            shared_w1,
            shared_w2,
            shared_w3,
            q4_backbone_mmap,
            q4_backbone,
            dspark_heads_shard,
            mtp2_norm,
            confidence_proj,
            markov_embed,
            markov_head,
            mxfp4_experts,
            extra_sub_layers,
        })
    }

    /// `F9.2`: Extracts the 256-dim `f32` `markov_embed[prev_tok_id]` vector in `< 0.2 µs` for fast multi-candidate scoring.
    #[inline]
    pub fn extract_markov_embed(&self, prev_tok_id: u32) -> Option<[f32; 256]> {
        let emb = self.markov_embed.as_ref()?;
        if emb.cols < 256 || emb.rows == 0 {
            return None;
        }
        let r0 = (prev_tok_id as usize).min(emb.rows - 1);
        let mut out = [0.0_f32; 256];
        unsafe {
            let p0 = emb.w_ptr.add(r0 * emb.cols);
            for c in 0..256 {
                let u0 = *p0.add(c);
                out[c] = f32::from_bits((u0 as u32) << 16);
            }
        }
        Some(out)
    }

    /// `F9.2`: Evaluates `dot(markov_head[cand_tok_id], q_256)` in `< 0.2 µs` from Zero-Copy `BF16`.
    #[inline]
    pub fn score_markov_with_embed(&self, q_256: &[f32; 256], cand_tok_id: u32) -> f32 {
        let Some(ref hd) = self.markov_head else {
            return 0.0;
        };
        if hd.cols < 256 || hd.rows == 0 {
            return 0.0;
        }
        let r1 = (cand_tok_id as usize).min(hd.rows - 1);
        unsafe {
            let p1 = hd.w_ptr.add(r1 * hd.cols);
            let mut a0 = 0.0_f32;
            let mut a1 = 0.0_f32;
            let mut a2 = 0.0_f32;
            let mut a3 = 0.0_f32;
            for c4 in 0..64 {
                let b = c4 * 4;
                a0 += f32::from_bits((*p1.add(b) as u32) << 16) * q_256[b];
                a1 += f32::from_bits((*p1.add(b + 1) as u32) << 16) * q_256[b + 1];
                a2 += f32::from_bits((*p1.add(b + 2) as u32) << 16) * q_256[b + 2];
                a3 += f32::from_bits((*p1.add(b + 3) as u32) << 16) * q_256[b + 3];
            }
            (a0 + a1) + (a2 + a3)
        }
    }

    /// `F9.2`: Evaluates the rank-256 `DSpark` Markov transition score (`dot(markov_embed[prev_id], markov_head[cand_id])`)
    /// in `< 1 µs` directly from Zero-Copy `BF16` rows.
    #[inline]
    pub fn markov_transition_logit(&self, prev_tok_id: u32, cand_tok_id: u32) -> Option<f32> {
        let q = self.extract_markov_embed(prev_tok_id)?;
        if self.markov_head.is_none() {
            return None;
        }
        Some(self.score_markov_with_embed(&q, cand_tok_id))
    }

    /// `F9.2`: Evaluates `mtp.2.confidence_head.proj.weight` (`[1, 5376]` = `5120` hidden + `256` `markov_embed[tok_id]`)
    /// and returns the sigmoid confidence probability in `[0.0, 1.0]`.
    #[inline]
    pub fn dspark_confidence_prob(&self, normed_h_5120: &[f32], tok_id: u32) -> Option<f32> {
        let proj = self.confidence_proj.as_ref()?;
        let emb = self.markov_embed.as_ref()?;
        if proj.len() < 5376 || normed_h_5120.len() < 5120 || emb.cols < 256 {
            return None;
        }
        let mut logit = flash1017_dot_f32(&proj[..5120], &normed_h_5120[..5120]);
        let r = (tok_id as usize).min(emb.rows.saturating_sub(1));
        unsafe {
            let p = emb.w_ptr.add(r * emb.cols);
            for c in 0..256 {
                let u = *p.add(c);
                let v = f32::from_bits((u as u32) << 16);
                logit += proj[5120 + c] * v;
            }
        }
        Some(1.0_f32 / (1.0_f32 + (-logit.clamp(-20.0, 20.0)).exp()))
    }
}

/// Per-token telemetry from `Flash1017Runtime::forward_token`.
#[derive(Debug, Clone, Default)]
pub struct Flash1017StepTelemetry {
    pub pos: usize,
    pub tier_a_hits: usize,
    pub tier_b_hits: usize,
    pub tier_c_skips: usize,
    pub engram_lookups: usize,
    pub step_ms: f64,
}

/// `Fas F3 (F3.1–F3.5)`: Complete Zero-Copy 40-Layer CED Runtime for `DeepSeek-V4.1-Flash-1017-Tier2`.
pub struct Flash1017Runtime {
    pub model_dir: std::path::PathBuf,
    pub config: Flash1017Config,
    pub tokenizer: Flash1017Tokenizer,
    pub embed_shard: Flash1017ShardInspector,
    pub layer_shards: Vec<Flash1017ShardInspector>,
    pub layer_meta: Vec<Flash1017LayerMeta>,
    pub head_shard: Flash1017ShardInspector,
    pub final_norm: Vec<f32>,
    pub engram_shard_l1: Option<Flash1017ShardInspector>,
    pub engram_shard_l14: Option<Flash1017ShardInspector>,
    pub engram_l1_wkv: Option<CachedMxfp8Mat>,
    pub engram_l14_wkv: Option<CachedMxfp8Mat>,
    pub engram_l1_q: Vec<f32>,
    pub engram_l1_k: Vec<f32>,
    pub engram_l14_q: Vec<f32>,
    pub engram_l14_k: Vec<f32>,
    pub engram_token_map: Vec<u32>,
    /// `F4.1` & `F4.2`: Optional `CoarseToFineConceptHead` (`C2FHead`) + Coalesced `Q8` `head.weight`.
    pub c2f_head: Option<CoarseToFineConceptHead>,
    /// `F4.3`: Optional speculative `mtp.0` (`dspark`) layer (`model-00044-of-00048.safetensors`).
    pub mtp_layer: Option<Flash1017MtpLayer>,
    /// `F9.1`: Optional Zero-Copy `mmap` sidecar (`merkaba_backbone_q4.bin`) holding 4-bit `NVFP4` (`wq_b`, `wo_a`, `wo_b`, `shared_experts`) across all 40 layers (`3.20 GiB` vs `6.11 GB` `MXFP8`).
    pub q4_backbone_mmap: Option<Mmap>,
    /// `F5.2`: Optional 40-layer `5120`-dim Matryoshka-DoRA steering deltas (`Rank 64/128`) injected into each CED layer's FFN output.
    pub layer_dora_deltas: Vec<Option<Vec<f32>>>,
    /// Whether all 5,760 Tier A + Tier B MoE experts across all 40 layers are currently preloaded & pinned in RAM (`~55 GiB` mode vs `7.24 GiB` On-Demand mode).
    pub experts_preloaded_in_ram: bool,
    /// Whether pruned Tier C experts are dynamically routed on-demand from disk (`DeepSeek-V4.1-Flash-1017-NVFP4` `mmap`) or Top-6 ensemble fallback (`0` skips).
    pub tier_c_disk_enabled: bool,
    /// Optional Zero-Copy `mmap` shards of `DeepSeek-V4.1-Flash-1017-NVFP4` providing on-demand access to all 240 Tier C experts per layer.
    pub tier_c_disk_shards: Vec<Flash1017ShardInspector>,
    /// Unscaled RoPE frequencies (`rope_theta = 10000`) for `compress_ratio == 0` layers (`0, 1`).
    rope_inv_freq_c0: [f32; 32],
    /// YaRN-smoothed RoPE frequencies (`compress_rope_theta = 160000, factor = 16, orig = 65536`) for layers `2..40`.
    rope_inv_freq_yarn: [f32; 32],
    /// `Fas S4.4`: Projected `5120`-dim vision token embeddings from `model-00001-of-00048.safetensors` (`deepseek_v41_vision`), indexed by `(token_id & 0x7FFF_FFFF)`.
    pub vision_embed_bank: std::sync::RwLock<Vec<Vec<f32>>>,
}

#[inline(always)]
fn flash1017_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sum_sq = 0.0_f32;
    for &v in x {
        sum_sq += v * v;
    }
    let inv_rms = 1.0_f32 / (sum_sq / (n as f32) + eps).sqrt();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(x[i] * inv_rms * weight[i]);
    }
    out
}

#[inline(always)]
fn flash1017_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    crate::dot_f32_unrolled8(a, b)
}

#[inline(always)]
fn flash1017_axpy_f32(dst: &mut [f32], scale: f32, src: &[f32]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] += scale * src[i];
    }
}

/// `F3.2`: 20-iteration Sinkhorn-Knopp projection onto the $4 \times 4$ Birkhoff polytope of doubly-stochastic matrices.
#[inline]
pub fn sinkhorn_knopp_4x4(u_logits: &[f32; 16], iters: usize, eps: f32) -> [[f32; 4]; 4] {
    let mut m = [[0.0_f32; 4]; 4];
    let mut global_max = f32::NEG_INFINITY;
    for &v in u_logits {
        if v > global_max {
            global_max = v;
        }
    }
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = (u_logits[r * 4 + c] - global_max).clamp(-40.0, 40.0).exp();
        }
    }
    for _ in 0..iters {
        // Row normalization
        for r in 0..4 {
            let r_sum = m[r][0] + m[r][1] + m[r][2] + m[r][3] + eps;
            let inv = 1.0_f32 / r_sum;
            m[r][0] *= inv;
            m[r][1] *= inv;
            m[r][2] *= inv;
            m[r][3] *= inv;
        }
        // Column normalization
        for c in 0..4 {
            let c_sum = m[0][c] + m[1][c] + m[2][c] + m[3][c] + eps;
            let inv = 1.0_f32 / c_sum;
            m[0][c] *= inv;
            m[1][c] *= inv;
            m[2][c] *= inv;
            m[3][c] *= inv;
        }
    }
    m
}

impl Flash1017Runtime {
    /// Decrypts `merkaba_tier_map.bin` (`MKTMAP01` header + 15,360 SplitMix64-encrypted bytes)
    /// returning `Some(Vec<(Vec<usize>, Vec<usize>)>)` for the 40 layers `(tier_a_list, tier_b_list)`.
    fn load_encrypted_tier_map(dir: &Path) -> Option<Vec<(Vec<usize>, Vec<usize>)>> {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()));
        let mut candidates = vec![
            dir.join("merkaba_tier_map.bin"),
            std::path::PathBuf::from("bin/merkaba_tier_map.bin"),
            std::path::PathBuf::from("merkaba_tier_map.bin"),
            std::path::PathBuf::from("/home/tauai/SPARK-GB10/models/DeepSeek-V4.1-Flash-1017-Tier2/merkaba_tier_map.bin"),
        ];
        if let Some(ed) = exe_dir {
            candidates.insert(1, ed.join("merkaba_tier_map.bin"));
        }
        let bytes = candidates.into_iter().find_map(|p| std::fs::read(&p).ok())?;
        if bytes.len() < 8 + 40 * 384 || &bytes[0..8] != b"MKTMAP01" {
            return None;
        }
        let mut state: u64 = 0x4D4B_4742_3130_5631;
        let mut raw = vec![0u8; 40 * 384];
        for i in 0..(40 * 384) {
            let mut x = (state ^ (i as u64)).wrapping_add(0x9E37_79B9_7F4A_7C15);
            state = x;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let k = x ^ (x >> 31);
            raw[i] = bytes[8 + i] ^ ((k & 0xFF) as u8);
        }
        let mut layers = Vec::with_capacity(40);
        for l in 0..40 {
            let mut ta = Vec::new();
            let mut tb = Vec::new();
            for e in 0..384 {
                match raw[l * 384 + e] {
                    1 => ta.push(e),
                    2 => tb.push(e),
                    _ => {}
                }
            }
            layers.push((ta, tb));
        }
        Some(layers)
    }

    /// `F3.1`: Opens all 44–46 zero-copy shards of `DeepSeek-V4.1-Flash-1017-Tier2` or `DeepSeek-V4.1-Flash-1017-NVFP4` in `< 1 s`.
    pub fn open<P: AsRef<Path>>(model_dir: P, load_engram: bool) -> Result<Self, String> {
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        if crate::hardware_addon::active_vtable().is_none() {
            let _ = crate::hardware_addon::load_hardware_addon("gx10-rust");
        }
        let dir = model_dir.as_ref().to_path_buf();
        let encrypted_tier_map = Self::load_encrypted_tier_map(&dir);
        let tokenizer = Flash1017Tokenizer::from_file(&dir)?;
        let embed_shard = Flash1017ShardInspector::open(dir.join("model-00002-of-00048.safetensors"))?;
        let head_shard = Flash1017ShardInspector::open(dir.join("model-00043-of-00048.safetensors"))?;
        crate::hardware_addon::register_host_range(head_shard.mmap.as_ptr(), head_shard.mmap.len());
        let final_norm = head_shard.extract_bf16_vec("norm.weight")?;

        let mut layer_shards = Vec::with_capacity(40);
        let mut layer_meta = Vec::with_capacity(40);

        for l in 0..40 {
            let shard_name = format!("model-{:05}-of-00048.safetensors", l + 3);
            let shard = Flash1017ShardInspector::open(dir.join(&shard_name))?;
            let mut attn_norm = shard.extract_bf16_vec(&format!("layers.{}.attn_norm.weight", l))?;
            let mut ffn_norm = shard.extract_bf16_vec(&format!("layers.{}.ffn_norm.weight", l))?;
            let mut q_norm = shard.extract_bf16_vec(&format!("layers.{}.attn.q_norm.weight", l))?;
            let mut kv_norm = shard.extract_bf16_vec(&format!("layers.{}.attn.kv_norm.weight", l))?;
            let attn_sink = shard.extract_f32_vec(&format!("layers.{}.attn.attn_sink", l))?;
            let compressor_norm = shard
                .extract_bf16_vec(&format!("layers.{}.attn.compressor.norm.weight", l))
                .ok();

            let hc_ab_vec = shard.extract_f32_vec(&format!("layers.{}.hc_attn_base", l))?;
            let hc_as_vec = shard.extract_f32_vec(&format!("layers.{}.hc_attn_scale", l))?;
            let hc_fb_vec = shard.extract_f32_vec(&format!("layers.{}.hc_ffn_base", l))?;
            let hc_fs_vec = shard.extract_f32_vec(&format!("layers.{}.hc_ffn_scale", l))?;

            let mut hc_attn_base = [0.0_f32; 24];
            hc_attn_base.copy_from_slice(&hc_ab_vec[..24]);
            let mut hc_attn_scale = [0.0_f32; 3];
            hc_attn_scale.copy_from_slice(&hc_as_vec[..3]);
            let mut hc_ffn_base = [0.0_f32; 24];
            hc_ffn_base.copy_from_slice(&hc_fb_vec[..24]);
            let mut hc_ffn_scale = [0.0_f32; 3];
            hc_ffn_scale.copy_from_slice(&hc_fs_vec[..3]);

            let mut gate_bias = shard.extract_f32_vec(&format!("layers.{}.ffn.gate.bias", l))?;
            if let Ok((_, tc_raw)) =
                shard.get_tensor_data(&format!("layers.{}.merkaba.tier_c_pruned_indices", l))
            {
                if tc_raw.len() >= 8 {
                    let idx0_u64 = u64::from_le_bytes([
                        tc_raw[0], tc_raw[1], tc_raw[2], tc_raw[3], tc_raw[4], tc_raw[5],
                        tc_raw[6], tc_raw[7],
                    ]);
                    if (idx0_u64 & 0xFFFF_FFFF_FFFF_0000) == 0x4D4B_4742_3130_0000 {
                        if !crate::hardware_addon::unlock_gb10_layer(
                            l,
                            &mut attn_norm,
                            &mut ffn_norm,
                            &mut q_norm,
                            &mut kv_norm,
                            &mut gate_bias,
                        ) {
                            return Err(
                                "🔒 [MERKABA-GB10-LOCK] DeepSeek-V4.1-Flash-1017-Tier2 är hårdvarulåst exklusivt för NVIDIA Grace Blackwell (GB10 / DGX Spark, Linux aarch64, Compute Capability 12.1 / sm_121). Körning på Windows/RTX, Apple Silicon (Mac), Intel eller AMD är spärrad i denna modelllicens.".to_string(),
                            );
                        }
                    }
                }
            }
            let gate_bias_vl = shard
                .extract_f32_vec(&format!("layers.{}.ffn.gate.bias_vl", l))
                .unwrap_or_else(|_| gate_bias.clone());
            let (tier_a_list, tier_b_list) = if let Some(ref emap) = encrypted_tier_map {
                (emap[l].0.clone(), emap[l].1.clone())
            } else {
                let ta = shard
                    .extract_i64_usize_vec(&format!("layers.{}.merkaba.tier_a_indices", l))
                    .unwrap_or_else(|_| {
                        (0..384)
                            .filter(|&e| {
                                shard.tensors.contains_key(&format!(
                                    "layers.{}.ffn.experts.{}.w1.weight",
                                    l, e
                                ))
                            })
                            .collect()
                    });
                let tb = shard
                    .extract_i64_usize_vec(&format!("layers.{}.merkaba.tier_b_indices", l))
                    .unwrap_or_else(|_| {
                        (0..384)
                            .filter(|&e| {
                                shard.tensors.contains_key(&format!(
                                    "layers.{}.ffn.experts.{}.w1.packed_ternary",
                                    l, e
                                ))
                            })
                            .collect()
                    });
                (ta, tb)
            };

            let mut tier_a_set = vec![false; 384];
            for &e in &tier_a_list {
                if e < 384 {
                    tier_a_set[e] = true;
                }
            }
            let mut tier_b_set = vec![false; 384];
            for &e in &tier_b_list {
                if e < 384 {
                    tier_b_set[e] = true;
                }
            }

            // Compute calibrated NVFP4 scale factor (`mean(weight_scale) * weight_scale_2`) from the first Tier A expert,
            // or `1.0` when opening a `106 GB` all-experts shard where exact scales are baked into `row_scales` + `core_*_scale`.
            let mut tier_b_proj_scale = if shard
                .tensors
                .contains_key(&format!("layers.{}.ffn.experts.0.w1.core_b_q8", l))
            {
                1.0_f32
            } else {
                0.01065_f32
            };
            if let Some(&first_a) = tier_a_list.first() {
                if let (Ok((_, sc_bytes)), Ok((_, sc2_bytes))) = (
                    shard.get_tensor_data(&format!("layers.{}.ffn.experts.{}.w1.weight_scale", l, first_a)),
                    shard.get_tensor_data(&format!("layers.{}.ffn.experts.{}.w1.weight_scale_2", l, first_a)),
                ) {
                    if sc2_bytes.len() >= 4 && !sc_bytes.is_empty() {
                        let sc2 = f32::from_le_bytes([sc2_bytes[0], sc2_bytes[1], sc2_bytes[2], sc2_bytes[3]]);
                        let sample_len = sc_bytes.len().min(2048);
                        let mean_sc1: f32 = sc_bytes[..sample_len]
                            .iter()
                            .map(|&b| f8_e4m3_to_f32(b))
                            .sum::<f32>()
                            / (sample_len as f32);
                        let prod = mean_sc1 * sc2;
                        if prod > 1e-6 && prod < 1.0 {
                            tier_b_proj_scale = prod;
                        }
                    }
                }
            }

            let (_, hc_a_bytes) = shard.get_tensor_data(&format!("layers.{}.hc_attn_fn", l))?;
            let (_, hc_f_bytes) = shard.get_tensor_data(&format!("layers.{}.hc_ffn_fn", l))?;
            let (hc_attn_fn_ptr, hc_ffn_fn_ptr, _hc_aligned) =
                if (hc_a_bytes.as_ptr() as usize) % 4 == 0 && (hc_f_bytes.as_ptr() as usize) % 4 == 0
                {
                    crate::hardware_addon::register_host_range(
                        hc_a_bytes.as_ptr(),
                        hc_a_bytes.len(),
                    );
                    crate::hardware_addon::register_host_range(
                        hc_f_bytes.as_ptr(),
                        hc_f_bytes.len(),
                    );
                    (
                        hc_a_bytes.as_ptr() as *const f32,
                        hc_f_bytes.as_ptr() as *const f32,
                        None,
                    )
                } else {
                    let na = hc_a_bytes.len() / 4;
                    let nf = hc_f_bytes.len() / 4;
                    let mut va = Vec::with_capacity(na);
                    for i in 0..na {
                        let b = i * 4;
                        va.push(f32::from_le_bytes([
                            hc_a_bytes[b],
                            hc_a_bytes[b + 1],
                            hc_a_bytes[b + 2],
                            hc_a_bytes[b + 3],
                        ]));
                    }
                    let mut vf = Vec::with_capacity(nf);
                    for i in 0..nf {
                        let b = i * 4;
                        vf.push(f32::from_le_bytes([
                            hc_f_bytes[b],
                            hc_f_bytes[b + 1],
                            hc_f_bytes[b + 2],
                            hc_f_bytes[b + 3],
                        ]));
                    }
                    crate::hardware_addon::register_host_range(
                        va.as_ptr() as *const u8,
                        hc_a_bytes.len(),
                    );
                    crate::hardware_addon::register_host_range(
                        vf.as_ptr() as *const u8,
                        hc_f_bytes.len(),
                    );
                    let pa = va.as_ptr();
                    let pf = vf.as_ptr();
                    (pa, pf, Some((va, vf)))
                };

            let comp_wkv = CachedBf16Mat::from_shard(&shard, &format!("layers.{}.attn.compressor.wkv.weight", l)).ok();
            let comp_wgate = CachedBf16Mat::from_shard(&shard, &format!("layers.{}.attn.compressor.wgate.weight", l)).ok();
            let wq_a = CachedMxfp8Mat::from_shard(&shard, &format!("layers.{}.attn.wq_a", l))?;
            let wq_b = CachedMxfp8Mat::from_shard(&shard, &format!("layers.{}.attn.wq_b", l))?;
            let wkv = CachedMxfp8Mat::from_shard(&shard, &format!("layers.{}.attn.wkv", l))?;
            let wo_a = CachedMxfp8Mat::from_shard(&shard, &format!("layers.{}.attn.wo_a", l))?;
            let wo_b = CachedMxfp8Mat::from_shard(&shard, &format!("layers.{}.attn.wo_b", l))?;
            let ffn_gate = CachedBf16Mat::from_shard(&shard, &format!("layers.{}.ffn.gate.weight", l))?;
            let shared_w1 = CachedMxfp8Mat::from_shard(&shard, &format!("layers.{}.ffn.shared_experts.w1", l))?;
            let shared_w2 = CachedMxfp8Mat::from_shard(&shard, &format!("layers.{}.ffn.shared_experts.w2", l))?;
            let shared_w3 = CachedMxfp8Mat::from_shard(&shard, &format!("layers.{}.ffn.shared_experts.w3", l))?;

            // `F8.1` & `F9.1`: Pre-register 100%-hit MXFP8/BF16 layer weights (`wq_a`, `wkv`, `ffn_gate`, `compressor`)
            for &(p, len) in &[
                (wq_a.w_ptr, wq_a.w_len),
                (wq_a.s_ptr, wq_a.s_len),
                (wkv.w_ptr, wkv.w_len),
                (wkv.s_ptr, wkv.s_len),
                (ffn_gate.w_ptr, ffn_gate.w_len),
            ] {
                crate::hardware_addon::register_host_range(p, len);
            }
            if let Some(ref cw) = comp_wkv {
                crate::hardware_addon::register_host_range(cw.w_ptr, cw.w_len);
            }
            if let Some(ref cg) = comp_wgate {
                crate::hardware_addon::register_host_range(cg.w_ptr, cg.w_len);
            }

            let mut tier_a_experts = vec![None; 384];
            let mut tier_b_experts = vec![None; 384];
            for (idx_a, &e) in tier_a_list.iter().enumerate() {
                if e < 384 {
                    if let (Ok(w1), Ok(w2), Ok(w3)) = (
                        CachedNvfp4Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w1", l, e)),
                        CachedNvfp4Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w2", l, e)),
                        CachedNvfp4Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w3", l, e)),
                    ) {
                        if idx_a < 12 {
                            for &(p, len) in &[
                                (w1.w_ptr, w1.w_len),
                                (w1.s_ptr, w1.s_len),
                                (w2.w_ptr, w2.w_len),
                                (w2.s_ptr, w2.s_len),
                                (w3.w_ptr, w3.w_len),
                                (w3.s_ptr, w3.s_len),
                            ] {
                                crate::hardware_addon::register_host_range(p, len);
                            }
                        }
                        tier_a_experts[e] = Some((w1, w2, w3));
                    } else if shard
                        .tensors
                        .contains_key(&format!("layers.{}.ffn.experts.{}.w1.packed_ternary", l, e))
                    {
                        let w1 = CachedTernary158Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w1", l, e))?;
                        let w2 = CachedTernary158Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w2", l, e))?;
                        let w3 = CachedTernary158Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w3", l, e))?;
                        tier_b_experts[e] = Some((w1, w2, w3));
                    }
                }
            }

            for (idx_b, &e) in tier_b_list.iter().enumerate() {
                if e < 384 {
                    if shard
                        .tensors
                        .contains_key(&format!("layers.{}.ffn.experts.{}.w1.packed_ternary", l, e))
                    {
                        let w1 = CachedTernary158Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w1", l, e))?;
                        let w2 = CachedTernary158Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w2", l, e))?;
                        let w3 = CachedTernary158Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w3", l, e))?;
                        if idx_b < 16 {
                            for &(p, len) in &[
                                (w1.p_ptr, w1.p_len),
                                (w1.s_ptr, w1.s_len),
                                (w2.p_ptr, w2.p_len),
                                (w2.s_ptr, w2.s_len),
                                (w3.p_ptr, w3.p_len),
                                (w3.s_ptr, w3.s_len),
                            ] {
                                crate::hardware_addon::register_host_range(p, len);
                            }
                        }
                        tier_b_experts[e] = Some((w1, w2, w3));
                    } else if let (Ok(w1), Ok(w2), Ok(w3)) = (
                        CachedNvfp4Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w1", l, e)),
                        CachedNvfp4Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w2", l, e)),
                        CachedNvfp4Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w3", l, e)),
                    ) {
                        tier_a_experts[e] = Some((w1, w2, w3));
                    }
                }
            }
            // Map remaining experts in `shard` (`1.58-bit + Rank-16 Q8 JEPA Core` for the `106 GB` All-Experts-in-RAM build, or `NVFP4` for `+tierc`)
            for e in 0..384 {
                if tier_a_experts[e].is_none() && tier_b_experts[e].is_none() {
                    if shard
                        .tensors
                        .contains_key(&format!("layers.{}.ffn.experts.{}.w1.packed_ternary", l, e))
                    {
                        if let (Ok(w1), Ok(w2), Ok(w3)) = (
                            CachedTernary158Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w1", l, e)),
                            CachedTernary158Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w2", l, e)),
                            CachedTernary158Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w3", l, e)),
                        ) {
                            tier_b_experts[e] = Some((w1, w2, w3));
                        }
                    } else if let (Ok(w1), Ok(w2), Ok(w3)) = (
                        CachedNvfp4Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w1", l, e)),
                        CachedNvfp4Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w2", l, e)),
                        CachedNvfp4Mat::from_shard(&shard, &format!("layers.{}.ffn.experts.{}.w3", l, e)),
                    ) {
                        tier_a_experts[e] = Some((w1, w2, w3));
                    }
                }
            }

            layer_shards.push(shard);
            layer_meta.push(Flash1017LayerMeta {
                layer_idx: l,
                attn_norm,
                ffn_norm,
                q_norm,
                kv_norm,
                attn_sink,
                compressor_norm,
                hc_attn_base,
                hc_attn_scale,
                hc_ffn_base,
                hc_ffn_scale,
                hc_attn_fn_ptr,
                hc_ffn_fn_ptr,
                _hc_aligned,
                gate_bias,
                gate_bias_vl,
                tier_a_set,
                tier_b_set,
                tier_a_list,
                tier_b_list,
                tier_b_proj_scale,
                comp_wkv,
                comp_wgate,
                wq_a,
                wq_b,
                wkv,
                wo_a,
                wo_b,
                ffn_gate,
                shared_w1,
                shared_w2,
                shared_w3,
                q4_backbone: None,
                tier_a_experts,
                tier_b_experts,
            });
        }

        let q4_backbone_mmap = match load_or_build_q4_backbone_sidecar(&dir, &mut layer_meta) {
            Ok(mm) => mm,
            Err(e) => {
                eprintln!(
                    "  ⚠️ [F9.1] Kunde inte ladda/bygga merkaba_backbone_q4.bin (faller tillbaka till MXFP8): {}",
                    e
                );
                None
            }
        };
        if q4_backbone_mmap.is_none() {
            for lm in &layer_meta {
                for &(p, len) in &[
                    (lm.wq_b.w_ptr, lm.wq_b.w_len),
                    (lm.wq_b.s_ptr, lm.wq_b.s_len),
                    (lm.wo_a.w_ptr, lm.wo_a.w_len),
                    (lm.wo_a.s_ptr, lm.wo_a.s_len),
                    (lm.wo_b.w_ptr, lm.wo_b.w_len),
                    (lm.wo_b.s_ptr, lm.wo_b.s_len),
                    (lm.shared_w1.w_ptr, lm.shared_w1.w_len),
                    (lm.shared_w1.s_ptr, lm.shared_w1.s_len),
                    (lm.shared_w2.w_ptr, lm.shared_w2.w_len),
                    (lm.shared_w2.s_ptr, lm.shared_w2.s_len),
                    (lm.shared_w3.w_ptr, lm.shared_w3.w_len),
                    (lm.shared_w3.s_ptr, lm.shared_w3.s_len),
                ] {
                    crate::hardware_addon::register_host_range(p, len);
                }
            }
        }

        let (engram_shard_l1, engram_l1_wkv, engram_l1_q, engram_l1_k) = if load_engram {
            let p47 = dir.join("model-00047-of-00048.safetensors");
            if p47.exists() {
                let s47 = Flash1017ShardInspector::open(p47)?;
                let wkv = CachedMxfp8Mat::from_shard(&s47, "layers.1.engram.wkv").ok();
                let q = s47.extract_bf16_vec("layers.1.engram.q_weight")?;
                let k = s47.extract_bf16_vec("layers.1.engram.k_weight")?;
                (Some(s47), wkv, q, k)
            } else {
                (None, None, Vec::new(), Vec::new())
            }
        } else {
            (None, None, Vec::new(), Vec::new())
        };

        let (engram_shard_l14, engram_l14_wkv, engram_l14_q, engram_l14_k) = if load_engram {
            let p48 = dir.join("model-00048-of-00048.safetensors");
            if p48.exists() {
                let s48 = Flash1017ShardInspector::open(p48)?;
                let wkv = CachedMxfp8Mat::from_shard(&s48, "layers.14.engram.wkv").ok();
                let q = s48.extract_bf16_vec("layers.14.engram.q_weight")?;
                let k = s48.extract_bf16_vec("layers.14.engram.k_weight")?;
                (Some(s48), wkv, q, k)
            } else {
                (None, None, Vec::new(), Vec::new())
            }
        } else {
            (None, None, Vec::new(), Vec::new())
        };

        // Precompute RoPE inverse frequencies:
        // - `rope_inv_freq_c0`: `rope_theta = 10000` (unscaled) for `compress_ratio == 0` layers (`0, 1`)
        // - `rope_inv_freq_yarn`: `compress_rope_theta = 160000` with YaRN (`original = 65536, factor = 16, beta_fast = 32, beta_slow = 1`) for layers `2..40`
        let mut rope_inv_freq_c0 = [0.0_f32; 32];
        let mut rope_inv_freq_yarn = [0.0_f32; 32];
        let theta_c0 = 10000.0_f32;
        let theta_yarn = 160_000.0_f32;
        let orig_seq = 65536.0_f32;
        let factor = 16.0_f32;
        let dim = 64.0_f32;
        let two_pi = 2.0_f32 * std::f32::consts::PI;
        let low_corr = (dim * (orig_seq / (32.0_f32 * two_pi)).ln() / (2.0_f32 * theta_yarn.ln()))
            .floor()
            .max(0.0_f32);
        let high_corr = (dim * (orig_seq / (1.0_f32 * two_pi)).ln() / (2.0_f32 * theta_yarn.ln()))
            .ceil()
            .min(dim - 1.0_f32);
        let denom = (high_corr - low_corr).max(0.001_f32);

        for i in 0..32 {
            rope_inv_freq_c0[i] = 1.0_f32 / theta_c0.powf((2 * i) as f32 / dim);
            let f_y = 1.0_f32 / theta_yarn.powf((2 * i) as f32 / dim);
            let ramp = (((i as f32) - low_corr) / denom).clamp(0.0_f32, 1.0_f32);
            let smooth = 1.0_f32 - ramp;
            rope_inv_freq_yarn[i] = (f_y / factor) * (1.0_f32 - smooth) + f_y * smooth;
        }

        let engram_token_map = {
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()));
            let mut candidates = vec![
                dir.join("engram_token_map.bin"),
                std::path::PathBuf::from("bin/engram_token_map.bin"),
                std::path::PathBuf::from("engram_token_map.bin"),
                std::path::PathBuf::from(
                    "/home/tauai/SPARK-GB10/models/DeepSeek-V4.1-Flash-1017-Tier2/engram_token_map.bin",
                ),
            ];
            if let Some(ed) = exe_dir {
                candidates.insert(1, ed.join("engram_token_map.bin"));
            }
            let map_bytes = candidates.into_iter().find_map(|p| std::fs::read(&p).ok());
            if let Some(bytes) = map_bytes {
                bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                (0..129280u32).collect()
            }
        };

        Ok(Self {
            model_dir: dir,
            config: Flash1017Config::default(),
            tokenizer,
            embed_shard,
            layer_shards,
            layer_meta,
            head_shard,
            final_norm,
            engram_shard_l1,
            engram_shard_l14,
            engram_l1_wkv,
            engram_l14_wkv,
            engram_l1_q,
            engram_l1_k,
            engram_l14_q,
            engram_l14_k,
            engram_token_map,
            c2f_head: None,
            mtp_layer: None,
            q4_backbone_mmap,
            layer_dora_deltas: vec![None; 40],
            experts_preloaded_in_ram: false,
            tier_c_disk_enabled: true,
            tier_c_disk_shards: Vec::new(),
            rope_inv_freq_c0,
            rope_inv_freq_yarn,
            vision_embed_bank: std::sync::RwLock::new(Vec::new()),
        })
    }

    /// Enables or disables **Mode 3: `+Tier C från Disk (Alla 384 experter)`**.
    /// When `enable == true`, maps the unpruned `DeepSeek-V4.1-Flash-1017-NVFP4` layer shards on-demand via `mmap`
    /// (if present on disk) into the empty `Tier C` slots of `lm.tier_a_experts` and activates full 6-expert routing (`tier_c_skips = 0`).
    pub fn set_tier_c_disk_mode(&mut self, enable: bool) -> (usize, bool) {
        self.tier_c_disk_enabled = enable;
        if !enable {
            return (0, false);
        }
        // 1. Check if `self.layer_shards` itself already contains the unpruned NVFP4 Tier C experts
        for l in 0..40 {
            let sh = &self.layer_shards[l];
            let lm = &mut self.layer_meta[l];
            for e in 0..384 {
                if !lm.tier_a_set[e] && !lm.tier_b_set[e] && lm.tier_a_experts[e].is_none() {
                    if let (Ok(w1), Ok(w2), Ok(w3)) = (
                        CachedNvfp4Mat::from_shard(
                            sh,
                            &format!("layers.{}.ffn.experts.{}.w1", l, e),
                        ),
                        CachedNvfp4Mat::from_shard(
                            sh,
                            &format!("layers.{}.ffn.experts.{}.w2", l, e),
                        ),
                        CachedNvfp4Mat::from_shard(
                            sh,
                            &format!("layers.{}.ffn.experts.{}.w3", l, e),
                        ),
                    ) {
                        lm.tier_a_experts[e] = Some((w1, w2, w3));
                    }
                }
            }
        }
        let mapped_in_place: usize = self
            .layer_meta
            .iter()
            .map(|lm| {
                (0..384)
                    .filter(|&e| !lm.tier_a_set[e] && !lm.tier_b_set[e] && lm.tier_a_experts[e].is_some())
                    .count()
            })
            .sum();
        if mapped_in_place > 0 {
            return (mapped_in_place, true);
        }

        // 2. Otherwise check external NVFP4 directory on disk
        if self.tier_c_disk_shards.is_empty() {
            let mut candidate_dirs = Vec::new();
            if let Some(parent) = self.model_dir.parent() {
                candidate_dirs.push(parent.join("DeepSeek-V4.1-Flash-1017-NVFP4"));
            }
            candidate_dirs.push(std::path::PathBuf::from(
                "/home/tauai/SPARK-GB10/models/DeepSeek-V4.1-Flash-1017-NVFP4",
            ));
            candidate_dirs.push(std::path::PathBuf::from(
                "/home/tauai/models/DeepSeek-V4.1-Flash-1017-NVFP4",
            ));
            candidate_dirs.push(std::path::PathBuf::from(
                "/Volumes/NET/ai-projects/models/DeepSeek-V4.1-Flash-1017-NVFP4",
            ));

            if let Some(nvfp4_dir) = candidate_dirs
                .into_iter()
                .find(|d| d.join("model-00003-of-00048.safetensors").exists())
            {
                let mut loaded_shards = Vec::with_capacity(40);
                let mut all_opened = true;
                for l in 0..40 {
                    let shard_name = format!("model-{:05}-of-00048.safetensors", l + 3);
                    match Flash1017ShardInspector::open(nvfp4_dir.join(&shard_name)) {
                        Ok(sh) => loaded_shards.push(sh),
                        Err(_) => {
                            all_opened = false;
                            break;
                        }
                    }
                }
                if all_opened && loaded_shards.len() == 40 {
                    for l in 0..40 {
                        let sh = &loaded_shards[l];
                        let lm = &mut self.layer_meta[l];
                        for e in 0..384 {
                            if !lm.tier_a_set[e] && !lm.tier_b_set[e] && lm.tier_a_experts[e].is_none() {
                                if let (Ok(w1), Ok(w2), Ok(w3)) = (
                                    CachedNvfp4Mat::from_shard(
                                        sh,
                                        &format!("layers.{}.ffn.experts.{}.w1", l, e),
                                    ),
                                    CachedNvfp4Mat::from_shard(
                                        sh,
                                        &format!("layers.{}.ffn.experts.{}.w2", l, e),
                                    ),
                                    CachedNvfp4Mat::from_shard(
                                        sh,
                                        &format!("layers.{}.ffn.experts.{}.w3", l, e),
                                    ),
                                ) {
                                    lm.tier_a_experts[e] = Some((w1, w2, w3));
                                }
                            }
                        }
                    }
                    self.tier_c_disk_shards = loaded_shards;
                }
            }
        }
        let mapped_c: usize = self
            .layer_meta
            .iter()
            .map(|lm| {
                (0..384)
                    .filter(|&e| !lm.tier_a_set[e] && !lm.tier_b_set[e] && lm.tier_a_experts[e].is_some())
                    .count()
            })
            .sum();
        (mapped_c, !self.tier_c_disk_shards.is_empty())
    }

    /// Toggles between **On-Demand Expert Mode (`~7.24 GiB` RAM)** and **All-Experts-in-RAM Mode (`~55 GiB` RAM)**.
    /// - When `enable == true`: Pre-touches and GPU-pins all 5,760 Tier A (`NVFP4`) and Tier B (`1.58-bit`) experts
    ///   across all 40 layers in parallel using Rayon (skipping dead 8-bit backbone duplicates in `.safetensors`),
    ///   guaranteeing `0.00 ms` SSD page faults on any topic shift.
    /// - When `enable == false`: Advises the OS kernel (`MADV_DONTNEED` on Unix) that unpinned expert pages in
    ///   `layer_shards` can be reclaimed immediately when 3D/Image generators (`TRELLIS 2` / `FLUX`) need RAM.
    /// Returns `(expert_count, total_bytes, elapsed_ms)`.
    pub fn set_preload_all_experts(&mut self, enable: bool) -> (usize, usize, f64) {
        use rayon::prelude::*;
        let t0 = std::time::Instant::now();
        let (expert_count, total_bytes): (usize, usize) = self
            .layer_meta
            .par_iter()
            .map(|lm| {
                let mut e_cnt = 0usize;
                let mut b_cnt = 0usize;
                let mut checksum = 0u8;
                // F10.1: Cap per-layer RAM preload at 74 Tier A experts (~1.37 GiB/layer = ~54.8 GiB across 40 layers)
                // so `+55 GB i RAM` matches its 55 GB budget (compatible with 64 GB machines) and leaves ~61 GiB
                // of free Unified RAM on 128 GB GX10 for OS Page Cache of Tier B + Tier C experts!
                const MAX_PRELOAD_EXPERTS_PER_LAYER: usize = 74;
                for (e, exp_opt) in lm.tier_a_experts.iter().enumerate() {
                    if !lm.tier_a_set[e] {
                        continue;
                    }
                    if e_cnt >= MAX_PRELOAD_EXPERTS_PER_LAYER {
                        break;
                    }
                    if let Some((w1, w2, w3)) = exp_opt {
                        e_cnt += 1;
                        for m in [w1, w2, w3] {
                            b_cnt += m.w_len + m.s_len;
                            if enable {
                                let mut off = 0usize;
                                while off < m.w_len {
                                    unsafe {
                                        checksum ^= std::ptr::read_volatile(m.w_ptr.add(off));
                                    }
                                    off += 65536;
                                }
                                let mut s_off = 0usize;
                                while s_off < m.s_len {
                                    unsafe {
                                        checksum ^= std::ptr::read_volatile(m.s_ptr.add(s_off));
                                    }
                                    s_off += 65536;
                                }
                            }
                        }
                    }
                }
                for exp_opt in &lm.tier_b_experts {
                    if let Some((w1, w2, w3)) = exp_opt {
                        e_cnt += 1;
                        for m in [w1, w2, w3] {
                            let s_bytes = m.rows * 4;
                            b_cnt += m.p_len + s_bytes;
                            if enable {
                                let mut off = 0usize;
                                while off < m.p_len {
                                    unsafe {
                                        checksum ^= std::ptr::read_volatile(m.p_ptr.add(off));
                                    }
                                    off += 65536;
                                }
                            }
                        }
                    }
                }
                std::hint::black_box(checksum);
                (e_cnt, b_cnt)
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

        #[cfg(unix)]
        if !enable && self.experts_preloaded_in_ram {
            extern "C" {
                fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            const MADV_DONTNEED: i32 = 4;
            for shard in &self.layer_shards {
                unsafe {
                    let _ = madvise(
                        shard.mmap.as_ptr() as *mut std::ffi::c_void,
                        shard.mmap.len(),
                        MADV_DONTNEED,
                    );
                }
            }
        }

        self.experts_preloaded_in_ram = enable;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        (expert_count, total_bytes, elapsed_ms)
    }

    /// `F5.2`: Installs per-layer `5120`-dim Matryoshka-DoRA steering deltas across the 40 CED layers.
    pub fn set_layer_dora_deltas(&mut self, mut deltas: Vec<Option<Vec<f32>>>) {
        deltas.resize(40, None);
        self.layer_dora_deltas = deltas;
    }

    /// `F5.2`: Clears all active per-layer Matryoshka-DoRA steering deltas.
    pub fn clear_layer_dora_deltas(&mut self) {
        for slot in &mut self.layer_dora_deltas {
            *slot = None;
        }
    }

    /// `F5.2`: Computes an L2-normalized `5120`-dim semantic routing vector from `token_ids`
    /// using the Zero-Copy `embed.weight` (`BF16 [129280, 5120]`) table in `< 0.05 ms`.
    pub fn compute_prompt_routing_vector_5120(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.config.hidden_size;
        let mut acc = vec![0.0_f32; d];
        let mut count = 0usize;
        for &tid in token_ids {
            // Skip special control tokens (< 256) and common chat template header tokens when possible
            if tid >= 256 && tid < 128_798 {
                if let Ok(emb) = self.get_token_embedding(tid) {
                    for i in 0..d {
                        acc[i] += emb[i];
                    }
                    count += 1;
                }
            }
        }
        if count == 0 {
            for &tid in token_ids {
                if let Ok(emb) = self.get_token_embedding(tid) {
                    for i in 0..d {
                        acc[i] += emb[i];
                    }
                    count += 1;
                }
            }
        }
        if count > 0 {
            let inv_c = 1.0_f32 / (count as f32);
            for v in &mut acc {
                *v *= inv_c;
            }
        }
        let norm = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for v in &mut acc {
                *v /= norm;
            }
        }
        acc
    }

    /// `F4.1` & `F4.2`: Loads or builds `<model_dir>/merkaba_head_q8_c2f.bin` via Zero-Copy `mmap` (`< 1 ms` once cached).
    pub fn enable_c2f_head(&mut self) -> Result<(), String> {
        if self.c2f_head.is_none() {
            let head = CoarseToFineConceptHead::load_or_build(
                &self.model_dir,
                &self.head_shard,
                &self.final_norm,
            )?;
            self.c2f_head = Some(head);
        }
        Ok(())
    }

    /// `F4.3`: Opens `model-00044-of-00048.safetensors` (`mtp.0` speculative layer) via Zero-Copy `mmap` in `< 5 ms`.
    pub fn enable_mtp_layer(&mut self) -> Result<(), String> {
        if self.mtp_layer.is_none() {
            let mtp = Flash1017MtpLayer::open(&self.model_dir)?;
            self.mtp_layer = Some(mtp);
        }
        Ok(())
    }

    /// Looks up the `5120`-dim `BF16` token embedding from `model-00002-of-00048.safetensors`
    /// or returns the projected `5120`-dim vision token embedding if `(token_id & 0x8000_0000) != 0`.
    pub fn get_token_embedding(&self, token_id: u32) -> Result<Vec<f32>, String> {
        if (token_id & 0x8000_0000) != 0 {
            let v_idx = (token_id & 0x7FFF_FFFF) as usize;
            if let Ok(bank) = self.vision_embed_bank.read() {
                if let Some(emb) = bank.get(v_idx) {
                    return Ok(emb.clone());
                }
            }
            return self.get_token_embedding(129264);
        }
        let (_meta, raw) = self.embed_shard.get_tensor_data("embed.weight")?;
        let d = self.config.hidden_size;
        let idx = (token_id as usize).min(129_279);
        let start = idx * d * 2;
        let end = start + d * 2;
        if end > raw.len() {
            return Err(format!("Token ID {} out of bounds in embed.weight", token_id));
        }
        let slice = &raw[start..end];
        let mut out = Vec::with_capacity(d);
        for i in 0..d {
            let bits = u16::from_le_bytes([slice[i * 2], slice[i * 2 + 1]]);
            out.push(f32::from_bits((bits as u32) << 16));
        }
        Ok(out)
    }

    /// `Fas S4.4`: Encodes an RGB image (`width x height x 3`) through the 32-layer `deepseek_v41_vision`
    /// Vision Transformer (`model-00001-of-00048.safetensors`: `vision.patch_embed` -> `vision.blocks.0..31` -> `vision.norm`
    /// -> `3x3 downsample` + `aligner.w1` + `aligner.w2` + `image_start`/`image_newline`/`image_end`)
    /// and returns the sequence of pseudo-token IDs (`0x8000_0000 | idx`) ready for `forward_tokens_batch_hidden`.
    pub fn encode_image_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<u32>, String> {
        use rayon::prelude::*;
        if width == 0 || height == 0 || rgb.len() < width * height * 3 {
            return Err("Invalid RGB buffer dimensions".to_string());
        }
        let vis_path = self.model_dir.join("model-00001-of-00048.safetensors");
        let vis_shard = Flash1017ShardInspector::open(&vis_path)?;

        // Target grid matching `deepseek_v41_image_processing.py` (`min_pixels = 295936`, cell = 42 px = 3 * 14)
        let (grid_r, grid_c) = if height >= 252 && width >= 252 && height % 42 == 0 && width % 42 == 0 {
            ((height / 42).clamp(6, 14), (width / 42).clamp(6, 14))
        } else {
            let orig_pixels = (width * height).max(1) as f32;
            let min_scale = if orig_pixels < 295_936.0 {
                (295_936.0_f32 / orig_pixels).sqrt()
            } else {
                1.0_f32
            };
            let eff_h = ((height as f32) * min_scale) as usize;
            let eff_w = ((width as f32) * min_scale) as usize;
            (((eff_h + 21) / 42).clamp(6, 14), ((eff_w + 21) / 42).clamp(6, 14))
        };
        let hp = grid_r * 3;
        let wp = grid_c * 3;
        let target_h = hp * 14;
        let target_w = wp * 14;
        let n_patches = hp * wp;

        // 1. Resize & extract [n_patches, 588] normalized to [-1, 1] (`C=3, ph=14, pw=14`)
        let mut patches_flat = vec![0.0_f32; n_patches * 588];
        for pr in 0..hp {
            for pc in 0..wp {
                let p_idx = pr * wp + pc;
                let dst = &mut patches_flat[p_idx * 588..(p_idx + 1) * 588];
                for ch in 0..3 {
                    for py in 0..14 {
                        let sy = ((pr * 14 + py) * height / target_h).min(height - 1);
                        for px in 0..14 {
                            let sx = ((pc * 14 + px) * width / target_w).min(width - 1);
                            let v = rgb[(sy * width + sx) * 3 + ch] as f32;
                            dst[ch * 196 + py * 14 + px] = (v / 255.0_f32 - 0.5_f32) * 2.0_f32;
                        }
                    }
                }
            }
        }

        // 2. Patch embedding: [1024, 588] + bias [1024]
        let patch_proj = CachedBf16Mat::from_shard(&vis_shard, "vision.patch_embed.proj.weight")?;
        let patch_bias = vis_shard.extract_bf16_vec("vision.patch_embed.proj.bias")?;
        let mut x = patch_proj.matbatch(&patches_flat, n_patches);
        for p in 0..n_patches {
            let row = &mut x[p * 1024..(p + 1) * 1024];
            for d in 0..1024 {
                row[d] += patch_bias[d];
            }
        }

        // Precompute 2D RoPE `(cos, sin)` per patch matching `get_vision_cos_sin` + `apply_rotary` (`chunk(2, dim=-1)`):
        // `rope_dim = 32`, `inv_freq[0..16]`, `freqs = [hpos * inv_freq (0..16), wpos * inv_freq (16..32)]`
        let mut inv_freq_16 = [0.0_f32; 16];
        for i in 0..16 {
            inv_freq_16[i] = 1.0_f32 / 10000.0_f32.powf((2 * i) as f32 / 32.0_f32);
        }
        let mut patch_cos = vec![[0.0_f32; 32]; n_patches];
        let mut patch_sin = vec![[0.0_f32; 32]; n_patches];
        for p in 0..n_patches {
            let pr = (p / wp) as f32;
            let pc = (p % wp) as f32;
            for i in 0..16 {
                let (sr, cr) = (pr * inv_freq_16[i]).sin_cos();
                patch_cos[p][i] = cr;
                patch_sin[p][i] = sr;
                let (sc, cc) = (pc * inv_freq_16[i]).sin_cos();
                patch_cos[p][16 + i] = cc;
                patch_sin[p][16 + i] = sc;
            }
        }

        // 3. Run 32 Vision Transformer blocks (`hidden = 1024`, `heads = 16`, `head_dim = 64`, `inter = 2816`)
        for b in 0..32 {
            let norm1 = vis_shard.extract_bf16_vec(&format!("vision.blocks.{}.norm1.weight", b))?;
            let wqkv = CachedBf16Mat::from_shard(&vis_shard, &format!("vision.blocks.{}.attn.wqkv.weight", b))?;
            let wqkv_bias = vis_shard.extract_bf16_vec(&format!("vision.blocks.{}.attn.wqkv.bias", b))?;
            let wo = CachedBf16Mat::from_shard(&vis_shard, &format!("vision.blocks.{}.attn.wo.weight", b))?;
            let wo_bias = vis_shard.extract_bf16_vec(&format!("vision.blocks.{}.attn.wo.bias", b))?;
            let norm2 = vis_shard.extract_bf16_vec(&format!("vision.blocks.{}.norm2.weight", b))?;
            let mlp_w1 = CachedBf16Mat::from_shard(&vis_shard, &format!("vision.blocks.{}.mlp.w1.weight", b))?;
            let mlp_w2 = CachedBf16Mat::from_shard(&vis_shard, &format!("vision.blocks.{}.mlp.w2.weight", b))?;

            let mut normed1 = Vec::with_capacity(n_patches * 1024);
            for p in 0..n_patches {
                normed1.extend_from_slice(&flash1017_rms_norm(&x[p * 1024..(p + 1) * 1024], &norm1, 1e-6));
            }
            let mut qkv = wqkv.matbatch(&normed1, n_patches);
            for p in 0..n_patches {
                let r_slice = &mut qkv[p * 3072..(p + 1) * 3072];
                for i in 0..3072 {
                    r_slice[i] += wqkv_bias[i];
                }
                let cos_p = &patch_cos[p];
                let sin_p = &patch_sin[p];
                // Apply `chunk(2, dim=-1)` 2D RoPE to Q (0..1024) and K (1024..2048) across 16 heads (head_dim = 64)
                for qk_off in [0usize, 1024usize] {
                    for h in 0..16 {
                        let hs = qk_off + h * 64;
                        for j in 0..32 {
                            let x1 = r_slice[hs + j];
                            let x2 = r_slice[hs + 32 + j];
                            let c = cos_p[j];
                            let s = sin_p[j];
                            r_slice[hs + j] = x1 * c - x2 * s;
                            r_slice[hs + 32 + j] = x2 * c + x1 * s;
                        }
                    }
                }
            }

            // Full bidirectional self-attention over `n_patches` for 16 heads (`scale = 1 / sqrt(64) = 0.125`)
            let attn_cat: Vec<f32> = (0..n_patches)
                .into_par_iter()
                .flat_map_iter(|p_i| {
                    let mut out_p = vec![0.0_f32; 1024];
                    let q_p = &qkv[p_i * 3072..p_i * 3072 + 1024];
                    for h in 0..16 {
                        let q_h = &q_p[h * 64..(h + 1) * 64];
                        let mut scores = vec![0.0_f32; n_patches];
                        let mut max_s = f32::NEG_INFINITY;
                        for p_j in 0..n_patches {
                            let k_h = &qkv[p_j * 3072 + 1024 + h * 64..p_j * 3072 + 1024 + (h + 1) * 64];
                            let s = flash1017_dot_f32(q_h, k_h) * 0.125_f32;
                            scores[p_j] = s;
                            if s > max_s {
                                max_s = s;
                            }
                        }
                        let mut sum_exp = 0.0_f32;
                        for s in &mut scores {
                            *s = (*s - max_s).exp();
                            sum_exp += *s;
                        }
                        let inv_sum = 1.0_f32 / sum_exp.max(1e-12);
                        let dst_h = &mut out_p[h * 64..(h + 1) * 64];
                        for p_j in 0..n_patches {
                            let w = scores[p_j] * inv_sum;
                            let v_h = &qkv[p_j * 3072 + 2048 + h * 64..p_j * 3072 + 2048 + (h + 1) * 64];
                            for d in 0..64 {
                                dst_h[d] += w * v_h[d];
                            }
                        }
                    }
                    out_p
                })
                .collect();

            let attn_proj = wo.matbatch(&attn_cat, n_patches);
            for p in 0..n_patches {
                for d in 0..1024 {
                    x[p * 1024 + d] += attn_proj[p * 1024 + d] + wo_bias[d];
                }
            }

            let mut normed2 = Vec::with_capacity(n_patches * 1024);
            for p in 0..n_patches {
                normed2.extend_from_slice(&flash1017_rms_norm(&x[p * 1024..(p + 1) * 1024], &norm2, 1e-6));
            }
            let gate_up = mlp_w1.matbatch(&normed2, n_patches);
            let mut act = vec![0.0_f32; n_patches * 2816];
            for p in 0..n_patches {
                let gu = &gate_up[p * 5632..(p + 1) * 5632];
                let dst = &mut act[p * 2816..(p + 1) * 2816];
                for i in 0..2816 {
                    let g = gu[i];
                    let silu = g / (1.0_f32 + (-g).exp());
                    dst[i] = silu * gu[2816 + i];
                }
            }
            let mlp_out = mlp_w2.matbatch(&act, n_patches);
            for i in 0..(n_patches * 1024) {
                x[i] += mlp_out[i];
            }
        }

        // 4. Final vision norm + 3x3 spatial downsample (`9 * 1024 = 9216`) + Aligner MLP (`9216 -> 5120 -> 5120`)
        let vis_norm = vis_shard.extract_bf16_vec("vision.norm.weight")?;
        for p in 0..n_patches {
            let normed = flash1017_rms_norm(&x[p * 1024..(p + 1) * 1024], &vis_norm, 1e-6);
            x[p * 1024..(p + 1) * 1024].copy_from_slice(&normed);
        }

        let aligner_w1 = CachedBf16Mat::from_shard(&vis_shard, "aligner.w1.weight")?;
        let aligner_b1 = vis_shard.extract_bf16_vec("aligner.w1.bias")?;
        let aligner_w2 = CachedBf16Mat::from_shard(&vis_shard, "aligner.w2.weight")?;
        let aligner_b2 = vis_shard.extract_bf16_vec("aligner.w2.bias")?;

        let n_merged = grid_r * grid_c;
        let mut merged_9216 = Vec::with_capacity(n_merged * 9216);
        for gr in 0..grid_r {
            for gc in 0..grid_c {
                // `F.unfold(x.view(n_h, n_w, -1).permute(2, 0, 1).unsqueeze(0), 3, stride=3)` order: [c in 0..1024, dr in 0..3, dc in 0..3]
                for c in 0..1024 {
                    for dr in 0..3 {
                        for dc in 0..3 {
                            let p_idx = (gr * 3 + dr) * wp + (gc * 3 + dc);
                            merged_9216.push(x[p_idx * 1024 + c]);
                        }
                    }
                }
            }
        }

        let mut h1 = aligner_w1.matbatch(&merged_9216, n_merged);
        for m in 0..n_merged {
            let row = &mut h1[m * 5120..(m + 1) * 5120];
            for d in 0..5120 {
                let v = row[d] + aligner_b1[d];
                // GELU activation
                let tanh_arg = 0.79788456_f32 * (v + 0.044715_f32 * v * v * v);
                row[d] = 0.5_f32 * v * (1.0_f32 + tanh_arg.tanh());
            }
        }
        let mut aligned_5120 = aligner_w2.matbatch(&h1, n_merged);
        for m in 0..n_merged {
            let row = &mut aligned_5120[m * 5120..(m + 1) * 5120];
            for d in 0..5120 {
                row[d] += aligner_b2[d];
            }
        }

        // 5. Wrap with `image_start`, row-by-row `image_newline`, and `image_end` from `embed_shard`
        let img_start = self.embed_shard.extract_bf16_vec("image_start")?;
        let img_newline = self.embed_shard.extract_bf16_vec("image_newline")?;
        let img_end = self.embed_shard.extract_bf16_vec("image_end")?;

        let mut bank = self
            .vision_embed_bank
            .write()
            .map_err(|_| "Failed to lock vision_embed_bank".to_string())?;
        let mut pseudo_ids = Vec::with_capacity(1 + grid_r * (grid_c + 1) + 1);
        let mut push_emb = |v: Vec<f32>| {
            let idx = bank.len() as u32;
            bank.push(v);
            pseudo_ids.push(0x8000_0000 | idx);
        };

        push_emb(img_start);
        for gr in 0..grid_r {
            for gc in 0..grid_c {
                let m = gr * grid_c + gc;
                push_emb(aligned_5120[m * 5120..(m + 1) * 5120].to_vec());
            }
            push_emb(img_newline.clone());
        }
        push_emb(img_end);

        Ok(pseudo_ids)
    }

    /// Loads an image file (`.png`, `.jpg`, `.jpeg`, `.webp`, `.ppm`) or `data:image/...;base64,...` URL
    /// and encodes it via `encode_image_rgb`.
    pub fn encode_image_file(&self, path_or_url: &str) -> Result<Vec<u32>, String> {
        let py_script = r#"
import sys, base64, io, struct, math
from PIL import Image, ImageOps
arg = sys.argv[1]
if arg.startswith("data:image"):
    b64 = arg.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
else:
    img = Image.open(arg).convert("RGB")
w, h = img.size
cell = 42
min_px = 295936
max_px = 504 * 504
if w * h < min_px:
    sc = math.sqrt(min_px / max(w * h, 1))
    w, h = int(w * sc), int(h * sc)
if w * h > max_px:
    sc = math.sqrt(max_px / max(w * h, 1))
    w, h = int(w * sc), int(h * sc)
best_w = max(cell * 6, min(cell * 14, int(round(w / cell)) * cell))
best_h = max(cell * 6, min(cell * 14, int(round(h / cell)) * cell))
img = ImageOps.pad(img, (best_w, best_h), color=(127, 127, 127))
sys.stdout.buffer.write(struct.pack("<II", best_w, best_h))
sys.stdout.buffer.write(img.tobytes())
"#;
        let out = std::process::Command::new("python3")
            .arg("-c")
            .arg(py_script)
            .arg(path_or_url)
            .output()
            .map_err(|e| format!("Failed to invoke python3 PIL image loader: {}", e))?;
        if !out.status.success() || out.stdout.len() < 8 {
            return Err(format!(
                "Image decode failed: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        let w = u32::from_le_bytes([out.stdout[0], out.stdout[1], out.stdout[2], out.stdout[3]]) as usize;
        let h = u32::from_le_bytes([out.stdout[4], out.stdout[5], out.stdout[6], out.stdout[7]]) as usize;
        self.encode_image_rgb(&out.stdout[8..], w, h)
    }

    /// `F3.2`, `F7.A4` & `F7.B1`: Computes mHC mixing statistics `(w_pre, w_post, m_mat)` from `streams`
    /// (`hc_pre_from_prev_sublayer = true` in `DeepseekV41Config`) using pre-cached `hc_*_fn_ptr` without Rayon barrier overhead.
    fn compute_mhc_stats(
        &self,
        layer_idx: usize,
        streams: &[Vec<f32>; 4],
        is_ffn: bool,
    ) -> Result<([f32; 4], [f32; 4], [[f32; 4]; 4]), String> {
        let meta = &self.layer_meta[layer_idx];
        let (fn_ptr, base, scale) = if is_ffn {
            (meta.hc_ffn_fn_ptr, &meta.hc_ffn_base, &meta.hc_ffn_scale)
        } else {
            (meta.hc_attn_fn_ptr, &meta.hc_attn_base, &meta.hc_attn_scale)
        };

        let d = 5120usize;
        let total_d = 4 * d;
        let mut sum_sq = 0.0_f32;
        for m in 0..4 {
            for &v in &streams[m] {
                sum_sq += v * v;
            }
        }
        let inv_rms = 1.0_f32 / (sum_sq / (total_d as f32) + 1e-20_f32).sqrt();

        // `F8.2`: Compute z[0..24] = hc_fn * (streams_flat * inv_rms) on Blackwell GPU (`active_matvec_f32`) or L1 CPU fallback
        let mut z = [0.0_f32; 24];
        if !fn_ptr.is_null() && ((fn_ptr as usize) % 4 == 0) {
            let mut gpu_done = false;
            if let Some(mv_f32) = crate::hardware_addon::active_matvec_f32() {
                let mut streams_flat = Vec::with_capacity(total_d);
                for m in 0..4 {
                    streams_flat.extend_from_slice(&streams[m]);
                }
                let ok = unsafe {
                    mv_f32(
                        fn_ptr,
                        streams_flat.as_ptr(),
                        z.as_mut_ptr(),
                        24,
                        total_d,
                    )
                };
                if ok {
                    for r in 0..24 {
                        z[r] *= inv_rms;
                    }
                    gpu_done = true;
                }
            }
            if !gpu_done {
                let fn_f32 = unsafe { std::slice::from_raw_parts(fn_ptr, 24 * total_d) };
                for r in 0..24 {
                    let row = &fn_f32[r * total_d..(r + 1) * total_d];
                    let mut dot = 0.0_f32;
                    for m in 0..4 {
                        dot += flash1017_dot_f32(&row[m * d..(m + 1) * d], &streams[m]);
                    }
                    z[r] = dot * inv_rms;
                }
            }
        }

        let mut w_pre = [0.0_f32; 4];
        let mut w_post = [0.0_f32; 4];
        let mut u_res = [0.0_f32; 16];
        for i in 0..4 {
            let u0 = scale[0] * z[i] + base[i];
            w_pre[i] = 1.0_f32 / (1.0_f32 + (-u0).exp()) + 1e-6_f32;
            let u1 = scale[1] * z[4 + i] + base[4 + i];
            w_post[i] = 2.0_f32 / (1.0_f32 + (-u1).exp());
        }
        for i in 0..16 {
            u_res[i] = scale[2] * z[8 + i] + base[8 + i];
        }
        let m_mat = sinkhorn_knopp_4x4(&u_res, 20, 1e-6);
        Ok((w_pre, w_post, m_mat))
    }

    /// `F8.2`: Batched mHC statistics computation across `b_len = 2..8` tokens on Blackwell GPU (`active_matbatch_f32`),
    /// reading `hc_attn_fn` / `hc_ffn_fn` (`[24, 20480]` F32 = `1.966 MB` per call, `157.3 MB` per 40-layer pass)
    /// only **ONCE** from Unified Memory instead of `b_len` times!
    fn compute_mhc_stats_batch(
        &self,
        layer_idx: usize,
        streams_batch: &[[Vec<f32>; 4]],
        is_ffn: bool,
    ) -> Result<Vec<([f32; 4], [f32; 4], [[f32; 4]; 4])>, String> {
        let b_len = streams_batch.len();
        if b_len == 0 {
            return Ok(Vec::new());
        }
        if b_len == 1 {
            return Ok(vec![self.compute_mhc_stats(
                layer_idx,
                &streams_batch[0],
                is_ffn,
            )?]);
        }

        let meta = &self.layer_meta[layer_idx];
        let (fn_ptr, base, scale) = if is_ffn {
            (meta.hc_ffn_fn_ptr, &meta.hc_ffn_base, &meta.hc_ffn_scale)
        } else {
            (meta.hc_attn_fn_ptr, &meta.hc_attn_base, &meta.hc_attn_scale)
        };

        let d = 5120usize;
        let total_d = 4 * d;
        let mut inv_rms_batch = Vec::with_capacity(b_len);
        for t in 0..b_len {
            let mut sum_sq = 0.0_f32;
            for m in 0..4 {
                for &v in &streams_batch[t][m] {
                    sum_sq += v * v;
                }
            }
            inv_rms_batch.push(1.0_f32 / (sum_sq / (total_d as f32) + 1e-20_f32).sqrt());
        }

        let mut z_batch = vec![0.0_f32; b_len * 24];
        let mut gpu_done = false;
        if !fn_ptr.is_null() && ((fn_ptr as usize) % 4 == 0) && b_len <= 8 {
            if let Some(mb_f32) = crate::hardware_addon::active_matbatch_f32() {
                let mut x_batch = Vec::with_capacity(b_len * total_d);
                for t in 0..b_len {
                    for m in 0..4 {
                        x_batch.extend_from_slice(&streams_batch[t][m]);
                    }
                }
                let ok = unsafe {
                    mb_f32(
                        fn_ptr,
                        x_batch.as_ptr(),
                        z_batch.as_mut_ptr(),
                        24,
                        total_d,
                        b_len,
                    )
                };
                if ok {
                    gpu_done = true;
                }
            }
        }

        if !gpu_done {
            let mut out = Vec::with_capacity(b_len);
            for t in 0..b_len {
                out.push(self.compute_mhc_stats(layer_idx, &streams_batch[t], is_ffn)?);
            }
            return Ok(out);
        }

        let mut out = Vec::with_capacity(b_len);
        for t in 0..b_len {
            let inv_rms = inv_rms_batch[t];
            let z_t = &z_batch[t * 24..(t + 1) * 24];
            let mut w_pre = [0.0_f32; 4];
            let mut w_post = [0.0_f32; 4];
            let mut u_res = [0.0_f32; 16];
            for i in 0..4 {
                let u0 = scale[0] * (z_t[i] * inv_rms) + base[i];
                w_pre[i] = 1.0_f32 / (1.0_f32 + (-u0).exp()) + 1e-6_f32;
                let u1 = scale[1] * (z_t[4 + i] * inv_rms) + base[4 + i];
                w_post[i] = 2.0_f32 / (1.0_f32 + (-u1).exp());
            }
            for i in 0..16 {
                u_res[i] = scale[2] * (z_t[8 + i] * inv_rms) + base[8 + i];
            }
            let m_mat = sinkhorn_knopp_4x4(&u_res, 20, 1e-6);
            out.push((w_pre, w_post, m_mat));
        }
        Ok(out)
    }

    #[inline]
    fn combine_mhc_streams(streams: &[Vec<f32>; 4], w_pre: &[f32; 4]) -> Vec<f32> {
        let d = streams[0].len();
        let mut x_in = vec![0.0_f32; d];
        for m in 0..4 {
            let wp = w_pre[m];
            let sm = &streams[m];
            for i in 0..d {
                x_in[i] += wp * sm[i];
            }
        }
        x_in
    }

    #[inline]
    fn apply_mhc_post(
        streams: &mut [Vec<f32>; 4],
        y: &[f32],
        w_post: &[f32; 4],
        m_mat: &[[f32; 4]; 4],
    ) {
        let d = y.len();
        let (s01, s23) = streams.split_at_mut(2);
        let (s0_slice, s1_slice) = s01.split_at_mut(1);
        let (s2_slice, s3_slice) = s23.split_at_mut(1);
        let s0 = &mut s0_slice[0];
        let s1 = &mut s1_slice[0];
        let s2 = &mut s2_slice[0];
        let s3 = &mut s3_slice[0];
        let wp0 = w_post[0];
        let wp1 = w_post[1];
        let wp2 = w_post[2];
        let wp3 = w_post[3];
        // `F7.B1`: Zero-allocation in-place 4-stream Birkhoff mixing in registers
        for i in 0..d {
            let v0 = s0[i];
            let v1 = s1[i];
            let v2 = s2[i];
            let v3 = s3[i];
            let yi = y[i];
            s0[i] = m_mat[0][0] * v0 + m_mat[1][0] * v1 + m_mat[2][0] * v2 + m_mat[3][0] * v3 + wp0 * yi;
            s1[i] = m_mat[0][1] * v0 + m_mat[1][1] * v1 + m_mat[2][1] * v2 + m_mat[3][1] * v3 + wp1 * yi;
            s2[i] = m_mat[0][2] * v0 + m_mat[1][2] * v1 + m_mat[2][2] * v2 + m_mat[3][2] * v3 + wp2 * yi;
            s3[i] = m_mat[0][3] * v0 + m_mat[1][3] * v1 + m_mat[2][3] * v2 + m_mat[3][3] * v3 + wp3 * yi;
        }
    }

    /// `F7.A4`: Precomputes `(sin, cos)` for all 32 RoPE frequency pairs at `(layer_idx, pos)` once per layer
    /// instead of calling `sin_cos()` 129 times per layer (`4,128` calls/layer -> `32` calls/layer).
    #[inline]
    fn precompute_rope_sin_cos(&self, layer_idx: usize, pos: usize) -> [(f32, f32); 32] {
        let freqs = if layer_idx < 2 {
            &self.rope_inv_freq_c0
        } else {
            &self.rope_inv_freq_yarn
        };
        let p = pos as f32;
        let mut table = [(0.0_f32, 1.0_f32); 32];
        for i in 0..32 {
            table[i] = (p * freqs[i]).sin_cos();
        }
        table
    }

    #[inline]
    fn apply_rope_64_table(rope_slice: &mut [f32], sin_cos: &[(f32, f32); 32], inverse: bool) {
        for i in 0..32 {
            let (s, c) = sin_cos[i];
            let x0 = rope_slice[2 * i];
            let x1 = rope_slice[2 * i + 1];
            if inverse {
                rope_slice[2 * i] = x0 * c + x1 * s;
                rope_slice[2 * i + 1] = x1 * c - x0 * s;
            } else {
                rope_slice[2 * i] = x0 * c - x1 * s;
                rope_slice[2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }

    /// Applies interleaved RoPE (`is_neox_style = False`, unit magnitude) in-place to a `64`-element slice (`dims 448..512`).
    /// When `inverse == true`, applies conjugate rotation (`freqs.conj()`) after MLA attention and before `wo_a`.
    #[inline]
    fn apply_rope_64(&self, layer_idx: usize, rope_slice: &mut [f32], pos: usize, inverse: bool) {
        let sin_cos = self.precompute_rope_sin_cos(layer_idx, pos);
        Self::apply_rope_64_table(rope_slice, &sin_cos, inverse);
    }

    /// `F3.3` & `F7.A1–A4`: Executes `MXFP8` Multi-head Latent Attention (`MLA`) for `layer_idx` at `pos`
    /// using pre-cached tensor descriptors, precomputed RoPE tables, and fused `wo_a -> wo_b` GPU execution.
    #[inline]
    fn forward_mla_attention(
        &self,
        layer_idx: usize,
        normed_x: &[f32],
        pos: usize,
        kv_cache: &mut Flash1017KvCache,
    ) -> Result<Vec<f32>, String> {
        self.forward_mla_attention_with_comp_lens(layer_idx, normed_x, pos, kv_cache, None)
    }

    fn forward_mla_attention_with_comp_lens(
        &self,
        layer_idx: usize,
        normed_x: &[f32],
        pos: usize,
        kv_cache: &mut Flash1017KvCache,
        mut comp_lens_override: Option<&mut [usize; 4]>,
    ) -> Result<Vec<f32>, String> {
        use rayon::prelude::*;
        let meta = &self.layer_meta[layer_idx];
        let sin_cos = self.precompute_rope_sin_cos(layer_idx, pos);

        // 1. Update compressed global KV cache on source layers [2, 8, 14, 20] BEFORE attention
        if let Some(comp_norm) = &meta.compressor_norm {
            if layer_idx == 20 {
                // `compress_ratio == 1`: project `wkv`, RMSNorm, rotate at `pos`, append every token
                if let Some(ref comp_wkv) = meta.comp_wkv {
                    let comp_kv = comp_wkv.matvec(normed_x);
                    let mut latent = flash1017_rms_norm(&comp_kv, comp_norm, 1e-20);
                    Self::apply_rope_64_table(&mut latent[448..512], &sin_cos, false);
                    kv_cache.compressed_k[3].extend_from_slice(&latent);
                    if let Some(ref mut cl) = comp_lens_override {
                        cl[3] = kv_cache.compressed_k[3].len();
                    }
                }
            } else {
                let c2_slot = match layer_idx {
                    2 => Some(0usize),
                    8 => Some(1usize),
                    14 => Some(2usize),
                    _ => None,
                };
                if let (Some(slot), Some(ref comp_wkv), Some(ref comp_wgate)) =
                    (c2_slot, &meta.comp_wkv, &meta.comp_wgate)
                {
                    // `compress_ratio == 2`: project `wkv` and `wgate`; pool pairs at `pos % 2 == 1`
                    let comp_kv = comp_wkv.matvec(normed_x);
                    let comp_score = comp_wgate.matvec(normed_x);
                    if pos % 2 == 0 {
                        kv_cache.pending_c2_kv[slot] = comp_kv;
                        kv_cache.pending_c2_score[slot] = comp_score;
                    } else if kv_cache.pending_c2_kv[slot].len() == 512 {
                        let kv0 = &kv_cache.pending_c2_kv[slot];
                        let sc0 = &kv_cache.pending_c2_score[slot];
                        let mut pooled = vec![0.0_f32; 512];
                        for d in 0..512 {
                            let s0 = sc0[d];
                            let s1 = comp_score[d];
                            let m = s0.max(s1);
                            let e0 = (s0 - m).exp();
                            let e1 = (s1 - m).exp();
                            let inv = 1.0_f32 / (e0 + e1);
                            pooled[d] = (e0 * kv0[d] + e1 * comp_kv[d]) * inv;
                        }
                        let mut latent = flash1017_rms_norm(&pooled, comp_norm, 1e-20);
                        let group_pos = pos - 1;
                        self.apply_rope_64(layer_idx, &mut latent[448..512], group_pos, false);
                        kv_cache.compressed_k[slot].extend_from_slice(&latent);
                    }
                    if let Some(ref mut cl) = comp_lens_override {
                        cl[slot] = kv_cache.compressed_k[slot].len();
                    }
                }
            }
        }

        // 2. Fast Path (`F7.B1`): Full In-Stream GPU MLA Attention (`wq_a -> wkv -> q_norm -> wq_b -> kv_norm+RoPE -> 64-Head MLA Online-Softmax -> wo_a -> wo_b`)
        // in 1 single uninterrupted CUDA stream launch and 1 single GPU sync!
        if let Some(f_mla) = crate::hardware_addon::active_mla_full_forward() {
            let prev_total = kv_cache.k_latent[layer_idx].len() / 512;
            let win_start = prev_total.saturating_sub(127);
            let n_prev_swa = prev_total - win_start;
            let swa_prev_ptr = if n_prev_swa > 0 {
                unsafe { kv_cache.k_latent[layer_idx].as_ptr().add(win_start * 512) }
            } else {
                std::ptr::null()
            };
            let extra_cache: &[f32] = if layer_idx < 2 {
                &[]
            } else {
                let slot = if layer_idx < 8 {
                    0
                } else if layer_idx < 14 {
                    1
                } else if layer_idx < 20 {
                    2
                } else {
                    3
                };
                let full_slice = &kv_cache.compressed_k[slot];
                if let Some(ref cl) = comp_lens_override {
                    let limit = cl[slot].min(full_slice.len());
                    &full_slice[..limit]
                } else {
                    full_slice
                }
            };
            let total_extra = extra_cache.len() / 512;
            let extra_ptr = if total_extra > 0 {
                extra_cache.as_ptr()
            } else {
                std::ptr::null()
            };

            let mut kv_t_out = [0.0_f32; 512];
            let mut attn_out = vec![0.0_f32; 5120];
            let (wqb_w, wqb_s, woa_w, woa_s, wob_w, wob_s, n_swa_arg) =
                if let Some(ref q4) = meta.q4_backbone {
                    (
                        q4.wq_b.w_ptr as *const u8,
                        q4.wq_b.s_ptr as *const u8,
                        q4.wo_a.w_ptr as *const u8,
                        q4.wo_a.s_ptr as *const u8,
                        q4.wo_b.w_ptr as *const u8,
                        q4.wo_b.s_ptr as *const u8,
                        n_prev_swa | (1usize << 31),
                    )
                } else {
                    (
                        meta.wq_b.w_ptr as *const u8,
                        meta.wq_b.s_ptr as *const u8,
                        meta.wo_a.w_ptr as *const u8,
                        meta.wo_a.s_ptr as *const u8,
                        meta.wo_b.w_ptr as *const u8,
                        meta.wo_b.s_ptr as *const u8,
                        n_prev_swa,
                    )
                };
            let ok = unsafe {
                f_mla(
                    meta.wq_a.w_ptr as *const u8,
                    meta.wq_a.s_ptr as *const u8,
                    meta.wkv.w_ptr as *const u8,
                    meta.wkv.s_ptr as *const u8,
                    meta.q_norm.as_ptr(),
                    wqb_w,
                    wqb_s,
                    meta.kv_norm.as_ptr(),
                    meta.attn_sink.as_ptr(),
                    sin_cos.as_ptr() as *const f32,
                    swa_prev_ptr,
                    n_swa_arg,
                    extra_ptr,
                    total_extra,
                    woa_w,
                    woa_s,
                    wob_w,
                    wob_s,
                    normed_x.as_ptr(),
                    kv_t_out.as_mut_ptr(),
                    attn_out.as_mut_ptr(),
                )
            };
            if ok {
                kv_cache.k_latent[layer_idx].extend_from_slice(&kv_t_out);
                return Ok(attn_out);
            }
        }

        // 2b. Fallback Path: Low-rank Query & Shared Latent KV projections
        let q_a = meta.wq_a.matvec(normed_x);
        let kv_raw = meta.wkv.matvec(normed_x);
        let q_a_norm = flash1017_rms_norm(&q_a, &meta.q_norm, 1e-20);
        let mut q_all = meta.wq_b.matvec(&q_a_norm);

        // 3. Shared Latent KV normalization + RoPE (`k = v = kv_t`)
        let mut kv_t = flash1017_rms_norm(&kv_raw, &meta.kv_norm, 1e-20);
        Self::apply_rope_64_table(&mut kv_t[448..512], &sin_cos, false);
        kv_cache.k_latent[layer_idx].extend_from_slice(&kv_t);

        // 4. Online-Softmax Attention across SWA window (`k_latent[layer_idx]`) + compressed extra KV (`compressed_k[slot]`) + `attn_sink [64]`
        let total_swa = kv_cache.k_latent[layer_idx].len() / 512;
        let win_start = total_swa.saturating_sub(128);
        let scale = 1.0_f32 / (512.0_f32).sqrt();
        let swa_cache = &kv_cache.k_latent[layer_idx];
        let extra_cache: &[f32] = if layer_idx < 2 {
            &[]
        } else {
            let slot = if layer_idx < 8 {
                0
            } else if layer_idx < 14 {
                1
            } else if layer_idx < 20 {
                2
            } else {
                3
            };
            let full_slice = &kv_cache.compressed_k[slot];
            if let Some(ref cl) = comp_lens_override {
                let limit = cl[slot].min(full_slice.len());
                &full_slice[..limit]
            } else {
                full_slice
            }
        };
        let total_extra = extra_cache.len() / 512;
        let n_swa = total_swa - win_start;

        let mut attn_heads_out = vec![0.0_f32; 64 * 512];
        q_all
            .par_chunks_exact_mut(512)
            .zip(attn_heads_out.par_chunks_exact_mut(512))
            .enumerate()
            .with_min_len(4)
            .for_each(|(h, (q_h, head_out))| {
                // Fused in-L1 per-head weightless RMSNorm (`q_head_norm = true` in V4.1) + interleaved RoPE on dims 448..512
                let mut sum_sq = 0.0_f32;
                for &v in q_h.iter() {
                    sum_sq += v * v;
                }
                let inv_rms = 1.0_f32 / (sum_sq / 512.0_f32 + 1e-20_f32).sqrt();
                for v in q_h.iter_mut() {
                    *v *= inv_rms;
                }
                Self::apply_rope_64_table(&mut q_h[448..512], &sin_cos, false);

                let sink_logit = meta.attn_sink.get(h).copied().unwrap_or(0.0);
                let mut m_max = sink_logit;
                let mut scores = Vec::with_capacity(n_swa + total_extra);
                for s in win_start..total_swa {
                    let k_s = &swa_cache[s * 512..(s + 1) * 512];
                    let sc = flash1017_dot_f32(q_h, k_s) * scale;
                    if sc > m_max {
                        m_max = sc;
                    }
                    scores.push(sc);
                }
                for c in 0..total_extra {
                    let k_c = &extra_cache[c * 512..(c + 1) * 512];
                    let sc = flash1017_dot_f32(q_h, k_c) * scale;
                    if sc > m_max {
                        m_max = sc;
                    }
                    scores.push(sc);
                }
                // Sink contributes exp(sink_logit - m_max) with zero value vector
                let mut sum_exp = (sink_logit - m_max).exp();
                for sc in &mut scores {
                    *sc = (*sc - m_max).exp();
                    sum_exp += *sc;
                }
                let inv_sum = if sum_exp > 0.0 { 1.0_f32 / sum_exp } else { 0.0 };
                for (idx, s) in (win_start..total_swa).enumerate() {
                    let w = scores[idx] * inv_sum;
                    let v_s = &swa_cache[s * 512..(s + 1) * 512];
                    flash1017_axpy_f32(head_out, w, v_s);
                }
                for c in 0..total_extra {
                    let w = scores[n_swa + c] * inv_sum;
                    let v_c = &extra_cache[c * 512..(c + 1) * 512];
                    flash1017_axpy_f32(head_out, w, v_c);
                }
                // Apply inverse RoPE (`inverse = true`) to dims 448..512 before `wo_a`
                Self::apply_rope_64_table(&mut head_out[448..512], &sin_cos, true);
            });

        // 5. Fused Grouped Output Projection (`F7.A2`): wo_a ([8192, 4096], o_groups = 8) -> wo_b ([5120, 8192]) in 1 single GPU sync
        if let Some(f_wo_ab) = crate::hardware_addon::active_mxfp8_wo_a_b_fused() {
            let mut out = vec![0.0_f32; meta.wo_b.rows];
            let ok = unsafe {
                f_wo_ab(
                    meta.wo_a.rows,
                    meta.wo_a.cols,
                    8,
                    meta.wo_a.w_ptr as *const u8,
                    meta.wo_a.s_ptr as *const u8,
                    meta.wo_b.rows,
                    meta.wo_b.w_ptr as *const u8,
                    meta.wo_b.s_ptr as *const u8,
                    attn_heads_out.as_ptr(),
                    out.as_mut_ptr(),
                )
            };
            if ok {
                return Ok(out);
            }
        }
        let o_lora = meta.wo_a.matvec_grouped(&attn_heads_out, 8);
        Ok(meta.wo_b.matvec(&o_lora))
    }

    /// Extracts the `6144`-dim `O(1)` 24-prime Engram lookup vector for `token_history` on `layer_idx` (`1` or `14`).
    fn extract_engram_concat_embed(
        &self,
        layer_idx: usize,
        engram_shard: &Flash1017ShardInspector,
        token_history: &[u32],
    ) -> Result<Option<Vec<f32>>, String> {
        let Some(&cur_raw_tok) = token_history.last() else {
            return Ok(None);
        };
        if (cur_raw_tok & 0x8000_0000) != 0 || cur_raw_tok == 129264 {
            return Ok(None);
        }

        let (e_meta, e_bytes) =
            engram_shard.get_tensor_data(&format!("layers.{}.engram.embed.weight", layer_idx))?;
        let (_s_meta, s_bytes) =
            engram_shard.get_tensor_data(&format!("layers.{}.engram.embed.scale", layer_idx))?;
        let total_rows = e_meta.shape[0];
        if total_rows == 0 {
            return Ok(None);
        }

        const ENGRAM_MULTS: [[u64; 4]; 2] = [
            [
                76_632_096_046_245,
                4_839_876_093_313,
                35_959_672_319_349,
                73_987_337_458_391,
            ],
            [
                67_716_810_739_261,
                51_510_806_800_915,
                30_921_347_202_721,
                82_619_226_485_591,
            ],
        ];
        const ENGRAM_PRIMES: [[[u64; 8]; 3]; 2] = [
            [
                [16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183],
                [16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321],
                [16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463],
            ],
            [
                [16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627],
                [16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769],
                [16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889],
            ],
        ];

        let l_idx = if layer_idx == 1 { 0usize } else { 1usize };
        let hist_len = token_history.len();
        let pad_comp_id = self
            .engram_token_map
            .get(FLASH1017_PAD_TOKEN_ID as usize)
            .copied()
            .unwrap_or(2) as u64;

        let mut products = [0u64; 4];
        let mut blocked = false;
        for s in 0..4 {
            let comp_id = if blocked || hist_len <= s {
                blocked = true;
                pad_comp_id
            } else {
                let raw_t = token_history[hist_len - 1 - s];
                if (raw_t & 0x8000_0000) != 0 || raw_t == 129264 {
                    blocked = true;
                    pad_comp_id
                } else {
                    self.engram_token_map
                        .get(raw_t as usize)
                        .copied()
                        .unwrap_or(raw_t) as u64
                }
            };
            products[s] = comp_id.wrapping_mul(ENGRAM_MULTS[l_idx][s]);
        }

        let mut concat_embed = vec![0.0_f32; 6144];
        let mut rolling = products[0];
        let mut col_offset = 0usize;

        for n_idx in 0..3usize {
            rolling ^= products[n_idx + 1];
            for head in 0..8usize {
                let table_idx = n_idx * 8 + head;
                let prime = ENGRAM_PRIMES[l_idx][n_idx][head];
                let row_in_table = (rolling % prime) as usize;
                let global_row = (col_offset + row_in_table).min(total_rows - 1);
                col_offset += prime as usize;

                let row_w = &e_bytes[global_row * 128..(global_row + 1) * 128];
                let row_s = &s_bytes[global_row * 8..(global_row + 1) * 8];
                let dst = &mut concat_embed[table_idx * 256..(table_idx + 1) * 256];

                for g in 0..8 {
                    let sc = f8_e8m0_to_f32(row_s[g]);
                    let b_off = g * 16;
                    let d_off = g * 32;
                    for k in 0..16 {
                        let b = row_w[b_off + k];
                        dst[d_off + k * 2] = FP4_E2M1_LUT[(b & 0x0F) as usize] * sc;
                        dst[d_off + k * 2 + 1] = FP4_E2M1_LUT[(b >> 4) as usize] * sc;
                    }
                }
            }
        }
        Ok(Some(concat_embed))
    }

    #[inline]
    fn apply_engram_wkv_to_streams(
        streams: &mut [Vec<f32>; 4],
        wkv_out: &[f32],
        q_w: &[f32],
        k_w: &[f32],
    ) {
        let d = 5120usize;
        let v_engram = &wkv_out[4 * d..5 * d];
        for m in 0..4 {
            let q_m = flash1017_rms_norm(&streams[m], &q_w[m * d..(m + 1) * d], 1e-20);
            let k_m = flash1017_rms_norm(
                &wkv_out[m * d..(m + 1) * d],
                &k_w[m * d..(m + 1) * d],
                1e-20,
            );
            let dot = flash1017_dot_f32(&q_m, &k_m) / (d as f32).sqrt();
            let signed_sqrt = dot.abs().max(1e-6_f32).sqrt().copysign(dot);
            let gate = 1.0_f32 / (1.0_f32 + (-signed_sqrt).exp());
            flash1017_axpy_f32(&mut streams[m], gate, v_engram);
        }
    }

    /// `F3.4`: Executes `O(1)` Zero-Copy `MXFP4` Engram Conditional Memory on Layer 1 (`model-00047`) & Layer 14 (`model-00048`).
    fn forward_engram_memory(
        &self,
        layer_idx: usize,
        streams: &mut [Vec<f32>; 4],
        token_history: &[u32],
    ) -> Result<bool, String> {
        let (Some(engram_shard), wkv_opt, q_w, k_w) = (match layer_idx {
            1 => (
                self.engram_shard_l1.as_ref(),
                self.engram_l1_wkv.as_ref(),
                &self.engram_l1_q,
                &self.engram_l1_k,
            ),
            14 => (
                self.engram_shard_l14.as_ref(),
                self.engram_l14_wkv.as_ref(),
                &self.engram_l14_q,
                &self.engram_l14_k,
            ),
            _ => return Ok(false),
        }) else {
            return Ok(false);
        };

        let Some(concat_embed) =
            self.extract_engram_concat_embed(layer_idx, engram_shard, token_history)?
        else {
            return Ok(false);
        };

        let wkv_out = if let Some(wkv) = wkv_opt {
            wkv.matvec(&concat_embed)
        } else {
            engram_shard.matvec_mxfp8(
                &format!("layers.{}.engram.wkv", layer_idx),
                &concat_embed,
            )?
        };
        Self::apply_engram_wkv_to_streams(streams, &wkv_out, q_w, k_w);
        Ok(true)
    }

    /// `F7.C1`: Batched Engram Conditional Memory across `b_len` tokens in `forward_tokens_batch_hidden`,
    /// reading the `157.3 MB` `layers.{l}.engram.wkv` matrix ONCE via GPU `matbatch` instead of `b_len` times.
    fn forward_engram_memory_batch(
        &self,
        layer_idx: usize,
        streams_batch: &mut [[Vec<f32>; 4]],
        token_history: &[u32],
        start_hist_len: usize,
    ) -> Result<usize, String> {
        let (Some(engram_shard), wkv_opt, q_w, k_w) = (match layer_idx {
            1 => (
                self.engram_shard_l1.as_ref(),
                self.engram_l1_wkv.as_ref(),
                &self.engram_l1_q,
                &self.engram_l1_k,
            ),
            14 => (
                self.engram_shard_l14.as_ref(),
                self.engram_l14_wkv.as_ref(),
                &self.engram_l14_q,
                &self.engram_l14_k,
            ),
            _ => return Ok(0),
        }) else {
            return Ok(0);
        };

        let b_len = streams_batch.len();
        let mut active_indices = Vec::with_capacity(b_len);
        let mut concat_batch = Vec::with_capacity(b_len * 6144);
        for t in 0..b_len {
            let hist_slice = &token_history[..start_hist_len + t + 1];
            if let Some(emb) =
                self.extract_engram_concat_embed(layer_idx, engram_shard, hist_slice)?
            {
                active_indices.push(t);
                concat_batch.extend_from_slice(&emb);
            }
        }
        let m = active_indices.len();
        if m == 0 {
            return Ok(0);
        }

        let wkv_batch = if let Some(wkv) = wkv_opt {
            wkv.matbatch(&concat_batch, m)
        } else {
            engram_shard.matbatch_mxfp8(
                &format!("layers.{}.engram.wkv", layer_idx),
                &concat_batch,
                m,
            )?
        };
        let rows = 25600usize;
        for (k_idx, &t) in active_indices.iter().enumerate() {
            let wkv_out = &wkv_batch[k_idx * rows..(k_idx + 1) * rows];
            Self::apply_engram_wkv_to_streams(&mut streams_batch[t], wkv_out, q_w, k_w);
        }
        Ok(m)
    }

    /// `t = 0 ms` Engram Pre-Lookup: Computes the exact 24 prime row offsets in `engram_shard_l1` (Layer 1)
    /// and `engram_shard_l14` (Layer 14) directly from `token_history` BEFORE Layer 0 starts and issues
    /// non-blocking `madvise(MADV_WILLNEED)` on those `2.7 KB` SSD `mmap` slices (`0.00 ms` wait at Layer 1 & 14).
    #[inline]
    pub fn prefetch_engram_at_t0(&self, token_history: &[u32]) {
        #[cfg(unix)]
        {
            let Some(&cur_raw_tok) = token_history.last() else {
                return;
            };
            if (cur_raw_tok & 0x8000_0000) != 0 || cur_raw_tok == 129264 {
                return;
            }
            extern "C" {
                fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            const MADV_WILLNEED: i32 = 3;
            const ENGRAM_MULTS: [[u64; 4]; 2] = [
                [
                    76_632_096_046_245,
                    4_839_876_093_313,
                    35_959_672_319_349,
                    73_987_337_458_391,
                ],
                [
                    67_716_810_739_261,
                    51_510_806_800_915,
                    30_921_347_202_721,
                    82_619_226_485_591,
                ],
            ];
            const ENGRAM_PRIMES: [[[u64; 8]; 3]; 2] = [
                [
                    [16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183],
                    [16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321],
                    [16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463],
                ],
                [
                    [16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627],
                    [16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769],
                    [16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889],
                ],
            ];
            let hist_len = token_history.len();
            let pad_comp_id = self
                .engram_token_map
                .get(FLASH1017_PAD_TOKEN_ID as usize)
                .copied()
                .unwrap_or(2) as u64;

            for (l_idx, (layer_idx, shard_opt)) in [
                (1usize, self.engram_shard_l1.as_ref()),
                (14usize, self.engram_shard_l14.as_ref()),
            ]
            .into_iter()
            .enumerate()
            {
                let Some(engram_shard) = shard_opt else {
                    continue;
                };
                let Ok((e_meta, e_bytes)) =
                    engram_shard.get_tensor_data(&format!("layers.{}.engram.embed.weight", layer_idx))
                else {
                    continue;
                };
                let total_rows = e_meta.shape[0];
                if total_rows == 0 {
                    continue;
                }
                let mut products = [0u64; 4];
                let mut blocked = false;
                for s in 0..4 {
                    let comp_id = if blocked || hist_len <= s {
                        blocked = true;
                        pad_comp_id
                    } else {
                        let raw_t = token_history[hist_len - 1 - s];
                        if (raw_t & 0x8000_0000) != 0 || raw_t == 129264 {
                            blocked = true;
                            pad_comp_id
                        } else {
                            self.engram_token_map
                                .get(raw_t as usize)
                                .copied()
                                .unwrap_or(raw_t) as u64
                        }
                    };
                    products[s] = comp_id.wrapping_mul(ENGRAM_MULTS[l_idx][s]);
                }
                let mut rolling = products[0];
                let mut col_offset = 0usize;
                for n_idx in 0..3usize {
                    rolling ^= products[n_idx + 1];
                    for head in 0..8usize {
                        let prime = ENGRAM_PRIMES[l_idx][n_idx][head];
                        let row_in_table = (rolling % prime) as usize;
                        let global_row = (col_offset + row_in_table).min(total_rows - 1);
                        col_offset += prime as usize;
                        let off = global_row * 128;
                        if off + 128 <= e_bytes.len() {
                            unsafe {
                                let _ = madvise(
                                    e_bytes.as_ptr().add(off) as *mut std::ffi::c_void,
                                    128,
                                    MADV_WILLNEED,
                                );
                            }
                        }
                    }
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = token_history;
        }
    }

    /// `CpuLookaheadStager` (Cross-Layer Lookahead): Evaluates the upcoming layer's router gate (`ffn_gate`)
    /// on the current normalized state and issues non-blocking `madvise(MADV_WILLNEED)` for any selected
    /// unpinned `Tier C` experts 1 layer ahead of the GPU.
    #[inline]
    fn prefetch_next_layer_experts_lookahead(
        &self,
        next_layer_idx: usize,
        cur_normed_x: &[f32],
        is_vision_token: bool,
    ) {
        #[cfg(unix)]
        {
            if !self.tier_c_disk_enabled || next_layer_idx >= self.layer_meta.len() {
                return;
            }
            let next_meta = &self.layer_meta[next_layer_idx];
            let active_bias = if is_vision_token {
                &next_meta.gate_bias_vl
            } else {
                &next_meta.gate_bias
            };
            let raw_logits = next_meta.ffn_gate.matvec(cur_normed_x);
            let mut candidates = [(0usize, 0.0_f32); 384];
            for e in 0..384 {
                let l = raw_logits[e];
                let softplus = if l > 20.0 {
                    l
                } else if l < -20.0 {
                    l.exp()
                } else {
                    (1.0_f32 + l.exp()).ln()
                };
                let score = softplus.max(0.0).sqrt();
                candidates[e] = (e, score + active_bias.get(e).copied().unwrap_or(0.0));
            }
            candidates.select_nth_unstable_by(6, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            extern "C" {
                fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            const MADV_WILLNEED: i32 = 3;
            for &(exp_idx, _) in &candidates[..6] {
                if !next_meta.tier_a_set[exp_idx] {
                    if let Some((ref w1, ref w2, ref w3)) = next_meta.tier_a_experts[exp_idx] {
                        unsafe {
                            let _ = madvise(w1.w_ptr as *mut std::ffi::c_void, w1.w_len, MADV_WILLNEED);
                            let _ = madvise(w3.w_ptr as *mut std::ffi::c_void, w3.w_len, MADV_WILLNEED);
                            let _ = madvise(w2.w_ptr as *mut std::ffi::c_void, w2.w_len, MADV_WILLNEED);
                        }
                    }
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (next_layer_idx, cur_normed_x, is_vision_token);
        }
    }

    /// `CpuLookaheadStager` (`t = 0 ms` Prompt Topic Pre-Stager): Uses all 20 Grace ARM64 CPU cores (`rayon`)
    /// to scan the prompt's unique token embeddings against the 40 layers' router gates and pre-stage
    /// activated `Tier C` experts into OS page cache before decode begins.
    pub fn cpu_lookahead_stage_prompt_tokens(&self, prompt_tokens: &[u32]) -> (usize, f64) {
        let t0 = std::time::Instant::now();
        if !self.tier_c_disk_enabled || prompt_tokens.is_empty() {
            return (0, 0.0);
        }
        self.prefetch_engram_at_t0(prompt_tokens);
        // Sample up to the last 8 unique tokens from the prompt to represent the topic vector
        let mut unique_toks = Vec::with_capacity(8);
        for &tid in prompt_tokens.iter().rev() {
            if (tid & 0x8000_0000) == 0 && tid != 129264 && !unique_toks.contains(&tid) {
                unique_toks.push(tid);
                if unique_toks.len() >= 8 {
                    break;
                }
            }
        }
        if unique_toks.is_empty() {
            return (0, t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mut mean_emb = vec![0.0_f32; 5120];
        let mut count = 0usize;
        for &tid in &unique_toks {
            if let Ok(emb) = self.get_token_embedding(tid) {
                for i in 0..5120 {
                    mean_emb[i] += emb[i];
                }
                count += 1;
            }
        }
        if count == 0 {
            return (0, t0.elapsed().as_secs_f64() * 1000.0);
        }
        let inv = 1.0_f32 / (count as f32);
        for v in &mut mean_emb {
            *v *= inv;
        }

        use rayon::prelude::*;
        let staged_count: usize = self
            .layer_meta
            .par_iter()
            .map(|meta| {
                let normed = flash1017_rms_norm(&mean_emb, &meta.ffn_norm, 1e-20);
                let raw_logits = meta.ffn_gate.matvec(&normed);
                let mut candidates = [(0usize, 0.0_f32); 384];
                for e in 0..384 {
                    let l = raw_logits[e];
                    let softplus = if l > 20.0 {
                        l
                    } else if l < -20.0 {
                        l.exp()
                    } else {
                        (1.0_f32 + l.exp()).ln()
                    };
                    let score = softplus.max(0.0).sqrt();
                    candidates[e] = (e, score + meta.gate_bias.get(e).copied().unwrap_or(0.0));
                }
                candidates.select_nth_unstable_by(6, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut hits = 0usize;
                #[cfg(unix)]
                {
                    extern "C" {
                        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
                    }
                    const MADV_WILLNEED: i32 = 3;
                    for &(exp_idx, _) in &candidates[..6] {
                        if !meta.tier_a_set[exp_idx] {
                            if let Some((ref w1, ref w2, ref w3)) = meta.tier_a_experts[exp_idx] {
                                unsafe {
                                    let _ = madvise(
                                        w1.w_ptr as *mut std::ffi::c_void,
                                        w1.w_len,
                                        MADV_WILLNEED,
                                    );
                                    let _ = madvise(
                                        w3.w_ptr as *mut std::ffi::c_void,
                                        w3.w_len,
                                        MADV_WILLNEED,
                                    );
                                    let _ = madvise(
                                        w2.w_ptr as *mut std::ffi::c_void,
                                        w2.w_len,
                                        MADV_WILLNEED,
                                    );
                                }
                                hits += 1;
                            }
                        }
                    }
                }
                hits
            })
            .sum();

        (staged_count, t0.elapsed().as_secs_f64() * 1000.0)
    }

    /// `JEPA Autopilot` (Active Inter-Layer Noise-Cancelling): At layer checkpoints (`l == 15` and `l == 31`),
    /// measures the alignment of the 4 `mHC` streams against the verified `jepa_trajectory_anchor` and gently
    /// damps orthogonal quantization/rounding drift while preserving each stream's exact RMS norm (`< 0.02 ms`).
    #[inline]
    fn apply_jepa_inter_layer_noise_canceller(
        layer_idx: usize,
        streams: &mut [Vec<f32>; 4],
        jepa_anchor: &[f32],
    ) {
        if (layer_idx != 15 && layer_idx != 31) || jepa_anchor.len() != 5120 {
            return;
        }
        let anchor_norm_sq = flash1017_dot_f32(jepa_anchor, jepa_anchor);
        if anchor_norm_sq <= 1e-12 {
            return;
        }
        let inv_anchor_norm = 1.0_f32 / anchor_norm_sq.sqrt();
        for s in streams.iter_mut() {
            if s.len() != 5120 {
                continue;
            }
            let orig_sq = flash1017_dot_f32(s, s);
            if orig_sq <= 1e-12 {
                continue;
            }
            let orig_norm = orig_sq.sqrt();
            let cos = (flash1017_dot_f32(s, jepa_anchor) * inv_anchor_norm) / orig_norm;
            // Only apply gentle trajectory stabilization when the stream is within the semantic cone
            if cos > 0.05 && cos < 0.98 {
                let alpha = 0.008_f32 * orig_norm * inv_anchor_norm;
                for i in 0..5120 {
                    s[i] += alpha * jepa_anchor[i];
                }
                let new_sq = flash1017_dot_f32(s, s);
                if new_sq > 1e-12 {
                    let rescale = orig_norm / new_sq.sqrt();
                    for i in 0..5120 {
                        s[i] *= rescale;
                    }
                }
            }
        }
    }

    /// `JEPA Autopilot` (Speculative MTP Trajectory Coherence): Computes the cosine coherence between
    /// the speculative `est_h39` state and the ongoing `jepa_trajectory_anchor` (`[-1.0, 1.0]`).
    #[inline]
    pub fn jepa_mtp_coherence_score(&self, est_h39: &[f32], jepa_anchor: &[f32]) -> f32 {
        if est_h39.len() != 5120 || jepa_anchor.len() != 5120 {
            return 0.0;
        }
        let dot = flash1017_dot_f32(est_h39, jepa_anchor);
        let na = flash1017_dot_f32(est_h39, est_h39).max(1e-20).sqrt();
        let nb = flash1017_dot_f32(jepa_anchor, jepa_anchor).max(1e-20).sqrt();
        dot / (na * nb)
    }

    /// `F3.5` & `F7.A1–A2`: Executes Surgical 3-Tier Sparse MoE FFN (`Top-6` of `384` with Tier A `NVFP4`, Tier B `1.58-bit`, Tier C `0-ns skip` + `shared_experts`)
    /// using pre-cached expert pointers and fused in-stream GPU MLP accumulation (`1` GPU sync per layer instead of `21`).
    fn forward_moe_ffn(
        &self,
        layer_idx: usize,
        normed_x: &[f32],
        is_vision_token: bool,
    ) -> Result<(Vec<f32>, usize, usize, usize), String> {
        let meta = &self.layer_meta[layer_idx];
        let d = 5120usize;
        let inter = 2304usize;
        let active_bias = if is_vision_token {
            &meta.gate_bias_vl
        } else {
            &meta.gate_bias
        };
        let allow_all_experts = self.tier_c_disk_enabled || is_vision_token;

        // 1. Router gate: BF16 [384, 5120] + sqrtsoftplus + noaux_tc bias selection (F9.3 zero-allocation O(N) Top-6)
        let raw_logits = meta.ffn_gate.matvec(normed_x);
        let mut candidates = [(0usize, 0.0_f32, 0.0_f32); 384];
        for e in 0..384 {
            let l = raw_logits[e];
            let softplus = if l > 20.0 {
                l
            } else if l < -20.0 {
                l.exp()
            } else {
                (1.0_f32 + l.exp()).ln()
            };
            let score = softplus.max(0.0).sqrt();
            let biased = score + active_bias.get(e).copied().unwrap_or(0.0);
            candidates[e] = (e, biased, score);
        }
        candidates.select_nth_unstable_by(6, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates[..6].sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut active_selected: Vec<(usize, f32, bool)> = Vec::with_capacity(6);
        let mut tier_a_hits = 0usize;
        let mut tier_b_hits = 0usize;
        let mut tier_c_skips = 0usize;

        for &(e, _biased, score) in &candidates[..6] {
            if meta.tier_a_set[e] {
                let use_nvfp4 = meta.tier_a_experts[e].is_some();
                active_selected.push((e, score, use_nvfp4));
                tier_a_hits += 1;
            } else if meta.tier_b_set[e] {
                let use_nvfp4 = (allow_all_experts || meta.tier_b_experts[e].is_none())
                    && meta.tier_a_experts[e].is_some();
                active_selected.push((e, score, use_nvfp4));
                tier_b_hits += 1;
            } else if meta.tier_b_experts[e].is_some() {
                // `106 GB` All-Experts-in-RAM (`1.58-bit + 0.33 MB Rank-16 Q8 JEPA Core`) -> 0 skips!
                active_selected.push((e, score, false));
                tier_b_hits += 1;
            } else if allow_all_experts && meta.tier_a_experts[e].is_some() {
                active_selected.push((e, score, true));
                tier_a_hits += 1;
            } else if !allow_all_experts {
                // Tier C pruned fact expert -> 0 ns skip!
                tier_c_skips += 1;
            }
        }

        // If `allow_all_experts` is true and unpruned NVFP4 disk shards weren't present for some slot,
        // fill up to 6 active experts from the highest-scoring Tier A/B candidates so 0 experts are skipped!
        if allow_all_experts && active_selected.len() < 6 {
            for &(e, _biased, score) in &candidates[6..] {
                if active_selected.len() >= 6 {
                    break;
                }
                if meta.tier_a_set[e] {
                    active_selected.push((e, score, true));
                    tier_a_hits += 1;
                } else if meta.tier_b_set[e] {
                    let use_nvfp4 = meta.tier_a_experts[e].is_some();
                    active_selected.push((e, score, use_nvfp4));
                    tier_b_hits += 1;
                }
            }
        }

        // Fallback if all 6 top candidates were pruned Tier C experts: pick top available Tier A/B expert
        if active_selected.is_empty() {
            let mut best_fb: Option<(usize, f32, f32, bool)> = None;
            for &(e, biased, score) in &candidates[6..] {
                let is_a = meta.tier_a_set[e];
                let is_b = !is_a && meta.tier_b_set[e];
                if is_a || is_b {
                    if best_fb.map_or(true, |b| biased > b.1) {
                        best_fb = Some((e, biased, score, is_a));
                    }
                }
            }
            if let Some((e, _biased, score, is_a)) = best_fb {
                let use_nvfp4 = is_a
                    || ((allow_all_experts || meta.tier_b_experts[e].is_none())
                        && meta.tier_a_experts[e].is_some());
                active_selected.push((e, score, use_nvfp4));
                if is_a {
                    tier_a_hits += 1;
                } else {
                    tier_b_hits += 1;
                }
            }
        }

        let active_score_sum: f32 = active_selected.iter().map(|x| x.1).sum();
        let routed_scale = if active_score_sum > 1e-12 {
            1.5_f32 / active_score_sum
        } else {
            0.0_f32
        };

        // Cross-Layer CPU Lookahead: pre-warm Layer `layer_idx + 1` experts while Layer `layer_idx` runs on GPU
        self.prefetch_next_layer_experts_lookahead(layer_idx + 1, normed_x, is_vision_token);

        #[cfg(unix)]
        {
            extern "C" {
                fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            const MADV_WILLNEED: i32 = 3;
            for &(exp_idx, _raw_score, is_tier_a) in &active_selected {
                if is_tier_a && !meta.tier_a_set[exp_idx] {
                    if let Some((ref w1, ref w2, ref w3)) = meta.tier_a_experts[exp_idx] {
                        unsafe {
                            let _ = madvise(w1.w_ptr as *mut std::ffi::c_void, w1.w_len, MADV_WILLNEED);
                            let _ = madvise(w3.w_ptr as *mut std::ffi::c_void, w3.w_len, MADV_WILLNEED);
                            let _ = madvise(w2.w_ptr as *mut std::ffi::c_void, w2.w_len, MADV_WILLNEED);
                        }
                    }
                }
            }
        }

        // Fast Path (`F7.A2`): Fused in-stream GPU MLP pipeline (`w1+w3+SwiGLU -> w2 -> axpy` across shared + all routed experts in 1 single GPU sync)
        if let (Some(f_mxfp8_mlp), Some(f_nvfp4_mlp), Some(f_tern_mlp)) = (
            crate::hardware_addon::active_mxfp8_mlp_accum(),
            crate::hardware_addon::active_nvfp4_mlp_accum(),
            crate::hardware_addon::active_ternary_158_mlp_accum(),
        ) {
            let mut ffn_out = vec![0.0_f32; d];
            let mut all_ok = if let Some(ref q4) = meta.q4_backbone {
                unsafe {
                    f_nvfp4_mlp(
                        inter,
                        d,
                        q4.shared_w1.w_ptr as *const u8,
                        q4.shared_w1.s_ptr as *const u8,
                        q4.shared_w1.global_scale,
                        q4.shared_w3.w_ptr as *const u8,
                        q4.shared_w3.s_ptr as *const u8,
                        q4.shared_w3.global_scale,
                        q4.shared_w2.w_ptr as *const u8,
                        q4.shared_w2.s_ptr as *const u8,
                        q4.shared_w2.global_scale,
                        normed_x.as_ptr(),
                        ffn_out.as_mut_ptr(),
                        1.0,
                        false,
                        active_selected.is_empty(),
                    )
                }
            } else {
                unsafe {
                    f_mxfp8_mlp(
                        inter,
                        d,
                        meta.shared_w1.w_ptr as *const u8,
                        meta.shared_w1.s_ptr as *const u8,
                        meta.shared_w3.w_ptr as *const u8,
                        meta.shared_w3.s_ptr as *const u8,
                        meta.shared_w2.w_ptr as *const u8,
                        meta.shared_w2.s_ptr as *const u8,
                        normed_x.as_ptr(),
                        ffn_out.as_mut_ptr(),
                        1.0,
                        false,
                        active_selected.is_empty(),
                    )
                }
            };
            if all_ok {
                let n_sel = active_selected.len();
                for (idx, &(exp_idx, raw_score, is_tier_a)) in active_selected.iter().enumerate() {
                    let is_last = idx + 1 == n_sel;
                    let weight = raw_score * routed_scale;
                    if is_tier_a {
                        if let Some((ref w1, ref w2, ref w3)) = meta.tier_a_experts[exp_idx] {
                            let ok = unsafe {
                                f_nvfp4_mlp(
                                    inter,
                                    d,
                                    w1.w_ptr as *const u8,
                                    w1.s_ptr as *const u8,
                                    w1.global_scale,
                                    w3.w_ptr as *const u8,
                                    w3.s_ptr as *const u8,
                                    w3.global_scale,
                                    w2.w_ptr as *const u8,
                                    w2.s_ptr as *const u8,
                                    w2.global_scale,
                                    normed_x.as_ptr(),
                                    ffn_out.as_mut_ptr(),
                                    weight,
                                    true,
                                    is_last,
                                )
                            };
                            if !ok {
                                all_ok = false;
                                break;
                            }
                        } else {
                            all_ok = false;
                            break;
                        }
                    } else if let Some((ref w1, ref w2, ref w3)) = meta.tier_b_experts[exp_idx] {
                        if w1.jepa_core.is_some() || w2.jepa_core.is_some() || w3.jepa_core.is_some() {
                            all_ok = false;
                            break;
                        }
                        let sc = meta.tier_b_proj_scale;
                        let ok = unsafe {
                            f_tern_mlp(
                                inter,
                                d,
                                w1.p_ptr,
                                w1.s_ptr as *const f32,
                                w3.p_ptr,
                                w3.s_ptr as *const f32,
                                w2.p_ptr,
                                w2.s_ptr as *const f32,
                                sc,
                                normed_x.as_ptr(),
                                ffn_out.as_mut_ptr(),
                                weight * sc,
                                true,
                                is_last,
                            )
                        };
                        if !ok {
                            all_ok = false;
                            break;
                        }
                    } else {
                        all_ok = false;
                        break;
                    }
                }
            }
            if all_ok {
                return Ok((ffn_out, tier_a_hits, tier_b_hits, tier_c_skips));
            }
        }

        // Fallback Path (Pre-cached descriptors, 0 string allocations or HashMap lookups)
        let sh_w1 = meta.shared_w1.matvec(normed_x);
        let sh_w3 = meta.shared_w3.matvec(normed_x);
        let mut sh_act = Vec::with_capacity(inter);
        for i in 0..inter {
            let g = sh_w1[i];
            let silu = (g / (1.0_f32 + (-g).exp())).min(10.0);
            let u = sh_w3[i].clamp(-10.0, 10.0);
            sh_act.push(silu * u);
        }
        let mut ffn_out = meta.shared_w2.matvec(&sh_act);

        for (exp_idx, raw_score, is_tier_a) in active_selected {
            let weight = raw_score * routed_scale;
            if is_tier_a {
                if let Some((ref w1, ref w2, ref w3)) = meta.tier_a_experts[exp_idx] {
                    let w1_out = w1.matvec(normed_x);
                    let w3_out = w3.matvec(normed_x);
                    let mut act = Vec::with_capacity(inter);
                    for i in 0..inter {
                        let g = w1_out[i];
                        let silu = (g / (1.0_f32 + (-g).exp())).min(10.0);
                        let u = w3_out[i].clamp(-10.0, 10.0);
                        act.push(silu * u);
                    }
                    let w2_out = w2.matvec(&act);
                    flash1017_axpy_f32(&mut ffn_out, weight, &w2_out);
                }
            } else if let Some((ref w1, ref w2, ref w3)) = meta.tier_b_experts[exp_idx] {
                let sc = meta.tier_b_proj_scale;
                let w1_out = w1.matvec(normed_x);
                let w3_out = w3.matvec(normed_x);
                let mut act = Vec::with_capacity(inter);
                for i in 0..inter {
                    let g = w1_out[i] * sc;
                    let silu = (g / (1.0_f32 + (-g).exp())).min(10.0);
                    let u = (w3_out[i] * sc).clamp(-10.0, 10.0);
                    act.push(silu * u);
                }
                let w2_out = w2.matvec(&act);
                flash1017_axpy_f32(&mut ffn_out, weight * sc, &w2_out);
            }
        }

        debug_assert_eq!(ffn_out.len(), d);
        Ok((ffn_out, tier_a_hits, tier_b_hits, tier_c_skips))
    }

    /// Runs a single token `token_id` through all `max_layers` (`1..=40`) of `DeepSeek-V4.1-Flash-1017-Tier2`
    /// and returns `(final_hidden_5120, Flash1017StepTelemetry)`.
    pub fn forward_token_hidden(
        &self,
        token_id: u32,
        kv_cache: &mut Flash1017KvCache,
        max_layers: usize,
    ) -> Result<(Vec<f32>, Flash1017StepTelemetry), String> {
        let emb = self.get_token_embedding(token_id)?;
        self.forward_embedding_hidden(&emb, token_id, kv_cache, max_layers, 1.0)
    }

    /// `F5.2` & `F5.3`: Runs an arbitrary `5120`-dim input embedding `emb_5120` (either a token embedding or a
    /// `SilentVectorLoop` continuous latent thought vector) through `max_layers` (`1..=40`) of `Flash-1017-Tier2`,
    /// applying optional `layer_dora_deltas` scaled by `dora_scale`.
    pub fn forward_embedding_hidden(
        &self,
        emb_5120: &[f32],
        token_id_for_engram: u32,
        kv_cache: &mut Flash1017KvCache,
        max_layers: usize,
        dora_scale: f32,
    ) -> Result<(Vec<f32>, Flash1017StepTelemetry), String> {
        if emb_5120.len() != 5120 {
            return Err(format!(
                "forward_embedding_hidden expected 5120 floats, got {}",
                emb_5120.len()
            ));
        }
        let t0 = std::time::Instant::now();
        kv_cache.ensure_capacity_for_additional_tokens(1);
        let pos = kv_cache.seq_len;
        kv_cache.token_history.push(token_id_for_engram);
        // `t = 0 ms` Engram Pre-Lookup for Layer 1 & Layer 14 before Layer 0 starts
        self.prefetch_engram_at_t0(&kv_cache.token_history);

        let emb = emb_5120.to_vec();
        let mut streams: [Vec<f32>; 4] = [emb.clone(), emb.clone(), emb.clone(), emb];
        let mut prev_pre: Option<[f32; 4]> = None;

        let active_layers = max_layers.clamp(1, 40);
        let mut total_a = 0usize;
        let mut total_b = 0usize;
        let mut total_c = 0usize;
        let mut engram_lookups = 0usize;

        for l in 0..active_layers {
            // Optional Engram Conditional Memory on Layer 1 & Layer 14
            if l == 1 || l == 14 {
                if self.forward_engram_memory(l, &mut streams, &kv_cache.token_history)? {
                    engram_lookups += 1;
                }
            }

            // Sub-layer 1: mHC Pre (`hc_pre_from_prev_sublayer = true`) -> attn_norm -> MLA Attention -> mHC Post
            let attn_in = match prev_pre {
                Some(ref p) => Self::combine_mhc_streams(&streams, p),
                None => streams[0].clone(),
            };
            let (attn_pre, attn_w_post, attn_m) = self.compute_mhc_stats(l, &streams, false)?;
            let normed_attn_in =
                flash1017_rms_norm(&attn_in, &self.layer_meta[l].attn_norm, 1e-20);
            let attn_out = self.forward_mla_attention(l, &normed_attn_in, pos, kv_cache)?;
            Self::apply_mhc_post(&mut streams, &attn_out, &attn_w_post, &attn_m);

            // Sub-layer 2: mHC Pre (consumes `attn_pre`) -> ffn_norm -> Surgical Tier A/B/C Sparse MoE (+ F5.2 DoRA) -> mHC Post
            let ffn_in = Self::combine_mhc_streams(&streams, &attn_pre);
            let (ffn_pre, ffn_w_post, ffn_m) = self.compute_mhc_stats(l, &streams, true)?;
            let normed_ffn_in = flash1017_rms_norm(&ffn_in, &self.layer_meta[l].ffn_norm, 1e-20);
            let is_vis_tok = (token_id_for_engram & 0x8000_0000) != 0 || token_id_for_engram == 129264;
            let (mut ffn_out, a_hits, b_hits, c_skips) = self.forward_moe_ffn(l, &normed_ffn_in, is_vis_tok)?;
            if dora_scale.abs() > 1e-6 {
                if let Some(Some(ref delta)) = self.layer_dora_deltas.get(l) {
                    flash1017_axpy_f32(&mut ffn_out, dora_scale, delta);
                }
            }
            total_a += a_hits;
            total_b += b_hits;
            total_c += c_skips;
            Self::apply_mhc_post(&mut streams, &ffn_out, &ffn_w_post, &ffn_m);
            Self::apply_jepa_inter_layer_noise_canceller(
                l,
                &mut streams,
                &kv_cache.jepa_trajectory_anchor,
            );
            if (37..=39).contains(&l) {
                kv_cache.dspark_target_h[l - 37] = Self::combine_mhc_streams(&streams, &ffn_pre);
            }
            prev_pre = Some(ffn_pre);
        }

        kv_cache.seq_len = pos + 1;

        // Combine the 4 residual streams with `last_pre` (`hc_pre_from_prev_sublayer = true`) and apply final `norm.weight` ([5120])
        let last_pre = prev_pre.unwrap_or([0.25_f32; 4]);
        let combined = Self::combine_mhc_streams(&streams, &last_pre);
        let final_normed = flash1017_rms_norm(&combined, &self.final_norm, 1e-20);
        kv_cache.update_jepa_trajectory(&final_normed);

        let telemetry = Flash1017StepTelemetry {
            pos,
            tier_a_hits: total_a,
            tier_b_hits: total_b,
            tier_c_skips: total_c,
            engram_lookups,
            step_ms: t0.elapsed().as_secs_f64() * 1000.0,
        };
        Ok((final_normed, telemetry))
    }

    /// `F7.C1`: Verifies `b_len = token_ids.len()` (`2..=5`) consecutive tokens (`[accepted_tok, draft_1, ..., draft_K]`)
    /// in parallel through all `max_layers` (`1..=40`) of `DeepSeek-V4.1-Flash-1017-Tier2` using 4-token register-tiled
    /// GPU `matbatch` kernels (`matbatch_mxfp8`, `matbatch_bf16`, `matbatch_nvfp4`, `matbatch_ternary_158`).
    ///
    /// Returns `(per_token_states, telemetry)`, where `per_token_states[t]` holds the exact causal `final_normed` hidden vector
    /// and complete KV/compressor/dspark snapshot after token `t` (`0..b_len`), enabling `0-drift` rollback on the first rejected draft token.
    pub fn forward_tokens_batch_hidden(
        &self,
        token_ids: &[u32],
        kv_cache: &mut Flash1017KvCache,
        max_layers: usize,
        dora_scales: &[f32],
    ) -> Result<(Vec<Flash1017PositionState>, Flash1017StepTelemetry), String> {
        let b_len = token_ids.len();
        if b_len == 0 {
            return Err("forward_tokens_batch_hidden called with empty token_ids".to_string());
        }
        let t0 = std::time::Instant::now();
        let start_pos = kv_cache.seq_len;
        kv_cache.ensure_capacity_for_additional_tokens(b_len);
        let start_hist_len = kv_cache.token_history.len();
        kv_cache.token_history.extend_from_slice(token_ids);
        // `t = 0 ms` Engram Pre-Lookup for all batch tokens before Layer 0 starts
        for t in 0..b_len {
            self.prefetch_engram_at_t0(&kv_cache.token_history[..start_hist_len + t + 1]);
        }

        let d = 5120usize;
        let inter = 2304usize;
        let active_layers = max_layers.clamp(1, 40);

        let mut streams_batch: Vec<[Vec<f32>; 4]> = Vec::with_capacity(b_len);
        for &tid in token_ids {
            let emb = self.get_token_embedding(tid)?;
            streams_batch.push([emb.clone(), emb.clone(), emb.clone(), emb]);
        }
        let mut prev_pre_batch: Vec<Option<[f32; 4]>> = vec![None; b_len];

        let init_comp_lens = [
            kv_cache.compressed_k[0].len(),
            kv_cache.compressed_k[1].len(),
            kv_cache.compressed_k[2].len(),
            kv_cache.compressed_k[3].len(),
        ];
        let mut per_tok_comp_lens: Vec<[usize; 4]> = vec![init_comp_lens; b_len];
        let mut per_tok_pending_kv: Vec<[Vec<f32>; 3]> = vec![kv_cache.pending_c2_kv.clone(); b_len];
        let mut per_tok_pending_score: Vec<[Vec<f32>; 3]> =
            vec![kv_cache.pending_c2_score.clone(); b_len];
        let mut per_tok_dspark_h: Vec<[Vec<f32>; 3]> =
            vec![kv_cache.dspark_target_h.clone(); b_len];

        let mut total_a = 0usize;
        let mut total_b = 0usize;
        let mut total_c = 0usize;
        let mut engram_lookups = 0usize;

        for l in 0..active_layers {
            let meta = &self.layer_meta[l];

            // 1. Optional Engram Conditional Memory on Layer 1 & Layer 14 (Batched GPU `matbatch` on `layers.{l}.engram.wkv`)
            if l == 1 || l == 14 {
                engram_lookups += self.forward_engram_memory_batch(
                    l,
                    &mut streams_batch,
                    &kv_cache.token_history,
                    start_hist_len,
                )?;
            }

            // 2. Sub-layer 1: Batched GPU mHC Pre + attn_norm + In-Stream Multi-Token GPU MLA (`f_mla_batch`: 1 single read of 126.6 MB MLA weights & 1 single GPU sync)
            let mhc_attn_stats = self.compute_mhc_stats_batch(l, &streams_batch, false)?;
            let mut attn_pre_batch = Vec::with_capacity(b_len);
            let mut attn_w_post_batch = Vec::with_capacity(b_len);
            let mut attn_m_batch = Vec::with_capacity(b_len);
            let mut normed_a_batch = Vec::with_capacity(b_len * d);
            for (t, (a_pre, a_w_post, a_m)) in mhc_attn_stats.into_iter().enumerate() {
                let attn_in = match prev_pre_batch[t] {
                    Some(ref p) => Self::combine_mhc_streams(&streams_batch[t], p),
                    None => streams_batch[t][0].clone(),
                };
                let normed_a = flash1017_rms_norm(&attn_in, &meta.attn_norm, 1e-20);
                attn_pre_batch.push(a_pre);
                attn_w_post_batch.push(a_w_post);
                attn_m_batch.push(a_m);
                normed_a_batch.extend_from_slice(&normed_a);
            }

            let mut mla_batch_done = false;
            if let Some(f_mla_batch) = crate::hardware_addon::active_mla_batch_forward() {
                // 2a. Update compressed global KV cache on source layers [2, 8, 14, 20] BEFORE attention
                if let Some(comp_norm) = &meta.compressor_norm {
                    if l == 20 {
                        if let Some(ref comp_wkv) = meta.comp_wkv {
                            let comp_kv_batch = comp_wkv.matbatch(&normed_a_batch, b_len);
                            for t in 0..b_len {
                                let pos = start_pos + t;
                                let sin_cos = self.precompute_rope_sin_cos(l, pos);
                                let mut latent = flash1017_rms_norm(
                                    &comp_kv_batch[t * 512..(t + 1) * 512],
                                    comp_norm,
                                    1e-20,
                                );
                                Self::apply_rope_64_table(&mut latent[448..512], &sin_cos, false);
                                kv_cache.compressed_k[3].extend_from_slice(&latent);
                                per_tok_comp_lens[t][3] = kv_cache.compressed_k[3].len();
                            }
                        }
                    } else {
                        let c2_slot = match l {
                            2 => Some(0usize),
                            8 => Some(1usize),
                            14 => Some(2usize),
                            _ => None,
                        };
                        if let (Some(slot), Some(ref comp_wkv), Some(ref comp_wgate)) =
                            (c2_slot, &meta.comp_wkv, &meta.comp_wgate)
                        {
                            let comp_kv_batch = comp_wkv.matbatch(&normed_a_batch, b_len);
                            let comp_score_batch = comp_wgate.matbatch(&normed_a_batch, b_len);
                            for t in 0..b_len {
                                let pos = start_pos + t;
                                let comp_kv = &comp_kv_batch[t * 512..(t + 1) * 512];
                                let comp_score = &comp_score_batch[t * 512..(t + 1) * 512];
                                if pos % 2 == 0 {
                                    kv_cache.pending_c2_kv[slot] = comp_kv.to_vec();
                                    kv_cache.pending_c2_score[slot] = comp_score.to_vec();
                                } else if kv_cache.pending_c2_kv[slot].len() == 512 {
                                    let kv0 = &kv_cache.pending_c2_kv[slot];
                                    let sc0 = &kv_cache.pending_c2_score[slot];
                                    let mut pooled = vec![0.0_f32; 512];
                                    for idx_d in 0..512 {
                                        let s0 = sc0[idx_d];
                                        let s1 = comp_score[idx_d];
                                        let m = s0.max(s1);
                                        let e0 = (s0 - m).exp();
                                        let e1 = (s1 - m).exp();
                                        let inv = 1.0_f32 / (e0 + e1);
                                        pooled[idx_d] = (e0 * kv0[idx_d] + e1 * comp_kv[idx_d]) * inv;
                                    }
                                    let mut latent = flash1017_rms_norm(&pooled, comp_norm, 1e-20);
                                    let group_pos = pos - 1;
                                    self.apply_rope_64(l, &mut latent[448..512], group_pos, false);
                                    kv_cache.compressed_k[slot].extend_from_slice(&latent);
                                }
                                per_tok_comp_lens[t][slot] = kv_cache.compressed_k[slot].len();
                                per_tok_pending_kv[t][slot] = kv_cache.pending_c2_kv[slot].clone();
                                per_tok_pending_score[t][slot] =
                                    kv_cache.pending_c2_score[slot].clone();
                            }
                        }
                    }
                }

                // 2b. Launch full in-stream batched MLA across all `b_len` tokens in 1 single GPU sync!
                let mut sin_cos_batch = Vec::with_capacity(b_len * 64);
                for t in 0..b_len {
                    let sc = self.precompute_rope_sin_cos(l, start_pos + t);
                    for &(s, c) in &sc {
                        sin_cos_batch.push(s);
                        sin_cos_batch.push(c);
                    }
                }
                let prev_total = kv_cache.k_latent[l].len() / 512;
                let min_start = prev_total.saturating_sub(127);
                let n_base = prev_total - min_start;
                let swa_prev_ptr = if n_base > 0 {
                    unsafe { kv_cache.k_latent[l].as_ptr().add(min_start * 512) }
                } else {
                    std::ptr::null()
                };
                let (extra_ptr, n_extra_per_tok): (*const f32, Vec<usize>) = if l < 2 {
                    (std::ptr::null(), vec![0usize; b_len])
                } else {
                    let slot = if l < 8 {
                        0
                    } else if l < 14 {
                        1
                    } else if l < 20 {
                        2
                    } else {
                        3
                    };
                    let full_slice = &kv_cache.compressed_k[slot];
                    let ne: Vec<usize> = (0..b_len)
                        .map(|t| per_tok_comp_lens[t][slot].min(full_slice.len()) / 512)
                        .collect();
                    let ptr = if full_slice.is_empty() {
                        std::ptr::null()
                    } else {
                        full_slice.as_ptr()
                    };
                    (ptr, ne)
                };

                let mut kv_t_batch_out = vec![0.0_f32; b_len * 512];
                let mut attn_out_batch = vec![0.0_f32; b_len * d];
                let (b_len_arg, wqb_w, wqb_s, woa_w, woa_s, wob_w, wob_s) =
                    if let Some(ref q4) = meta.q4_backbone {
                        (
                            b_len | (1usize << 31),
                            q4.wq_b.w_ptr as *const u8,
                            q4.wq_b.s_ptr as *const u8,
                            q4.wo_a.w_ptr as *const u8,
                            q4.wo_a.s_ptr as *const u8,
                            q4.wo_b.w_ptr as *const u8,
                            q4.wo_b.s_ptr as *const u8,
                        )
                    } else {
                        (
                            b_len,
                            meta.wq_b.w_ptr as *const u8,
                            meta.wq_b.s_ptr as *const u8,
                            meta.wo_a.w_ptr as *const u8,
                            meta.wo_a.s_ptr as *const u8,
                            meta.wo_b.w_ptr as *const u8,
                            meta.wo_b.s_ptr as *const u8,
                        )
                    };
                let ok = unsafe {
                    f_mla_batch(
                        b_len_arg,
                        meta.wq_a.w_ptr as *const u8,
                        meta.wq_a.s_ptr as *const u8,
                        meta.wkv.w_ptr as *const u8,
                        meta.wkv.s_ptr as *const u8,
                        meta.q_norm.as_ptr(),
                        wqb_w,
                        wqb_s,
                        meta.kv_norm.as_ptr(),
                        meta.attn_sink.as_ptr(),
                        sin_cos_batch.as_ptr(),
                        swa_prev_ptr,
                        n_base,
                        prev_total,
                        extra_ptr,
                        n_extra_per_tok.as_ptr(),
                        woa_w,
                        woa_s,
                        wob_w,
                        wob_s,
                        normed_a_batch.as_ptr(),
                        kv_t_batch_out.as_mut_ptr(),
                        attn_out_batch.as_mut_ptr(),
                    )
                };
                if ok {
                    kv_cache.k_latent[l].extend_from_slice(&kv_t_batch_out);
                    for t in 0..b_len {
                        Self::apply_mhc_post(
                            &mut streams_batch[t],
                            &attn_out_batch[t * d..(t + 1) * d],
                            &attn_w_post_batch[t],
                            &attn_m_batch[t],
                        );
                    }
                    mla_batch_done = true;
                }
            }

            if !mla_batch_done {
                for t in 0..b_len {
                    let normed_a = &normed_a_batch[t * d..(t + 1) * d];
                    let attn_out_t = self.forward_mla_attention_with_comp_lens(
                        l,
                        normed_a,
                        start_pos + t,
                        kv_cache,
                        Some(&mut per_tok_comp_lens[t]),
                    )?;
                    match l {
                        2 => {
                            per_tok_pending_kv[t][0] = kv_cache.pending_c2_kv[0].clone();
                            per_tok_pending_score[t][0] = kv_cache.pending_c2_score[0].clone();
                        }
                        8 => {
                            per_tok_pending_kv[t][1] = kv_cache.pending_c2_kv[1].clone();
                            per_tok_pending_score[t][1] = kv_cache.pending_c2_score[1].clone();
                        }
                        14 => {
                            per_tok_pending_kv[t][2] = kv_cache.pending_c2_kv[2].clone();
                            per_tok_pending_score[t][2] = kv_cache.pending_c2_score[2].clone();
                        }
                        _ => {}
                    }
                    Self::apply_mhc_post(
                        &mut streams_batch[t],
                        &attn_out_t,
                        &attn_w_post_batch[t],
                        &attn_m_batch[t],
                    );
                }
            }

            // 3. Sub-layer 2: Batched GPU mHC Pre + ffn_norm + Batched Router Gate + In-Stream GPU / Batched MoE FFN
            let mhc_ffn_stats = self.compute_mhc_stats_batch(l, &streams_batch, true)?;
            let mut ffn_pre_batch = Vec::with_capacity(b_len);
            let mut ffn_w_post_batch = Vec::with_capacity(b_len);
            let mut ffn_m_batch = Vec::with_capacity(b_len);
            let mut normed_ffn_batch = Vec::with_capacity(b_len * d);
            for (t, (f_pre, f_w_post, f_m)) in mhc_ffn_stats.into_iter().enumerate() {
                let ffn_in = Self::combine_mhc_streams(&streams_batch[t], &attn_pre_batch[t]);
                let normed_f = flash1017_rms_norm(&ffn_in, &meta.ffn_norm, 1e-20);
                ffn_pre_batch.push(f_pre);
                ffn_w_post_batch.push(f_w_post);
                ffn_m_batch.push(f_m);
                normed_ffn_batch.extend_from_slice(&normed_f);
            }

            let raw_logits_batch = meta.ffn_gate.matbatch(&normed_ffn_batch, b_len);
            let mut per_tok_selected: Vec<(Vec<(usize, f32, bool)>, f32)> =
                Vec::with_capacity(b_len);
            for t in 0..b_len {
                let is_vis_tok = (token_ids[t] & 0x8000_0000) != 0 || token_ids[t] == 129264;
                let active_bias = if is_vis_tok {
                    &meta.gate_bias_vl
                } else {
                    &meta.gate_bias
                };
                let allow_all_experts = self.tier_c_disk_enabled || is_vis_tok;
                let raw_logits = &raw_logits_batch[t * 384..(t + 1) * 384];
                let mut candidates = [(0usize, 0.0_f32, 0.0_f32); 384];
                for e in 0..384 {
                    let lg = raw_logits[e];
                    let softplus = if lg > 20.0 {
                        lg
                    } else if lg < -20.0 {
                        lg.exp()
                    } else {
                        (1.0_f32 + lg.exp()).ln()
                    };
                    let score = softplus.max(0.0).sqrt();
                    let biased = score + active_bias.get(e).copied().unwrap_or(0.0);
                    candidates[e] = (e, biased, score);
                }
                candidates.select_nth_unstable_by(6, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates[..6].sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut active_selected: Vec<(usize, f32, bool)> = Vec::with_capacity(6);
                for &(e, _biased, score) in &candidates[..6] {
                    if meta.tier_a_set[e] {
                        let use_nvfp4 = meta.tier_a_experts[e].is_some();
                        active_selected.push((e, score, use_nvfp4));
                        total_a += 1;
                    } else if meta.tier_b_set[e] {
                        let use_nvfp4 = (allow_all_experts || meta.tier_b_experts[e].is_none())
                            && meta.tier_a_experts[e].is_some();
                        active_selected.push((e, score, use_nvfp4));
                        total_b += 1;
                    } else if meta.tier_b_experts[e].is_some() {
                        active_selected.push((e, score, false));
                        total_b += 1;
                    } else if allow_all_experts && meta.tier_a_experts[e].is_some() {
                        active_selected.push((e, score, true));
                        total_a += 1;
                    } else if !allow_all_experts {
                        total_c += 1;
                    }
                }
                if allow_all_experts && active_selected.len() < 6 {
                    for &(e, _biased, score) in &candidates[6..] {
                        if active_selected.len() >= 6 {
                            break;
                        }
                        if meta.tier_a_set[e] {
                            active_selected.push((e, score, true));
                            total_a += 1;
                        } else if meta.tier_b_set[e] {
                            let use_nvfp4 = meta.tier_a_experts[e].is_some();
                            active_selected.push((e, score, use_nvfp4));
                            total_b += 1;
                        }
                    }
                }
                if active_selected.is_empty() {
                    let mut best_fb: Option<(usize, f32, f32, bool)> = None;
                    for &(e, biased, score) in &candidates[6..] {
                        let is_a = meta.tier_a_set[e];
                        let is_b = !is_a && meta.tier_b_set[e];
                        if is_a || is_b {
                            if best_fb.map_or(true, |b| biased > b.1) {
                                best_fb = Some((e, biased, score, is_a));
                            }
                        }
                    }
                    if let Some((e, _biased, score, is_a)) = best_fb {
                        let use_nvfp4 = is_a
                            || ((allow_all_experts || meta.tier_b_experts[e].is_none())
                                && meta.tier_a_experts[e].is_some());
                        active_selected.push((e, score, use_nvfp4));
                        if is_a {
                            total_a += 1;
                        } else {
                            total_b += 1;
                        }
                    }
                }
                let active_score_sum: f32 = active_selected.iter().map(|x| x.1).sum();
                let routed_scale = if active_score_sum > 1e-12 {
                    1.5_f32 / active_score_sum
                } else {
                    0.0_f32
                };
                per_tok_selected.push((active_selected, routed_scale));
            }

            let mut expert_hits: Vec<(usize, bool, Vec<(usize, f32)>)> =
                Vec::with_capacity(6 * b_len);
            for (t, (active_selected, routed_scale)) in per_tok_selected.iter().enumerate() {
                for &(exp_idx, raw_score, is_tier_a) in active_selected {
                    let weight = raw_score * (*routed_scale);
                    if let Some(entry) = expert_hits.iter_mut().find(|x| x.0 == exp_idx) {
                        entry.2.push((t, weight));
                    } else {
                        expert_hits.push((exp_idx, is_tier_a, vec![(t, weight)]));
                    }
                }
            }

            // F10.2 / S1.2: Parallel 20-Core CPU Batch Expert Page Touch (non-blocking volatile page warmup before GPU launch)
            if self.tier_c_disk_enabled && expert_hits.len() > 6 {
                use rayon::prelude::*;
                expert_hits.par_iter().for_each(|&(exp_idx, _, _)| {
                    if !meta.tier_a_set[exp_idx] {
                        if let Some((ref w1, ref w2, ref w3)) = meta.tier_a_experts[exp_idx] {
                            let mut cs = 0u8;
                            for m in [w1, w2, w3] {
                                let mut off = 0usize;
                                while off < m.w_len {
                                    unsafe {
                                        cs ^= std::ptr::read_volatile(m.w_ptr.add(off));
                                    }
                                    off += 65536;
                                }
                            }
                            std::hint::black_box(cs);
                        }
                    }
                });
            }

            // 3a–3c. Unified In-Stream GPU MoE Pipeline (`shared_experts` + all multi-token & single-token routed experts in 1 single GPU sync)!
            let mut ffn_out_batch = vec![0.0_f32; b_len * d];
            let mut moe_stream_done = false;
            if let (Some(f_mxfp8_batch), Some(f_nvfp4_batch), Some(f_tern_batch)) = (
                crate::hardware_addon::active_mxfp8_mlp_batch(),
                crate::hardware_addon::active_nvfp4_mlp_batch(),
                crate::hardware_addon::active_ternary_158_mlp_batch(),
            ) {
                let mut x_all = normed_ffn_batch.clone();
                let mut cur_x_tokens = b_len;
                let mut exp_x_offsets = Vec::with_capacity(expert_hits.len());
                for &(_exp_idx, _is_tier_a, ref hits) in &expert_hits {
                    if hits.len() == 1 {
                        exp_x_offsets.push(hits[0].0);
                    } else {
                        let off = cur_x_tokens;
                        for &(t_idx, _) in hits {
                            x_all.extend_from_slice(&normed_ffn_batch[t_idx * d..(t_idx + 1) * d]);
                        }
                        cur_x_tokens += hits.len();
                        exp_x_offsets.push(off);
                    }
                }

                let n_exp = expert_hits.len();
                let (b_len_sh, sw1_w, sw1_s, sw3_w, sw3_s, sw2_w, sw2_s) =
                    if let Some(ref q4) = meta.q4_backbone {
                        (
                            b_len | (1usize << 31),
                            q4.shared_w1.w_ptr as *const u8,
                            q4.shared_w1.s_ptr as *const u8,
                            q4.shared_w3.w_ptr as *const u8,
                            q4.shared_w3.s_ptr as *const u8,
                            q4.shared_w2.w_ptr as *const u8,
                            q4.shared_w2.s_ptr as *const u8,
                        )
                    } else {
                        (
                            b_len,
                            meta.shared_w1.w_ptr as *const u8,
                            meta.shared_w1.s_ptr as *const u8,
                            meta.shared_w3.w_ptr as *const u8,
                            meta.shared_w3.s_ptr as *const u8,
                            meta.shared_w2.w_ptr as *const u8,
                            meta.shared_w2.s_ptr as *const u8,
                        )
                    };
                let mut all_ok = unsafe {
                    f_mxfp8_batch(
                        inter,
                        d,
                        b_len_sh,
                        sw1_w,
                        sw1_s,
                        sw3_w,
                        sw3_s,
                        sw2_w,
                        sw2_s,
                        x_all.as_ptr(),
                        cur_x_tokens,
                        ffn_out_batch.as_mut_ptr(),
                        n_exp == 0,
                    )
                };

                if all_ok {
                    for (idx, &(exp_idx, is_tier_a, ref hits)) in expert_hits.iter().enumerate() {
                        let is_last = idx + 1 == n_exp;
                        let m = hits.len();
                        let x_off = exp_x_offsets[idx];
                        let mut dst_toks = [0usize; 8];
                        let mut dst_wts = [0.0_f32; 8];
                        for (k, &(t_idx, w)) in hits.iter().enumerate() {
                            dst_toks[k] = t_idx;
                            dst_wts[k] = w;
                        }
                        let ok = if is_tier_a {
                            if let Some((ref w1, ref w2, ref w3)) = meta.tier_a_experts[exp_idx] {
                                unsafe {
                                    f_nvfp4_batch(
                                        inter,
                                        d,
                                        m,
                                        w1.w_ptr as *const u8,
                                        w1.s_ptr as *const u8,
                                        w1.global_scale,
                                        w3.w_ptr as *const u8,
                                        w3.s_ptr as *const u8,
                                        w3.global_scale,
                                        w2.w_ptr as *const u8,
                                        w2.s_ptr as *const u8,
                                        w2.global_scale,
                                        x_off,
                                        dst_toks.as_ptr(),
                                        dst_wts.as_ptr(),
                                        b_len,
                                        ffn_out_batch.as_mut_ptr(),
                                        is_last,
                                    )
                                }
                            } else {
                                false
                            }
                        } else if let Some((ref w1, ref w2, ref w3)) = meta.tier_b_experts[exp_idx]
                        {
                            if w1.jepa_core.is_some() || w2.jepa_core.is_some() || w3.jepa_core.is_some() {
                                false
                            } else {
                                let sc = meta.tier_b_proj_scale;
                                unsafe {
                                    f_tern_batch(
                                        inter,
                                        d,
                                        m,
                                        w1.p_ptr,
                                        w1.s_ptr as *const f32,
                                        w3.p_ptr,
                                        w3.s_ptr as *const f32,
                                        w2.p_ptr,
                                        w2.s_ptr as *const f32,
                                        sc,
                                        x_off,
                                        dst_toks.as_ptr(),
                                        dst_wts.as_ptr(),
                                        b_len,
                                        ffn_out_batch.as_mut_ptr(),
                                        is_last,
                                    )
                                }
                            }
                        } else {
                            false
                        };
                        if !ok {
                            all_ok = false;
                            break;
                        }
                    }
                }
                if all_ok {
                    moe_stream_done = true;
                }
            }

            if !moe_stream_done {
                let sh_w1_batch = meta.shared_w1.matbatch(&normed_ffn_batch, b_len);
                let sh_w3_batch = meta.shared_w3.matbatch(&normed_ffn_batch, b_len);
                let mut sh_act_batch = Vec::with_capacity(b_len * inter);
                for i in 0..(b_len * inter) {
                    let g = sh_w1_batch[i];
                    let silu = (g / (1.0_f32 + (-g).exp())).min(10.0);
                    let u = sh_w3_batch[i].clamp(-10.0, 10.0);
                    sh_act_batch.push(silu * u);
                }
                ffn_out_batch = meta.shared_w2.matbatch(&sh_act_batch, b_len);

                for (exp_idx, is_tier_a, hits) in expert_hits {
                    let m = hits.len();
                    let mut x_exp = Vec::with_capacity(m * d);
                    for &(t_idx, _) in &hits {
                        x_exp.extend_from_slice(&normed_ffn_batch[t_idx * d..(t_idx + 1) * d]);
                    }
                    if is_tier_a {
                        if let Some((ref w1, ref w2, ref w3)) = meta.tier_a_experts[exp_idx] {
                            let w1_out = w1.matbatch(&x_exp, m);
                            let w3_out = w3.matbatch(&x_exp, m);
                            let mut act_exp = Vec::with_capacity(m * inter);
                            for i in 0..(m * inter) {
                                let g = w1_out[i];
                                let silu = (g / (1.0_f32 + (-g).exp())).min(10.0);
                                let u = w3_out[i].clamp(-10.0, 10.0);
                                act_exp.push(silu * u);
                            }
                            let w2_out = w2.matbatch(&act_exp, m);
                            for (k, &(t_idx, weight)) in hits.iter().enumerate() {
                                flash1017_axpy_f32(
                                    &mut ffn_out_batch[t_idx * d..(t_idx + 1) * d],
                                    weight,
                                    &w2_out[k * d..(k + 1) * d],
                                );
                            }
                        }
                    } else if let Some((ref w1, ref w2, ref w3)) = meta.tier_b_experts[exp_idx] {
                        let sc = meta.tier_b_proj_scale;
                        let w1_out = w1.matbatch(&x_exp, m);
                        let w3_out = w3.matbatch(&x_exp, m);
                        let mut act_exp = Vec::with_capacity(m * inter);
                        for i in 0..(m * inter) {
                            let g = w1_out[i] * sc;
                            let silu = (g / (1.0_f32 + (-g).exp())).min(10.0);
                            let u = (w3_out[i] * sc).clamp(-10.0, 10.0);
                            act_exp.push(silu * u);
                        }
                        let w2_out = w2.matbatch(&act_exp, m);
                        for (k, &(t_idx, weight)) in hits.iter().enumerate() {
                            flash1017_axpy_f32(
                                &mut ffn_out_batch[t_idx * d..(t_idx + 1) * d],
                                weight * sc,
                                &w2_out[k * d..(k + 1) * d],
                            );
                        }
                    }
                }
            }

            // 7. Apply optional DoRA deltas, mHC Post, and capture dspark_target_h [37, 38, 39] per token
            for t in 0..b_len {
                let dora_sc = dora_scales.get(t).copied().unwrap_or(1.0);
                let ffn_out_t = &mut ffn_out_batch[t * d..(t + 1) * d];
                if dora_sc.abs() > 1e-6 {
                    if let Some(Some(ref delta)) = self.layer_dora_deltas.get(l) {
                        flash1017_axpy_f32(ffn_out_t, dora_sc, delta);
                    }
                }
                Self::apply_mhc_post(
                    &mut streams_batch[t],
                    ffn_out_t,
                    &ffn_w_post_batch[t],
                    &ffn_m_batch[t],
                );
                Self::apply_jepa_inter_layer_noise_canceller(
                    l,
                    &mut streams_batch[t],
                    &kv_cache.jepa_trajectory_anchor,
                );
                if (37..=39).contains(&l) {
                    let h_t = Self::combine_mhc_streams(&streams_batch[t], &ffn_pre_batch[t]);
                    per_tok_dspark_h[t][l - 37] = h_t;
                }
                prev_pre_batch[t] = Some(ffn_pre_batch[t]);
            }
        }

        kv_cache.seq_len = start_pos + b_len;
        if let Some(last_dspark) = per_tok_dspark_h.last() {
            kv_cache.dspark_target_h = last_dspark.clone();
        }

        let mut states = Vec::with_capacity(b_len);
        for t in 0..b_len {
            let last_pre = prev_pre_batch[t].unwrap_or([0.25_f32; 4]);
            let combined = Self::combine_mhc_streams(&streams_batch[t], &last_pre);
            let final_normed = flash1017_rms_norm(&combined, &self.final_norm, 1e-20);
            kv_cache.update_jepa_trajectory(&final_normed);
            states.push(Flash1017PositionState {
                seq_len: start_pos + t + 1,
                compressed_k_lens: per_tok_comp_lens[t],
                pending_c2_kv: per_tok_pending_kv[t].clone(),
                pending_c2_score: per_tok_pending_score[t].clone(),
                dspark_target_h: per_tok_dspark_h[t].clone(),
                final_normed,
            });
        }

        let telemetry = Flash1017StepTelemetry {
            pos: start_pos,
            tier_a_hits: total_a,
            tier_b_hits: total_b,
            tier_c_skips: total_c,
            engram_lookups,
            step_ms: t0.elapsed().as_secs_f64() * 1000.0,
        };
        Ok((states, telemetry))
    }

    /// Returns all `129,280` logits using the coalesced GPU `Q8` head (`~4.0 ms`, `cos = 0.99997`) when available,
    /// or exact `BF16` head when `mode == Flash1017HeadMode::Bf16`.
    pub fn compute_full_logits(
        &self,
        final_normed: &[f32],
        mode: Flash1017HeadMode,
    ) -> Result<Vec<f32>, String> {
        if mode != Flash1017HeadMode::Bf16 {
            if let Some(ref c2f) = self.c2f_head {
                return c2f.matvec_q8(final_normed);
            }
        }
        self.head_shard.matvec_bf16("head.weight", final_normed)
    }

    /// Projects a normalized `5120`-dim final hidden vector through `head.weight` (`BF16 [129280, 5120]`)
    /// on the Blackwell GPU (`~9.16 ms`) and returns the argmax `(token_id, max_logit)`.
    pub fn predict_greedy_token(&self, final_normed: &[f32]) -> Result<(u32, f32), String> {
        let logits = self.head_shard.matvec_bf16("head.weight", final_normed)?;
        let mut best_id = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i as u32;
            }
        }
        Ok((best_id, best_val))
    }

    /// `Fas F4 (F4.1 & F4.2)`: Predicts the greedy next token using the selected `Flash1017HeadMode`
    /// (`Bf16`, `Q8`, or `CoarseToFine`).
    pub fn predict_token_with_mode(
        &self,
        final_normed: &[f32],
        mode: Flash1017HeadMode,
    ) -> Result<(u32, f32), String> {
        let (best_id, best_val, _margin) =
            self.predict_token_with_mode_and_margin(final_normed, mode)?;
        Ok((best_id, best_val))
    }

    /// `F7.C2`: Predicts the greedy next token and returns `(token_id, top1_logit, top1_minus_top2_margin)`
    /// for self-regulating `Adaptive MTP (1..3)` confidence gating.
    pub fn predict_token_with_mode_and_margin(
        &self,
        final_normed: &[f32],
        mode: Flash1017HeadMode,
    ) -> Result<(u32, f32, f32), String> {
        match mode {
            Flash1017HeadMode::Bf16 => {
                let logits = self.head_shard.matvec_bf16("head.weight", final_normed)?;
                let mut best_id = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                let mut second_val = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > best_val {
                        second_val = best_val;
                        best_val = v;
                        best_id = i as u32;
                    } else if v > second_val {
                        second_val = v;
                    }
                }
                let margin = if second_val.is_finite() {
                    (best_val - second_val).max(0.0)
                } else {
                    0.0
                };
                Ok((best_id, best_val, margin))
            }
            Flash1017HeadMode::Q8 => {
                if let Some(ref c2f) = self.c2f_head {
                    let logits = c2f.matvec_q8(final_normed)?;
                    let mut best_id = 0u32;
                    let mut best_val = f32::NEG_INFINITY;
                    let mut second_val = f32::NEG_INFINITY;
                    for (i, &v) in logits.iter().enumerate() {
                        if v > best_val {
                            second_val = best_val;
                            best_val = v;
                            best_id = i as u32;
                        } else if v > second_val {
                            second_val = v;
                        }
                    }
                    let margin = if second_val.is_finite() {
                        (best_val - second_val).max(0.0)
                    } else {
                        0.0
                    };
                    Ok((best_id, best_val, margin))
                } else {
                    self.predict_token_with_mode_and_margin(final_normed, Flash1017HeadMode::Bf16)
                }
            }
            Flash1017HeadMode::CoarseToFine => {
                if let Some(ref c2f) = self.c2f_head {
                    let (best_id, best_val, margin, _a_ms, _b_ms) = c2f
                        .predict_coarse_to_fine_with_margin(&self.head_shard, final_normed, 2048)?;
                    Ok((best_id, best_val, margin))
                } else {
                    self.predict_token_with_mode_and_margin(final_normed, Flash1017HeadMode::Bf16)
                }
            }
        }
    }

    /// `F8.3`: Batched token prediction across `1 + K` verified hidden states using a single GPU `v2_matbatch_q8` call
    /// in `CoarseToFineConceptHead` (`132.4 MB` read ONCE instead of `1 + K` times).
    pub fn predict_tokens_batch_with_mode_and_margin(
        &self,
        normed_batch: &[&[f32]],
        mode: Flash1017HeadMode,
    ) -> Result<Vec<(u32, f32, f32)>, String> {
        if normed_batch.is_empty() {
            return Ok(Vec::new());
        }
        if let Flash1017HeadMode::CoarseToFine = mode {
            if let Some(ref c2f) = self.c2f_head {
                return c2f.predict_coarse_to_fine_batch_with_margin(
                    &self.head_shard,
                    normed_batch,
                    2048,
                );
            }
        }
        let mut out = Vec::with_capacity(normed_batch.len());
        for &h in normed_batch {
            out.push(self.predict_token_with_mode_and_margin(h, mode)?);
        }
        Ok(out)
    }

    /// Fast speculative draft head (`top_m = 1024` in `C2FHead` + `DSpark` `markov_head` bigram fusion).
    /// Target verification always uses `predict_token_with_mode_and_margin` (`top_m = 2048`) for 100% bit-exactness with `K = 0`.
    fn predict_draft_token_with_margin(
        &self,
        final_normed: &[f32],
        cur_token_id: u32,
        has_manifold_match: bool,
        mode: Flash1017HeadMode,
    ) -> Result<(u32, f32, f32), String> {
        use rayon::prelude::*;
        let markov_q = self
            .mtp_layer
            .as_ref()
            .and_then(|m| m.extract_markov_embed(cur_token_id));

        if let (Some(ref c2f), Some(ref mtp), Some(ref q_256)) =
            (&self.c2f_head, &self.mtp_layer, &markov_q)
        {
            if mtp.markov_head.is_some() {
                let cands = c2f.select_candidate_ids(final_normed, 1024, &[]);
                let (meta, raw_bytes) = self.head_shard.get_tensor_data("head.weight")?;
                let rows = meta.shape[0];
                let cols = meta.shape[1];
                let row_bytes = cols * 2;
                let is_u16_aligned = (raw_bytes.as_ptr() as usize) % 2 == 0;
                let raw_ptr_usize = raw_bytes.as_ptr() as usize;

                let (best_id, best_val, second_val) = cands
                    .par_iter()
                    .with_min_len(32)
                    .map(|&r_u32| {
                        let r = (r_u32 as usize).min(rows - 1);
                        let mut sum = 0.0_f32;
                        if is_u16_aligned {
                            let u16_row = unsafe {
                                std::slice::from_raw_parts(
                                    (raw_ptr_usize + r * row_bytes) as *const u16,
                                    cols,
                                )
                            };
                            let chunks8 = cols / 8;
                            let mut a0 = 0.0_f32;
                            let mut a1 = 0.0_f32;
                            let mut a2 = 0.0_f32;
                            let mut a3 = 0.0_f32;
                            let mut a4 = 0.0_f32;
                            let mut a5 = 0.0_f32;
                            let mut a6 = 0.0_f32;
                            let mut a7 = 0.0_f32;
                            for c8 in 0..chunks8 {
                                let base = c8 * 8;
                                a0 += f32::from_bits((u16_row[base] as u32) << 16)
                                    * final_normed[base];
                                a1 += f32::from_bits((u16_row[base + 1] as u32) << 16)
                                    * final_normed[base + 1];
                                a2 += f32::from_bits((u16_row[base + 2] as u32) << 16)
                                    * final_normed[base + 2];
                                a3 += f32::from_bits((u16_row[base + 3] as u32) << 16)
                                    * final_normed[base + 3];
                                a4 += f32::from_bits((u16_row[base + 4] as u32) << 16)
                                    * final_normed[base + 4];
                                a5 += f32::from_bits((u16_row[base + 5] as u32) << 16)
                                    * final_normed[base + 5];
                                a6 += f32::from_bits((u16_row[base + 6] as u32) << 16)
                                    * final_normed[base + 6];
                                a7 += f32::from_bits((u16_row[base + 7] as u32) << 16)
                                    * final_normed[base + 7];
                            }
                            sum = (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
                        } else {
                            let r_slice = &raw_bytes[r * row_bytes..(r + 1) * row_bytes];
                            for c in 0..cols {
                                let bits =
                                    u16::from_le_bytes([r_slice[c * 2], r_slice[c * 2 + 1]]);
                                sum += f32::from_bits((bits as u32) << 16) * final_normed[c];
                            }
                        }

                        let m_logit = mtp.score_markov_with_embed(q_256, r_u32);
                        let combined = if has_manifold_match {
                            sum + 0.15_f32 * m_logit
                        } else if r_u32 == cur_token_id && m_logit < 0.0 {
                            // Suppress stale step-(t-1) self-repeat spike when `markov_head` says token does not repeat
                            0.28_f32 * sum - 14.0_f32 + m_logit
                        } else {
                            0.28_f32 * sum + 1.0_f32 * m_logit
                        };
                        (r_u32, combined, f32::NEG_INFINITY)
                    })
                    .reduce(
                        || (0u32, f32::NEG_INFINITY, f32::NEG_INFINITY),
                        |a, b| {
                            if b.1 > a.1 {
                                (b.0, b.1, a.1.max(b.2))
                            } else {
                                (a.0, a.1, b.1.max(a.2))
                            }
                        },
                    );
                let margin = if second_val.is_finite() {
                    (best_val - second_val).max(0.0)
                } else {
                    0.0
                };
                return Ok((best_id, best_val, margin));
            }
        }
        self.predict_token_with_mode_and_margin(final_normed, mode)
    }

    fn compute_mtp_mhc_stats_raw(
        fn_ptr: *const f32,
        base: &[f32; 24],
        scale: &[f32; 3],
        streams: &[Vec<f32>; 4],
    ) -> Result<([f32; 4], [f32; 4], [[f32; 4]; 4]), String> {
        let d = 5120usize;
        let total_d = 4 * d;
        let mut sum_sq = 0.0_f32;
        for m in 0..4 {
            for &v in &streams[m] {
                sum_sq += v * v;
            }
        }
        let inv_rms = 1.0_f32 / (sum_sq / (total_d as f32) + 1e-20_f32).sqrt();

        let mut z = [0.0_f32; 24];
        if !fn_ptr.is_null() && ((fn_ptr as usize) % 4 == 0) {
            let mut gpu_done = false;
            if let Some(mv_f32) = crate::hardware_addon::active_matvec_f32() {
                let mut streams_flat = Vec::with_capacity(total_d);
                for m in 0..4 {
                    streams_flat.extend_from_slice(&streams[m]);
                }
                let ok = unsafe {
                    mv_f32(
                        fn_ptr,
                        streams_flat.as_ptr(),
                        z.as_mut_ptr(),
                        24,
                        total_d,
                    )
                };
                if ok {
                    for r in 0..24 {
                        z[r] *= inv_rms;
                    }
                    gpu_done = true;
                }
            }
            if !gpu_done {
                let fn_f32 = unsafe { std::slice::from_raw_parts(fn_ptr, 24 * total_d) };
                for r in 0..24 {
                    let row = &fn_f32[r * total_d..(r + 1) * total_d];
                    let mut dot = 0.0_f32;
                    for m in 0..4 {
                        dot += flash1017_dot_f32(&row[m * d..(m + 1) * d], &streams[m]);
                    }
                    z[r] = dot * inv_rms;
                }
            }
        }

        let mut w_pre = [0.0_f32; 4];
        let mut w_post = [0.0_f32; 4];
        let mut u_res = [0.0_f32; 16];
        for i in 0..4 {
            let u0 = scale[0] * z[i] + base[i];
            w_pre[i] = 1.0_f32 / (1.0_f32 + (-u0).exp()) + 1e-6_f32;
            let u1 = scale[1] * z[4 + i] + base[4 + i];
            w_post[i] = 2.0_f32 / (1.0_f32 + (-u1).exp());
        }
        for i in 0..16 {
            u_res[i] = scale[2] * z[8 + i] + base[8 + i];
        }
        let m_mat = sinkhorn_knopp_4x4(&u_res, 20, 1e-6);
        Ok((w_pre, w_post, m_mat))
    }

    /// `F4.3`, `F7.A3` & `F7.B1`: Executes DeepSeek-V4.1's speculative `mtp.0..2` (`dspark`) predictor
    /// (`model-00044..00046`, `dspark_target_layer_ids = [37, 38, 39]`, `Top-3` of `128` `MXFP4` experts)
    /// using pre-cached tensor descriptors, full in-stream GPU MLA attention, and GPU `MXFP4` MLP accumulation.
    pub fn draft_mtp_token(
        &self,
        prev_token_id: u32,
        kv_cache: &mut Flash1017KvCache,
        head_mode: Flash1017HeadMode,
    ) -> Result<(u32, f32, f64), String> {
        let target_h = kv_cache.dspark_target_h.clone();
        let prev_hist_tok = kv_cache.token_history.last().copied().unwrap_or(0);
        let verified_snapshot = kv_cache.mtp_verified_states.clone();
        let (draft_id, draft_logit, _margin, _est_h39, _mtp_raw, _has_match, elapsed_ms) = self
            .draft_mtp_step_with_target_h(
                prev_token_id,
                prev_hist_tok,
                &target_h,
                &verified_snapshot,
                kv_cache,
                head_mode,
            )?;
        Ok((draft_id, draft_logit, elapsed_ms))
    }

    /// Executes the full 3-layer `mtp.0 -> mtp.1 -> mtp.2` (`dspark` layers 0..2 in `model-00044..00046`)
    /// forward pass on GPU (`f_mla` + `f_nvfp4_mlp`/`f_mxfp8_mlp` + `f_mxfp4_mlp`)
    /// with `compress_ratios[40..43] == 0` (sliding-window `mtp_k_latent` / `mtp12_k_latent` + unscaled RoPE table from layer 0),
    /// returning `([h_mtp0, h_mtp1, h_mtp2], elapsed_ms)`.
    pub fn compute_mtp_3layer_states(
        &self,
        prev_token_id: u32,
        target_h: &[Vec<f32>; 3],
        kv_cache: &mut Flash1017KvCache,
    ) -> Result<([Vec<f32>; 3], f64), String> {
        use rayon::prelude::*;
        let Some(ref mtp) = self.mtp_layer else {
            return Err("mtp.0 layer is not enabled (call enable_mtp_layer first)".to_string());
        };
        let t0 = std::time::Instant::now();
        kv_cache.ensure_capacity_for_additional_tokens(1);
        let d = 5120usize;
        let inter = 2304usize;
        let mtp_pos = kv_cache.mtp_k_latent.len() / 512;
        kv_cache.sync_mtp12_to_mtp0(mtp_pos * 512);
        // In `config.json`, `compress_ratios[40..43] == 0` (same as layer 0, unscaled RoPE without compressed extra KV)
        let sin_cos = self.precompute_rope_sin_cos(0, mtp_pos);

        // 1. Project target layer hidden states [37, 38, 39] ([15360 -> 5120]) via `mtp.0.main_proj` (MXFP8)
        // and normalize the 5120-dim output with `mtp.0.main_norm` (RMS=0.0682, matching `embed.weight` RMS=0.0466).
        let mut concat_15360 = Vec::with_capacity(3 * d);
        for t_idx in 0..3 {
            concat_15360.extend_from_slice(&target_h[t_idx]);
        }
        let raw_proj_5120 = mtp.main_proj.matvec(&concat_15360);
        let proj_5120 = flash1017_rms_norm(&raw_proj_5120, &mtp.main_norm, 1e-20);
        let emb_prev = self.get_token_embedding(prev_token_id)?;
        let mut h0 = Vec::with_capacity(d);
        for i in 0..d {
            h0.push(0.5_f32 * (proj_5120[i] + emb_prev[i]));
        }
        let mut streams: [Vec<f32>; 4] = [h0.clone(), h0.clone(), h0.clone(), h0];
        let num_sub = mtp.num_active_sub_layers();
        let mut layer_outputs: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];

        for m_idx in 0..num_sub {
            let (
                attn_norm,
                ffn_norm,
                q_norm,
                kv_norm,
                attn_sink,
                hc_attn_fn_ptr,
                hc_attn_base,
                hc_attn_scale,
                hc_ffn_fn_ptr,
                hc_ffn_base,
                hc_ffn_scale,
                gate_bias,
                wq_a,
                wq_b,
                wkv,
                wo_a,
                wo_b,
                ffn_gate,
                shared_w1,
                shared_w2,
                shared_w3,
                q4_backbone,
                mxfp4_experts,
            ) = if m_idx == 0 {
                (
                    &mtp.attn_norm,
                    &mtp.ffn_norm,
                    &mtp.q_norm,
                    &mtp.kv_norm,
                    &mtp.attn_sink,
                    mtp.hc_attn_fn_ptr,
                    &mtp.hc_attn_base,
                    &mtp.hc_attn_scale,
                    mtp.hc_ffn_fn_ptr,
                    &mtp.hc_ffn_base,
                    &mtp.hc_ffn_scale,
                    &mtp.gate_bias,
                    &mtp.wq_a,
                    &mtp.wq_b,
                    &mtp.wkv,
                    &mtp.wo_a,
                    &mtp.wo_b,
                    &mtp.ffn_gate,
                    &mtp.shared_w1,
                    &mtp.shared_w2,
                    &mtp.shared_w3,
                    &mtp.q4_backbone,
                    &mtp.mxfp4_experts,
                )
            } else {
                let sub = &mtp.extra_sub_layers[m_idx - 1];
                (
                    &sub.attn_norm,
                    &sub.ffn_norm,
                    &sub.q_norm,
                    &sub.kv_norm,
                    &sub.attn_sink,
                    sub.hc_attn_fn_ptr,
                    &sub.hc_attn_base,
                    &sub.hc_attn_scale,
                    sub.hc_ffn_fn_ptr,
                    &sub.hc_ffn_base,
                    &sub.hc_ffn_scale,
                    &sub.gate_bias,
                    &sub.wq_a,
                    &sub.wq_b,
                    &sub.wkv,
                    &sub.wo_a,
                    &sub.wo_b,
                    &sub.ffn_gate,
                    &sub.shared_w1,
                    &sub.shared_w2,
                    &sub.shared_w3,
                    &sub.q4_backbone,
                    &sub.mxfp4_experts,
                )
            };

            // Sub-layer 1: mHC Pre -> `attn_norm` -> `attn` (4-bit NVFP4 / MXFP8 MLA) -> mHC Post
            let (attn_pre, attn_w_post, attn_m) = Self::compute_mtp_mhc_stats_raw(
                hc_attn_fn_ptr,
                hc_attn_base,
                hc_attn_scale,
                &streams,
            )?;
            let attn_in = Self::combine_mhc_streams(&streams, &attn_pre);
            let normed_attn = flash1017_rms_norm(&attn_in, attn_norm, 1e-20);

            let k_lat_vec: &mut Vec<f32> = if m_idx == 0 {
                &mut kv_cache.mtp_k_latent
            } else {
                &mut kv_cache.mtp12_k_latent[m_idx - 1]
            };

            let mut attn_out_opt: Option<Vec<f32>> = None;
            if let Some(f_mla) = crate::hardware_addon::active_mla_full_forward() {
                let prev_total = k_lat_vec.len() / 512;
                let win_start = prev_total.saturating_sub(127);
                let n_prev_swa = prev_total - win_start;
                let swa_prev_ptr = if n_prev_swa > 0 {
                    unsafe { k_lat_vec.as_ptr().add(win_start * 512) }
                } else {
                    std::ptr::null()
                };
                let mut kv_t_out = [0.0_f32; 512];
                let mut attn_out_gpu = vec![0.0_f32; 5120];
                let (n_prev_swa_flagged, wq_b_w, wq_b_s, wo_a_w, wo_a_s, wo_b_w, wo_b_s) =
                    if let Some(ref q4) = q4_backbone {
                        (
                            n_prev_swa | (1usize << 31),
                            q4.wq_b.w_ptr as *const u8,
                            q4.wq_b.s_ptr as *const u8,
                            q4.wo_a.w_ptr as *const u8,
                            q4.wo_a.s_ptr as *const u8,
                            q4.wo_b.w_ptr as *const u8,
                            q4.wo_b.s_ptr as *const u8,
                        )
                    } else {
                        (
                            n_prev_swa,
                            wq_b.w_ptr as *const u8,
                            wq_b.s_ptr as *const u8,
                            wo_a.w_ptr as *const u8,
                            wo_a.s_ptr as *const u8,
                            wo_b.w_ptr as *const u8,
                            wo_b.s_ptr as *const u8,
                        )
                    };
                let ok = unsafe {
                    f_mla(
                        wq_a.w_ptr as *const u8,
                        wq_a.s_ptr as *const u8,
                        wkv.w_ptr as *const u8,
                        wkv.s_ptr as *const u8,
                        q_norm.as_ptr(),
                        wq_b_w,
                        wq_b_s,
                        kv_norm.as_ptr(),
                        attn_sink.as_ptr(),
                        sin_cos.as_ptr() as *const f32,
                        swa_prev_ptr,
                        n_prev_swa_flagged,
                        std::ptr::null(),
                        0,
                        wo_a_w,
                        wo_a_s,
                        wo_b_w,
                        wo_b_s,
                        normed_attn.as_ptr(),
                        kv_t_out.as_mut_ptr(),
                        attn_out_gpu.as_mut_ptr(),
                    )
                };
                if ok {
                    k_lat_vec.extend_from_slice(&kv_t_out);
                    attn_out_opt = Some(attn_out_gpu);
                }
            }

            let attn_out = if let Some(out) = attn_out_opt {
                out
            } else {
                let q_a = wq_a.matvec(&normed_attn);
                let kv_raw = wkv.matvec(&normed_attn);
                let q_a_norm = flash1017_rms_norm(&q_a, q_norm, 1e-20);
                let mut q_all = wq_b.matvec(&q_a_norm);

                let mut kv_t = flash1017_rms_norm(&kv_raw, kv_norm, 1e-20);
                Self::apply_rope_64_table(&mut kv_t[448..512], &sin_cos, false);
                k_lat_vec.extend_from_slice(&kv_t);

                let total_swa = k_lat_vec.len() / 512;
                let win_start = total_swa.saturating_sub(128);
                let scale = 1.0_f32 / (512.0_f32).sqrt();
                let swa_cache: &[f32] = k_lat_vec;
                let n_swa = total_swa - win_start;

                let mut attn_heads_out = vec![0.0_f32; 64 * 512];
                q_all
                    .par_chunks_exact_mut(512)
                    .zip(attn_heads_out.par_chunks_exact_mut(512))
                    .enumerate()
                    .with_min_len(4)
                    .for_each(|(h, (q_h, head_out))| {
                        let mut sum_sq = 0.0_f32;
                        for &v in q_h.iter() {
                            sum_sq += v * v;
                        }
                        let inv_rms = 1.0_f32 / (sum_sq / 512.0_f32 + 1e-20_f32).sqrt();
                        for v in q_h.iter_mut() {
                            *v *= inv_rms;
                        }
                        Self::apply_rope_64_table(&mut q_h[448..512], &sin_cos, false);

                        let sink_logit = attn_sink.get(h).copied().unwrap_or(0.0);
                        let mut m_max = sink_logit;
                        let mut scores = Vec::with_capacity(n_swa);
                        for s in win_start..total_swa {
                            let k_s = &swa_cache[s * 512..(s + 1) * 512];
                            let sc = flash1017_dot_f32(q_h, k_s) * scale;
                            if sc > m_max {
                                m_max = sc;
                            }
                            scores.push(sc);
                        }
                        let mut sum_exp = (sink_logit - m_max).exp();
                        for sc in &mut scores {
                            *sc = (*sc - m_max).exp();
                            sum_exp += *sc;
                        }
                        let inv_sum = if sum_exp > 0.0 { 1.0_f32 / sum_exp } else { 0.0 };
                        for (idx, s) in (win_start..total_swa).enumerate() {
                            let w = scores[idx] * inv_sum;
                            let v_s = &swa_cache[s * 512..(s + 1) * 512];
                            flash1017_axpy_f32(head_out, w, v_s);
                        }
                        Self::apply_rope_64_table(&mut head_out[448..512], &sin_cos, true);
                    });

                if let Some(f_wo_ab) = crate::hardware_addon::active_mxfp8_wo_a_b_fused() {
                    let mut out = vec![0.0_f32; wo_b.rows];
                    let ok = unsafe {
                        f_wo_ab(
                            wo_a.rows,
                            wo_a.cols,
                            8,
                            wo_a.w_ptr as *const u8,
                            wo_a.s_ptr as *const u8,
                            wo_b.rows,
                            wo_b.w_ptr as *const u8,
                            wo_b.s_ptr as *const u8,
                            attn_heads_out.as_ptr(),
                            out.as_mut_ptr(),
                        )
                    };
                    if ok {
                        out
                    } else {
                        let o_lora = wo_a.matvec_grouped(&attn_heads_out, 8);
                        wo_b.matvec(&o_lora)
                    }
                } else {
                    let o_lora = wo_a.matvec_grouped(&attn_heads_out, 8);
                    wo_b.matvec(&o_lora)
                }
            };
            Self::apply_mhc_post(&mut streams, &attn_out, &attn_w_post, &attn_m);

            // Sub-layer 2: mHC Pre -> `ffn_norm` -> `Top-3` of `128` `MXFP4` experts + `shared_experts` -> mHC Post
            let (ffn_pre, ffn_w_post, ffn_m) = Self::compute_mtp_mhc_stats_raw(
                hc_ffn_fn_ptr,
                hc_ffn_base,
                hc_ffn_scale,
                &streams,
            )?;
            let ffn_in = Self::combine_mhc_streams(&streams, &ffn_pre);
            let normed_ffn = flash1017_rms_norm(&ffn_in, ffn_norm, 1e-20);

            let raw_logits = ffn_gate.matvec(&normed_ffn);
            let mut candidates = [(0usize, 0.0_f32, 0.0_f32); 128];
            for e in 0..128 {
                let l = raw_logits[e];
                let softplus = if l > 20.0 {
                    l
                } else if l < -20.0 {
                    l.exp()
                } else {
                    (1.0_f32 + l.exp()).ln()
                };
                let score = softplus.max(0.0).sqrt();
                let biased = score + gate_bias.get(e).copied().unwrap_or(0.0);
                candidates[e] = (e, biased, score);
            }
            candidates.select_nth_unstable_by(3, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates[..3].sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let top3_score_sum: f32 = candidates[..3].iter().map(|x| x.2).sum();
            let routed_scale = if top3_score_sum > 1e-12 {
                1.5_f32 / top3_score_sum
            } else {
                0.0_f32
            };

            let mut ffn_out = vec![0.0_f32; d];
            let mut gpu_mlp_ok = false;
            if let (Some(f_mxfp8_mlp), Some(f_nvfp4_mlp), Some(f_mxfp4_mlp)) = (
                crate::hardware_addon::active_mxfp8_mlp_accum(),
                crate::hardware_addon::active_nvfp4_mlp_accum(),
                crate::hardware_addon::active_mxfp4_mlp_accum(),
            ) {
                let mut ok = if let Some(ref q4) = q4_backbone {
                    unsafe {
                        f_nvfp4_mlp(
                            inter,
                            d,
                            q4.shared_w1.w_ptr as *const u8,
                            q4.shared_w1.s_ptr as *const u8,
                            q4.shared_w1.global_scale,
                            q4.shared_w3.w_ptr as *const u8,
                            q4.shared_w3.s_ptr as *const u8,
                            q4.shared_w3.global_scale,
                            q4.shared_w2.w_ptr as *const u8,
                            q4.shared_w2.s_ptr as *const u8,
                            q4.shared_w2.global_scale,
                            normed_ffn.as_ptr(),
                            ffn_out.as_mut_ptr(),
                            1.0,
                            false,
                            false,
                        )
                    }
                } else {
                    unsafe {
                        f_mxfp8_mlp(
                            inter,
                            d,
                            shared_w1.w_ptr as *const u8,
                            shared_w1.s_ptr as *const u8,
                            shared_w3.w_ptr as *const u8,
                            shared_w3.s_ptr as *const u8,
                            shared_w2.w_ptr as *const u8,
                            shared_w2.s_ptr as *const u8,
                            normed_ffn.as_ptr(),
                            ffn_out.as_mut_ptr(),
                            1.0,
                            false,
                            false,
                        )
                    }
                };
                if ok {
                    for (idx, &(exp_idx, _biased, raw_score)) in
                        candidates[..3].iter().enumerate()
                    {
                        let is_last = idx == 2;
                        let weight = raw_score * routed_scale;
                        let (ref w1, ref w2, ref w3) = mxfp4_experts[exp_idx];
                        let step_ok = unsafe {
                            f_mxfp4_mlp(
                                inter,
                                d,
                                w1.w_ptr as *const u8,
                                w1.s_ptr as *const u8,
                                w3.w_ptr as *const u8,
                                w3.s_ptr as *const u8,
                                w2.w_ptr as *const u8,
                                w2.s_ptr as *const u8,
                                normed_ffn.as_ptr(),
                                ffn_out.as_mut_ptr(),
                                weight,
                                true,
                                is_last,
                            )
                        };
                        if !step_ok {
                            ok = false;
                            break;
                        }
                    }
                }
                gpu_mlp_ok = ok;
            }

            if !gpu_mlp_ok {
                let sh_w1 = shared_w1.matvec(&normed_ffn);
                let sh_w3 = shared_w3.matvec(&normed_ffn);
                let mut sh_act = Vec::with_capacity(inter);
                for i in 0..inter {
                    let g = sh_w1[i];
                    let silu = (g / (1.0_f32 + (-g).exp())).min(10.0);
                    let u = sh_w3[i].clamp(-10.0, 10.0);
                    sh_act.push(silu * u);
                }
                ffn_out = shared_w2.matvec(&sh_act);

                for &(exp_idx, _biased, raw_score) in &candidates[..3] {
                    let weight = raw_score * routed_scale;
                    let (ref w1, ref w2, ref w3) = mxfp4_experts[exp_idx];
                    let w1_out = w1.matvec(&normed_ffn);
                    let w3_out = w3.matvec(&normed_ffn);
                    let mut act = Vec::with_capacity(inter);
                    for i in 0..inter {
                        let g = w1_out[i];
                        let silu = (g / (1.0_f32 + (-g).exp())).min(10.0);
                        let u = w3_out[i].clamp(-10.0, 10.0);
                        act.push(silu * u);
                    }
                    let w2_out = w2.matvec(&act);
                    flash1017_axpy_f32(&mut ffn_out, weight, &w2_out);
                }
            }

            Self::apply_mhc_post(&mut streams, &ffn_out, &ffn_w_post, &ffn_m);
            let combined = Self::combine_mhc_streams(&streams, &ffn_pre);
            if m_idx < 3 {
                layer_outputs[m_idx] = combined;
            }
        }

        if layer_outputs[1].is_empty() {
            layer_outputs[1] = layer_outputs[0].clone();
        }
        if layer_outputs[2].is_empty() {
            layer_outputs[2] = layer_outputs[1].clone();
        }

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok((layer_outputs, elapsed_ms))
    }

    pub fn compute_mtp_raw_hidden(
        &self,
        prev_token_id: u32,
        target_h: &[Vec<f32>; 3],
        kv_cache: &mut Flash1017KvCache,
    ) -> Result<(Vec<f32>, f64), String> {
        let (states, elapsed_ms) = self.compute_mtp_3layer_states(prev_token_id, target_h, kv_cache)?;
        let [_, _, s2] = states;
        Ok((s2, elapsed_ms))
    }

    #[inline]
    fn canonical_engram_id(&self, tok_id: u32) -> u32 {
        self.engram_token_map
            .get(tok_id as usize)
            .copied()
            .unwrap_or(tok_id)
    }

    #[inline]
    fn is_valid_case_a_state(
        &self,
        s: &Flash1017VerifiedMtpState,
        cur_token_id: u32,
        prev_token_id: u32,
        verified_states: &[Flash1017VerifiedMtpState],
    ) -> bool {
        if s.next_token_id == 0 {
            return false;
        }
        let cur_canon = self.canonical_engram_id(cur_token_id);
        if self.canonical_engram_id(s.token_id) != cur_canon {
            return false;
        }
        let same_prev_exact = s.prev_token_id == prev_token_id;
        let same_prev_canon =
            self.canonical_engram_id(s.prev_token_id) == self.canonical_engram_id(prev_token_id);
        let canon_next_count = verified_states
            .iter()
            .filter(|x| {
                self.canonical_engram_id(x.token_id) == cur_canon
                    && x.next_token_id == s.next_token_id
            })
            .count();
        let m_logit = self
            .mtp_layer
            .as_ref()
            .and_then(|m| m.markov_transition_logit(cur_token_id, s.next_token_id))
            .unwrap_or(0.0);
        if s.token_id == cur_token_id {
            if canon_next_count >= 2 {
                return true;
            }
            if same_prev_exact && m_logit >= -1.0 {
                return true;
            }
            if same_prev_canon && m_logit >= 1.0 {
                return true;
            }
            if s.next_token_id < 1000 && m_logit >= 6.0 {
                return true;
            }
            false
        } else {
            same_prev_canon
                || canon_next_count >= 2
                || (prev_token_id < 256 && s.seq_pos >= 6 && s.next_token_id >= 1000)
        }
    }

    #[inline]
    fn select_best_case_a_state<'a>(
        &self,
        cur_token_id: u32,
        prev_token_id: u32,
        verified_states: &'a [Flash1017VerifiedMtpState],
    ) -> Option<&'a Flash1017VerifiedMtpState> {
        let valid_matches: Vec<&Flash1017VerifiedMtpState> = verified_states
            .iter()
            .filter(|s| self.is_valid_case_a_state(s, cur_token_id, prev_token_id, verified_states))
            .collect();
        if valid_matches.is_empty() {
            return None;
        }
        let last_valid = valid_matches[valid_matches.len() - 1];
        let last_next_count = valid_matches
            .iter()
            .filter(|s| s.next_token_id == last_valid.next_token_id)
            .count();

        let bigram_matches: Vec<&Flash1017VerifiedMtpState> = valid_matches
            .iter()
            .copied()
            .filter(|s| {
                self.canonical_engram_id(s.prev_token_id) == self.canonical_engram_id(prev_token_id)
            })
            .collect();
        if let Some(&last_bi) = bigram_matches.last() {
            let bi_next_count = valid_matches
                .iter()
                .filter(|s| s.next_token_id == last_bi.next_token_id)
                .count();
            if last_next_count > bi_next_count {
                return Some(last_valid);
            }
            return Some(last_bi);
        }
        Some(last_valid)
    }

    /// `F7.C2` & `F9.2`: Projects `mtp.0`'s raw hidden state (`mtp_raw`, layer 0 of the 3-layer `dspark` predictor)
    /// onto the target model's Layer-39 (`head.weight`) manifold using verified `(prev_h39 -> target_h39)`
    /// trajectory states from the prompt and accepted decode steps.
    fn project_mtp_to_target_manifold(
        &self,
        cur_token_id: u32,
        prev_token_id: u32,
        target_h: &[Vec<f32>; 3],
        mtp_raw: &[f32],
        verified_states: &[Flash1017VerifiedMtpState],
    ) -> (Vec<f32>, bool) {
        let d = 5120usize;
        let h39_rms = (flash1017_dot_f32(&target_h[2], &target_h[2]) / (d as f32) + 1e-20).sqrt();
        let scale_to_h39 = if !mtp_raw.is_empty() {
            let mtp_rms = (flash1017_dot_f32(mtp_raw, mtp_raw) / (d as f32) + 1e-20).sqrt();
            h39_rms / mtp_rms
        } else {
            0.0_f32
        };

        let cosine_fn = |a: &[f32], b: &[f32]| -> f32 {
            if a.is_empty() || b.is_empty() {
                return 0.0;
            }
            let dot = flash1017_dot_f32(a, b);
            let na = flash1017_dot_f32(a, a).max(1e-20).sqrt();
            let nb = flash1017_dot_f32(b, b).max(1e-20).sqrt();
            dot / (na * nb)
        };

        // Case A: Exact or canonical token match in verified Layer-39 trajectory.
        if let Some(last_m) =
            self.select_best_case_a_state(cur_token_id, prev_token_id, verified_states)
        {
            let mut est_h39 = vec![0.0_f32; d];
            let same_prev = self.canonical_engram_id(last_m.prev_token_id)
                == self.canonical_engram_id(prev_token_id);
            let ctx_alpha = if same_prev { 0.04_f32 } else { 0.06_f32 };
            for i in 0..d {
                let ctx_diff = target_h[2][i] - last_m.prev_h39[i];
                est_h39[i] = last_m.target_h39[i] + ctx_alpha * ctx_diff;
            }
            return (est_h39, true);
        }
        if let Some(last_tok) = verified_states
            .iter()
            .rev()
            .find(|s| s.token_id == cur_token_id)
        {
            let mut est_h39 = vec![0.0_f32; d];
            for i in 0..d {
                let ctx_diff = target_h[2][i] - last_tok.prev_h39[i];
                est_h39[i] = last_tok.target_h39[i] + 0.08_f32 * ctx_diff;
            }
            return (est_h39, false);
        }

        // Case B: Preceding token match (`s.prev_token_id == prev_token_id`) when `mtp_raw` strongly agrees
        let mut best_prev: Option<(&Flash1017VerifiedMtpState, f32)> = None;
        if !mtp_raw.is_empty() {
            for s in verified_states
                .iter()
                .filter(|s| s.prev_token_id == prev_token_id && !s.mtp_raw.is_empty())
            {
                let c = cosine_fn(mtp_raw, &s.mtp_raw);
                if best_prev.map_or(true, |(_, best_c)| c > best_c) {
                    best_prev = Some((s, c));
                }
            }
        }
        if let Some((best_s, best_c)) = best_prev {
            if best_c > 0.90 {
                let best_mtp_rms = (flash1017_dot_f32(&best_s.mtp_raw, &best_s.mtp_raw) / (d as f32)
                    + 1e-20)
                    .sqrt();
                let best_scale = h39_rms / best_mtp_rms;
                let mut est_h39 = vec![0.0_f32; d];
                for i in 0..d {
                    let ctx_diff = target_h[2][i] - best_s.prev_h39[i];
                    let mtp_diff = mtp_raw[i] * scale_to_h39 - best_s.mtp_raw[i] * best_scale;
                    est_h39[i] = best_s.target_h39[i] + 0.25_f32 * ctx_diff + 0.12_f32 * mtp_diff;
                }
                return (est_h39, true);
            }
        }

        // Case C: Global cosine match in verified states or residual continuation fallback
        let mut best_any: Option<(&Flash1017VerifiedMtpState, f32)> = None;
        if !mtp_raw.is_empty() {
            for s in verified_states.iter().filter(|s| !s.mtp_raw.is_empty()) {
                let c = cosine_fn(mtp_raw, &s.mtp_raw);
                if best_any.map_or(true, |(_, best_c)| c > best_c) {
                    best_any = Some((s, c));
                }
            }
        }
        let mut est_h39 = vec![0.0_f32; d];
        if let Some((best_s, best_c)) = best_any {
            if best_c > 0.90 {
                for i in 0..d {
                    let step_delta = best_s.target_h39[i] - best_s.prev_h39[i];
                    est_h39[i] = 0.65_f32 * best_s.target_h39[i]
                        + 0.35_f32 * (target_h[2][i] + step_delta)
                        + 0.05_f32 * (mtp_raw.get(i).copied().unwrap_or(0.0) * scale_to_h39);
                }
                return (est_h39, false);
            }
        }
        for i in 0..d {
            est_h39[i] = target_h[2][i]
                + 0.04_f32 * (mtp_raw.get(i).copied().unwrap_or(0.0) * scale_to_h39);
        }
        (est_h39, false)
    }

    /// Executes one `mtp.0..2` / `DSpark` speculative step given explicit `target_h` (`[layer_37, layer_38, layer_39]`
    /// or shifted rolling hidden states during chained `K > 1` drafting), returning:
    /// `(draft_token_id, top1_logit, top1_minus_top2_margin, est_h39, mtp_raw, next_rolling_h, has_manifold_match, elapsed_ms)`.
    fn draft_mtp_step_with_target_h(
        &self,
        cur_token_id: u32,
        prev_token_id: u32,
        target_h: &[Vec<f32>; 3],
        verified_states: &[Flash1017VerifiedMtpState],
        kv_cache: &mut Flash1017KvCache,
        head_mode: Flash1017HeadMode,
    ) -> Result<(u32, f32, f32, Vec<f32>, Vec<f32>, bool, f64), String> {
        let t0 = std::time::Instant::now();
        let full_3layer_mtp = self
            .mtp_layer
            .as_ref()
            .map_or(false, |m| m.num_active_sub_layers() == 3);
        let has_valid_case_a = self
            .select_best_case_a_state(cur_token_id, prev_token_id, verified_states)
            .is_some();
        let has_any_tok_match = verified_states.iter().any(|s| s.token_id == cur_token_id);

        let (mtp_states_3, mtp_raw) = if has_valid_case_a || has_any_tok_match {
            if kv_cache.mtp_k_latent.len() >= 512 {
                let start = kv_cache.mtp_k_latent.len() - 512;
                kv_cache.mtp_k_latent.extend_from_within(start..);
            } else {
                kv_cache
                    .mtp_k_latent
                    .resize(kv_cache.mtp_k_latent.len() + 512, 0.0);
            }
            let new_len = kv_cache.mtp_k_latent.len();
            kv_cache.sync_mtp12_to_mtp0(new_len);
            (
                [Vec::new(), Vec::new(), Vec::new()],
                Vec::new(),
            )
        } else {
            let (st3, _raw_ms) = self.compute_mtp_3layer_states(cur_token_id, target_h, kv_cache)?;
            let raw = st3[2].clone();
            (st3, raw)
        };

        let (est_h39, has_manifold_match, mtp_final_normed) = {
            let (est, has_m) = self.project_mtp_to_target_manifold(
                cur_token_id,
                prev_token_id,
                target_h,
                &mtp_raw,
                verified_states,
            );
            let normed = if full_3layer_mtp && !has_m && !mtp_raw.is_empty() {
                let norm_w = self
                    .mtp_layer
                    .as_ref()
                    .and_then(|m| m.mtp2_norm.as_deref())
                    .unwrap_or(&self.final_norm);
                let normed_mtp2 = flash1017_rms_norm(&mtp_raw, norm_w, 1e-20);
                let normed_est = flash1017_rms_norm(&est, &self.final_norm, 1e-20);
                let mut blended = Vec::with_capacity(5120);
                for i in 0..5120 {
                    blended.push(0.75_f32 * normed_est[i] + 0.25_f32 * normed_mtp2[i]);
                }
                blended
            } else {
                flash1017_rms_norm(&est, &self.final_norm, 1e-20)
            };
            (est, has_m, normed)
        };

        // `F9.2` DSpark Fast Continuation Head: When `cur_token_id` has a verified `Case A` continuation
        // in `verified_states`, directly emit the verified continuation token (`0.002 ms`).
        let mut fast_draft_opt: Option<(u32, f32, f32)> = None;
        if let Some(best_s) =
            self.select_best_case_a_state(cur_token_id, prev_token_id, verified_states)
        {
            if best_s.next_token_id != 0 {
                let markov_ok = self
                    .mtp_layer
                    .as_ref()
                    .and_then(|m| m.markov_transition_logit(cur_token_id, best_s.next_token_id))
                    .map_or(true, |ml| ml > -5.0);
                if markov_ok {
                    fast_draft_opt = Some((
                        best_s.next_token_id,
                        best_s.next_logit.max(20.0),
                        best_s.next_margin.max(1.0),
                    ));
                }
            }
        }

        let (draft_id, draft_logit, margin) = if let Some(fd) = fast_draft_opt {
            fd
        } else {
            self.predict_draft_token_with_margin(
                &mtp_final_normed,
                cur_token_id,
                has_manifold_match,
                head_mode,
            )?
        };
        let _ = mtp_states_3;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok((
            draft_id,
            draft_logit,
            margin,
            est_h39,
            mtp_raw,
            has_manifold_match,
            elapsed_ms,
        ))
    }

    /// `F7.C2`: Chains `mtp.0` for `K = 1..=4` (`Fixed(K)`) or self-regulating `Adaptive (1..3)` steps.
    pub fn draft_mtp_chain(
        &self,
        first_token_id: u32,
        kv_cache: &mut Flash1017KvCache,
        mtp_mode: Flash1017MtpMode,
        head_mode: Flash1017HeadMode,
        max_draft_limit: usize,
    ) -> Result<(Vec<u32>, f64), String> {
        let (draft_ids, _mtp_raws, total_draft_ms) = self.draft_mtp_chain_with_raw(
            first_token_id,
            kv_cache,
            mtp_mode,
            head_mode,
            max_draft_limit,
        )?;
        Ok((draft_ids, total_draft_ms))
    }

    pub fn draft_mtp_chain_with_raw(
        &self,
        first_token_id: u32,
        kv_cache: &mut Flash1017KvCache,
        mtp_mode: Flash1017MtpMode,
        head_mode: Flash1017HeadMode,
        max_draft_limit: usize,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>, f64), String> {
        let max_k = mtp_mode.max_draft_tokens().min(max_draft_limit).min(4);
        if max_k == 0 || self.mtp_layer.is_none() {
            return Ok((Vec::new(), Vec::new(), 0.0));
        }
        let full_3layer_mtp = self
            .mtp_layer
            .as_ref()
            .map_or(false, |m| m.num_active_sub_layers() == 3);

        let mut draft_ids = Vec::with_capacity(max_k);
        let mut draft_mtp_raws = Vec::with_capacity(max_k);
        let mut total_draft_ms = 0.0_f64;
        let mut rolling_target_h = kv_cache.dspark_target_h.clone();
        let mut prev_token_id = kv_cache.token_history.last().copied().unwrap_or(0);
        let mut cur_token_id = first_token_id;
        let mut chain_states = kv_cache.mtp_verified_states.clone();

        for step in 0..max_k {
            if matches!(mtp_mode, Flash1017MtpMode::Adaptive) {
                let has_case_a = self
                    .select_best_case_a_state(cur_token_id, prev_token_id, &chain_states)
                    .is_some();
                if !has_case_a {
                    break;
                }
            }
            let step_mtp_k_len = kv_cache.mtp_k_latent.len();
            let (draft_id, logit, margin, est_h39, mtp_raw, _has_match, step_ms) = self
                .draft_mtp_step_with_target_h(
                    cur_token_id,
                    prev_token_id,
                    &rolling_target_h,
                    &chain_states,
                    kv_cache,
                    head_mode,
                )?;
            total_draft_ms += step_ms;

            if matches!(mtp_mode, Flash1017MtpMode::Adaptive) {
                let verified_transition = chain_states.iter().any(|s| {
                    s.next_token_id == draft_id
                        && self.is_valid_case_a_state(s, cur_token_id, prev_token_id, &chain_states)
                });
                let m_logit = self
                    .mtp_layer
                    .as_ref()
                    .and_then(|m| m.markov_transition_logit(cur_token_id, draft_id))
                    .unwrap_or(0.0);
                let is_self_repeat = draft_id == cur_token_id && margin >= 0.35 && m_logit >= 0.0;
                let strong_markov =
                    cur_token_id < 1000 && draft_id < 1000 && m_logit >= 7.0 && margin >= 0.25;
                let conf_prob = if full_3layer_mtp && !mtp_raw.is_empty() {
                    self.mtp_layer
                        .as_ref()
                        .and_then(|m| {
                            let norm_w = m.mtp2_norm.as_deref().unwrap_or(&self.final_norm);
                            let n_mtp2 = flash1017_rms_norm(&mtp_raw, norm_w, 1e-20);
                            m.dspark_confidence_prob(&n_mtp2, draft_id)
                        })
                        .unwrap_or(0.0)
                } else {
                    0.0
                };
                let jepa_coh =
                    self.jepa_mtp_coherence_score(&est_h39, &kv_cache.jepa_trajectory_anchor);
                let neural_3layer_ok = full_3layer_mtp
                    && ((conf_prob >= 0.72 && margin >= 1.5)
                        || (jepa_coh >= 0.65 && conf_prob >= 0.62 && margin >= 1.2));

                let accept_step =
                    verified_transition || is_self_repeat || strong_markov || neural_3layer_ok;

                if !accept_step {
                    if step == 0 {
                        draft_mtp_raws.push(mtp_raw);
                    } else {
                        kv_cache.mtp_k_latent.truncate(step_mtp_k_len);
                        kv_cache.sync_mtp12_to_mtp0(step_mtp_k_len);
                    }
                    break;
                }
            }

            draft_ids.push(draft_id);
            draft_mtp_raws.push(mtp_raw.clone());

            chain_states.push(Flash1017VerifiedMtpState {
                seq_pos: kv_cache.seq_len + step,
                token_id: cur_token_id,
                prev_token_id,
                prev_h39: rolling_target_h[2].clone(),
                mtp_raw,
                target_h39: est_h39.clone(),
                next_token_id: draft_id,
                next_logit: logit,
                next_margin: margin,
            });

            if matches!(mtp_mode, Flash1017MtpMode::Adaptive) {
                let prev_states = &chain_states[..chain_states.len() - 1];
                let next_has_verified = self
                    .select_best_case_a_state(draft_id, cur_token_id, prev_states)
                    .is_some();
                if !next_has_verified {
                    break;
                }
            }

            if step + 1 < max_k {
                for i in 0..5120 {
                    let delta = est_h39[i] - rolling_target_h[2][i];
                    rolling_target_h[0][i] += delta;
                    rolling_target_h[1][i] += delta;
                    rolling_target_h[2][i] = est_h39[i];
                }
                prev_token_id = cur_token_id;
                cur_token_id = draft_id;
            }
        }

        Ok((draft_ids, draft_mtp_raws, total_draft_ms))
    }

    /// Runs `forward_token_hidden` for `tok_id` AND (when `self.mtp_layer.is_some()` and `max_layers == 40`)
    /// updates `kv_cache.mtp_k_latent` and records a `Flash1017VerifiedMtpState` entry so `mtp.0..2` is warm
    /// and calibrated on the prompt's Layer-39 manifold.
    pub fn forward_prefill_token_with_mtp(
        &self,
        tok_id: u32,
        kv_cache: &mut Flash1017KvCache,
        max_layers: usize,
    ) -> Result<(Vec<f32>, Flash1017StepTelemetry), String> {
        if let Some(prev_state) = kv_cache.mtp_verified_states.last_mut() {
            if prev_state.next_token_id == 0 {
                prev_state.next_token_id = tok_id;
            }
        }
        let seq_pos = kv_cache.seq_len;
        let prev_tok = kv_cache.token_history.last().copied().unwrap_or(0);
        let prev_dspark_h = kv_cache.dspark_target_h.clone();
        let mtp_raw_opt = if self.mtp_layer.is_some() && max_layers == 40 {
            let raw = self
                .compute_mtp_raw_hidden(tok_id, &prev_dspark_h, kv_cache)
                .ok()
                .map(|(r, _)| r)
                .unwrap_or_default();
            Some(raw)
        } else {
            None
        };
        let (h, tel) = self.forward_token_hidden(tok_id, kv_cache, max_layers)?;
        if let Some(mtp_raw) = mtp_raw_opt {
            kv_cache.mtp_verified_states.push(Flash1017VerifiedMtpState {
                seq_pos,
                token_id: tok_id,
                prev_token_id: prev_tok,
                prev_h39: prev_dspark_h[2].clone(),
                mtp_raw,
                target_h39: kv_cache.dspark_target_h[2].clone(),
                next_token_id: 0,
                next_logit: 0.0,
                next_margin: 0.0,
            });
        }
        Ok((h, tel))
    }

    /// `F7.C1–F7.C3`: Generates `max_tokens` tokens using the specified `Flash1017MtpMode`
    /// (`Off`, `Fixed(1..4)`, or `Adaptive`) and returns detailed performance & verification telemetry.
    pub fn generate_with_mtp(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        max_layers: usize,
        head_mode: Flash1017HeadMode,
        mtp_mode: Flash1017MtpMode,
    ) -> Result<Flash1017MtpDuelRow, String> {
        if prompt_ids.is_empty() {
            return Err("generate_with_mtp requires at least 1 prompt token".to_string());
        }
        let max_tokens = max_tokens.max(1);
        let max_layers = max_layers.clamp(1, 40);

        let mut kv_cache = Flash1017KvCache::new();
        let t_prefill = std::time::Instant::now();
        let mut first_normed = vec![0.0_f32; 5120];
        let mut prefill_a = 0usize;
        let mut prefill_b = 0usize;
        let mut prefill_c = 0usize;
        let mut prefill_engram = 0usize;
        for &tok_id in prompt_ids {
            let (h, tel) = self.forward_prefill_token_with_mtp(tok_id, &mut kv_cache, max_layers)?;
            first_normed = h;
            prefill_a += tel.tier_a_hits;
            prefill_b += tel.tier_b_hits;
            prefill_c += tel.tier_c_skips;
            prefill_engram += tel.engram_lookups;
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

        self.decode_with_mtp_from_prefill(
            &first_normed,
            kv_cache,
            prefill_ms,
            (prefill_a, prefill_b, prefill_c, prefill_engram),
            max_tokens,
            max_layers,
            head_mode,
            mtp_mode,
        )
    }

    pub fn decode_with_mtp_from_prefill(
        &self,
        first_normed: &[f32],
        mut kv_cache: Flash1017KvCache,
        prefill_ms: f64,
        prefill_counts: (usize, usize, usize, usize),
        max_tokens: usize,
        max_layers: usize,
        head_mode: Flash1017HeadMode,
        mtp_mode: Flash1017MtpMode,
    ) -> Result<Flash1017MtpDuelRow, String> {
        let max_tokens = max_tokens.max(1);
        let max_layers = max_layers.clamp(1, 40);
        let (mut tier_a_hits, mut tier_b_hits, mut tier_c_skips, mut engram_lookups) =
            prefill_counts;

        let t_head0 = std::time::Instant::now();
        let (first_tok, first_logit, first_margin) =
            self.predict_token_with_mode_and_margin(first_normed, head_mode)?;
        let mut total_head_ms = t_head0.elapsed().as_secs_f64() * 1000.0;
        if let Some(last_state) = kv_cache.mtp_verified_states.last_mut() {
            if last_state.next_token_id == 0 {
                last_state.next_token_id = first_tok;
                last_state.next_logit = first_logit;
                last_state.next_margin = first_margin;
            }
        }

        let mut generated_ids = Vec::with_capacity(max_tokens + 4);
        generated_ids.push(first_tok);

        let target_decode_tokens = max_tokens.saturating_sub(1);
        if target_decode_tokens == 0 {
            let decoded_text = self.tokenizer.decode(&generated_ids);
            return Ok(Flash1017MtpDuelRow {
                mode_label: mtp_mode.label().to_string(),
                generated_ids,
                decoded_text,
                prefill_ms,
                total_decode_ms: 0.0,
                total_draft_ms: 0.0,
                total_verify_ms: 0.0,
                total_head_ms,
                verify_steps: 0,
                total_drafted_tokens: 0,
                total_accepted_drafts: 0,
                acceptance_rate_pct: 0.0,
                mean_tokens_per_step: 1.0,
                effective_ms_per_tok: 0.0,
                effective_tok_per_s: 0.0,
                speedup_vs_k0: 1.0,
                exact_match_k0: true,
                tier_a_hits,
                tier_b_hits,
                tier_c_skips,
                engram_lookups,
            });
        }

        let use_speculative = !matches!(mtp_mode, Flash1017MtpMode::Off)
            && self.mtp_layer.is_some()
            && max_layers == 40;

        let t_decode_start = std::time::Instant::now();
        let mut verify_steps = 0usize;
        let mut total_drafted_tokens = 0usize;
        let mut total_accepted_drafts = 0usize;
        let mut total_draft_ms = 0.0_f64;
        let mut total_verify_ms = 0.0_f64;
        let mut total_emitted_decode_tokens = 0usize;
        let mut cur_tok = first_tok;

        if !use_speculative {
            while total_emitted_decode_tokens < target_decode_tokens {
                let t_ver = std::time::Instant::now();
                let (states, tel) =
                    self.forward_tokens_batch_hidden(&[cur_tok], &mut kv_cache, max_layers, &[])?;
                let final_normed = states.into_iter().next().map(|s| s.final_normed).unwrap_or_default();
                total_verify_ms += t_ver.elapsed().as_secs_f64() * 1000.0;
                tier_a_hits += tel.tier_a_hits;
                tier_b_hits += tel.tier_b_hits;
                tier_c_skips += tel.tier_c_skips;
                engram_lookups += tel.engram_lookups;

                let t_h = std::time::Instant::now();
                let (next_tok, _logit, _margin) =
                    self.predict_token_with_mode_and_margin(&final_normed, head_mode)?;
                total_head_ms += t_h.elapsed().as_secs_f64() * 1000.0;

                verify_steps += 1;
                total_emitted_decode_tokens += 1;
                cur_tok = next_tok;
                if generated_ids.len() < max_tokens {
                    generated_ids.push(cur_tok);
                }
            }
        } else {
            while total_emitted_decode_tokens < target_decode_tokens {
                let mtp_k_base_len = kv_cache.mtp_k_latent.len();
                let prev_tok_before_batch = kv_cache.token_history.last().copied().unwrap_or(0);
                let prev_h39_before_batch = kv_cache.dspark_target_h[2].clone();

                let (draft_ids, draft_mtp_raws, step_draft_ms) = self.draft_mtp_chain_with_raw(
                    cur_tok,
                    &mut kv_cache,
                    mtp_mode,
                    head_mode,
                    4,
                )?;
                total_draft_ms += step_draft_ms;
                let k = draft_ids.len();
                total_drafted_tokens += k;

                if k == 0 {
                    let expected_mtp_len = kv_cache.seq_len * 512;
                    if kv_cache.mtp_k_latent.len() < expected_mtp_len {
                        if kv_cache.mtp_k_latent.len() >= 512 {
                            let start = kv_cache.mtp_k_latent.len() - 512;
                            kv_cache.mtp_k_latent.extend_from_within(start..);
                        } else {
                            kv_cache.mtp_k_latent.resize(expected_mtp_len, 0.0);
                        }
                    } else {
                        kv_cache.mtp_k_latent.truncate(expected_mtp_len);
                    }
                    kv_cache.sync_mtp12_to_mtp0(expected_mtp_len);
                    let t_ver = std::time::Instant::now();
                    let (states, tel) =
                        self.forward_tokens_batch_hidden(&[cur_tok], &mut kv_cache, max_layers, &[])?;
                    let final_normed = states.into_iter().next().map(|s| s.final_normed).unwrap_or_default();
                    total_verify_ms += t_ver.elapsed().as_secs_f64() * 1000.0;
                    tier_a_hits += tel.tier_a_hits;
                    tier_b_hits += tel.tier_b_hits;
                    tier_c_skips += tel.tier_c_skips;
                    engram_lookups += tel.engram_lookups;

                    let t_h = std::time::Instant::now();
                    let (next_tok, next_logit, next_margin) =
                        self.predict_token_with_mode_and_margin(&final_normed, head_mode)?;
                    total_head_ms += t_h.elapsed().as_secs_f64() * 1000.0;

                    kv_cache.mtp_verified_states.push(Flash1017VerifiedMtpState {
                        seq_pos: kv_cache.seq_len.saturating_sub(1),
                        token_id: cur_tok,
                        prev_token_id: prev_tok_before_batch,
                        prev_h39: prev_h39_before_batch,
                        mtp_raw: draft_mtp_raws.into_iter().next().unwrap_or_default(),
                        target_h39: kv_cache.dspark_target_h[2].clone(),
                        next_token_id: next_tok,
                        next_logit,
                        next_margin,
                    });

                    verify_steps += 1;
                    total_emitted_decode_tokens += 1;
                    cur_tok = next_tok;
                    if generated_ids.len() < max_tokens {
                        generated_ids.push(cur_tok);
                    }
                    continue;
                }

                let mut batch_tokens = Vec::with_capacity(1 + k);
                batch_tokens.push(cur_tok);
                batch_tokens.extend_from_slice(&draft_ids);

                let t_ver = std::time::Instant::now();
                let (states, tel) = self.forward_tokens_batch_hidden(
                    &batch_tokens,
                    &mut kv_cache,
                    max_layers,
                    &[],
                )?;
                total_verify_ms += t_ver.elapsed().as_secs_f64() * 1000.0;
                tier_a_hits += tel.tier_a_hits;
                tier_b_hits += tel.tier_b_hits;
                tier_c_skips += tel.tier_c_skips;
                engram_lookups += tel.engram_lookups;

                let t_h = std::time::Instant::now();
                let normed_refs: Vec<&[f32]> = states
                    .iter()
                    .map(|s| s.final_normed.as_slice())
                    .collect();
                let batch_preds =
                    self.predict_tokens_batch_with_mode_and_margin(&normed_refs, head_mode)?;
                let mut n_acc = 0usize;
                let mut bonus_tok = 0u32;
                for j in 0..=k {
                    let (pred_tok, _logit, _margin) = batch_preds[j];
                    if j < k {
                        if pred_tok == draft_ids[j] {
                            n_acc += 1;
                            if generated_ids.len() < max_tokens {
                                generated_ids.push(draft_ids[j]);
                            }
                        } else {
                            let keep_mtp =
                                (mtp_k_base_len + (j + 1) * 512).min(kv_cache.mtp_k_latent.len());
                            kv_cache.restore_position_state(&states[j], keep_mtp);
                            bonus_tok = pred_tok;
                            break;
                        }
                    } else {
                        bonus_tok = pred_tok;
                    }
                }
                total_head_ms += t_h.elapsed().as_secs_f64() * 1000.0;

                // Keep `mtp_k_latent`, `mtp12_k_latent`, and `mtp_verified_states` 1:1 aligned with all `n_acc + 1` accepted tokens
                let keep_mtp_pre =
                    (mtp_k_base_len + k * 512).min(kv_cache.mtp_k_latent.len());
                if n_acc < k {
                    let keep_mtp_len =
                        (mtp_k_base_len + (n_acc + 1) * 512).min(kv_cache.mtp_k_latent.len());
                    kv_cache.mtp_k_latent.truncate(keep_mtp_len);
                    kv_cache.sync_mtp12_to_mtp0(keep_mtp_len);
                } else {
                    kv_cache.mtp_k_latent.truncate(keep_mtp_pre);
                    if kv_cache.mtp_k_latent.len() >= 512 {
                        let start = kv_cache.mtp_k_latent.len() - 512;
                        kv_cache.mtp_k_latent.extend_from_within(start..);
                    } else {
                        kv_cache
                            .mtp_k_latent
                            .resize(keep_mtp_pre + 512, 0.0);
                    }
                    let new_mtp_len = kv_cache.mtp_k_latent.len();
                    kv_cache.sync_mtp12_to_mtp0(new_mtp_len);
                }
                for idx in 0..=n_acc {
                    let seq_pos = states[idx].seq_len - 1;
                    let tok_id = batch_tokens[idx];
                    let prev_tok_id = if idx == 0 {
                        prev_tok_before_batch
                    } else {
                        batch_tokens[idx - 1]
                    };
                    let prev_h39 = if idx == 0 {
                        prev_h39_before_batch.clone()
                    } else {
                        states[idx - 1].dspark_target_h[2].clone()
                    };
                    let mtp_raw = if idx < draft_mtp_raws.len() {
                        draft_mtp_raws[idx].clone()
                    } else {
                        Vec::new()
                    };
                    let (pred_next_id, pred_next_logit, pred_next_margin) = batch_preds[idx];
                    kv_cache.mtp_verified_states.push(Flash1017VerifiedMtpState {
                        seq_pos,
                        token_id: tok_id,
                        prev_token_id: prev_tok_id,
                        prev_h39,
                        mtp_raw,
                        target_h39: states[idx].dspark_target_h[2].clone(),
                        next_token_id: pred_next_id,
                        next_logit: pred_next_logit,
                        next_margin: pred_next_margin,
                    });
                }

                verify_steps += 1;
                total_accepted_drafts += n_acc;
                total_emitted_decode_tokens += n_acc + 1;
                cur_tok = bonus_tok;
                if generated_ids.len() < max_tokens {
                    generated_ids.push(bonus_tok);
                }
            }
        }

        let total_decode_ms = t_decode_start.elapsed().as_secs_f64() * 1000.0;
        let steps_f = (verify_steps.max(1)) as f64;
        let emitted_f = (total_emitted_decode_tokens.max(1)) as f64;
        let acceptance_rate_pct = if total_drafted_tokens > 0 {
            (total_accepted_drafts as f64 / total_drafted_tokens as f64) * 100.0
        } else {
            0.0
        };
        let mean_tokens_per_step = emitted_f / steps_f;
        let effective_ms_per_tok = total_decode_ms / emitted_f;
        let effective_tok_per_s = if effective_ms_per_tok > 0.0 {
            1000.0 / effective_ms_per_tok
        } else {
            0.0
        };
        let decoded_text = self.tokenizer.decode(&generated_ids);

        Ok(Flash1017MtpDuelRow {
            mode_label: mtp_mode.label().to_string(),
            generated_ids,
            decoded_text,
            prefill_ms,
            total_decode_ms,
            total_draft_ms,
            total_verify_ms,
            total_head_ms,
            verify_steps,
            total_drafted_tokens,
            total_accepted_drafts,
            acceptance_rate_pct,
            mean_tokens_per_step,
            effective_ms_per_tok,
            effective_tok_per_s,
            speedup_vs_k0: 1.0,
            exact_match_k0: true,
            tier_a_hits,
            tier_b_hits,
            tier_c_skips,
            engram_lookups,
        })
    }

    /// `F7.C3`: Runs the Great MTP Duel (`K = 0, 1, 2, 3, 4` + `Adaptive (1..3)`) side-by-side
    /// on the exact same prompt, verifying 100% bit-exact output token match against `K = 0`
    /// and computing speedup and effective `tok/s` for each mode.
    pub fn benchmark_mtp_duel(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        max_layers: usize,
        head_mode: Flash1017HeadMode,
    ) -> Result<Vec<Flash1017MtpDuelRow>, String> {
        if prompt_ids.is_empty() {
            return Err("benchmark_mtp_duel requires at least 1 prompt token".to_string());
        }
        let duel_max_tokens = max_tokens.max(13); // 1 prefill token + 12 decode tokens (lcm(1,2,3,4) = 12)
        let max_layers = max_layers.clamp(1, 40);

        // Prefill prompt once and snapshot KV state so all 6 modes start from 100% identical state
        let mut base_kv_cache = Flash1017KvCache::new();
        let t_prefill = std::time::Instant::now();
        let mut first_normed = vec![0.0_f32; 5120];
        let mut prefill_a = 0usize;
        let mut prefill_b = 0usize;
        let mut prefill_c = 0usize;
        let mut prefill_engram = 0usize;
        for &tok_id in prompt_ids {
            let (h, tel) =
                self.forward_prefill_token_with_mtp(tok_id, &mut base_kv_cache, max_layers)?;
            first_normed = h;
            prefill_a += tel.tier_a_hits;
            prefill_b += tel.tier_b_hits;
            prefill_c += tel.tier_c_skips;
            prefill_engram += tel.engram_lookups;
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let prefill_counts = (prefill_a, prefill_b, prefill_c, prefill_engram);

        let modes = [
            Flash1017MtpMode::Off,
            Flash1017MtpMode::Fixed(1),
            Flash1017MtpMode::Fixed(2),
            Flash1017MtpMode::Fixed(3),
            Flash1017MtpMode::Fixed(4),
            Flash1017MtpMode::Adaptive,
        ];
        // `F8.1`: 1-pass untimed warm-up so all active expert pages along the duel trajectory are 100% pinned
        // before the timed comparison across `K = 0..4` + `Adaptive` begins.
        let _ = self.decode_with_mtp_from_prefill(
            &first_normed,
            base_kv_cache.clone_preallocated(),
            prefill_ms,
            prefill_counts,
            duel_max_tokens,
            max_layers,
            head_mode,
            Flash1017MtpMode::Fixed(4),
        );

        let mut rows = Vec::with_capacity(modes.len());
        let mut k0_ids: Vec<u32> = Vec::new();
        let mut k0_ms_per_tok = 0.0_f64;

        for &mode in &modes {
            let mut row = self.decode_with_mtp_from_prefill(
                &first_normed,
                base_kv_cache.clone_preallocated(),
                prefill_ms,
                prefill_counts,
                duel_max_tokens,
                max_layers,
                head_mode,
                mode,
            )?;
            if matches!(mode, Flash1017MtpMode::Off) {
                k0_ids = row.generated_ids.clone();
                k0_ms_per_tok = row.effective_ms_per_tok;
                row.speedup_vs_k0 = 1.0;
                row.exact_match_k0 = true;
            } else {
                let cmp_len = k0_ids.len().min(row.generated_ids.len());
                row.exact_match_k0 =
                    cmp_len > 0 && row.generated_ids[..cmp_len] == k0_ids[..cmp_len];
                row.speedup_vs_k0 = if row.effective_ms_per_tok > 0.0 && k0_ms_per_tok > 0.0 {
                    k0_ms_per_tok / row.effective_ms_per_tok
                } else {
                    1.0
                };
            }
            rows.push(row);
        }

        Ok(rows)
    }

    /// `Fas F4 (F4.1–F4.3)`: Runs a comprehensive side-by-side benchmark comparing:
    /// 1. Baseline `BF16` `head.weight` (`1.324 GB`)
    /// 2. `F4.1` Coalesced `Q8` `head.weight` (`662 MB`) + cosine similarity vs `BF16`
    /// 3. `F4.2` Two-Stage `CoarseToFineConceptHead` (`C2FHead`: Stage A `[129280, 1024]` `Q8` GPU Sketch + Stage B Top-2048 Exact `BF16`)
    /// 4. `F4.3` `mtp.0` speculative draft token (`ms/tok`) + 2-token `matbatch` verification vs sequential verification.
    pub fn benchmark_fas_f4(
        &self,
        final_normed: &[f32],
        prev_token_id: u32,
        kv_cache: &mut Flash1017KvCache,
    ) -> Result<Flash1017F4BenchReport, String> {
        // 1. Warm up + measure BF16 head
        let _ = self.head_shard.matvec_bf16("head.weight", final_normed)?;
        let t_bf16 = std::time::Instant::now();
        let bf16_logits = self.head_shard.matvec_bf16("head.weight", final_normed)?;
        let bf16_ms = t_bf16.elapsed().as_secs_f64() * 1000.0;
        let mut bf16_token = 0u32;
        let mut bf16_logit = f32::NEG_INFINITY;
        for (i, &v) in bf16_logits.iter().enumerate() {
            if v > bf16_logit {
                bf16_logit = v;
                bf16_token = i as u32;
            }
        }

        // 2. Measure F4.1 Coalesced Q8 head & F4.2 CoarseToFineConceptHead
        let mut q8_ms = 0.0_f64;
        let mut q8_token = bf16_token;
        let mut q8_logit = bf16_logit;
        let mut q8_cosine_sim = 1.0_f32;
        let mut c2f_stage_a_ms = 0.0_f64;
        let mut c2f_stage_b_ms = 0.0_f64;
        let mut c2f_total_ms = 0.0_f64;
        let mut c2f_token = bf16_token;
        let mut c2f_logit = bf16_logit;

        if let Some(ref c2f) = self.c2f_head {
            let _ = c2f.matvec_q8(final_normed)?;
            let t_q8 = std::time::Instant::now();
            let q8_logits = c2f.matvec_q8(final_normed)?;
            q8_ms = t_q8.elapsed().as_secs_f64() * 1000.0;

            let mut best_q8 = f32::NEG_INFINITY;
            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for i in 0..bf16_logits.len() {
                let a = bf16_logits[i] as f64;
                let b = q8_logits[i] as f64;
                dot += a * b;
                na += a * a;
                nb += b * b;
                if q8_logits[i] > best_q8 {
                    best_q8 = q8_logits[i];
                    q8_token = i as u32;
                }
            }
            q8_logit = best_q8;
            q8_cosine_sim = (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32;

            let _ = c2f.predict_coarse_to_fine(&self.head_shard, final_normed, 2048)?;
            let t_c2f = std::time::Instant::now();
            let (cid, clog, sa_ms, sb_ms) =
                c2f.predict_coarse_to_fine(&self.head_shard, final_normed, 2048)?;
            c2f_total_ms = t_c2f.elapsed().as_secs_f64() * 1000.0;
            c2f_stage_a_ms = sa_ms;
            c2f_stage_b_ms = sb_ms;
            c2f_token = cid;
            c2f_logit = clog;
        }

        // 3. Measure F4.3 Speculative mtp.0 draft step
        let (mtp_draft_token, mtp_draft_logit, mtp_draft_ms) = if self.mtp_layer.is_some() {
            let saved_mtp_len = kv_cache.mtp_k_latent.len();
            let _ = self.draft_mtp_token(prev_token_id, kv_cache, Flash1017HeadMode::CoarseToFine)?;
            kv_cache.mtp_k_latent.truncate(saved_mtp_len);
            let res =
                self.draft_mtp_token(prev_token_id, kv_cache, Flash1017HeadMode::CoarseToFine)?;
            kv_cache.mtp_k_latent.truncate(saved_mtp_len);
            res
        } else {
            (0, 0.0, 0.0)
        };

        // 4. Measure 2-token matbatch verification vs 2x sequential matvec on Layer 0 (extrapolated across 40 layers)
        let s0 = &self.layer_shards[0];
        let mut x2 = Vec::with_capacity(2 * 5120);
        x2.extend_from_slice(final_normed);
        x2.extend_from_slice(final_normed);
        let exp0 = self.layer_meta[0].tier_a_list.first().copied().unwrap_or(0);
        let p_w1 = format!("layers.0.ffn.experts.{}.w1", exp0);
        let p_w3 = format!("layers.0.ffn.experts.{}.w3", exp0);

        // Sequential 2 tokens
        let t_seq = std::time::Instant::now();
        for _ in 0..2 {
            let _ = s0.matvec_mxfp8("layers.0.attn.wq_a", final_normed)?;
            let _ = s0.matvec_mxfp8("layers.0.attn.wkv", final_normed)?;
            let _ = s0.matvec_mxfp8("layers.0.ffn.shared_experts.w1", final_normed)?;
            let _ = s0.matvec_mxfp8("layers.0.ffn.shared_experts.w3", final_normed)?;
            let _ = s0.matvec_nvfp4(&p_w1, final_normed)?;
            let _ = s0.matvec_nvfp4(&p_w3, final_normed)?;
        }
        let seq_verify_2tok_ms = t_seq.elapsed().as_secs_f64() * 1000.0 * 40.0;

        // Batched 2 tokens via matbatch_mxfp8 + matbatch_nvfp4 (warm up batch buffers first)
        let _ = s0.matbatch_mxfp8("layers.0.attn.wq_a", &x2, 2)?;
        let _ = s0.matbatch_nvfp4(&p_w1, &x2, 2)?;
        let t_mb = std::time::Instant::now();
        let _ = s0.matbatch_mxfp8("layers.0.attn.wq_a", &x2, 2)?;
        let _ = s0.matbatch_mxfp8("layers.0.attn.wkv", &x2, 2)?;
        let _ = s0.matbatch_mxfp8("layers.0.ffn.shared_experts.w1", &x2, 2)?;
        let _ = s0.matbatch_mxfp8("layers.0.ffn.shared_experts.w3", &x2, 2)?;
        let _ = s0.matbatch_nvfp4(&p_w1, &x2, 2)?;
        let _ = s0.matbatch_nvfp4(&p_w3, &x2, 2)?;
        let matbatch_verify_2tok_ms = t_mb.elapsed().as_secs_f64() * 1000.0 * 40.0;

        Ok(Flash1017F4BenchReport {
            bf16_ms,
            bf16_token,
            bf16_logit,
            q8_ms,
            q8_token,
            q8_logit,
            q8_cosine_sim,
            c2f_stage_a_ms,
            c2f_stage_b_ms,
            c2f_total_ms,
            c2f_token,
            c2f_logit,
            mtp_draft_ms,
            mtp_draft_token,
            mtp_draft_logit,
            matbatch_verify_2tok_ms,
            seq_verify_2tok_ms,
        })
    }
}

// ============================================================================
// 106 GB ALL-EXPERTS-IN-RAM COMPRESSOR (`1.58-Bit` + `0.33 MB Rank-16 Q8 JEPA`)
// ============================================================================

/// Compressed `1.58-bit` ternary matrix + `Block-64 U8` scales + `Rank-16 Q8 JEPA` principal residual core.
#[derive(Debug, Clone)]
pub struct CompressedJepaMatrix158 {
    pub rows: usize,
    pub cols: usize,
    pub packed_ternary: Vec<u8>,
    pub row_scales: Vec<f32>,
    pub block_scales: Vec<u8>,
    pub core_b_q8: Vec<u8>,
    pub core_b_scale: Vec<f32>,
    pub core_a_q8: Vec<u8>,
    pub core_a_scale: Vec<f32>,
    pub orig_frob_sq: f64,
    pub ternary_err_sq: f64,
    pub jepa_err_sq: f64,
}

/// Telemetry report after compressing a Flash-1017 layer shard to the `106 GB` (`1.58-bit + Block-64 + Rank-16 Q8 JEPA`) format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flash1017Jepa106ShardReport {
    pub layer_idx: usize,
    pub experts_compressed: usize,
    pub original_file_bytes: u64,
    pub compressed_file_bytes: u64,
    pub saved_bytes: u64,
    pub savings_percent: f32,
    pub ternary_only_rel_frob_err: f32,
    pub jepa_recovered_rel_frob_err: f32,
    pub cosine_similarity: f32,
    pub elapsed_secs: f64,
}

#[inline]
fn orthonormalize_cols_16(mat: &mut [[f32; 16]]) {
    let n = mat.len();
    if n == 0 {
        return;
    }
    for k in 0..16 {
        for j in 0..k {
            let mut dot = 0.0_f64;
            for r in 0..n {
                dot += (mat[r][k] as f64) * (mat[r][j] as f64);
            }
            let d = dot as f32;
            for r in 0..n {
                mat[r][k] -= d * mat[r][j];
            }
        }
        let mut norm_sq = 0.0_f64;
        for r in 0..n {
            let v = mat[r][k] as f64;
            norm_sq += v * v;
        }
        let inv_norm = if norm_sq > 1e-24 {
            (1.0 / norm_sq.sqrt()) as f32
        } else {
            0.0_f32
        };
        for r in 0..n {
            mat[r][k] *= inv_norm;
        }
    }
}

/// Compresses an `NVFP4` matrix (`w_bytes` `[rows, packed_cols]`, `sc_bytes` `[rows, cols/16]` `F8_E4M3`, `sc2_bytes` `[1]` `F32`)
/// into `1.58-bit` ternary (`packed_ternary` + `row_scales`) plus a `Rank-16 Q8 JEPA` principal residual core
/// (`core_b_q8` `[rows, 16]`, `core_b_scale` `[16]`, `core_a_q8` `[16, cols]`, `core_a_scale` `[16]`).
pub fn compress_nvfp4_matrix_to_158_jepa_q8(
    w_bytes: &[u8],
    sc_bytes: &[u8],
    sc2_bytes: &[u8],
    rows: usize,
    packed_cols: usize,
    seed: u64,
) -> CompressedJepaMatrix158 {
    let cols = packed_cols * 2;
    let groups_per_row = cols / 16;
    let global_scale = if sc2_bytes.len() >= 4 {
        f32::from_le_bytes([sc2_bytes[0], sc2_bytes[1], sc2_bytes[2], sc2_bytes[3]])
    } else {
        1.0_f32
    };

    let packed_ternary_cols = (cols + 3) / 4;
    let blocks_per_row = cols / 64;
    let mut packed_ternary = vec![0u8; rows * packed_ternary_cols];
    let mut row_scales = Vec::with_capacity(rows);
    let mut block_scales = vec![0u8; rows * blocks_per_row];
    let mut residual = vec![0.0_f32; rows * cols];

    let mut orig_frob_sq = 0.0_f64;
    let mut ternary_err_sq = 0.0_f64;

    // 1. Dequantize true NVFP4 weights -> Block-64 Lloyd-Max LSQ 1.58-bit ternary -> form residual R = W - W_1.58
    let mut blk_gammas = vec![0.0_f32; blocks_per_row];
    for r in 0..rows {
        let w_row = &w_bytes[r * packed_cols..(r + 1) * packed_cols];
        let sc_row = &sc_bytes[r * groups_per_row..(r + 1) * groups_per_row];
        let res_row = &mut residual[r * cols..(r + 1) * cols];

        for g in 0..groups_per_row {
            let g_sc = f8_e4m3_to_f32(sc_row[g]) * global_scale;
            let b_base = g * 8;
            let c_base = g * 16;
            for k in 0..8 {
                let b = w_row[b_base + k];
                let w0 = FP4_E2M1_LUT[(b & 0x0F) as usize] * g_sc;
                let w1 = FP4_E2M1_LUT[(b >> 4) as usize] * g_sc;
                res_row[c_base + k * 2] = w0;
                res_row[c_base + k * 2 + 1] = w1;
                orig_frob_sq += (w0 as f64) * (w0 as f64) + (w1 as f64) * (w1 as f64);
            }
        }

        let p_row_start = r * packed_ternary_cols;
        let mut max_gamma = 0.0_f32;

        for blk in 0..blocks_per_row {
            let c_start = blk * 64;
            let w_blk = &res_row[c_start..c_start + 64];
            let mut sq_sum = 0.0_f32;
            for &w in w_blk {
                sq_sum += w * w;
            }
            let rms = (sq_sum * (1.0_f32 / 64.0_f32)).sqrt();
            if rms <= 1e-12 {
                blk_gammas[blk] = 0.0;
                let p_blk = &mut packed_ternary[p_row_start + blk * 16..p_row_start + (blk + 1) * 16];
                p_blk.fill(0x55); // four 0-trits (`code = 1`) packed per byte
                continue;
            }

            // Test 4 candidate thresholds around Lloyd-Max optimum (`0.42..0.78 * rms`) to maximize (w·t)^2 / (t·t)
            let mut best_score = -1.0_f32;
            let mut best_gamma = 0.0_f32;
            let mut best_thresh = 0.58_f32 * rms;
            for &factor in &[0.42_f32, 0.54_f32, 0.66_f32, 0.78_f32] {
                let th = factor * rms;
                let mut dot_wt = 0.0_f32;
                let mut cnt = 0u32;
                for &w in w_blk {
                    if w > th {
                        dot_wt += w;
                        cnt += 1;
                    } else if w < -th {
                        dot_wt -= w;
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    let score = (dot_wt * dot_wt) / (cnt as f32);
                    if score > best_score {
                        best_score = score;
                        best_gamma = dot_wt / (cnt as f32);
                        best_thresh = th;
                    }
                }
            }

            blk_gammas[blk] = best_gamma;
            if best_gamma > max_gamma {
                max_gamma = best_gamma;
            }

            let p_blk_start = p_row_start + blk * 16;
            for i in 0..64 {
                let w = w_blk[i];
                let code: u8 = if w > best_thresh {
                    2
                } else if w < -best_thresh {
                    0
                } else {
                    1
                };
                packed_ternary[p_blk_start + (i >> 2)] |= code << ((i & 3) * 2);
            }
        }

        let row_sc = if max_gamma > 1e-12 {
            max_gamma / 255.0_f32
        } else {
            0.0_f32
        };
        row_scales.push(row_sc);

        const LUT: [f32; 4] = [-1.0, 0.0, 1.0, 0.0];
        let bs_row = &mut block_scales[r * blocks_per_row..(r + 1) * blocks_per_row];
        let p_row = &packed_ternary[p_row_start..p_row_start + packed_ternary_cols];

        for blk in 0..blocks_per_row {
            let u8_sc = if row_sc > 0.0 {
                (blk_gammas[blk] / row_sc).round().clamp(0.0, 255.0) as u8
            } else {
                0
            };
            bs_row[blk] = u8_sc;
            let q_gamma = (u8_sc as f32) * row_sc;
            let p_blk = &p_row[blk * 16..(blk + 1) * 16];
            let res_blk = &mut res_row[blk * 64..(blk + 1) * 64];
            for (i, &pb) in p_blk.iter().enumerate() {
                let base = i * 4;
                let r0 = res_blk[base] - LUT[(pb & 3) as usize] * q_gamma;
                let r1 = res_blk[base + 1] - LUT[((pb >> 2) & 3) as usize] * q_gamma;
                let r2 = res_blk[base + 2] - LUT[((pb >> 4) & 3) as usize] * q_gamma;
                let r3 = res_blk[base + 3] - LUT[((pb >> 6) & 3) as usize] * q_gamma;
                res_blk[base] = r0;
                res_blk[base + 1] = r1;
                res_blk[base + 2] = r2;
                res_blk[base + 3] = r3;
                ternary_err_sq += (r0 as f64) * (r0 as f64)
                    + (r1 as f64) * (r1 as f64)
                    + (r2 as f64) * (r2 as f64)
                    + (r3 as f64) * (r3 as f64);
            }
        }
    }

    // 2. Two-sweep Randomized Subspace Iteration (`Rank = 16`) on residual matrix R (`[rows, cols]`)
    let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut p = vec![[0.0_f32; 16]; cols];
    for c in 0..cols {
        for k in 0..16 {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            p[c][k] = (((z >> 40) as i32 - (1 << 23)) as f32) * (1.0_f32 / ((1 << 23) as f32));
        }
    }
    orthonormalize_cols_16(&mut p);

    let mut q = vec![[0.0_f32; 16]; rows];
    let mut z = vec![[0.0_f32; 16]; cols];

    for sweep in 0..2 {
        // q = R * p (`[rows, 16]`)
        for r in 0..rows {
            let r_row = &residual[r * cols..(r + 1) * cols];
            let mut acc = [0.0_f32; 16];
            for c in 0..cols {
                let v = r_row[c];
                let p_row = &p[c];
                for k in 0..16 {
                    acc[k] += v * p_row[k];
                }
            }
            q[r] = acc;
        }
        orthonormalize_cols_16(&mut q);

        // z = R^T * q (`[cols, 16]`)
        z.fill([0.0_f32; 16]);
        for r in 0..rows {
            let r_row = &residual[r * cols..(r + 1) * cols];
            let q_row = q[r];
            for c in 0..cols {
                let v = r_row[c];
                let z_row = &mut z[c];
                for k in 0..16 {
                    z_row[k] += v * q_row[k];
                }
            }
        }
        if sweep == 0 {
            p.copy_from_slice(&z);
            orthonormalize_cols_16(&mut p);
        }
    }

    // 3. Quantize orthonormal left subspace `q` (`[rows, 16]`) -> `core_b_q8` (`[rows, 16]`) + `core_b_scale` (`[16]`)
    //    and right projected vectors `z` (`[cols, 16]`) -> `core_a_q8` (`[16, cols]`) + `core_a_scale` (`[16]`)
    let mut core_b_q8 = vec![0u8; rows * 16];
    let mut core_b_scale = vec![0.0_f32; 16];
    let mut core_a_q8 = vec![0u8; 16 * cols];
    let mut core_a_scale = vec![0.0_f32; 16];
    let mut captured_frob_sq = 0.0_f64;

    for k in 0..16 {
        let mut max_u = 0.0_f32;
        for r in 0..rows {
            let a = q[r][k].abs();
            if a > max_u {
                max_u = a;
            }
        }
        let b_sc = if max_u > 1e-12 { max_u / 127.0_f32 } else { 0.0_f32 };
        core_b_scale[k] = b_sc;
        if b_sc > 0.0 {
            let inv_b = 1.0_f32 / b_sc;
            for r in 0..rows {
                let q8 = (q[r][k] * inv_b).round().clamp(-127.0, 127.0) as i8;
                core_b_q8[r * 16 + k] = q8 as u8;
            }
        }

        let mut max_z = 0.0_f32;
        let mut col_norm_sq = 0.0_f64;
        for c in 0..cols {
            let zv = z[c][k];
            let a = zv.abs();
            if a > max_z {
                max_z = a;
            }
            col_norm_sq += (zv as f64) * (zv as f64);
        }
        captured_frob_sq += col_norm_sq;
        let a_sc = if max_z > 1e-12 { max_z / 127.0_f32 } else { 0.0_f32 };
        core_a_scale[k] = a_sc;
        if a_sc > 0.0 {
            let inv_a = 1.0_f32 / a_sc;
            let a_row_off = k * cols;
            for c in 0..cols {
                let q8 = (z[c][k] * inv_a).round().clamp(-127.0, 127.0) as i8;
                core_a_q8[a_row_off + c] = q8 as u8;
            }
        }
    }

    let jepa_err_sq = (ternary_err_sq - captured_frob_sq).max(0.0);

    CompressedJepaMatrix158 {
        rows,
        cols,
        packed_ternary,
        row_scales,
        block_scales,
        core_b_q8,
        core_b_scale,
        core_a_q8,
        core_a_scale,
        orig_frob_sq,
        ternary_err_sq,
        jepa_err_sq,
    }
}

/// Pure-Rust `106 GB` All-Experts-in-RAM Compressor (`1.58-Bit` + `0.33 MB Rank-16 Q8 JEPA Core` across all 384 experts).
pub struct Flash1017Jepa106Compressor;

impl Flash1017Jepa106Compressor {
    /// Compresses a single Flash-1017 MoE layer shard (`model-00003`..`model-00042`, layers `0..39`)
    /// so all 384 experts are stored as `1.58-bit` ternary + `0.33 MB Rank-16 Q8 JEPA Core` (`~2.10 GB` per layer shard).
    pub fn compress_layer_shard(
        input_shard_path: &Path,
        output_shard_path: &Path,
    ) -> Result<Flash1017Jepa106ShardReport, String> {
        use rayon::prelude::*;
        use std::io::{BufWriter, Write};

        let t0 = std::time::Instant::now();
        let inspector = Flash1017ShardInspector::open(input_shard_path)?;
        let summary = inspector.summarize()?;
        let layer_idx = summary.layer_idx;
        let num_experts = summary.detected_experts;

        // 1. Retain all non-MoE tensors (`MLA Attention`, `shared_experts`, `gate`, `norms`, `hc_*`) 100% untouched
        let mut zero_copy_tensors: Vec<(String, String, Vec<usize>, &[u8])> = Vec::new();
        let mut sorted_names: Vec<&String> = inspector.tensors.keys().collect();
        sorted_names.sort();

        for name in sorted_names {
            if !name.contains(".ffn.experts.") && !name.contains(".merkaba.") {
                let (meta, slice) = inspector.get_tensor_data(name)?;
                zero_copy_tensors.push((
                    name.clone(),
                    meta.dtype.clone(),
                    meta.shape.clone(),
                    slice,
                ));
            }
        }

        // 2. Compress all 384 experts in parallel across all CPU cores via Rayon
        struct CompressedExpert106 {
            expert_idx: usize,
            w1: CompressedJepaMatrix158,
            w2: CompressedJepaMatrix158,
            w3: CompressedJepaMatrix158,
        }

        let experts: Vec<CompressedExpert106> = (0..num_experts)
            .into_par_iter()
            .map(|exp_idx| {
                let p_w1 = format!("layers.{}.ffn.experts.{}.w1", layer_idx, exp_idx);
                let p_w2 = format!("layers.{}.ffn.experts.{}.w2", layer_idx, exp_idx);
                let p_w3 = format!("layers.{}.ffn.experts.{}.w3", layer_idx, exp_idx);

                let (m1, w1_b) = inspector.get_tensor_data(&format!("{}.weight", p_w1))?;
                let (_, sc1_b) = inspector.get_tensor_data(&format!("{}.weight_scale", p_w1))?;
                let (_, sc2_1_b) = inspector.get_tensor_data(&format!("{}.weight_scale_2", p_w1))?;

                let (m2, w2_b) = inspector.get_tensor_data(&format!("{}.weight", p_w2))?;
                let (_, sc2_b) = inspector.get_tensor_data(&format!("{}.weight_scale", p_w2))?;
                let (_, sc2_2_b) = inspector.get_tensor_data(&format!("{}.weight_scale_2", p_w2))?;

                let (m3, w3_b) = inspector.get_tensor_data(&format!("{}.weight", p_w3))?;
                let (_, sc3_b) = inspector.get_tensor_data(&format!("{}.weight_scale", p_w3))?;
                let (_, sc2_3_b) = inspector.get_tensor_data(&format!("{}.weight_scale_2", p_w3))?;

                let seed_base = ((layer_idx as u64) << 16) ^ ((exp_idx as u64) << 4);
                let w1 = compress_nvfp4_matrix_to_158_jepa_q8(
                    w1_b,
                    sc1_b,
                    sc2_1_b,
                    m1.shape[0],
                    m1.shape[1],
                    seed_base ^ 1,
                );
                let w2 = compress_nvfp4_matrix_to_158_jepa_q8(
                    w2_b,
                    sc2_b,
                    sc2_2_b,
                    m2.shape[0],
                    m2.shape[1],
                    seed_base ^ 2,
                );
                let w3 = compress_nvfp4_matrix_to_158_jepa_q8(
                    w3_b,
                    sc3_b,
                    sc2_3_b,
                    m3.shape[0],
                    m3.shape[1],
                    seed_base ^ 3,
                );

                Ok(CompressedExpert106 {
                    expert_idx: exp_idx,
                    w1,
                    w2,
                    w3,
                })
            })
            .collect::<Result<Vec<CompressedExpert106>, String>>()?;

        let mut total_orig_sq = 0.0_f64;
        let mut total_tern_err_sq = 0.0_f64;
        let mut total_jepa_err_sq = 0.0_f64;

        struct OwnedTensor {
            name: String,
            dtype: String,
            shape: Vec<usize>,
            data: Vec<u8>,
        }
        let mut owned_tensors: Vec<OwnedTensor> = Vec::with_capacity(num_experts * 18 + 3);

        let f32_slice_to_bytes = |v: &[f32]| -> Vec<u8> {
            let mut b = Vec::with_capacity(v.len() * 4);
            for &x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b
        };

        let mut push_jepa_mat = |prefix: String, m: CompressedJepaMatrix158| {
            total_orig_sq += m.orig_frob_sq;
            total_tern_err_sq += m.ternary_err_sq;
            total_jepa_err_sq += m.jepa_err_sq;

            owned_tensors.push(OwnedTensor {
                name: format!("{}.packed_ternary", prefix),
                dtype: "U8".to_string(),
                shape: vec![m.rows, m.cols / 4],
                data: m.packed_ternary,
            });
            owned_tensors.push(OwnedTensor {
                name: format!("{}.row_scales", prefix),
                dtype: "F32".to_string(),
                shape: vec![m.rows],
                data: f32_slice_to_bytes(&m.row_scales),
            });
            owned_tensors.push(OwnedTensor {
                name: format!("{}.block_scales", prefix),
                dtype: "U8".to_string(),
                shape: vec![m.rows, m.cols / 64],
                data: m.block_scales,
            });
            owned_tensors.push(OwnedTensor {
                name: format!("{}.core_b_q8", prefix),
                dtype: "I8".to_string(),
                shape: vec![m.rows, 16],
                data: m.core_b_q8,
            });
            owned_tensors.push(OwnedTensor {
                name: format!("{}.core_b_scale", prefix),
                dtype: "F32".to_string(),
                shape: vec![16],
                data: f32_slice_to_bytes(&m.core_b_scale),
            });
            owned_tensors.push(OwnedTensor {
                name: format!("{}.core_a_q8", prefix),
                dtype: "I8".to_string(),
                shape: vec![16, m.cols],
                data: m.core_a_q8,
            });
            owned_tensors.push(OwnedTensor {
                name: format!("{}.core_a_scale", prefix),
                dtype: "F32".to_string(),
                shape: vec![16],
                data: f32_slice_to_bytes(&m.core_a_scale),
            });
        };

        for exp in experts {
            let e = exp.expert_idx;
            push_jepa_mat(format!("layers.{}.ffn.experts.{}.w1", layer_idx, e), exp.w1);
            push_jepa_mat(format!("layers.{}.ffn.experts.{}.w2", layer_idx, e), exp.w2);
            push_jepa_mat(format!("layers.{}.ffn.experts.{}.w3", layer_idx, e), exp.w3);
        }

        // Write index metadata: all 384 experts active in Tier B (`1.58-bit + JEPA`), 0 pruned
        let mut all_indices_bytes = Vec::with_capacity(num_experts * 8);
        for e in 0..num_experts {
            all_indices_bytes.extend_from_slice(&(e as i64).to_le_bytes());
        }
        owned_tensors.push(OwnedTensor {
            name: format!("layers.{}.merkaba.tier_a_indices", layer_idx),
            dtype: "I64".to_string(),
            shape: vec![0],
            data: Vec::new(),
        });
        owned_tensors.push(OwnedTensor {
            name: format!("layers.{}.merkaba.tier_b_indices", layer_idx),
            dtype: "I64".to_string(),
            shape: vec![num_experts],
            data: all_indices_bytes,
        });
        owned_tensors.push(OwnedTensor {
            name: format!("layers.{}.merkaba.tier_c_pruned_indices", layer_idx),
            dtype: "I64".to_string(),
            shape: vec![0],
            data: Vec::new(),
        });

        // Build 8-byte-aligned safetensors header & write atomically
        let mut header_map = serde_json::Map::new();
        let mut current_offset: usize = 0;

        for (name, dtype, shape, slice) in &zero_copy_tensors {
            let start = current_offset;
            let end = start + slice.len();
            current_offset = (end + 7) & !7;
            let mut meta = serde_json::Map::new();
            meta.insert("dtype".to_string(), serde_json::Value::String(dtype.clone()));
            meta.insert("shape".to_string(), serde_json::json!(shape));
            meta.insert("data_offsets".to_string(), serde_json::json!([start, end]));
            header_map.insert(name.clone(), serde_json::Value::Object(meta));
        }

        for ot in &owned_tensors {
            let start = current_offset;
            let end = start + ot.data.len();
            current_offset = (end + 7) & !7;
            let mut meta = serde_json::Map::new();
            meta.insert(
                "dtype".to_string(),
                serde_json::Value::String(ot.dtype.clone()),
            );
            meta.insert("shape".to_string(), serde_json::json!(ot.shape));
            meta.insert("data_offsets".to_string(), serde_json::json!([start, end]));
            header_map.insert(ot.name.clone(), serde_json::Value::Object(meta));
        }

        let header_json = serde_json::Value::Object(header_map);
        let mut header_str = serde_json::to_string(&header_json)
            .map_err(|e| format!("Failed to serialize 106GB safetensors header: {}", e))?;
        while header_str.len() % 8 != 0 {
            header_str.push(' ');
        }
        let header_bytes = header_str.as_bytes();
        let header_len = header_bytes.len() as u64;

        if let Some(parent) = output_shard_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp_output = output_shard_path.with_extension("safetensors.tmp");
        let file = File::create(&tmp_output)
            .map_err(|e| format!("Failed to create {}: {}", tmp_output.display(), e))?;
        let mut writer = BufWriter::with_capacity(32 * 1024 * 1024, file);

        writer
            .write_all(&header_len.to_le_bytes())
            .map_err(|e| format!("Write header len failed: {}", e))?;
        writer
            .write_all(header_bytes)
            .map_err(|e| format!("Write header JSON failed: {}", e))?;

        let zero_pad = [0u8; 8];
        for (_, _, _, slice) in &zero_copy_tensors {
            writer
                .write_all(slice)
                .map_err(|e| format!("Write zero-copy tensor failed: {}", e))?;
            let rem = slice.len() % 8;
            if rem != 0 {
                writer
                    .write_all(&zero_pad[..8 - rem])
                    .map_err(|e| format!("Write pad failed: {}", e))?;
            }
        }
        for ot in &owned_tensors {
            writer
                .write_all(&ot.data)
                .map_err(|e| format!("Write owned tensor failed: {}", e))?;
            let rem = ot.data.len() % 8;
            if rem != 0 {
                writer
                    .write_all(&zero_pad[..8 - rem])
                    .map_err(|e| format!("Write pad failed: {}", e))?;
            }
        }
        writer
            .flush()
            .map_err(|e| format!("Flush failed: {}", e))?;
        drop(writer);

        std::fs::rename(&tmp_output, output_shard_path).map_err(|e| {
            format!(
                "Failed to rename {} to {}: {}",
                tmp_output.display(),
                output_shard_path.display(),
                e
            )
        })?;

        let original_file_bytes = inspector.mmap.len() as u64;
        let compressed_file_bytes = std::fs::metadata(output_shard_path)
            .map(|m| m.len())
            .unwrap_or((8 + header_bytes.len() + current_offset) as u64);
        let saved_bytes = original_file_bytes.saturating_sub(compressed_file_bytes);
        let savings_percent = if original_file_bytes > 0 {
            (saved_bytes as f32 / original_file_bytes as f32) * 100.0
        } else {
            0.0
        };

        let orig_norm = total_orig_sq.sqrt().max(1e-12);
        let ternary_only_rel_frob_err = (total_tern_err_sq.sqrt() / orig_norm) as f32;
        let jepa_recovered_rel_frob_err = (total_jepa_err_sq.sqrt() / orig_norm) as f32;
        let cosine_similarity =
            (1.0_f64 - (total_jepa_err_sq / total_orig_sq.max(1e-12))).max(0.0).sqrt() as f32;

        Ok(Flash1017Jepa106ShardReport {
            layer_idx,
            experts_compressed: num_experts,
            original_file_bytes,
            compressed_file_bytes,
            saved_bytes,
            savings_percent,
            ternary_only_rel_frob_err,
            jepa_recovered_rel_frob_err,
            cosine_similarity,
            elapsed_secs: t0.elapsed().as_secs_f64(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_compress_nvfp4_matrix_to_158_jepa_q8_accuracy() {
        let rows = 64usize;
        let cols = 128usize;
        let packed_cols = cols / 2;
        let groups_per_row = cols / 16;

        let mut w_bytes = vec![0u8; rows * packed_cols];
        let mut sc_bytes = vec![0x38u8; rows * groups_per_row]; // 0x38 = 1.0 in F8_E4M3
        for (i, b) in w_bytes.iter_mut().enumerate() {
            let n0 = ((i * 7 + 3) % 16) as u8;
            let n1 = ((i * 13 + 5) % 16) as u8;
            *b = n0 | (n1 << 4);
        }
        for (i, s) in sc_bytes.iter_mut().enumerate() {
            *s = if i % 2 == 0 { 0x38 } else { 0x3C };
        }
        let sc2_bytes = 0.02_f32.to_le_bytes();

        let comp = compress_nvfp4_matrix_to_158_jepa_q8(
            &w_bytes,
            &sc_bytes,
            &sc2_bytes,
            rows,
            packed_cols,
            42,
        );
        assert_eq!(comp.packed_ternary.len(), rows * (cols / 4));
        assert_eq!(comp.row_scales.len(), rows);
        assert_eq!(comp.core_b_q8.len(), rows * 16);
        assert_eq!(comp.core_a_q8.len(), 16 * cols);
        assert!(
            comp.jepa_err_sq < comp.ternary_err_sq,
            "Rank-16 Q8 JEPA core must strictly reduce Frobenius quantization error ({} < {})",
            comp.jepa_err_sq,
            comp.ternary_err_sq
        );
    }

    #[test]
    fn test_inspect_shard_layer_5() {
        let shard_path = PathBuf::from(
            "/Volumes/NET/ai-projects/models/DeepSeek-V4.1-Flash-1017-NVFP4/model-00008-of-00048.safetensors",
        );
        if !shard_path.exists() {
            eprintln!("Shard file not yet downloaded, skipping test.");
            return;
        }

        let inspector = Flash1017ShardInspector::open(&shard_path).expect("Failed to open shard 8");
        let summary = inspector.summarize().expect("Failed to summarize shard 8");

        assert_eq!(summary.layer_idx, 5, "Shard 8 must be Layer 5");
        assert_eq!(summary.detected_experts, 384, "Must detect exactly 384 experts");
        assert!(!summary.is_protected, "Layer 5 is not an anchor layer");
        assert!(!summary.has_engram, "Layer 5 does not contain Engram");
        assert!(summary.has_attention, "Layer 5 must have attention weights");
        assert!(summary.has_attn_sink, "Layer 5 must have attn_sink");

        // Test expert 0 view
        let expert0 = inspector.get_expert_view(5, 0).expect("Failed to get expert 0 view");
        assert_eq!(expert0.expert_idx, 0);
        assert_eq!(expert0.w1_shape, vec![2304, 2560]);
        assert_eq!(expert0.w2_shape, vec![5120, 1152]);
        assert_eq!(expert0.w3_shape, vec![2304, 2560]);
        assert_eq!(expert0.total_bytes, 17_694_720); // 3 * 5,898,240 bytes = 16.87 MB

        // Test Frobenius energy calculation
        let energy0 = inspector
            .compute_expert_frobenius_energy(5, 0)
            .expect("Failed to compute Frobenius energy");
        assert!(energy0 > 0.0, "Energy must be positive");
        println!("\n========================================================");
        println!(">>> Shard 8 (Lager 5) Validerad med Zero-Copy mmap!");
        println!(">>> Detekterade experter: {}", summary.detected_experts);
        println!(">>> Total MoE storlek i lagret: {:.2} GB", summary.total_moe_bytes as f64 / 1e9);
        println!(">>> Expert 0 Frobenius-energi: {:.4}", energy0);
        println!("========================================================\n");
    }

    #[test]
    fn test_knowledge_card_parsing_and_atomic_checkpoint() {
        let raw_llm = r#"<think>
Jag ska skapa ett tvåspråkigt faktakort (Engelska #1 + Svenska #2) inom fysik/01_mekanik.
</think>
```json
[
  {
    "id": "newtons_andra_lag",
    "title": "Newton's Second Law of Motion / Newtons andra rörelselag (F = m * a)",
    "domain": "fysik",
    "category": "01_mekanik",
    "key_triggers": ["newton's second law", "force mass acceleration", "newtons andra lag", "f=ma", "kraft massa acceleration"],
    "fact_en": "Newton's second law states that the net force F (in Newtons, N) acting on a body equals its mass m (in kg) multiplied by its acceleration a (in m/s²): F = m · a.",
    "fact_sv": "Newtons andra lag beskriver att den resulterande kraften F (i Newton, N) som verkar på en kropp är lika med kroppens massa m (i kg) multiplicerad med dess acceleration a (i m/s²): F = m · a.",
    "fact_or_recipe": "Newton's second law states that the net force F (in Newtons, N) acting on a body equals its mass m (in kg) multiplied by its acceleration a (in m/s²): F = m · a.",
    "translations": {
      "de": "Das zweite newtonsche Gesetz besagt: F = m · a."
    },
    "components": ["Force / Kraft (F)", "Mass / Massa (m)", "Acceleration (a)"],
    "notes": "Valid in inertial reference frames at velocities far below the speed of light c."
  }
]
```"#;

        let cards = CrystallizedKnowledgeCard::parse_from_llm_output(raw_llm)
            .expect("Should parse LLM output with <think> and json fence");
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].id, "newtons_andra_lag");
        assert_eq!(cards[0].domain, "fysik");
        assert!(cards[0].fact_en.contains("Newton's second law"));
        assert!(cards[0].fact_sv.contains("Newtons andra lag"));
        assert_eq!(
            cards[0].translations.get("de").map(|s| s.as_str()),
            Some("Das zweite newtonsche Gesetz besagt: F = m · a.")
        );

        let dir_path = std::env::temp_dir().join("flash1017_test_db");
        let _ = std::fs::remove_dir_all(&dir_path);
        std::fs::create_dir_all(&dir_path).unwrap();
        let created_domains = init_core_domain_skeleton(&dir_path)
            .expect("Should initialize 20 core domains");
        assert_eq!(created_domains, CORE_DOMAINS.len());
        assert!(dir_path.join("fysik").join("domain_profile.json").exists());

        let saved_path = cards[0]
            .save_to_databases(&dir_path)
            .expect("Should save card atomically");
        assert!(saved_path.exists());

        let cp_path = dir_path.join("crystallization_checkpoint.json");
        let mut cp = CrystallizationCheckpoint::load_or_default(&cp_path);
        cp.record_saved_card(&cards[0], Some("fysik:mekanik"));
        cp.save_atomic(&cp_path).expect("Should save checkpoint");

        let loaded_cp = CrystallizationCheckpoint::load_or_default(&cp_path);
        assert_eq!(loaded_cp.total_cards_saved, 1);
        assert_eq!(loaded_cp.domain_card_counts.get("fysik"), Some(&1));
        let _ = std::fs::remove_dir_all(&dir_path);

        // Also verify compatibility with existing databases/kemi card if present, and ensure 20 core domains exist
        let real_db_root = PathBuf::from("/Volumes/NET/ai-projects/databases");
        if real_db_root.exists() {
            let _ = init_core_domain_skeleton(&real_db_root);
        }
        let existing_card_path = PathBuf::from(
            "/Volumes/NET/ai-projects/databases/kemi/01_periodiska_systemet/grundamnen/vatten_och_molmassa.json",
        );
        if existing_card_path.exists() {
            let content = std::fs::read_to_string(&existing_card_path).unwrap();
            let parsed = CrystallizedKnowledgeCard::parse_from_llm_output(&content)
                .expect("Existing vatten_och_molmassa.json must validate cleanly");
            assert_eq!(parsed[0].id, "vatten_och_molmassa");
        }
    }

    #[test]
    fn test_flash1017_tokenizer_and_sinkhorn() {
        // 1. Verify Sinkhorn-Knopp 4x4 doubly-stochastic manifold projection (F3.2)
        let logits = [
            2.8, -3.1, -3.0, -3.2, -3.1, 2.7, -3.1, -2.9, -3.0, -3.2, 2.9, -3.1, -3.2, -2.9, -3.1,
            2.8,
        ];
        let m = sinkhorn_knopp_4x4(&logits, 20, 1e-6);
        for r in 0..4 {
            let row_sum: f32 = (0..4).map(|c| m[r][c]).sum();
            assert!(
                (row_sum - 1.0).abs() < 2e-3,
                "Row {} sum must be ~1.0, got {}",
                r,
                row_sum
            );
            assert!(
                m[r][r] > 0.95,
                "Diagonal m[{}][{}] must dominate (>0.95), got {}",
                r,
                r,
                m[r][r]
            );
        }
        for c in 0..4 {
            let col_sum: f32 = (0..4).map(|r| m[r][c]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-4,
                "Col {} sum must be ~1.0, got {}",
                c,
                col_sum
            );
        }

        // 2. Verify Flash1017Tokenizer ByteLevel BPE encode/decode round-trip (F3.1)
        let tok_path = PathBuf::from(
            "/Volumes/NET/ai-projects/models/DeepSeek-V4.1-Flash-1017-Tier2/tokenizer.json",
        );
        if tok_path.exists() {
            let tok =
                Flash1017Tokenizer::from_file(&tok_path).expect("Load Flash1017 tokenizer.json");
            let sample = "Hej världen! 2 + 2 = 4.";
            let ids = tok.encode(sample);
            assert!(!ids.is_empty());
            let decoded = tok.decode(&ids);
            assert_eq!(decoded, sample);
            let chat_ids = tok.format_chat_prompt("Vad är 2+2?", false);
            assert_eq!(chat_ids[0], FLASH1017_BOS_TOKEN_ID);
            assert!(chat_ids.contains(&FLASH1017_USER_TOKEN_ID));
            assert!(chat_ids.contains(&FLASH1017_ASSISTANT_TOKEN_ID));
            assert!(chat_ids.contains(&FLASH1017_THINK_END_TOKEN_ID));
        }

        // 3. Verify Flash1017Runtime::open + 1-layer forward_token_hidden (F3.1 - F3.5)
        let tier2_dir = PathBuf::from(
            "/Volumes/NET/ai-projects/models/DeepSeek-V4.1-Flash-1017-Tier2",
        );
        if tier2_dir.join("model-00002-of-00048.safetensors").exists()
            && tier2_dir.join("model-00043-of-00048.safetensors").exists()
        {
            let t_open = std::time::Instant::now();
            let mut rt = match Flash1017Runtime::open(&tier2_dir, false) {
                Ok(r) => r,
                Err(e) if e.contains("[MERKABA-GB10-LOCK]") => {
                    println!(">>> Verified GB10 hardware lock on non-GB10 host: {}", e);
                    return;
                }
                Err(e) => panic!("Open Flash1017Runtime: {}", e),
            };
            let open_ms = t_open.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(rt.layer_shards.len(), 40);
            if tier2_dir.join("model-00044-of-00048.safetensors").exists() {
                rt.enable_mtp_layer().expect("Open mtp.0 layer");
                assert!(rt.mtp_layer.is_some());
            }
            let mut kv = Flash1017KvCache::new();
            let (h, tel) = rt
                .forward_token_hidden(FLASH1017_BOS_TOKEN_ID, &mut kv, 1)
                .expect("Forward 1 layer");
            assert_eq!(h.len(), 5120);
            assert!(h.iter().all(|v| v.is_finite()), "Hidden state must be finite");
            let (top_cand, top_score) = rt
                .head_shard
                .score_bf16_rows("head.weight", &h, &[0, 1, 2, 100, 1000])
                .expect("Score candidate BF16 rows");
            assert!(top_score.is_finite());
            println!(
                ">>> Flash1017Runtime opened 40 shards in {:.1} ms | L0 step: {:.1} ms (Tier A={}, B={}, C={}) | sample cand={} ({:.3})",
                open_ms, tel.step_ms, tel.tier_a_hits, tel.tier_b_hits, tel.tier_c_skips, top_cand, top_score
            );

            // 4. Verify Fas F5 (`F5.2` + `F5.3`): 5120d prompt routing vector, DoRA residual injection & `forward_embedding_hidden`
            let route_vec = rt.compute_prompt_routing_vector_5120(&[FLASH1017_BOS_TOKEN_ID]);
            assert_eq!(route_vec.len(), 5120);
            let norm: f32 = route_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "Routing vector should be L2-normalized");

            let mut deltas = vec![None; 40];
            deltas[0] = Some(vec![0.01_f32; 5120]);
            rt.set_layer_dora_deltas(deltas);
            let mut kv_dora = Flash1017KvCache::new();
            let (h_dora, _) = rt
                .forward_embedding_hidden(&route_vec, FLASH1017_BOS_TOKEN_ID, &mut kv_dora, 1, 1.0)
                .expect("Forward embedding hidden with DoRA");
            assert_eq!(h_dora.len(), 5120);
            assert!(h_dora.iter().all(|v| v.is_finite()));
            rt.clear_layer_dora_deltas();

            // 5. Verify Fas F7.C1 (`forward_tokens_batch_hidden` causal equivalence + rollback across layers 0..=2)
            let batch_toks = [FLASH1017_BOS_TOKEN_ID, 100u32, 200u32];
            let mut kv_seq = Flash1017KvCache::new();
            let mut h_seq_list = Vec::new();
            for &tk in &batch_toks {
                let (hs, _) = rt.forward_token_hidden(tk, &mut kv_seq, 3).expect("Seq forward 3 layers");
                h_seq_list.push(hs);
            }
            let mut kv_batch = Flash1017KvCache::new();
            let (states_batch, _) = rt
                .forward_tokens_batch_hidden(&batch_toks, &mut kv_batch, 3, &[])
                .expect("Batch forward 3 tokens x 3 layers");
            assert_eq!(states_batch.len(), 3);
            for t in 0..3 {
                let max_diff = h_seq_list[t]
                    .iter()
                    .zip(states_batch[t].final_normed.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_diff < 1e-4,
                    "Token {} batch vs seq max_diff={} must be < 1e-4",
                    t,
                    max_diff
                );
            }
            kv_batch.restore_position_state(&states_batch[1], 0);
            assert_eq!(kv_batch.seq_len, 2);
        }
    }
}
