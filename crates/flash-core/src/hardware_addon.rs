//! # `dev-xray::hardware_addon` — Drop-in Hardware Accelerator Add-on Loader (`MPS-HW-1.0`)
//!
//! Allows the pure-Rust base engine (`Merkaba01`) to dynamically load target-specific
//! hardware acceleration add-ons at runtime (`plugins/hardware/addon_*.dylib|.so|.dll`):
//! - 🍏 `addon_mac_silicon.dylib` (Apple M1/M2/M3/M4 ARM64 NEON + `Accelerate.framework` AMX/vDSP)
//! - 🟢 `addon_gx10_cuda.so` (NVIDIA Grace Blackwell GB10 CUDA/PTX + `NVFP4`)
//! - 🔵 `addon_intel.so` / `.dll` (Intel `AVX2` + `AVX-512 VNNI` + `Intel AMX` + Arc Vulkan)
//! - 🔴 `addon_amd.so` / `.dll` (AMD Ryzen `AVX-512 VNNI` + Radeon/Strix Halo Vulkan/ROCm)
//!
//! Uses a lock-free `AtomicPtr<HardwareAddonVTable>` so hot-path inner loops (`dot_i8_f32`,
//! `dot_q4_g32_f32`, `dot_f32`) incur zero mutex overhead (`1` relaxed pointer load).

use std::ffi::{c_char, c_int, c_void, CString};
use std::path::PathBuf;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

pub const MERKABA_HW_ABI_V1: u32 = 0x4D48_5701;
pub const MERKABA_HW_ABI_V2: u32 = 0x4D48_5702;
pub const MERKABA_HW_ABI_V3: u32 = 0x4D48_5703;

pub type MatvecQ4G32Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    packed_nibbles: *const u8,
    group_scales_bf16: *const u16,
    x: *const f32,
    out: *mut f32,
);

pub type MatvecQ8Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    weights_i8: *const i8,
    row_scales: *const f32,
    x: *const f32,
    out: *mut f32,
);

pub type MatvecBf16Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    weights_bf16: *const u16,
    x: *const f32,
    out: *mut f32,
);

pub type MatbatchQ4G32Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    seq_len: usize,
    packed_nibbles: *const u8,
    group_scales_bf16: *const u16,
    x_batch: *const f32,
    out_batch: *mut f32,
);

pub type MatbatchQ8Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    seq_len: usize,
    weights_i8: *const i8,
    row_scales: *const f32,
    x_batch: *const f32,
    out_batch: *mut f32,
);

/// `ABI V3`: Batched BF16 GEMM (`Y[seq_len, rows] = X[seq_len, cols] * W_bf16[rows, cols]^T`).
pub type MatbatchBf16Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    seq_len: usize,
    weights_bf16: *const u16,
    x_batch: *const f32,
    out_batch: *mut f32,
);

/// `ABI V3`: Flash-1017 Tier A `NVFP4` (`group_size = 16`, `U8` E2M1 nibbles + `F8_E4M3` group scales + `F32` `global_scale`).
pub type MatvecNvfp4Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    packed_fp4: *const u8,
    group_scales_fp8: *const u8,
    global_scale: f32,
    x: *const f32,
    out: *mut f32,
);

/// `ABI V3`: Batched Flash-1017 Tier A `NVFP4` GEMM.
pub type MatbatchNvfp4Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    seq_len: usize,
    packed_fp4: *const u8,
    group_scales_fp8: *const u8,
    global_scale: f32,
    x_batch: *const f32,
    out_batch: *mut f32,
);

/// `ABI V3`: Flash-1017 Tier B & `.merkaba` `1.58-bit Ternary` (`4` weights/byte: `0->-1, 1->0, 2->+1` + `F32` `row_scales`).
pub type MatvecTernary158Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    packed_ternary: *const u8,
    row_scales: *const f32,
    x: *const f32,
    out: *mut f32,
);

/// `ABI V3`: Batched Flash-1017 Tier B & `.merkaba` `1.58-bit Ternary` GEMM.
pub type MatbatchTernary158Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    seq_len: usize,
    packed_ternary: *const u8,
    row_scales: *const f32,
    x_batch: *const f32,
    out_batch: *mut f32,
);

/// `ABI V3`: Flash-1017 MLA Attention, Shared Experts & Engram `MXFP8` (`F8_E4M3` weights `[rows, cols]` + `F8_E8M0` `32x32` block scales).
pub type MatvecMxfp8Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    weights_fp8: *const u8,
    block_scales_e8m0: *const u8,
    x: *const f32,
    out: *mut f32,
);

/// `ABI V3`: Batched Flash-1017 `MXFP8` (`F8_E4M3` + `F8_E8M0` `32x32` block scales) GEMM.
pub type MatbatchMxfp8Fn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    seq_len: usize,
    weights_fp8: *const u8,
    block_scales_e8m0: *const u8,
    x_batch: *const f32,
    out_batch: *mut f32,
);

#[repr(C)]
pub struct HardwareAddonVTable {
    pub abi_version: u32,
    pub addon_id_ptr: *const u8,
    pub addon_id_len: usize,
    pub addon_desc_ptr: *const u8,
    pub addon_desc_len: usize,
    pub dot_f32: unsafe extern "C" fn(a: *const f32, b: *const f32, len: usize) -> f32,
    pub dot_i8_f32: unsafe extern "C" fn(w_row: *const i8, x: *const f32, len: usize) -> f32,
    pub dot_q4_g32_f32: unsafe extern "C" fn(
        row_nibbles: *const u8,
        row_scales_bf16: *const u16,
        x: *const f32,
        cols: usize,
    ) -> f32,
    pub dot_ternary_2bit_f32:
        unsafe extern "C" fn(row_packed: *const u8, x: *const f32, cols: usize) -> f32,
    // --- ABI V2 fields ---
    pub matvec_q4_g32: Option<MatvecQ4G32Fn>,
    pub matvec_q8: Option<MatvecQ8Fn>,
    pub matvec_bf16: Option<MatvecBf16Fn>,
    pub matbatch_q4_g32: Option<MatbatchQ4G32Fn>,
    pub matbatch_q8: Option<MatbatchQ8Fn>,
    // --- ABI V3 fields (Flash-1017 Tier 2 & Full GPU Batch Coverage) ---
    pub matbatch_bf16: Option<MatbatchBf16Fn>,
    pub matvec_nvfp4: Option<MatvecNvfp4Fn>,
    pub matbatch_nvfp4: Option<MatbatchNvfp4Fn>,
    pub matvec_ternary_158: Option<MatvecTernary158Fn>,
    pub matbatch_ternary_158: Option<MatbatchTernary158Fn>,
    pub matvec_mxfp8: Option<MatvecMxfp8Fn>,
    pub matbatch_mxfp8: Option<MatbatchMxfp8Fn>,
}

unsafe impl Sync for HardwareAddonVTable {}
unsafe impl Send for HardwareAddonVTable {}

impl HardwareAddonVTable {
    #[inline]
    pub fn addon_id(&self) -> &str {
        unsafe {
            let slice = std::slice::from_raw_parts(self.addon_id_ptr, self.addon_id_len);
            std::str::from_utf8(slice).unwrap_or("unknown-addon")
        }
    }

    #[inline]
    pub fn addon_description(&self) -> &str {
        unsafe {
            let slice = std::slice::from_raw_parts(self.addon_desc_ptr, self.addon_desc_len);
            std::str::from_utf8(slice).unwrap_or("Unknown Hardware Add-on")
        }
    }

    #[inline(always)]
    pub fn v2_matvec_q4_g32(&self) -> Option<MatvecQ4G32Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V2 { self.matvec_q4_g32 } else { None }
    }

    #[inline(always)]
    pub fn v2_matvec_q8(&self) -> Option<MatvecQ8Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V2 { self.matvec_q8 } else { None }
    }

    #[inline(always)]
    pub fn v2_matvec_bf16(&self) -> Option<MatvecBf16Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V2 { self.matvec_bf16 } else { None }
    }

    #[inline(always)]
    pub fn v2_matbatch_q4_g32(&self) -> Option<MatbatchQ4G32Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V2 { self.matbatch_q4_g32 } else { None }
    }

    #[inline(always)]
    pub fn v2_matbatch_q8(&self) -> Option<MatbatchQ8Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V2 { self.matbatch_q8 } else { None }
    }

    #[inline(always)]
    pub fn v3_matbatch_bf16(&self) -> Option<MatbatchBf16Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V3 { self.matbatch_bf16 } else { None }
    }

    #[inline(always)]
    pub fn v3_matvec_nvfp4(&self) -> Option<MatvecNvfp4Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V3 { self.matvec_nvfp4 } else { None }
    }

    #[inline(always)]
    pub fn v3_matbatch_nvfp4(&self) -> Option<MatbatchNvfp4Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V3 { self.matbatch_nvfp4 } else { None }
    }

    #[inline(always)]
    pub fn v3_matvec_ternary_158(&self) -> Option<MatvecTernary158Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V3 { self.matvec_ternary_158 } else { None }
    }

    #[inline(always)]
    pub fn v3_matbatch_ternary_158(&self) -> Option<MatbatchTernary158Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V3 { self.matbatch_ternary_158 } else { None }
    }

    #[inline(always)]
    pub fn v3_matvec_mxfp8(&self) -> Option<MatvecMxfp8Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V3 { self.matvec_mxfp8 } else { None }
    }

    #[inline(always)]
    pub fn v3_matbatch_mxfp8(&self) -> Option<MatbatchMxfp8Fn> {
        if self.abi_version >= MERKABA_HW_ABI_V3 { self.matbatch_mxfp8 } else { None }
    }
}

pub type FlashAttnF32Fn = unsafe extern "C" fn(
    q_batch: *const f32,
    k_flat: *const f32,
    v_flat: *const f32,
    out_batch: *mut f32,
    new_len: usize,
    total_kv_tokens: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_stride: usize,
    start_pos: usize,
    base_pos: usize,
    win: usize,
    is_full: bool,
    scale: f32,
) -> bool;

pub type Mxfp8GroupedMatvecFn = unsafe extern "C" fn(
    total_rows: usize,
    cols_per_group: usize,
    num_groups: usize,
    weights_f8_e4m3: *const u8,
    block_scales_e8m0: *const u8,
    x: *const f32,
    out: *mut f32,
) -> bool;

pub type Mxfp8WoABFn = unsafe extern "C" fn(
    total_rows_a: usize,
    cols_per_group_a: usize,
    num_groups_a: usize,
    wo_a_w: *const u8,
    wo_a_s: *const u8,
    rows_b: usize,
    wo_b_w: *const u8,
    wo_b_s: *const u8,
    x: *const f32,
    out: *mut f32,
) -> bool;

pub type Mxfp4MatvecFn = unsafe extern "C" fn(
    rows: usize,
    cols: usize,
    packed_fp4: *const u8,
    scales_e8m0: *const u8,
    x: *const f32,
    out: *mut f32,
) -> bool;

pub type Mxfp8MlpAccumFn = unsafe extern "C" fn(
    inter_rows: usize,
    hidden_cols: usize,
    w1_w: *const u8,
    w1_s: *const u8,
    w3_w: *const u8,
    w3_s: *const u8,
    w2_w: *const u8,
    w2_s: *const u8,
    x: *const f32,
    out: *mut f32,
    accum_weight: f32,
    accumulate: bool,
    sync_now: bool,
) -> bool;

pub type Nvfp4MlpAccumFn = unsafe extern "C" fn(
    inter_rows: usize,
    hidden_cols: usize,
    w1_fp4: *const u8,
    w1_s: *const u8,
    w1_sc2: f32,
    w3_fp4: *const u8,
    w3_s: *const u8,
    w3_sc2: f32,
    w2_fp4: *const u8,
    w2_s: *const u8,
    w2_sc2: f32,
    x: *const f32,
    out: *mut f32,
    accum_weight: f32,
    accumulate: bool,
    sync_now: bool,
) -> bool;

pub type Ternary158MlpAccumFn = unsafe extern "C" fn(
    inter_rows: usize,
    hidden_cols: usize,
    w1_p: *const u8,
    w1_s: *const f32,
    w3_p: *const u8,
    w3_s: *const f32,
    w2_p: *const u8,
    w2_s: *const f32,
    proj_scale: f32,
    x: *const f32,
    out: *mut f32,
    accum_weight: f32,
    accumulate: bool,
    sync_now: bool,
) -> bool;

pub type Mxfp4MlpAccumFn = unsafe extern "C" fn(
    inter_rows: usize,
    hidden_cols: usize,
    w1_fp4: *const u8,
    w1_s: *const u8,
    w3_fp4: *const u8,
    w3_s: *const u8,
    w2_fp4: *const u8,
    w2_s: *const u8,
    x: *const f32,
    out: *mut f32,
    accum_weight: f32,
    accumulate: bool,
    sync_now: bool,
) -> bool;

pub type MlaFullForwardFn = unsafe extern "C" fn(
    wq_a_w: *const u8,
    wq_a_s: *const u8,
    wkv_w: *const u8,
    wkv_s: *const u8,
    q_norm: *const f32,
    wq_b_w: *const u8,
    wq_b_s: *const u8,
    kv_norm: *const f32,
    attn_sink: *const f32,
    sin_cos: *const f32,
    swa_prev_ptr: *const f32,
    n_prev_swa: usize,
    extra_ptr: *const f32,
    n_extra: usize,
    wo_a_w: *const u8,
    wo_a_s: *const u8,
    wo_b_w: *const u8,
    wo_b_s: *const u8,
    normed_x: *const f32,
    kv_t_out: *mut f32,
    attn_out: *mut f32,
) -> bool;

pub type MlaBatchForwardFn = unsafe extern "C" fn(
    b_len: usize,
    wq_a_w: *const u8,
    wq_a_s: *const u8,
    wkv_w: *const u8,
    wkv_s: *const u8,
    q_norm: *const f32,
    wq_b_w: *const u8,
    wq_b_s: *const u8,
    kv_norm: *const f32,
    attn_sink: *const f32,
    sin_cos_batch: *const f32,
    swa_prev_ptr: *const f32,
    n_base: usize,
    prev_total: usize,
    extra_ptr: *const f32,
    n_extra_per_tok: *const usize,
    wo_a_w: *const u8,
    wo_a_s: *const u8,
    wo_b_w: *const u8,
    wo_b_s: *const u8,
    normed_x_batch: *const f32,
    kv_t_batch_out: *mut f32,
    attn_out_batch: *mut f32,
) -> bool;

pub type Mxfp8MlpBatchFn = unsafe extern "C" fn(
    inter_rows: usize,
    hidden_cols: usize,
    seq_len: usize,
    w1_w: *const u8,
    w1_s: *const u8,
    w3_w: *const u8,
    w3_s: *const u8,
    w2_w: *const u8,
    w2_s: *const u8,
    x_all: *const f32,
    x_total_tokens: usize,
    out_batch: *mut f32,
    sync_now: bool,
) -> bool;

pub type Nvfp4MlpBatchFn = unsafe extern "C" fn(
    inter_rows: usize,
    hidden_cols: usize,
    seq_len: usize,
    w1_fp4: *const u8,
    w1_s: *const u8,
    w1_sc2: f32,
    w3_fp4: *const u8,
    w3_s: *const u8,
    w3_sc2: f32,
    w2_fp4: *const u8,
    w2_s: *const u8,
    w2_sc2: f32,
    x_tok_offset: usize,
    dst_tok_indices: *const usize,
    dst_weights: *const f32,
    b_len: usize,
    out_batch: *mut f32,
    sync_now: bool,
) -> bool;

pub type Ternary158MlpBatchFn = unsafe extern "C" fn(
    inter_rows: usize,
    hidden_cols: usize,
    seq_len: usize,
    w1_p: *const u8,
    w1_s: *const f32,
    w3_p: *const u8,
    w3_s: *const f32,
    w2_p: *const u8,
    w2_s: *const f32,
    proj_scale: f32,
    x_tok_offset: usize,
    dst_tok_indices: *const usize,
    dst_weights: *const f32,
    b_len: usize,
    out_batch: *mut f32,
    sync_now: bool,
) -> bool;

pub type Q4G32MlpFusedFn = unsafe extern "C" fn(
    inter_rows: usize,
    hidden_cols: usize,
    w1_nib: *const u8,
    w1_sc: *const u16,
    w3_nib: *const u8,
    w3_sc: *const u16,
    w2_nib: *const u8,
    w2_sc: *const u16,
    x: *const f32,
    out: *mut f32,
) -> bool;

pub type RegisterShardBoundsFn = unsafe extern "C" fn(ptr: *const c_void, bytes: usize);
pub type RegisterHostRangeFn = unsafe extern "C" fn(ptr: *const c_void, bytes: usize);

pub type MatvecF32Fn = unsafe extern "C" fn(
    weights: *const f32,
    x: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
) -> bool;

pub type MatbatchF32Fn = unsafe extern "C" fn(
    weights: *const f32,
    x_batch: *const f32,
    out_batch: *mut f32,
    rows: usize,
    cols: usize,
    b_len: usize,
) -> bool;

pub type UnlockGb10LayerFn = unsafe extern "C" fn(
    layer_idx: usize,
    attn_norm: *mut f32,
    attn_len: usize,
    ffn_norm: *mut f32,
    ffn_len: usize,
    q_norm: *mut f32,
    q_len: usize,
    kv_norm: *mut f32,
    kv_len: usize,
    gate_bias: *mut f32,
    gate_len: usize,
) -> bool;

static ACTIVE_VTABLE: AtomicPtr<HardwareAddonVTable> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_FLASH_ATTN: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_MXFP8_GROUPED_MV: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_MXFP8_WO_A_B: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_MXFP4_MV: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_MXFP8_MLP: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_NVFP4_MLP: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_TERN158_MLP: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_MXFP4_MLP: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_MLA_FULL_FWD: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_MLA_BATCH_FWD: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_MXFP8_MLP_BATCH: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_NVFP4_MLP_BATCH: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_TERN158_MLP_BATCH: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_Q4_G32_MLP: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_MATVEC_F32: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_MATBATCH_F32: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_REG_SHARD_BOUNDS: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_REG_HOST_RANGE: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_UNLOCK_GB10_LAYER: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static REGISTERED_MMAPS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
static REGISTERED_HOST_RANGES: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
static LOADED_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(unix)]
const RTLD_NOW: c_int = 2;

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const c_char) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *mut c_void;
}

/// Fast lock-free check used by `dot_i8_f32_unrolled8`, `dot_f32_unrolled8`, and `PackedQ4GroupMatrix`.
#[inline(always)]
pub fn active_vtable() -> Option<&'static HardwareAddonVTable> {
    let ptr = ACTIVE_VTABLE.load(Ordering::Relaxed);
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { &*ptr })
    }
}

/// Fast lock-free check for optional GPU Online-Softmax FlashAttention (`F2.5`).
#[inline(always)]
pub fn active_flash_attn_f32() -> Option<FlashAttnF32Fn> {
    let ptr = ACTIVE_FLASH_ATTN.load(Ordering::Relaxed);
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { std::mem::transmute(ptr) })
    }
}

#[inline(always)]
pub fn active_mxfp8_grouped_matvec() -> Option<Mxfp8GroupedMatvecFn> {
    let ptr = ACTIVE_MXFP8_GROUPED_MV.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_mxfp8_wo_a_b_fused() -> Option<Mxfp8WoABFn> {
    let ptr = ACTIVE_MXFP8_WO_A_B.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_mxfp4_matvec() -> Option<Mxfp4MatvecFn> {
    let ptr = ACTIVE_MXFP4_MV.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_mxfp8_mlp_accum() -> Option<Mxfp8MlpAccumFn> {
    let ptr = ACTIVE_MXFP8_MLP.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_nvfp4_mlp_accum() -> Option<Nvfp4MlpAccumFn> {
    let ptr = ACTIVE_NVFP4_MLP.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_ternary_158_mlp_accum() -> Option<Ternary158MlpAccumFn> {
    let ptr = ACTIVE_TERN158_MLP.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_mxfp4_mlp_accum() -> Option<Mxfp4MlpAccumFn> {
    let ptr = ACTIVE_MXFP4_MLP.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_mla_full_forward() -> Option<MlaFullForwardFn> {
    let ptr = ACTIVE_MLA_FULL_FWD.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_mla_batch_forward() -> Option<MlaBatchForwardFn> {
    let ptr = ACTIVE_MLA_BATCH_FWD.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_mxfp8_mlp_batch() -> Option<Mxfp8MlpBatchFn> {
    let ptr = ACTIVE_MXFP8_MLP_BATCH.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_nvfp4_mlp_batch() -> Option<Nvfp4MlpBatchFn> {
    let ptr = ACTIVE_NVFP4_MLP_BATCH.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_ternary_158_mlp_batch() -> Option<Ternary158MlpBatchFn> {
    let ptr = ACTIVE_TERN158_MLP_BATCH.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_q4_g32_mlp_fused() -> Option<Q4G32MlpFusedFn> {
    let ptr = ACTIVE_Q4_G32_MLP.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_matvec_f32() -> Option<MatvecF32Fn> {
    let ptr = ACTIVE_MATVEC_F32.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

#[inline(always)]
pub fn active_matbatch_f32() -> Option<MatbatchF32Fn> {
    let ptr = ACTIVE_MATBATCH_F32.load(Ordering::Relaxed);
    if ptr.is_null() { None } else { Some(unsafe { std::mem::transmute(ptr) }) }
}

/// Pre-registers a host memory range `[ptr, ptr + bytes]` with the CUDA Unified Memory DMA engine.
pub fn register_host_range(ptr: *const u8, bytes: usize) {
    if ptr.is_null() || bytes == 0 {
        return;
    }
    let addr = ptr as usize;
    if let Ok(mut guard) = REGISTERED_HOST_RANGES.lock() {
        if !guard.iter().any(|&(a, b)| a == addr && b == bytes) {
            guard.push((addr, bytes));
        }
    }
    let fn_ptr = ACTIVE_REG_HOST_RANGE.load(Ordering::Relaxed);
    if !fn_ptr.is_null() {
        let reg_fn: RegisterHostRangeFn = unsafe { std::mem::transmute(fn_ptr) };
        unsafe { reg_fn(ptr as *const c_void, bytes) };
    }
}

/// Registers the `[ptr, ptr + bytes]` VMA bounds of an `mmap`-backed model shard with the active hardware add-on
/// so large `512 MB` contiguous chunks can be pinned safely without crossing VMA boundaries.
pub fn register_host_mmap_bounds(ptr: *const u8, bytes: usize) {
    if ptr.is_null() || bytes == 0 {
        return;
    }
    let addr = ptr as usize;
    if let Ok(mut guard) = REGISTERED_MMAPS.lock() {
        if !guard.iter().any(|&(a, b)| a == addr && b == bytes) {
            guard.push((addr, bytes));
        }
    }
    let fn_ptr = ACTIVE_REG_SHARD_BOUNDS.load(Ordering::Relaxed);
    if !fn_ptr.is_null() {
        let reg_fn: RegisterShardBoundsFn = unsafe { std::mem::transmute(fn_ptr) };
        unsafe { reg_fn(ptr as *const c_void, bytes) };
    }
}

/// Hardware-bound decryption of GB10-locked (`MERKABA_GB10_LOCK_V1`) layer normalization & router bias vectors.
/// Returns `true` only when `addon_gx10_rust` is loaded on an authentic NVIDIA Grace Blackwell GB10 (`Linux aarch64`, `sm_121`).
pub fn unlock_gb10_layer(
    layer_idx: usize,
    attn_norm: &mut [f32],
    ffn_norm: &mut [f32],
    q_norm: &mut [f32],
    kv_norm: &mut [f32],
    gate_bias: &mut [f32],
) -> bool {
    let fn_ptr = ACTIVE_UNLOCK_GB10_LAYER.load(Ordering::Relaxed);
    if fn_ptr.is_null() {
        return false;
    }
    let unlock_fn: UnlockGb10LayerFn = unsafe { std::mem::transmute(fn_ptr) };
    unsafe {
        unlock_fn(
            layer_idx,
            attn_norm.as_mut_ptr(),
            attn_norm.len(),
            ffn_norm.as_mut_ptr(),
            ffn_norm.len(),
            q_norm.as_mut_ptr(),
            q_norm.len(),
            kv_norm.as_mut_ptr(),
            kv_norm.len(),
            gate_bias.as_mut_ptr(),
            gate_bias.len(),
        )
    }
}

/// Resolves short aliases (`mac`, `intel`, `amd`, `gx10`) or file paths to the `.dylib`/`.so`/`.dll` file.
pub fn resolve_addon_path(alias_or_path: &str) -> Option<PathBuf> {
    let trimmed = alias_or_path.trim();
    let direct = PathBuf::from(trimmed);
    if direct.exists() && direct.is_file() {
        return Some(direct);
    }

    let filenames: Vec<&str> = match trimmed.to_ascii_lowercase().as_str() {
        "mac" | "apple" | "m4" | "m1" | "mac-silicon" | "addon-mac-silicon" => vec![
            "addon_mac_silicon.dylib",
            "libaddon_mac_silicon.dylib",
        ],
        "gx10-rust" | "gb10-rust" | "rust-gpu" | "addon-gx10-rust" => vec![
            "addon_gx10_rust.so",
            "addon_gx10_rust.dylib",
        ],
        "gx10-cuda" | "gb10-cuda" | "cuda" | "addon-gx10-blackwell" => vec![
            "addon_gx10_blackwell.so",
            "addon_gx10_blackwell.dylib",
        ],
        "gb10-cpu" | "gx10-cpu" | "grace" | "addon-gx10-grace-cpu" => vec![
            "addon_gx10_grace_cpu.so",
            "addon_gx10_grace_cpu.dylib",
        ],
        "gb10" | "gx10" | "nvidia" | "addon-gx10-nvidia" => vec![
            "addon_gx10_rust.so",
            "addon_gx10_rust.dylib",
            "addon_gx10_blackwell.so",
            "addon_gx10_grace_cpu.so",
            "addon_gx10_grace_cpu.dylib",
            "addon_gx10_cuda.so",
            "addon_gx10_cuda.dylib",
        ],
        "intel" | "b70" | "arc" | "i7-1360p" | "addon-intel" | "addon-intel-arc-b70" | "addon-intel-i7-1360p" => vec![
            "addon_intel.so",
            "addon_intel.dylib",
            "addon_intel.dll",
        ],
        "rtx" | "4060" | "4060ti" | "rtx4060ti" | "addon-rtx-4060ti-16gb" => vec![
            "addon_rtx_30_40.dll",
            "addon_rtx_30_40.so",
            "addon_rtx_30_40.dylib",
        ],
        "amd" | "rocm" | "addon-amd" => vec![
            "addon_amd.so",
            "addon_amd.dylib",
            "addon_amd.dll",
        ],
        _ => vec![trimmed],
    };

    let search_dirs = [
        PathBuf::from("/Volumes/NET/ai-projects/plugins/hardware"),
        PathBuf::from("plugins/hardware"),
        PathBuf::from("../plugins/hardware"),
        PathBuf::from("../../plugins/hardware"),
    ];

    for dir in &search_dirs {
        for fname in &filenames {
            let candidate = dir.join(fname);
            if candidate.exists() && candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Dynamically loads a hardware add-on (`.dylib`/`.so`) and activates its `HardwareAddonVTable`.
pub fn load_hardware_addon(alias_or_path: &str) -> Result<(String, String, PathBuf), String> {
    let path = resolve_addon_path(alias_or_path).ok_or_else(|| {
        format!(
            "Kunde inte hitta hårdvaru-tillägget '{}' i plugins/hardware/.",
            alias_or_path
        )
    })?;

    #[cfg(unix)]
    unsafe {
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| format!("Ogiltig sökväg: {e}"))?;
        let handle = dlopen(c_path.as_ptr(), RTLD_NOW);
        if handle.is_null() {
            return Err(format!(
                "dlopen misslyckades för '{}'",
                path.display()
            ));
        }
        let sym_name = CString::new("merkaba_hardware_addon_v1").unwrap();
        let sym = dlsym(handle, sym_name.as_ptr());
        if sym.is_null() {
            return Err(format!(
                "Symbolen 'merkaba_hardware_addon_v1' saknas i '{}'",
                path.display()
            ));
        }
        let entry_fn: extern "C" fn() -> *const HardwareAddonVTable =
            std::mem::transmute(sym);
        let vtable_ptr = entry_fn();
        if vtable_ptr.is_null() {
            return Err("merkaba_hardware_addon_v1 returnerade null".to_string());
        }
        let vtable = &*vtable_ptr;
        if !matches!(
            vtable.abi_version,
            MERKABA_HW_ABI_V1 | MERKABA_HW_ABI_V2 | MERKABA_HW_ABI_V3
        ) {
            return Err(format!(
                "ABI-version matchar inte (fick 0x{:08X}, förväntade V1..V3 0x{:08X}..0x{:08X})",
                vtable.abi_version, MERKABA_HW_ABI_V1, MERKABA_HW_ABI_V3
            ));
        }

        let fa_sym_name = CString::new("gx10_rust_flash_attn_f32").unwrap();
        let fa_sym = dlsym(handle, fa_sym_name.as_ptr());
        ACTIVE_FLASH_ATTN.store(fa_sym, Ordering::Release);

        let resolve_opt = |name: &str, slot: &AtomicPtr<c_void>| {
            let cs = CString::new(name).unwrap();
            let p = dlsym(handle, cs.as_ptr());
            slot.store(p, Ordering::Release);
        };
        resolve_opt("gx10_rust_mxfp8_grouped_matvec", &ACTIVE_MXFP8_GROUPED_MV);
        resolve_opt("gx10_rust_mxfp8_wo_a_b_fused", &ACTIVE_MXFP8_WO_A_B);
        resolve_opt("gx10_rust_mxfp4_matvec", &ACTIVE_MXFP4_MV);
        resolve_opt("gx10_rust_mxfp8_mlp_accum", &ACTIVE_MXFP8_MLP);
        resolve_opt("gx10_rust_nvfp4_mlp_accum", &ACTIVE_NVFP4_MLP);
        resolve_opt("gx10_rust_ternary_158_mlp_accum", &ACTIVE_TERN158_MLP);
        resolve_opt("gx10_rust_mxfp4_mlp_accum", &ACTIVE_MXFP4_MLP);
        resolve_opt("gx10_rust_mla_full_forward", &ACTIVE_MLA_FULL_FWD);
        resolve_opt("gx10_rust_mla_batch_forward", &ACTIVE_MLA_BATCH_FWD);
        resolve_opt("gx10_rust_mxfp8_mlp_batch", &ACTIVE_MXFP8_MLP_BATCH);
        resolve_opt("gx10_rust_nvfp4_mlp_batch", &ACTIVE_NVFP4_MLP_BATCH);
        resolve_opt("gx10_rust_ternary_158_mlp_batch", &ACTIVE_TERN158_MLP_BATCH);
        resolve_opt("gx10_rust_q4_g32_mlp_fused", &ACTIVE_Q4_G32_MLP);
        resolve_opt("gx10_rust_matvec_f32", &ACTIVE_MATVEC_F32);
        resolve_opt("gx10_rust_matbatch_f32", &ACTIVE_MATBATCH_F32);
        resolve_opt("gx10_rust_register_shard_bounds", &ACTIVE_REG_SHARD_BOUNDS);
        resolve_opt("gx10_rust_register_host_range", &ACTIVE_REG_HOST_RANGE);
        resolve_opt("gx10_rust_unlock_gb10_layer", &ACTIVE_UNLOCK_GB10_LAYER);
        let reg_ptr = ACTIVE_REG_SHARD_BOUNDS.load(Ordering::Relaxed);
        if !reg_ptr.is_null() {
            let reg_fn: RegisterShardBoundsFn = std::mem::transmute(reg_ptr);
            if let Ok(guard) = REGISTERED_MMAPS.lock() {
                for &(addr, bytes) in guard.iter() {
                    reg_fn(addr as *const c_void, bytes);
                }
            }
        }
        let reg_hr_ptr = ACTIVE_REG_HOST_RANGE.load(Ordering::Relaxed);
        if !reg_hr_ptr.is_null() {
            let reg_hr_fn: RegisterHostRangeFn = std::mem::transmute(reg_hr_ptr);
            if let Ok(guard) = REGISTERED_HOST_RANGES.lock() {
                for &(addr, bytes) in guard.iter() {
                    reg_hr_fn(addr as *const c_void, bytes);
                }
            }
        }

        ACTIVE_VTABLE.store(vtable_ptr as *mut HardwareAddonVTable, Ordering::Release);
        if let Ok(mut guard) = LOADED_PATH.lock() {
            *guard = Some(path.clone());
        }
        Ok((
            vtable.addon_id().to_string(),
            vtable.addon_description().to_string(),
            path,
        ))
    }

    #[cfg(windows)]
    unsafe {
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| format!("Ogiltig sökväg: {e}"))?;
        let handle = LoadLibraryA(c_path.as_ptr());
        if handle.is_null() {
            return Err(format!("LoadLibraryA misslyckades för '{}'", path.display()));
        }
        let sym_name = CString::new("merkaba_hardware_addon_v1").unwrap();
        let sym = GetProcAddress(handle, sym_name.as_ptr());
        if sym.is_null() {
            return Err(format!(
                "Symbolen 'merkaba_hardware_addon_v1' saknas i '{}'",
                path.display()
            ));
        }
        let entry_fn: extern "C" fn() -> *const HardwareAddonVTable = std::mem::transmute(sym);
        let vtable_ptr = entry_fn();
        if vtable_ptr.is_null() {
            return Err("merkaba_hardware_addon_v1 returnerade null".to_string());
        }
        let vtable = &*vtable_ptr;
        if !matches!(
            vtable.abi_version,
            MERKABA_HW_ABI_V1 | MERKABA_HW_ABI_V2 | MERKABA_HW_ABI_V3
        ) {
            return Err(format!(
                "ABI-version matchar inte (fick 0x{:08X}, förväntade V1..V3 0x{:08X}..0x{:08X})",
                vtable.abi_version, MERKABA_HW_ABI_V1, MERKABA_HW_ABI_V3
            ));
        }
        ACTIVE_FLASH_ATTN.store(std::ptr::null_mut(), Ordering::Release);
        ACTIVE_VTABLE.store(vtable_ptr as *mut HardwareAddonVTable, Ordering::Release);
        if let Ok(mut guard) = LOADED_PATH.lock() {
            *guard = Some(path.clone());
        }
        Ok((
            vtable.addon_id().to_string(),
            vtable.addon_description().to_string(),
            path,
        ))
    }

    #[cfg(all(not(unix), not(windows)))]
    {
        let _ = path;
        Err("Dynamisk laddning stöds inte på denna plattform.".to_string())
    }
}

/// Unloads/deactivates the active hardware add-on, immediately switching back to Pure Rust Universal CPU fallback.
pub fn unload_hardware_addon() -> Option<String> {
    let prev = active_vtable().map(|v| v.addon_id().to_string());
    ACTIVE_FLASH_ATTN.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_MXFP8_GROUPED_MV.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_MXFP8_WO_A_B.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_MXFP4_MV.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_MXFP8_MLP.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_NVFP4_MLP.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_TERN158_MLP.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_MXFP4_MLP.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_MLA_FULL_FWD.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_MLA_BATCH_FWD.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_MXFP8_MLP_BATCH.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_NVFP4_MLP_BATCH.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_TERN158_MLP_BATCH.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_Q4_G32_MLP.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_MATVEC_F32.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_MATBATCH_F32.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_REG_SHARD_BOUNDS.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_REG_HOST_RANGE.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_UNLOCK_GB10_LAYER.store(std::ptr::null_mut(), Ordering::Release);
    ACTIVE_VTABLE.store(std::ptr::null_mut(), Ordering::Release);
    if let Ok(mut guard) = LOADED_PATH.lock() {
        *guard = None;
    }
    prev
}

/// Returns `(is_loaded, addon_id, description, path, available_files)` for `/hardware status`.
pub fn hardware_addon_status() -> (bool, String, String, Option<PathBuf>, Vec<PathBuf>) {
    let mut available = Vec::new();
    let search_dirs = [
        PathBuf::from("/Volumes/NET/ai-projects/plugins/hardware"),
        PathBuf::from("plugins/hardware"),
        PathBuf::from("../plugins/hardware"),
    ];
    for dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                    if matches!(ext, "dylib" | "so" | "dll")
                        && !available.iter().any(|existing: &PathBuf| {
                            existing.file_name() == p.file_name()
                        })
                    {
                        available.push(p);
                    }
                }
            }
        }
    }

    let loaded_path = LOADED_PATH.lock().ok().and_then(|g| g.clone());
    if let Some(vt) = active_vtable() {
        (
            true,
            vt.addon_id().to_string(),
            vt.addon_description().to_string(),
            loaded_path,
            available,
        )
    } else {
        (
            false,
            "pure-rust-universal-cpu".to_string(),
            "100 % Ren Standard-Rust CPU (Reserv / Fallback utan hårdvarutillägg)".to_string(),
            None,
            available,
        )
    }
}

/// Result of running `/hardware bench` comparing Pure Rust CPU fallback vs the Hardware Add-on.
#[derive(Debug, Clone)]
pub struct HardwareBenchmarkReport {
    pub addon_id: String,
    pub addon_desc: String,
    pub rows: usize,
    pub cols: usize,
    pub f32_rust_us: f64,
    pub f32_addon_us: f64,
    pub q8_rust_us: f64,
    pub q8_addon_us: f64,
    pub q4_rust_us: f64,
    pub q4_addon_us: f64,
    pub max_abs_diff: f32,
}

/// Runs a live micro-benchmark (`1024 x 2048` matrix-vector pass) comparing Pure Rust Universal CPU
/// against the loaded Hardware Add-on (`addon_mac_silicon.dylib`, `addon_gx10_rust.so`, `addon_gx10_blackwell.so`, etc.).
pub fn benchmark_hardware_addon(path_or_alias: Option<&str>) -> Result<HardwareBenchmarkReport, String> {
    if let Some(target) = path_or_alias {
        load_hardware_addon(target)?;
    } else if active_vtable().is_none() {
        load_hardware_addon("mac")?;
    }
    let vt = active_vtable().ok_or("Inget hårdvaru-tillägg är aktivt.")?;

    let rows = 1024usize;
    let cols = 2048usize;
    let groups_per_row = cols / 32;
    let nib_bytes_per_row = cols / 2;

    let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    let f32_row: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.007).cos() * 0.25).collect();
    let i8_mat: Vec<i8> = (0..rows * cols)
        .map(|i| (((i * 7 + 3) as i32 % 127) - 63) as i8)
        .collect();
    let row_scales: Vec<f32> = vec![0.015625_f32; rows];
    let q4_mat_nibbles: Vec<u8> = (0..rows * nib_bytes_per_row)
        .map(|i| (((i * 5 + 1) & 0x0F) | ((((i * 3) + 3) & 0x0F) << 4)) as u8)
        .collect();
    let q4_mat_scales: Vec<u16> = vec![half::bf16::from_f32(0.03125).to_bits(); rows * groups_per_row];
    let mut out_buf: Vec<f32> = vec![0.0_f32; rows];

    // 1. FP32 row dot comparison
    let t0 = std::time::Instant::now();
    let mut sum_f32_rust = 0.0_f32;
    for _ in 0..rows {
        let mut s = 0.0_f32;
        for c in 0..cols {
            s += unsafe { *f32_row.get_unchecked(c) * *x.get_unchecked(c) };
        }
        sum_f32_rust += std::hint::black_box(s);
    }
    let f32_rust_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

    let t1 = std::time::Instant::now();
    let mut sum_f32_addon = 0.0_f32;
    for _ in 0..rows {
        let s = unsafe { (vt.dot_f32)(f32_row.as_ptr(), x.as_ptr(), cols) };
        sum_f32_addon += std::hint::black_box(s);
    }
    let f32_addon_us = t1.elapsed().as_secs_f64() * 1_000_000.0;

    // 2. Q8 matvec comparison (uses full-matrix GPU/SDOT kernel when ABI V2 is available)
    let t2 = std::time::Instant::now();
    let mut sum_q8_rust = 0.0_f32;
    for r in 0..rows {
        let row_off = r * cols;
        let mut s = 0.0_f32;
        for c in 0..cols {
            s += (i8_mat[row_off + c] as f32) * x[c];
        }
        sum_q8_rust += std::hint::black_box(s * row_scales[r]);
    }
    let q8_rust_us = t2.elapsed().as_secs_f64() * 1_000_000.0;

    let mut sum_q8_addon = 0.0_f32;
    let q8_addon_us = if vt.abi_version >= MERKABA_HW_ABI_V2 && vt.matvec_q8.is_some() {
        let mv_q8 = vt.matvec_q8.unwrap();
        unsafe {
            mv_q8(
                rows,
                cols,
                i8_mat.as_ptr(),
                row_scales.as_ptr(),
                x.as_ptr(),
                out_buf.as_mut_ptr(),
            );
        }
        let t3 = std::time::Instant::now();
        unsafe {
            mv_q8(
                rows,
                cols,
                i8_mat.as_ptr(),
                row_scales.as_ptr(),
                x.as_ptr(),
                out_buf.as_mut_ptr(),
            );
        }
        let elapsed = t3.elapsed().as_secs_f64() * 1_000_000.0;
        for &v in &out_buf {
            sum_q8_addon += v;
        }
        elapsed
    } else {
        let t3 = std::time::Instant::now();
        for r in 0..rows {
            let s = unsafe { (vt.dot_i8_f32)(i8_mat.as_ptr().add(r * cols), x.as_ptr(), cols) };
            sum_q8_addon += std::hint::black_box(s * row_scales[r]);
        }
        t3.elapsed().as_secs_f64() * 1_000_000.0
    };

    // 3. Q4_G32 matvec comparison (uses full-matrix GPU/SDOT kernel when ABI V2 is available)
    let t4 = std::time::Instant::now();
    let mut sum_q4_rust = 0.0_f32;
    for r in 0..rows {
        let nib_off = r * nib_bytes_per_row;
        let sc_off = r * groups_per_row;
        let mut total = 0.0_f32;
        for g in 0..groups_per_row {
            let sc = half::bf16::from_bits(q4_mat_scales[sc_off + g]).to_f32();
            let mut gs = 0.0_f32;
            for k in 0..16 {
                let b = q4_mat_nibbles[nib_off + g * 16 + k];
                let q0 = (b & 0x0F) as i32 - 8;
                let q1 = (b >> 4) as i32 - 8;
                gs += (q0 as f32) * x[g * 32 + k * 2] + (q1 as f32) * x[g * 32 + k * 2 + 1];
            }
            total += gs * sc;
        }
        sum_q4_rust += std::hint::black_box(total);
    }
    let q4_rust_us = t4.elapsed().as_secs_f64() * 1_000_000.0;

    let mut sum_q4_addon = 0.0_f32;
    let q4_addon_us = if vt.abi_version >= MERKABA_HW_ABI_V2 && vt.matvec_q4_g32.is_some() {
        let mv_q4 = vt.matvec_q4_g32.unwrap();
        unsafe {
            mv_q4(
                rows,
                cols,
                q4_mat_nibbles.as_ptr(),
                q4_mat_scales.as_ptr(),
                x.as_ptr(),
                out_buf.as_mut_ptr(),
            );
        }
        let t5 = std::time::Instant::now();
        unsafe {
            mv_q4(
                rows,
                cols,
                q4_mat_nibbles.as_ptr(),
                q4_mat_scales.as_ptr(),
                x.as_ptr(),
                out_buf.as_mut_ptr(),
            );
        }
        let elapsed = t5.elapsed().as_secs_f64() * 1_000_000.0;
        for &v in &out_buf {
            sum_q4_addon += v;
        }
        elapsed
    } else {
        let t5 = std::time::Instant::now();
        for r in 0..rows {
            let s = unsafe {
                (vt.dot_q4_g32_f32)(
                    q4_mat_nibbles.as_ptr().add(r * nib_bytes_per_row),
                    q4_mat_scales.as_ptr().add(r * groups_per_row),
                    x.as_ptr(),
                    cols,
                )
            };
            sum_q4_addon += std::hint::black_box(s);
        }
        t5.elapsed().as_secs_f64() * 1_000_000.0
    };

    let diff_f32 = ((sum_f32_rust - sum_f32_addon) / (rows as f32)).abs();
    let diff_q8 = ((sum_q8_rust - sum_q8_addon) / (rows as f32)).abs();
    let diff_q4 = ((sum_q4_rust - sum_q4_addon) / (rows as f32)).abs();
    let max_abs_diff = diff_f32.max(diff_q8).max(diff_q4);

    Ok(HardwareBenchmarkReport {
        addon_id: vt.addon_id().to_string(),
        addon_desc: vt.addon_description().to_string(),
        rows,
        cols,
        f32_rust_us,
        f32_addon_us,
        q8_rust_us,
        q8_addon_us,
        q4_rust_us,
        q4_addon_us,
        max_abs_diff,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_silicon_hardware_addon_live_load_bench_and_unload() {
        let target_addon = if cfg!(target_os = "macos") {
            "mac"
        } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            "gx10-rust"
        } else if cfg!(target_os = "windows") {
            "intel"
        } else {
            return;
        };

        let Ok((id, desc, path)) = load_hardware_addon(target_addon) else {
            println!("[Test Hardware Add-on] Skipping live load test (binary for '{}' not found in search path).", target_addon);
            return;
        };
        assert!(!id.is_empty());
        assert!(!desc.is_empty());
        assert!(path.exists());

        let rep = benchmark_hardware_addon(Some(target_addon)).expect("Benchmark should succeed");
        println!(
            "[Test Hardware Add-on ({})] f32: {:.1}us -> {:.1}us ({:.2}x) | Q8: {:.1}us -> {:.1}us ({:.2}x) | Q4_G32: {:.1}us -> {:.1}us ({:.2}x) | max_diff={:.6}",
            id,
            rep.f32_rust_us,
            rep.f32_addon_us,
            rep.f32_rust_us / rep.f32_addon_us.max(0.1),
            rep.q8_rust_us,
            rep.q8_addon_us,
            rep.q8_rust_us / rep.q8_addon_us.max(0.1),
            rep.q4_rust_us,
            rep.q4_addon_us,
            rep.q4_rust_us / rep.q4_addon_us.max(0.1),
            rep.max_abs_diff
        );
        assert!(
            rep.max_abs_diff < 5e-2,
            "Expected accurate hardware add-on output, got max_abs_diff={}",
            rep.max_abs_diff
        );

        let unloaded = unload_hardware_addon();
        assert_eq!(unloaded.as_deref(), Some(id.as_str()));
        assert!(active_vtable().is_none());
    }
}
