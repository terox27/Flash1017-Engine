use flash_core::{
    hardware_addon, Flash1017HeadMode, Flash1017Jepa106Compressor, Flash1017KvCache,
    Flash1017MtpMode, Flash1017Runtime, Flash1017VerifiedMtpState, FLASH1017_BOS_TOKEN_ID,
    FLASH1017_EOS_TOKEN_ID, FLASH1017_THINK_END_TOKEN_ID,
};
use serde_json::{json, Value};
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn default_model_dir() -> PathBuf {
    let candidates = [
        "/home/tauai/SPARK-GB10/models/DeepSeek-V4.1-Flash-1017-Tier2",
        "/home/tauai/SPARK-GB10/models/DeepSeek-V4.1-Flash-1017-NVFP4",
        "/Volumes/NET/ai-projects/models/DeepSeek-V4.1-Flash-1017-Tier2",
        "/Volumes/NET/ai-projects/models/DeepSeek-V4.1-Flash-1017-NVFP4",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from(".")
}

fn default_addon_path() -> String {
    let candidates = [
        "bin/addon_gx10_rust.so",
        "./addon_gx10_rust.so",
        "/home/tauai/SPARK-GB10/plugins/hardware/addon_gx10_rust.so",
        "plugins/hardware/addon_gx10_rust.so",
    ];
    for c in candidates {
        if PathBuf::from(c).exists() {
            return c.to_string();
        }
    }
    "gx10-rust".to_string()
}

fn parse_mtp_mode(s: &str) -> Flash1017MtpMode {
    match s.to_ascii_lowercase().as_str() {
        "0" | "off" | "none" | "false" => Flash1017MtpMode::Off,
        "1" | "k1" => Flash1017MtpMode::Fixed(1),
        "2" | "k2" => Flash1017MtpMode::Fixed(2),
        "3" | "k3" => Flash1017MtpMode::Fixed(3),
        "4" | "k4" => Flash1017MtpMode::Fixed(4),
        _ => Flash1017MtpMode::Adaptive,
    }
}

fn parse_head_mode(s: &str) -> Flash1017HeadMode {
    match s.to_ascii_lowercase().as_str() {
        "bf16" | "exact" => Flash1017HeadMode::Bf16,
        "q8" => Flash1017HeadMode::Q8,
        _ => Flash1017HeadMode::CoarseToFine,
    }
}

fn apply_memory_profile(rt: &mut Flash1017Runtime, profile: &str) -> String {
    match profile.to_ascii_lowercase().as_str() {
        "55gb" | "preload" => {
            let (cnt, bytes, ms) = rt.set_preload_all_experts(true);
            let (c_cnt, c_ok) = rt.set_tier_c_disk_mode(true);
            let msg = format!(
                "🚀 [Minnesprofil: +55 GB i RAM + Alla 384 experter (Tier A+B+C)] Värmde {} Tier A-experter ({:.2} GiB) i RAM på {:.1} ms + mappade {} Tier C-experter (aktiv={}).",
                cnt,
                bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                ms,
                c_cnt,
                c_ok
            );
            println!("{}", msg);
            "55gb".to_string()
        }
        "tierc" | "full" | "256" | "384" => {
            let (cnt, bytes, ms) = rt.set_preload_all_experts(true);
            let (c_cnt, c_ok) = rt.set_tier_c_disk_mode(true);
            let msg = format!(
                "💿 [Minnesprofil: +55 GB i RAM + Tier C från NVMe (Alla 384 experter)] Värmde {} Tier A ({:.2} GiB, {:.1} ms) + mappade {} Tier C-experter (aktiv={}).",
                cnt,
                bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                ms,
                c_cnt,
                c_ok
            );
            println!("{}", msg);
            "tierc".to_string()
        }
        "pruned" | "skip-tierc" => {
            let _ = rt.set_tier_c_disk_mode(false);
            let (_cnt, _bytes, ms) = rt.set_preload_all_experts(false);
            println!(
                "⚡ [Minnesprofil: 7 GB Pruned (Endast Tier A+B — Experimentell)] Klar på {:.1} ms (~7 GB aktivt RAM).",
                ms
            );
            "pruned".to_string()
        }
        _ => {
            let (_cnt, _bytes, ms) = rt.set_preload_all_experts(false);
            let (c_cnt, c_ok) = rt.set_tier_c_disk_mode(true);
            let msg = format!(
                "⚡ [Minnesprofil: 7 GB Turbo + Alla 384 experter (Tier A+B+C On-Demand)] Klar på {:.1} ms (~7 GB låst RAM + OS Page Cache, {} Tier C mappade, aktiv={}).",
                ms,
                c_cnt,
                c_ok
            );
            println!("{}", msg);
            "7gb".to_string()
        }
    }
}

#[derive(Clone, Debug)]
struct SamplingParams {
    pub max_context_tokens: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: u64,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
    pub enable_thinking: bool,
    pub logprobs: bool,
    pub top_logprobs: usize,
    pub images: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_context_tokens: 1_048_576,
            max_tokens: 8_192,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 42,
            stop: Vec::new(),
            ignore_eos: false,
            enable_thinking: false,
            logprobs: false,
            top_logprobs: 0,
            images: Vec::new(),
        }
    }
}

impl SamplingParams {
    pub fn is_pure_greedy(&self) -> bool {
        self.temperature <= 1e-5
            && (self.repeat_penalty - 1.0).abs() <= 1e-5
            && self.frequency_penalty.abs() <= 1e-5
            && self.presence_penalty.abs() <= 1e-5
            && !self.logprobs
            && self.top_logprobs == 0
    }

    pub fn from_json_with_defaults(&self, req: &Value) -> Self {
        let mut p = self.clone();
        if let Some(v) = req
            .get("max_context_tokens")
            .or_else(|| req.get("max_model_len"))
            .or_else(|| req.get("n_ctx"))
            .or_else(|| req.get("ctx_size"))
            .and_then(|v| v.as_u64())
        {
            p.max_context_tokens = (v as usize).clamp(128, 1_048_576);
        }
        if let Some(v) = req
            .get("max_tokens")
            .or_else(|| req.get("max_completion_tokens"))
            .or_else(|| req.get("max_output_tokens"))
            .or_else(|| req.get("n_predict"))
            .and_then(|v| v.as_u64())
        {
            p.max_tokens = v as usize;
        }
        if let Some(v) = req
            .get("temperature")
            .or_else(|| req.get("temp"))
            .and_then(|v| v.as_f64())
        {
            p.temperature = (v as f32).max(0.0);
        }
        if let Some(v) = req.get("top_p").and_then(|v| v.as_f64()) {
            p.top_p = (v as f32).clamp(0.0, 1.0);
        }
        if let Some(v) = req.get("top_k").and_then(|v| v.as_u64()) {
            p.top_k = v as usize;
        }
        if let Some(v) = req.get("min_p").and_then(|v| v.as_f64()) {
            p.min_p = (v as f32).clamp(0.0, 1.0);
        }
        if let Some(v) = req
            .get("repeat_penalty")
            .or_else(|| req.get("repetition_penalty"))
            .and_then(|v| v.as_f64())
        {
            p.repeat_penalty = (v as f32).max(0.1);
        }
        if let Some(v) = req.get("repeat_last_n").and_then(|v| v.as_u64()) {
            p.repeat_last_n = v as usize;
        }
        if let Some(v) = req.get("frequency_penalty").and_then(|v| v.as_f64()) {
            p.frequency_penalty = v as f32;
        }
        if let Some(v) = req.get("presence_penalty").and_then(|v| v.as_f64()) {
            p.presence_penalty = v as f32;
        }
        if let Some(v) = req.get("seed").and_then(|v| v.as_u64()) {
            p.seed = v;
        }
        if let Some(v) = req.get("ignore_eos").and_then(|v| v.as_bool()) {
            p.ignore_eos = v;
        }
        if let Some(v) = req
            .get("enable_thinking")
            .or_else(|| req.get("think"))
            .and_then(|v| v.as_bool())
        {
            p.enable_thinking = v;
        }
        if let Some(s) = req.get("stop") {
            if let Some(st_str) = s.as_str() {
                if !st_str.is_empty() {
                    p.stop.push(st_str.to_string());
                }
            } else if let Some(arr) = s.as_array() {
                for item in arr {
                    if let Some(st_str) = item.as_str() {
                        if !st_str.is_empty() {
                            p.stop.push(st_str.to_string());
                        }
                    }
                }
            }
        }
        if let Some(lp) = req.get("logprobs") {
            if let Some(b) = lp.as_bool() {
                p.logprobs = b;
            } else if let Some(n) = lp.as_u64() {
                p.logprobs = n > 0;
                p.top_logprobs = (n as usize).min(20);
            }
        }
        if let Some(tlp) = req.get("top_logprobs").and_then(|v| v.as_u64()) {
            p.top_logprobs = (tlp as usize).min(20);
            if p.top_logprobs > 0 {
                p.logprobs = true;
            }
        }
        if let Some(img) = req.get("image").and_then(|v| v.as_str()) {
            p.images.push(img.to_string());
        }
        if let Some(imgs) = req.get("images").and_then(|v| v.as_array()) {
            for item in imgs {
                if let Some(s) = item.as_str() {
                    p.images.push(s.to_string());
                }
            }
        }
        p
    }
}

struct GenerationOutput {
    text: String,
    finish_reason: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_ms: f64,
    decode_ms: f64,
    tok_per_s: f64,
    logprobs_json: Option<Value>,
}

fn splitmix64_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = z ^ (z >> 31);
    ((z >> 40) as u32 as f32) / ((1u32 << 24) as f32)
}

fn sample_token_with_params(
    rt: &Flash1017Runtime,
    mut logits: Vec<f32>,
    history: &[u32],
    params: &SamplingParams,
    rng_state: &mut u64,
) -> (u32, Option<Value>) {
    let win_len = params.repeat_last_n.min(history.len());
    let recent = &history[history.len() - win_len..];
    if (params.repeat_penalty - 1.0).abs() > 1e-5
        || params.frequency_penalty.abs() > 1e-5
        || params.presence_penalty.abs() > 1e-5
    {
        let mut counts = std::collections::HashMap::<u32, usize>::new();
        for &tid in recent {
            *counts.entry(tid).or_insert(0) += 1;
        }
        for (&tid, &cnt) in &counts {
            let idx = tid as usize;
            if idx < logits.len() {
                if (params.repeat_penalty - 1.0).abs() > 1e-5 {
                    if logits[idx] > 0.0 {
                        logits[idx] /= params.repeat_penalty;
                    } else {
                        logits[idx] *= params.repeat_penalty;
                    }
                }
                logits[idx] -= (cnt as f32) * params.frequency_penalty + params.presence_penalty;
            }
        }
    }
    if params.ignore_eos && (FLASH1017_EOS_TOKEN_ID as usize) < logits.len() {
        logits[FLASH1017_EOS_TOKEN_ID as usize] = f32::NEG_INFINITY;
    }

    let mut max_logit = f32::NEG_INFINITY;
    let mut best_id = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_logit {
            max_logit = v;
            best_id = i as u32;
        }
    }

    let logprobs_entry = if params.logprobs {
        let mut sum_exp = 0.0_f64;
        for &v in &logits {
            if v.is_finite() {
                sum_exp += ((v - max_logit) as f64).exp();
            }
        }
        let log_z = (max_logit as f64) + sum_exp.ln();
        let top_n = params.top_logprobs.max(1).min(20);
        let mut cands: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        cands.select_nth_unstable_by(top_n, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        cands[..top_n].sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_arr: Vec<Value> = cands[..top_n]
            .iter()
            .map(|&(tid, lg)| {
                json!({
                    "token": rt.tokenizer.decode(&[tid]),
                    "token_id": tid,
                    "logprob": (lg as f64) - log_z,
                })
            })
            .collect();
        Some((log_z, top_arr))
    } else {
        None
    };

    let chosen_id = if params.temperature <= 1e-5 {
        best_id
    } else {
        let inv_t = 1.0_f32 / params.temperature.max(1e-4);
        let mut cands: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.is_finite())
            .map(|(i, &v)| (i as u32, (v - max_logit) * inv_t))
            .collect();

        if params.top_k > 0 && params.top_k < cands.len() {
            let k = params.top_k;
            cands.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            cands.truncate(k);
        }
        cands.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (_, p) in &mut cands {
            *p = (*p).exp();
        }
        if params.min_p > 0.0 && !cands.is_empty() {
            let thresh = cands[0].1 * params.min_p;
            cands.retain(|&(_, p)| p >= thresh);
        }
        let total_p: f32 = cands.iter().map(|x| x.1).sum();
        if params.top_p < 1.0 && total_p > 0.0 {
            let limit = total_p * params.top_p;
            let mut acc = 0.0_f32;
            let mut keep = 0usize;
            for &(_, p) in &cands {
                acc += p;
                keep += 1;
                if acc >= limit {
                    break;
                }
            }
            cands.truncate(keep.max(1));
        }
        let norm_p: f32 = cands.iter().map(|x| x.1).sum();
        if norm_p <= 0.0 || cands.is_empty() {
            best_id
        } else {
            let r = splitmix64_f32(rng_state) * norm_p;
            let mut acc = 0.0_f32;
            let mut pick = cands[0].0;
            for &(tid, p) in &cands {
                acc += p;
                if r <= acc {
                    pick = tid;
                    break;
                }
            }
            pick
        }
    };

    let lp_val = logprobs_entry.map(|(log_z, top_arr)| {
        let chosen_lg = logits.get(chosen_id as usize).copied().unwrap_or(max_logit) as f64;
        json!({
            "token": rt.tokenizer.decode(&[chosen_id]),
            "token_id": chosen_id,
            "logprob": chosen_lg - log_z,
            "top_logprobs": top_arr,
        })
    });
    (chosen_id, lp_val)
}

fn check_stop_strings(text: &str, stops: &[String]) -> Option<usize> {
    let mut earliest: Option<usize> = None;
    for s in stops {
        if !s.is_empty() {
            if let Some(idx) = text.find(s) {
                earliest = Some(earliest.map_or(idx, |e| e.min(idx)));
            }
        }
    }
    earliest
}

fn is_repetitive_cycle(ids: &[u32], cand: u32) -> bool {
    let n = ids.len();
    if n >= 2 && ids[n - 1] == cand && ids[n - 2] == cand {
        return true;
    }
    if n >= 3 && ids[n - 2] == cand && ids[n - 1] == ids[n - 3] {
        return true;
    }
    if n >= 5 && ids[n - 3] == cand && ids[n - 1] == ids[n - 4] && ids[n - 2] == ids[n - 5] {
        return true;
    }
    if n >= 7
        && ids[n - 4] == cand
        && ids[n - 1] == ids[n - 5]
        && ids[n - 2] == ids[n - 6]
        && ids[n - 3] == ids[n - 7]
    {
        return true;
    }
    false
}

fn generate_from_prompt(
    rt: &Flash1017Runtime,
    prompt: &str,
    raw_prompt: bool,
    params: &SamplingParams,
    head_mode: Flash1017HeadMode,
    mtp_mode: Flash1017MtpMode,
) -> Result<GenerationOutput, String> {
    if !params.images.is_empty() {
        if let Ok(mut bank) = rt.vision_embed_bank.write() {
            bank.clear();
        }
    }
    let mut vision_ids: Vec<u32> = Vec::new();
    for img_src in &params.images {
        let ids = rt.encode_image_file(img_src)?;
        vision_ids.extend(ids);
    }

    let mut prompt_ids = if raw_prompt {
        let mut v = vec![FLASH1017_BOS_TOKEN_ID];
        v.extend(&vision_ids);
        v.extend(rt.tokenizer.encode(prompt));
        v
    } else if vision_ids.is_empty() {
        rt.tokenizer.format_chat_prompt(prompt, params.enable_thinking)
    } else {
        let mut base = rt
            .tokenizer
            .format_chat_prompt(&format!("\n\n{}", prompt), params.enable_thinking);
        // Insert vision tokens right after `<｜begin▁of▁sentence｜><｜User｜>` (index 2)
        let insert_idx = 2.min(base.len());
        base.splice(insert_idx..insert_idx, vision_ids);
        base
    };
    let max_ctx = params.max_context_tokens.clamp(128, 1_048_576);
    if prompt_ids.len() > max_ctx.saturating_sub(16) {
        let keep_tail = max_ctx.saturating_sub(17).max(64);
        let tail_start = prompt_ids.len().saturating_sub(keep_tail);
        let mut trimmed = Vec::with_capacity(1 + keep_tail);
        trimmed.push(FLASH1017_BOS_TOKEN_ID);
        trimmed.extend_from_slice(&prompt_ids[tail_start..]);
        prompt_ids = trimmed;
    }
    let prompt_tokens = prompt_ids.len();
    let max_tokens = params.max_tokens.min(max_ctx.saturating_sub(prompt_tokens).max(1));
    let max_layers = 40usize;

    let mut kv_cache = Flash1017KvCache::new();
    let (_staged_cnt, _staged_ms) = rt.cpu_lookahead_stage_prompt_tokens(&prompt_ids);
    let t_prefill = Instant::now();
    let mut last_hidden = vec![0.0_f32; 5120];
    let tail_mtp = if prompt_ids.len() > 16 {
        8usize
    } else {
        prompt_ids.len()
    };
    let batch_end = prompt_ids.len().saturating_sub(tail_mtp);
    if batch_end > 0 {
        for chunk in prompt_ids[..batch_end].chunks(4) {
            let (states, _tel) =
                rt.forward_tokens_batch_hidden(chunk, &mut kv_cache, max_layers, &[])?;
            if let Some(last) = states.last() {
                last_hidden.clone_from(&last.final_normed);
            }
        }
    }
    for &tid in &prompt_ids[batch_end..] {
        let (h, _tel) = rt.forward_prefill_token_with_mtp(tid, &mut kv_cache, max_layers)?;
        last_hidden = h;
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    let t_decode = Instant::now();
    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut token_logprobs: Vec<Value> = Vec::new();
    let mut rng_state = params.seed ^ 0x517C_C1B7_2722_0A95;
    let mut finish_reason = "length".to_string();

    let head_mode = if !params.images.is_empty() && head_mode == Flash1017HeadMode::CoarseToFine {
        Flash1017HeadMode::Q8
    } else {
        head_mode
    };
    let pure_greedy = params.is_pure_greedy();
    let use_speculative = pure_greedy
        && params.images.is_empty()
        && !matches!(mtp_mode, Flash1017MtpMode::Off)
        && rt.mtp_layer.is_some();

    if max_tokens > 0 {
        if !pure_greedy {
            // Full sampling / repeat-penalty / logprobs path
            let mut full_hist = prompt_ids.clone();
            let logits0 = rt.compute_full_logits(&last_hidden, head_mode)?;
            let (first_id, lp0) =
                sample_token_with_params(rt, logits0, &full_hist, params, &mut rng_state);
            if let Some(lp) = lp0 {
                token_logprobs.push(lp);
            }
            if first_id == FLASH1017_EOS_TOKEN_ID && !params.ignore_eos {
                finish_reason = "stop".to_string();
            } else {
                generated_ids.push(first_id);
                full_hist.push(first_id);
                let mut cur_tok = first_id;
                for _step in 1..max_tokens {
                    if !params.stop.is_empty() {
                        let cur_text = rt.tokenizer.decode(&generated_ids);
                        if check_stop_strings(&cur_text, &params.stop).is_some() {
                            finish_reason = "stop".to_string();
                            break;
                        }
                    }
                    let (h, _tel) = rt.forward_token_hidden(cur_tok, &mut kv_cache, max_layers)?;
                    let logits = rt.compute_full_logits(&h, head_mode)?;
                    let (next_id, lp) =
                        sample_token_with_params(rt, logits, &full_hist, params, &mut rng_state);
                    if let Some(l) = lp {
                        token_logprobs.push(l);
                    }
                    if (next_id == FLASH1017_EOS_TOKEN_ID && !params.ignore_eos)
                        || (!params.ignore_eos && is_repetitive_cycle(&generated_ids, next_id))
                    {
                        finish_reason = "stop".to_string();
                        break;
                    }
                    generated_ids.push(next_id);
                    full_hist.push(next_id);
                    cur_tok = next_id;
                }
            }
        } else {
            let (first_id, first_logit, first_margin) =
                rt.predict_token_with_mode_and_margin(&last_hidden, head_mode)?;
            if let Some(last_state) = kv_cache.mtp_verified_states.last_mut() {
                if last_state.next_token_id == 0 {
                    last_state.next_token_id = first_id;
                    last_state.next_logit = first_logit;
                    last_state.next_margin = first_margin;
                }
            }
            if first_id == FLASH1017_EOS_TOKEN_ID && !params.ignore_eos {
                finish_reason = "stop".to_string();
            } else {
                generated_ids.push(first_id);
                let first_piece = rt.tokenizer.decode(&[first_id]);
                let mut consecutive_ws = if first_piece.trim().is_empty() { 1usize } else { 0usize };

                if !use_speculative {
                    let mut cur_tok = first_id;
                    for _step in 1..max_tokens {
                        if !params.stop.is_empty() {
                            let cur_text = rt.tokenizer.decode(&generated_ids);
                            if check_stop_strings(&cur_text, &params.stop).is_some() {
                                finish_reason = "stop".to_string();
                                break;
                            }
                        }
                        let (h, _tel) = rt.forward_token_hidden(cur_tok, &mut kv_cache, max_layers)?;
                        last_hidden = h;
                        let (next_id, _logit) = rt.predict_token_with_mode(&last_hidden, head_mode)?;
                        if !params.ignore_eos
                            && (next_id == FLASH1017_EOS_TOKEN_ID
                                || is_repetitive_cycle(&generated_ids, next_id))
                        {
                            finish_reason = "stop".to_string();
                            break;
                        }
                        let piece = rt.tokenizer.decode(&[next_id]);
                        if piece.trim().is_empty() {
                            consecutive_ws += 1;
                            if consecutive_ws >= 3 && !params.ignore_eos {
                                finish_reason = "stop".to_string();
                                break;
                            }
                        } else {
                            consecutive_ws = 0;
                        }
                        generated_ids.push(next_id);
                        cur_tok = next_id;
                    }
                } else {
                    let mut cur_tok = first_id;
                    while generated_ids.len() < max_tokens {
                        if !params.stop.is_empty() {
                            let cur_text = rt.tokenizer.decode(&generated_ids);
                            if check_stop_strings(&cur_text, &params.stop).is_some() {
                                finish_reason = "stop".to_string();
                                break;
                            }
                        }
                        if !params.ignore_eos && generated_ids.len() >= 3 {
                            let n = generated_ids.len();
                            if is_repetitive_cycle(&generated_ids[..n - 1], generated_ids[n - 1]) {
                                finish_reason = "stop".to_string();
                                break;
                            }
                        }
                        let mtp_k_base_len = kv_cache.mtp_k_latent.len();
                        let prev_tok_before_batch =
                            kv_cache.token_history.last().copied().unwrap_or(0);
                        let prev_h39_before_batch = kv_cache.dspark_target_h[2].clone();
                        let (draft_ids, draft_mtp_raws, _draft_ms) =
                            rt.draft_mtp_chain_with_raw(cur_tok, &mut kv_cache, mtp_mode, head_mode, 4)?;
                        let k = draft_ids.len();
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
                            let (h, _tel) =
                                rt.forward_token_hidden(cur_tok, &mut kv_cache, max_layers)?;
                            let (next_id, next_logit, next_margin) =
                                rt.predict_token_with_mode_and_margin(&h, head_mode)?;
                            kv_cache.mtp_verified_states.push(
                                flash_core::flash1017::Flash1017VerifiedMtpState {
                                    seq_pos: kv_cache.seq_len.saturating_sub(1),
                                    token_id: cur_tok,
                                    prev_token_id: prev_tok_before_batch,
                                    prev_h39: prev_h39_before_batch,
                                    mtp_raw: draft_mtp_raws.into_iter().next().unwrap_or_default(),
                                    target_h39: kv_cache.dspark_target_h[2].clone(),
                                    next_token_id: next_id,
                                    next_logit,
                                    next_margin,
                                },
                            );
                            if !params.ignore_eos
                                && (next_id == FLASH1017_EOS_TOKEN_ID
                                    || is_repetitive_cycle(&generated_ids, next_id))
                            {
                                finish_reason = "stop".to_string();
                                break;
                            }
                            let piece = rt.tokenizer.decode(&[next_id]);
                            if piece.trim().is_empty() {
                                consecutive_ws += 1;
                                if consecutive_ws >= 3 && !params.ignore_eos {
                                    finish_reason = "stop".to_string();
                                    break;
                                }
                            } else {
                                consecutive_ws = 0;
                            }
                            generated_ids.push(next_id);
                            cur_tok = next_id;
                            continue;
                        }

                        let mut batch_tokens = Vec::with_capacity(1 + k);
                        batch_tokens.push(cur_tok);
                        batch_tokens.extend_from_slice(&draft_ids);
                        let (states, _tel) = rt.forward_tokens_batch_hidden(
                            &batch_tokens,
                            &mut kv_cache,
                            max_layers,
                            &[],
                        )?;

                        let normed_refs: Vec<&[f32]> =
                            states.iter().map(|s| s.final_normed.as_slice()).collect();
                        let batch_preds =
                            rt.predict_tokens_batch_with_mode_and_margin(&normed_refs, head_mode)?;
                        let mut n_acc = 0usize;
                        let mut bonus_tok = FLASH1017_EOS_TOKEN_ID;
                        let mut hit_limit_or_eos = false;
                        for j in 0..=k {
                            let (pred_tok, _logit, _margin) = batch_preds[j];
                            if j < k {
                                let is_cycle =
                                    !params.ignore_eos && is_repetitive_cycle(&generated_ids, draft_ids[j]);
                                if pred_tok == draft_ids[j]
                                    && (params.ignore_eos || pred_tok != FLASH1017_EOS_TOKEN_ID)
                                    && !is_cycle
                                {
                                    let piece = rt.tokenizer.decode(&[draft_ids[j]]);
                                    if piece.trim().is_empty() {
                                        consecutive_ws += 1;
                                        if consecutive_ws >= 3 && !params.ignore_eos {
                                            let keep_mtp = (mtp_k_base_len + (j + 1) * 512)
                                                .min(kv_cache.mtp_k_latent.len());
                                            kv_cache.restore_position_state(&states[j], keep_mtp);
                                            finish_reason = "stop".to_string();
                                            hit_limit_or_eos = true;
                                            break;
                                        }
                                    } else {
                                        consecutive_ws = 0;
                                    }
                                    n_acc += 1;
                                    generated_ids.push(draft_ids[j]);
                                    if generated_ids.len() >= max_tokens {
                                        let keep_mtp = (mtp_k_base_len + (j + 1) * 512)
                                            .min(kv_cache.mtp_k_latent.len());
                                        kv_cache.restore_position_state(&states[j], keep_mtp);
                                        hit_limit_or_eos = true;
                                        break;
                                    }
                                } else {
                                    let keep_mtp = (mtp_k_base_len + (j + 1) * 512)
                                        .min(kv_cache.mtp_k_latent.len());
                                    kv_cache.restore_position_state(&states[j], keep_mtp);
                                    if is_cycle {
                                        finish_reason = "stop".to_string();
                                        hit_limit_or_eos = true;
                                    } else {
                                        bonus_tok = pred_tok;
                                    }
                                    break;
                                }
                            } else {
                                bonus_tok = pred_tok;
                            }
                        }

                        let keep_mtp_pre =
                            (mtp_k_base_len + k * 512).min(kv_cache.mtp_k_latent.len());
                        if n_acc < k {
                            let keep_mtp_len = (mtp_k_base_len + (n_acc + 1) * 512)
                                .min(kv_cache.mtp_k_latent.len());
                            kv_cache.mtp_k_latent.truncate(keep_mtp_len);
                            kv_cache.sync_mtp12_to_mtp0(keep_mtp_len);
                        } else {
                            kv_cache.mtp_k_latent.truncate(keep_mtp_pre);
                            kv_cache.sync_mtp12_to_mtp0(keep_mtp_pre);
                            let _ = rt.compute_mtp_raw_hidden(
                                batch_tokens[k],
                                &states[k - 1].dspark_target_h,
                                &mut kv_cache,
                            );
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

                        if hit_limit_or_eos || generated_ids.len() >= max_tokens {
                            break;
                        }
                        if !params.ignore_eos
                            && (bonus_tok == FLASH1017_EOS_TOKEN_ID
                                || is_repetitive_cycle(&generated_ids, bonus_tok))
                        {
                            finish_reason = "stop".to_string();
                            break;
                        }
                        let piece = rt.tokenizer.decode(&[bonus_tok]);
                        if piece.trim().is_empty() {
                            consecutive_ws += 1;
                            if consecutive_ws >= 3 && !params.ignore_eos {
                                finish_reason = "stop".to_string();
                                break;
                            }
                        } else {
                            consecutive_ws = 0;
                        }
                        generated_ids.push(bonus_tok);
                        cur_tok = bonus_tok;
                    }
                }
            }
        }
    }

    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let clean_ids: Vec<u32> = generated_ids
        .into_iter()
        .filter(|&tid| {
            (params.ignore_eos || tid != FLASH1017_EOS_TOKEN_ID)
                && (params.enable_thinking || tid != FLASH1017_THINK_END_TOKEN_ID)
        })
        .collect();
    let mut text = rt.tokenizer.decode(&clean_ids);
    if let Some(cut) = check_stop_strings(&text, &params.stop) {
        text.truncate(cut);
        finish_reason = "stop".to_string();
    }
    let completion_tokens = clean_ids.len();
    let decode_tokens = completion_tokens.saturating_sub(1);
    let tok_per_s = if decode_tokens > 0 && decode_ms > 0.0 {
        (decode_tokens as f64) * 1000.0 / decode_ms
    } else if prefill_ms + decode_ms > 0.0 {
        ((prompt_tokens + decode_tokens).max(1) as f64) * 1000.0 / (prefill_ms + decode_ms)
    } else {
        0.0
    };
    let logprobs_json = if params.logprobs {
        Some(json!({ "content": token_logprobs }))
    } else {
        None
    };

    Ok(GenerationOutput {
        text: text.trim().to_string(),
        finish_reason,
        prompt_tokens,
        completion_tokens,
        prefill_ms,
        decode_ms,
        tok_per_s,
        logprobs_json,
    })
}

struct ServerState {
    rt: Flash1017Runtime,
    memory_mode: String,
    default_params: SamplingParams,
    head_mode: Flash1017HeadMode,
    mtp_mode: Flash1017MtpMode,
    api_key: Option<String>,
}

fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn write_json_response(stream: &mut TcpStream, status: &str, body: &Value) {
    let payload = serde_json::to_vec(body).unwrap_or_default();
    let header = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n",
        status,
        payload.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(&payload);
    let _ = stream.flush();
}

fn handle_http_client(mut stream: TcpStream, state: Arc<Mutex<ServerState>>) {
    let mut buf = [0u8; 65536];
    let mut raw = Vec::new();
    let mut header_end = None;
    let mut content_len = 0usize;

    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                if header_end.is_none() {
                    if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                        let h_end = pos + 4;
                        header_end = Some(h_end);
                        let h_str = String::from_utf8_lossy(&raw[..h_end]);
                        for line in h_str.lines() {
                            if let Some((k, v)) = line.split_once(':') {
                                if k.trim().eq_ignore_ascii_case("content-length") {
                                    content_len = v.trim().parse().unwrap_or(0);
                                }
                            }
                        }
                    }
                }
                if let Some(h_end) = header_end {
                    if raw.len() >= h_end + content_len {
                        break;
                    }
                }
            }
            Err(_) => break,
        }
    }

    let Some(h_end) = header_end else {
        return;
    };
    let header_str = String::from_utf8_lossy(&raw[..h_end]);
    let first_line = header_str.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let path = parts.next().unwrap_or("/");
    let body_bytes = &raw[h_end..raw.len().min(h_end + content_len)];

    if method == "OPTIONS" {
        write_json_response(&mut stream, "200 OK", &json!({"ok": true}));
        return;
    }

    // Optional API key check
    {
        let st = state.lock().unwrap();
        if let Some(ref expected_key) = st.api_key {
            let auth_ok = header_str.lines().any(|l| {
                if let Some((k, v)) = l.split_once(':') {
                    k.trim().eq_ignore_ascii_case("authorization")
                        && v.trim().trim_start_matches("Bearer ").trim() == expected_key
                } else {
                    false
                }
            });
            if !auth_ok && path != "/health" && path != "/v1/health" {
                write_json_response(
                    &mut stream,
                    "401 Unauthorized",
                    &json!({"error": {"message": "Invalid or missing Bearer API key", "type": "auth_error"}}),
                );
                return;
            }
        }
    }

    match (method, path) {
        ("GET", "/health") | ("GET", "/v1/health") | ("GET", "/") | ("GET", "/props") => {
            let st = state.lock().unwrap();
            write_json_response(
                &mut stream,
                "200 OK",
                &json!({
                    "status": "ok",
                    "engine": "flash1017-gx10",
                    "model": "DeepSeek-V4.1-Flash-1017-NVFP4",
                    "memory_mode": st.memory_mode,
                    "mtp_mode": st.mtp_mode.label(),
                    "max_model_len": st.default_params.max_context_tokens,
                    "n_ctx": st.default_params.max_context_tokens,
                    "max_input_tokens": st.default_params.max_context_tokens,
                    "max_output_tokens": st.default_params.max_tokens,
                }),
            );
        }
        ("GET", "/v1/models") | ("GET", "/models") => {
            let st = state.lock().unwrap();
            write_json_response(
                &mut stream,
                "200 OK",
                &json!({
                    "object": "list",
                    "data": [{
                        "id": "DeepSeek-V4.1-Flash-1017-NVFP4",
                        "object": "model",
                        "created": unix_ts(),
                        "owned_by": "flash1017-gx10",
                        "max_model_len": st.default_params.max_context_tokens,
                        "context_window": st.default_params.max_context_tokens,
                        "max_input_tokens": st.default_params.max_context_tokens,
                        "max_output_tokens": st.default_params.max_tokens
                    }]
                }),
            );
        }
        ("POST", "/v1/engine/mode") => {
            let req: Value = serde_json::from_slice(body_bytes).unwrap_or_else(|_| json!({}));
            let req_mode = req
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("7gb")
                .to_string();
            let mut st = state.lock().unwrap();
            let new_mode = apply_memory_profile(&mut st.rt, &req_mode);
            st.memory_mode = new_mode.clone();
            write_json_response(
                &mut stream,
                "200 OK",
                &json!({
                    "status": "ok",
                    "memory_mode": new_mode
                }),
            );
        }
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            let req: Value = match serde_json::from_slice(body_bytes) {
                Ok(v) => v,
                Err(e) => {
                    write_json_response(
                        &mut stream,
                        "400 Bad Request",
                        &json!({"error": {"message": format!("Invalid JSON: {}", e)}}),
                    );
                    return;
                }
            };
            let is_stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
            let mut combined_prompt = String::new();
            let mut msg_images: Vec<String> = Vec::new();
            if let Some(msgs) = req.get("messages").and_then(|v| v.as_array()) {
                for m in msgs {
                    let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                    let mut text_part = String::new();
                    if let Some(s) = m.get("content").and_then(|v| v.as_str()) {
                        text_part = s.to_string();
                    } else if let Some(arr) = m.get("content").and_then(|v| v.as_array()) {
                        for part in arr {
                            let ptype = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if ptype == "text" {
                                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                    text_part.push_str(t);
                                }
                            } else if ptype == "image_url" {
                                if let Some(u) = part
                                    .get("image_url")
                                    .and_then(|iu| iu.get("url").or(Some(iu)))
                                    .and_then(|v| v.as_str())
                                {
                                    msg_images.push(u.to_string());
                                }
                            }
                        }
                    }
                    if role == "system" {
                        if !text_part.trim().is_empty() {
                            combined_prompt.push_str(text_part.trim());
                            combined_prompt.push_str("\n\n");
                        }
                    } else if role == "user" {
                        combined_prompt.push_str(text_part.trim());
                        combined_prompt.push('\n');
                    } else if role == "assistant" {
                        combined_prompt.push_str(text_part.trim());
                        combined_prompt.push('\n');
                    }
                }
            } else if let Some(p) = req.get("prompt").and_then(|v| v.as_str()) {
                combined_prompt = p.to_string();
            }

            let (res, model_id) = {
                let mut st = state.lock().unwrap();
                if let Some(m) = req.get("memory_mode").and_then(|v| v.as_str()) {
                    if m != st.memory_mode {
                        st.memory_mode = apply_memory_profile(&mut st.rt, m);
                    }
                }
                let mut req_params = st.default_params.from_json_with_defaults(&req);
                req_params.images.extend(msg_images);
                let out = generate_from_prompt(
                    &st.rt,
                    combined_prompt.trim(),
                    false,
                    &req_params,
                    st.head_mode,
                    st.mtp_mode,
                );
                let mid = req
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("DeepSeek-V4.1-Flash-1017-NVFP4")
                    .to_string();
                (out, mid)
            };

            match res {
                Ok(gen) => {
                    let created = unix_ts();
                    if is_stream {
                        let sse_hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                        let _ = stream.write_all(sse_hdr.as_bytes());
                        let chunk = json!({
                            "id": format!("chatcmpl-{}", created),
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": gen.text},
                                "finish_reason": gen.finish_reason,
                                "logprobs": gen.logprobs_json
                            }]
                        });
                        let _ = stream.write_all(format!("data: {}\n\n", chunk).as_bytes());
                        let _ = stream.write_all(b"data: [DONE]\n\n");
                        let _ = stream.flush();
                    } else {
                        let resp = json!({
                            "id": format!("chatcmpl-{}", created),
                            "object": "chat.completion",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": gen.text
                                },
                                "finish_reason": gen.finish_reason,
                                "logprobs": gen.logprobs_json
                            }],
                            "usage": {
                                "prompt_tokens": gen.prompt_tokens,
                                "completion_tokens": gen.completion_tokens,
                                "total_tokens": gen.prompt_tokens + gen.completion_tokens
                            },
                            "timings": {
                                "prompt_ms": gen.prefill_ms,
                                "predicted_ms": gen.decode_ms,
                                "predicted_n": gen.completion_tokens,
                                "predicted_per_second": gen.tok_per_s
                            }
                        });
                        write_json_response(&mut stream, "200 OK", &resp);
                    }
                }
                Err(e) => {
                    write_json_response(
                        &mut stream,
                        "500 Internal Server Error",
                        &json!({"error": {"message": e}}),
                    );
                }
            }
        }
        ("POST", "/v1/completions") | ("POST", "/completions") => {
            let req: Value = match serde_json::from_slice(body_bytes) {
                Ok(v) => v,
                Err(e) => {
                    write_json_response(
                        &mut stream,
                        "400 Bad Request",
                        &json!({"error": {"message": format!("Invalid JSON: {}", e)}}),
                    );
                    return;
                }
            };
            let is_stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
            let prompt_str = req
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let raw_prompt = req.get("raw_prompt").and_then(|v| v.as_bool()).unwrap_or(true);

            let (res, model_id) = {
                let st = state.lock().unwrap();
                let req_params = st.default_params.from_json_with_defaults(&req);
                let out = generate_from_prompt(
                    &st.rt,
                    &prompt_str,
                    raw_prompt,
                    &req_params,
                    st.head_mode,
                    st.mtp_mode,
                );
                let mid = req
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("DeepSeek-V4.1-Flash-1017-NVFP4")
                    .to_string();
                (out, mid)
            };

            match res {
                Ok(gen) => {
                    let created = unix_ts();
                    if is_stream {
                        let sse_hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                        let _ = stream.write_all(sse_hdr.as_bytes());
                        let chunk = json!({
                            "id": format!("cmpl-{}", created),
                            "object": "text_completion",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "text": gen.text,
                                "finish_reason": gen.finish_reason,
                                "logprobs": gen.logprobs_json
                            }]
                        });
                        let _ = stream.write_all(format!("data: {}\n\n", chunk).as_bytes());
                        let _ = stream.write_all(b"data: [DONE]\n\n");
                        let _ = stream.flush();
                    } else {
                        let resp = json!({
                            "id": format!("cmpl-{}", created),
                            "object": "text_completion",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "text": gen.text,
                                "finish_reason": gen.finish_reason,
                                "logprobs": gen.logprobs_json
                            }],
                            "usage": {
                                "prompt_tokens": gen.prompt_tokens,
                                "completion_tokens": gen.completion_tokens,
                                "total_tokens": gen.prompt_tokens + gen.completion_tokens
                            },
                            "timings": {
                                "prompt_ms": gen.prefill_ms,
                                "predicted_ms": gen.decode_ms,
                                "predicted_n": gen.completion_tokens,
                                "predicted_per_second": gen.tok_per_s
                            }
                        });
                        write_json_response(&mut stream, "200 OK", &resp);
                    }
                }
                Err(e) => {
                    write_json_response(
                        &mut stream,
                        "500 Internal Server Error",
                        &json!({"error": {"message": e}}),
                    );
                }
            }
        }
        _ => {
            write_json_response(
                &mut stream,
                "404 Not Found",
                &json!({"error": {"message": format!("Endpoint {} {} not found", method, path)}}),
            );
        }
    }
}

fn print_help() {
    println!(
        "flash1017-gx10 — Standalone DeepSeek-V4.1-Flash-1017-NVFP4 Engine (vLLM / llama-server compatible)

ANVÄNDNING:
  1. Starta som OpenAI / vLLM / llama-server HTTP-server (för harnesses, agenter & verktyg):
     flash1017-gx10 serve -m <model_dir> --host 0.0.0.0 --port 8000 --memory-mode <7gb|55gb|tierc>

  2. Kör direkt från kommandoraden (som llama-cli):
     flash1017-gx10 -m <model_dir> -p \"Vad är Sveriges huvudstad?\" -n 128 --temp 0.6 --top-p 0.9

  3. Kör MTP-hastighetsbenchmark (K=0..4 + Adaptive):
     flash1017-gx10 bench -m <model_dir> --memory-mode <7gb|55gb|tierc>

  4. Komprimera 404 GB NVFP4 -> 106 GB (1.58-Bit + 0.33 MB Rank-16 Q8 JEPA på alla 384 experter):
     flash1017-gx10 compress-106gb -m <nvfp4_dir> -o <output_106gb_dir> [--layer <0..39>]

FLAGGOR:
  -m, --model <sökväg>          Sökväg till DeepSeek-V4.1-Flash-1017-NVFP4 / Tier2
  --host <ip>                   Bind-adress för server (standard: 0.0.0.0)
  --port <port>                 HTTP-port för server (standard: 8000)
  --memory-mode <läge>          Minnesprofil:
                                  • 7gb   : ~7 GB RAM (Snabbläge för logik/kod)
                                  • 55gb  : ~55 GB låst i RAM (för 64 GB / 128 GB maskiner)
                                  • tierc : 55 GB i RAM + alla 9 600 Tier C från NVMe med CPU Lookahead
  --mtp <adaptive|off|1..4>     Multi-Token Prediction-läge (standard: adaptive, 3 lager mtp.0..2)
  --head-mode <c2f|bf16|q8>     LM-Head-läge (standard: c2f)
  -c, --ctx-size, --max-model-len <tokens>
                                Max IN-context (standard: 1048576 / 1M tokens via YaRN MLA)
  -n, --n-predict, --max-tokens, --max-output-tokens <tokens>
                                Max UT-context / genererade tokens (standard: 8192)
  -p, --prompt, --once <text>   Kör en enskild prompt direkt och avsluta
  --raw-prompt                  Skicka rå prompt utan chat-template
  --temp, --temperature <f32>   Sampling-temperatur (0.0 = greedy, standard: 0.0)
  --top-p <f32>                 Nucleus sampling (standard: 1.0)
  --top-k <usize>               Top-K sampling (0 = av, standard: 0)
  --min-p <f32>                 Min-P tröskel (standard: 0.0)
  --repeat-penalty <f32>        Upprepningsstraff (standard: 1.0)
  --repeat-last-n <usize>       Fönster för repeat-penalty (standard: 64)
  --frequency-penalty <f32>     OpenAI frequency_penalty (-2.0..2.0, standard: 0.0)
  --presence-penalty <f32>      OpenAI presence_penalty (-2.0..2.0, standard: 0.0)
  -s, --seed <u64>              Slumpfrö för deterministisk sampling
  --stop <sträng>               Stoppsträng (kan anges flera gånger)
  --ignore-eos                  Fortsätt generera trots EOS-token
  --think, --enable-thinking    Tvinga modellen att starta med <think>-resonemang
  --logprobs                    Returnera token-logprobs (för benchmarks/harnesses)
  --top-logprobs <usize>        Antal top-logprobs per token (standard: 5)
  --no-engram                   Stäng av 33 GB N-Gram Engram-tabell (sparar RAM)
  --api-key <nyckel>            Kräv Bearer API-nyckel vid HTTP-anrop
  --addon <sökväg>              Sökväg till addon_gx10_rust.so"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help" || a == "help") {
        print_help();
        return;
    }

    let mut subcmd = "chat".to_string();
    let mut model_dir = default_model_dir();
    let mut addon_spec = default_addon_path();
    let mut mode = "7gb".to_string();
    let mut mtp_mode = Flash1017MtpMode::Adaptive;
    let mut head_mode = Flash1017HeadMode::CoarseToFine;
    let mut host = "0.0.0.0".to_string();
    let mut port: u16 = 8000;
    let mut once_prompt: Option<String> = None;
    let mut raw_prompt = false;
    let mut default_params = SamplingParams::default();
    let mut load_engram = true;
    let mut api_key: Option<String> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut target_layer: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "serve" | "server" | "--serve" | "--server" => {
                subcmd = "serve".to_string();
                i += 1;
            }
            "bench" | "--bench" | "--bench-mtp" => {
                subcmd = "bench".to_string();
                i += 1;
            }
            "compress-106gb" | "--compress-106gb" => {
                subcmd = "compress-106gb".to_string();
                i += 1;
            }
            "-o" | "--output" | "--output-dir" => {
                if let Some(v) = args.get(i + 1) {
                    output_dir = Some(PathBuf::from(v));
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--layer" | "-l" => {
                if let Some(v) = args.get(i + 1) {
                    target_layer = v.parse::<usize>().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "chat" => {
                subcmd = "chat".to_string();
                i += 1;
            }
            "-m" | "--model" | "--model-dir" => {
                if let Some(v) = args.get(i + 1) {
                    model_dir = PathBuf::from(v);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--addon" | "--hardware" => {
                if let Some(v) = args.get(i + 1) {
                    addon_spec = v.clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--mode" | "--profile" | "--memory-mode" => {
                if let Some(v) = args.get(i + 1) {
                    mode = v.clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--preload-experts" | "--55gb" => {
                mode = "55gb".to_string();
                i += 1;
            }
            "--full-tier-c" | "--tierc" => {
                mode = "tierc".to_string();
                i += 1;
            }
            "--mtp" => {
                if let Some(v) = args.get(i + 1) {
                    mtp_mode = parse_mtp_mode(v);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--head-mode" => {
                if let Some(v) = args.get(i + 1) {
                    head_mode = parse_head_mode(v);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--host" => {
                if let Some(v) = args.get(i + 1) {
                    host = v.clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--port" => {
                if let Some(v) = args.get(i + 1) {
                    port = v.parse().unwrap_or(8000);
                    subcmd = "serve".to_string();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "-p" | "--prompt" | "--once" => {
                if let Some(v) = args.get(i + 1) {
                    once_prompt = Some(v.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--raw-prompt" => {
                raw_prompt = true;
                i += 1;
            }
            "-c" | "--ctx-size" | "--max-model-len" | "--max-input-tokens" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.max_context_tokens =
                        v.parse().unwrap_or(1_048_576).clamp(128, 1_048_576);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "-n" | "--n-predict" | "--max-tokens" | "--max-output-tokens" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.max_tokens = v.parse().unwrap_or(8_192);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--temp" | "--temperature" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.temperature = v.parse().unwrap_or(0.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--top-p" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.top_p = v.parse().unwrap_or(1.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--top-k" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.top_k = v.parse().unwrap_or(0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--min-p" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.min_p = v.parse().unwrap_or(0.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--repeat-penalty" | "--repetition-penalty" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.repeat_penalty = v.parse().unwrap_or(1.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--repeat-last-n" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.repeat_last_n = v.parse().unwrap_or(64);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--frequency-penalty" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.frequency_penalty = v.parse().unwrap_or(0.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--presence-penalty" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.presence_penalty = v.parse().unwrap_or(0.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "-s" | "--seed" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.seed = v.parse().unwrap_or(42);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--stop" | "--reverse-prompt" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.stop.push(v.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--ignore-eos" => {
                default_params.ignore_eos = true;
                i += 1;
            }
            "--think" | "--enable-thinking" => {
                default_params.enable_thinking = true;
                i += 1;
            }
            "--logprobs" => {
                default_params.logprobs = true;
                i += 1;
            }
            "--top-logprobs" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.logprobs = true;
                    default_params.top_logprobs = v.parse().unwrap_or(5);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--no-engram" => {
                load_engram = false;
                i += 1;
            }
            "--engram" => {
                load_engram = true;
                i += 1;
            }
            "--image" | "-i" => {
                if let Some(v) = args.get(i + 1) {
                    default_params.images.push(v.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--api-key" => {
                if let Some(v) = args.get(i + 1) {
                    api_key = Some(v.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            other if !other.starts_with('-') => {
                let p = PathBuf::from(other);
                if p.exists() {
                    model_dir = p;
                }
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("==========================================================================");
    println!("  ⚡ FLASH-1017 STANDALONE TURBO ENGINE (OPEN-SOURCE + JEPA + LOOKAHEAD)");
    println!("==========================================================================");
    if let Ok((name, backend, _path)) = hardware_addon::load_hardware_addon(&addon_spec) {
        println!("  • Hårdvara : {} ({})", name, backend);
    }

    if subcmd == "compress-106gb" {
        let out_dir = output_dir.unwrap_or_else(|| {
            model_dir
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("DeepSeek-V4.1-Flash-1017-106GB-JEPA")
        });
        let _ = std::fs::create_dir_all(&out_dir);
        println!("  • Kompressor: 404 GB NVFP4 -> 106 GB (1.58-Bit + 0.33 MB Rank-16 Q8 JEPA)");
        println!("  • Källa     : {}", model_dir.display());
        println!("  • Mål       : {}", out_dir.display());
        println!("==========================================================================\n");

        let layers: Vec<usize> = if let Some(l) = target_layer {
            vec![l]
        } else {
            // Symlink non-MoE files (`tokenizer.json`, `config.json`, `model-00001..2`, `model-00043..48`)
            for name in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
                "model-00001-of-00048.safetensors",
                "model-00002-of-00048.safetensors",
                "model-00043-of-00048.safetensors",
                "model-00044-of-00048.safetensors",
                "model-00045-of-00048.safetensors",
                "model-00046-of-00048.safetensors",
                "model-00047-of-00048.safetensors",
                "model-00048-of-00048.safetensors",
            ] {
                let src = model_dir.join(name);
                let dst = out_dir.join(name);
                if src.exists() && !dst.exists() {
                    #[cfg(unix)]
                    {
                        let _ = std::os::unix::fs::symlink(&src, &dst);
                    }
                }
            }
            (0..40).collect()
        };

        let t_all = Instant::now();
        let mut total_orig: u64 = 0;
        let mut total_comp: u64 = 0;
        for l in layers {
            let shard_name = format!("model-{:05}-of-00048.safetensors", l + 3);
            let in_shard = model_dir.join(&shard_name);
            let out_shard = out_dir.join(&shard_name);
            if target_layer.is_none() && out_shard.exists() {
                if let Ok(meta) = std::fs::metadata(&out_shard) {
                    if meta.len() > 100_000_000 {
                        println!(
                            "  ⏭️  [Lager {:02}/39] {} finns redan ({:.2} GB) — hoppar över.",
                            l,
                            shard_name,
                            meta.len() as f64 / 1e9
                        );
                        continue;
                    }
                }
            }
            print!(
                "  ⚙️  [Lager {:02}/39] Komprimerar 384 experter i {} ... ",
                l, shard_name
            );
            let _ = io::stdout().flush();
            match Flash1017Jepa106Compressor::compress_layer_shard(&in_shard, &out_shard) {
                Ok(rep) => {
                    total_orig += rep.original_file_bytes;
                    total_comp += rep.compressed_file_bytes;
                    println!(
                        "KLAR på {:.1}s! {:.2} GB -> {:.2} GB (-{:.1}%) | Frob-fel: 1.58b={:.4} -> JEPA-Q8={:.4}",
                        rep.elapsed_secs,
                        rep.original_file_bytes as f64 / 1e9,
                        rep.compressed_file_bytes as f64 / 1e9,
                        rep.savings_percent,
                        rep.ternary_only_rel_frob_err,
                        rep.jepa_recovered_rel_frob_err
                    );
                }
                Err(e) => {
                    eprintln!("\n❌ Fel vid komprimering av lager {}: {}", l, e);
                    std::process::exit(1);
                }
            }
        }
        println!(
            "\n✅ Komprimering klar på {:.1}s! Totalt: {:.2} GB -> {:.2} GB.",
            t_all.elapsed().as_secs_f64(),
            total_orig as f64 / 1e9,
            total_comp as f64 / 1e9
        );
        return;
    }

    let t_open = Instant::now();
    let mut rt = match Flash1017Runtime::open(&model_dir, load_engram) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ Kunde inte öppna Flash-1017 i {:?}: {}", model_dir, e);
            std::process::exit(1);
        }
    };
    let _ = rt.enable_c2f_head();
    let _ = rt.enable_mtp_layer();
    let open_ms = t_open.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  • Modell   : {} (Öppnad via Zero-Copy mmap på {:.1} ms | Engram={})",
        model_dir.display(),
        open_ms,
        if load_engram { "AKTIV (33 GB)" } else { "AV" }
    );

    let active_mode = apply_memory_profile(&mut rt, &mode);
    let mtp_layers_n = rt
        .mtp_layer
        .as_ref()
        .map(|m| m.num_active_sub_layers())
        .unwrap_or(0);
    println!(
        "  • Turbo    : MTP={} ({}/3 lager) | Head={:?} | Temp={:.2} | TopP={:.2} | TopK={}",
        mtp_mode.label(),
        mtp_layers_n,
        head_mode,
        default_params.temperature,
        default_params.top_p,
        default_params.top_k
    );
    println!(
        "  • Context  : IN max {} tokens (1M YaRN MLA) | UT max {} tokens",
        default_params.max_context_tokens, default_params.max_tokens
    );
    println!("==========================================================================\n");

    if subcmd == "serve" {
        let bind_addr = format!("{}:{}", host, port);
        let listener = match TcpListener::bind(&bind_addr) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("❌ Kunde inte binda HTTP-server till {}: {}", bind_addr, e);
                std::process::exit(1);
            }
        };
        println!("🟢 OpenAI / vLLM / llama-server Drop-In API lyssnar på http://{}", bind_addr);
        println!("   • GET  http://{}/health", bind_addr);
        println!("   • GET  http://{}/v1/models", bind_addr);
        println!("   • POST http://{}/v1/chat/completions", bind_addr);
        println!("   • POST http://{}/v1/completions", bind_addr);
        println!("   • POST http://{}/v1/engine/mode  {{\"mode\":\"7gb\"|\"55gb\"|\"tierc\"}}\n", bind_addr);

        let shared = Arc::new(Mutex::new(ServerState {
            rt,
            memory_mode: active_mode,
            default_params,
            head_mode,
            mtp_mode,
            api_key,
        }));

        for conn in listener.incoming().flatten() {
            let st_clone = Arc::clone(&shared);
            std::thread::spawn(move || {
                handle_http_client(conn, st_clone);
            });
        }
        return;
    }

    if subcmd == "bench" {
        let p = once_prompt
            .as_deref()
            .unwrap_or("The capital of Sweden is");
        let mut prompt_ids = vec![FLASH1017_BOS_TOKEN_ID];
        prompt_ids.extend(rt.tokenizer.encode(p));
        match rt.benchmark_mtp_duel(&prompt_ids, default_params.max_tokens.max(13), 40, head_mode) {
            Ok(rows) => {
                for r in &rows {
                    println!(
                        "  [{:<14}] {:>6.2} tok/s | {:>5.1} ms/tok | Acc: {:>5.1}% | {:.2} tok/step | Match K0: {}",
                        r.mode_label,
                        r.effective_tok_per_s,
                        r.effective_ms_per_tok,
                        r.acceptance_rate_pct,
                        r.mean_tokens_per_step,
                        if r.exact_match_k0 { "✅" } else { "⚠️" }
                    );
                }
                if let Some(first) = rows.first() {
                    println!("\n  [Text]: \"{}{}\"", p, first.decoded_text);
                }
            }
            Err(e) => eprintln!("Bench error: {}", e),
        }
        return;
    }

    if let Some(p) = once_prompt {
        match generate_from_prompt(&rt, &p, raw_prompt, &default_params, head_mode, mtp_mode) {
            Ok(gen) => {
                println!("{}\n", gen.text);
                if let Some(lp) = &gen.logprobs_json {
                    println!("  [Logprobs]: {}", lp);
                }
                println!(
                    "  [⚡ {:.2} tok/s | {} tokens | Prefill: {:.1} ms | Finish: {}]",
                    gen.tok_per_s, gen.completion_tokens, gen.prefill_ms, gen.finish_reason
                );
            }
            Err(e) => eprintln!("⚠️ Fel: {}", e),
        }
        return;
    }

    let stdin = io::stdin();
    loop {
        print!("flash1017> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            "/quit" | "/exit" | "quit" | "exit" => break,
            "/7gb" | "/experts ondemand" => {
                apply_memory_profile(&mut rt, "7gb");
                continue;
            }
            "/55gb" | "/experts preload" => {
                apply_memory_profile(&mut rt, "55gb");
                continue;
            }
            "/tierc" | "/experts tierc" => {
                apply_memory_profile(&mut rt, "tierc");
                continue;
            }
            "/bench" => {
                let ids = rt.tokenizer.encode("The capital of Sweden is");
                if let Ok(rows) = rt.benchmark_mtp_duel(&ids, 13, 40, head_mode) {
                    for r in rows {
                        println!(
                            "  [{:<14}] {:>6.2} tok/s | {:>5.1} ms/tok | Acc: {:>5.1}%",
                            r.mode_label,
                            r.effective_tok_per_s,
                            r.effective_ms_per_tok,
                            r.acceptance_rate_pct
                        );
                    }
                }
                continue;
            }
            _ => {}
        }

        match generate_from_prompt(&rt, trimmed, false, &default_params, head_mode, mtp_mode) {
            Ok(gen) => {
                println!("\n{}\n", gen.text);
                println!(
                    "  [⚡ {:.2} tok/s | {} tokens | Prefill: {:.1} ms | Finish: {}]\n",
                    gen.tok_per_s, gen.completion_tokens, gen.prefill_ms, gen.finish_reason
                );
            }
            Err(e) => eprintln!("⚠️ Fel: {}\n", e),
        }
    }
}
