pub mod flash1017;
pub mod hardware_addon;

pub use flash1017::*;

#[inline(always)]
pub fn dot_f32_unrolled8(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 8;
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;
    for i in 0..chunks {
        let base = i * 8;
        s0 = a[base].mul_add(b[base], s0);
        s1 = a[base + 1].mul_add(b[base + 1], s1);
        s2 = a[base + 2].mul_add(b[base + 2], s2);
        s3 = a[base + 3].mul_add(b[base + 3], s3);
        s4 = a[base + 4].mul_add(b[base + 4], s4);
        s5 = a[base + 5].mul_add(b[base + 5], s5);
        s6 = a[base + 6].mul_add(b[base + 6], s6);
        s7 = a[base + 7].mul_add(b[base + 7], s7);
    }
    let mut total = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    for i in (chunks * 8)..len {
        total = a[i].mul_add(b[i], total);
    }
    total
}
