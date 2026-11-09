#![no_main]
//! Finite inputs of power-of-two length must round-trip and preserve energy.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut x: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .filter(|v| v.is_finite() && v.abs() < 1e100)
        .collect();
    if x.is_empty() {
        return;
    }
    x.truncate(1 << x.len().ilog2());

    let orig = x.clone();
    // Scale by the max magnitude so squares neither underflow nor overflow.
    let scale = orig.iter().fold(f64::MIN_POSITIVE, |m, v| m.max(v.abs()));
    let norm = |v: &[f64]| scale * v.iter().map(|a| (a / scale).powi(2)).sum::<f64>().sqrt();
    let tol = 1e-9 * norm(&orig) + 1e-300;

    slantlet::forward(&mut x).unwrap();
    assert!((norm(&x) - norm(&orig)).abs() <= tol, "energy changed");
    slantlet::inverse(&mut x).unwrap();
    for (a, b) in x.iter().zip(&orig) {
        assert!((a - b).abs() <= tol, "roundtrip {a} vs {b}");
    }
});
