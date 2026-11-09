#![no_main]
//! Finite inputs must round-trip and preserve energy: power-of-two lengths
//! through `forward`, arbitrary lengths through `forward_frames`.

use libfuzzer_sys::fuzz_target;

/// Euclidean norm, scaled by the vector's own max magnitude so squares
/// neither underflow nor overflow.
fn norm(v: &[f64]) -> f64 {
    let scale = v.iter().fold(f64::MIN_POSITIVE, |m, a| m.max(a.abs()));
    scale * v.iter().map(|a| (a / scale).powi(2)).sum::<f64>().sqrt()
}

fuzz_target!(|data: &[u8]| {
    let mut x: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .filter(|v| v.is_finite() && v.abs() < 1e100)
        .collect();
    if x.is_empty() {
        return;
    }
    let frame = 1usize << (data[0] % 13);
    let mut framed = x.clone();
    x.truncate(1 << x.len().ilog2());

    let orig = x.clone();
    let tol = 1e-9 * norm(&orig) + 1e-300;

    slantlet::forward(&mut x).unwrap();
    assert!((norm(&x) - norm(&orig)).abs() <= tol, "energy changed");
    slantlet::inverse(&mut x).unwrap();
    for (a, b) in x.iter().zip(&orig) {
        assert!((a - b).abs() <= tol, "roundtrip {a} vs {b}");
    }

    let orig = framed.clone();
    let tol = 1e-9 * norm(&orig) + 1e-300;
    slantlet::forward_frames(&mut framed, frame).unwrap();
    assert!(
        (norm(&framed) - norm(&orig)).abs() <= tol,
        "frames: energy changed"
    );
    slantlet::inverse_frames(&mut framed, frame).unwrap();
    for (a, b) in framed.iter().zip(&orig) {
        assert!((a - b).abs() <= tol, "frames roundtrip {a} vs {b}");
    }
});
