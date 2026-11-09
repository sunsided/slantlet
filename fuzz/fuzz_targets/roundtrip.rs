#![no_main]
//! Finite inputs must round-trip and preserve energy through every transform:
//! `forward` (power-of-two prefix), `forward_frames` (any length),
//! `forward_2d_frames` (any width × height) and `forward_2d_planar` (largest
//! square power-of-two prefix).

use libfuzzer_sys::fuzz_target;

/// Euclidean norm, scaled by the vector's own max magnitude so squares
/// neither underflow nor overflow.
fn norm(v: &[f64]) -> f64 {
    let scale = v.iter().fold(f64::MIN_POSITIVE, |m, a| m.max(a.abs()));
    scale * v.iter().map(|a| (a / scale).powi(2)).sum::<f64>().sqrt()
}

type Transform<'a> = &'a dyn Fn(&mut [f64]) -> Result<(), slantlet::Error>;

/// Asserts that `fwd` preserves energy and `inv` undoes it on `orig`.
fn check(orig: &[f64], fwd: Transform, inv: Transform, what: &str) {
    let tol = 1e-9 * norm(orig) + 1e-300;
    let mut y = orig.to_vec();
    fwd(&mut y).unwrap();
    assert!(
        (norm(&y) - norm(orig)).abs() <= tol,
        "{what}: energy changed"
    );
    inv(&mut y).unwrap();
    for (a, b) in y.iter().zip(orig) {
        assert!((a - b).abs() <= tol, "{what}: roundtrip {a} vs {b}");
    }
}

fuzz_target!(|data: &[u8]| {
    let x: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .filter(|v| v.is_finite() && v.abs() < 1e100)
        .collect();
    if x.is_empty() {
        return;
    }
    let frame = 1usize << (data[0] % 13);

    let pow2 = &x[..1 << x.len().ilog2()];
    check(
        pow2,
        &|v| slantlet::forward(v),
        &|v| slantlet::inverse(v),
        "1-D",
    );

    check(
        &x,
        &|v| slantlet::forward_frames(v, frame),
        &|v| slantlet::inverse_frames(v, frame),
        "frames",
    );

    let w = usize::from(data[1 % data.len()]) % x.len() + 1;
    let h = x.len() / w;
    check(
        &x[..w * h],
        &|v| slantlet::forward_2d_frames(v, w, h, frame),
        &|v| slantlet::inverse_2d_frames(v, w, h, frame),
        "2-D separable",
    );

    let side = 1usize << (x.len().ilog2() / 2);
    check(
        &x[..side * side],
        &|v| slantlet::forward_2d_planar(v, side),
        &|v| slantlet::inverse_2d_planar(v, side),
        "2-D planar",
    );
});
