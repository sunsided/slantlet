//! Subband-energy features for vibration-based fault detection.
//!
//! Follows the feature-extraction step of Alobaidy et al., "Real time fault
//! diagnosis in industrial robotics using discrete and slantlet wavelet
//! transformations", Sci. Rep. 15, 34145 (2025): slantlet-transform
//! accelerometer windows and use per-scale energies as classifier input.
//! The signals here are synthetic; only the recording length (3973 samples,
//! not a power of two) mirrors the paper.
//!
//! Run with `cargo run --example fault_features`.

use std::f64::consts::TAU;
use std::ops::Range;

const FS: f64 = 1000.0; // Hz
const LEN: usize = 3973; // samples per recording
const FRAME: usize = 256; // power-of-two analysis window

fn main() -> Result<(), slantlet::Error> {
    let healthy = features(&signal(false))?;
    let faulty = features(&signal(true))?;

    println!(
        "relative energy per scale ({FRAME}-sample frames, {} per recording)",
        LEN / FRAME
    );
    println!("{:<28}{:>10}{:>10}", "band", "healthy", "faulty");
    for (i, (h, f)) in healthy.iter().zip(&faulty).enumerate() {
        println!("{:<28}{h:>10.4}{f:>10.4}", band_label(i));
    }

    // Impacts excite the short, fine-scale slantlets.
    let finest = healthy.len() - 1;
    assert!(faulty[finest] > 10.0 * healthy[finest], "fault not visible");
    Ok(())
}

/// Mean relative energy per band over all full frames. The trailing
/// `LEN % FRAME` samples are dropped; zero-padding them would add an
/// artificial edge that leaks energy into the detail scales.
fn features(x: &[f64]) -> Result<Vec<f64>, slantlet::Error> {
    let bands = bands(FRAME);
    let mut acc = vec![0.0; bands.len()];
    let mut frames = 0;
    for chunk in x.chunks_exact(FRAME) {
        let mut c = chunk.to_vec();
        slantlet::forward(&mut c)?;
        let total = c.iter().map(|v| v * v).sum::<f64>().max(f64::MIN_POSITIVE);
        for (a, r) in acc.iter_mut().zip(&bands) {
            *a += c[r.clone()].iter().map(|v| v * v).sum::<f64>() / total;
        }
        frames += 1;
    }
    Ok(acc.into_iter().map(|a| a / f64::from(frames)).collect())
}

/// Coefficient ranges, coarse to fine: `[h, f]`, then the scale with blocks of
/// length `2m` at `[N/m, 2N/m)` for `m = N/2 … 2` (see the crate docs).
fn bands(n: usize) -> Vec<Range<usize>> {
    let mut v = Vec::new();
    v.push(0..2); // trend: h and f
    let mut m = n / 2;
    while m >= 2 {
        v.push(n / m..2 * n / m);
        m /= 2;
    }
    v
}

fn band_label(i: usize) -> String {
    if i == 0 {
        return "trend (h, f)".into();
    }
    let block = FRAME >> (i - 1); // filter length at this scale
    format!("{block:>3}-sample filters (~{:.0} Hz)", FS / block as f64)
}

/// Joint motion plus a motor tone and sensor noise; `fault` adds decaying
/// 180 Hz impacts every 80 ms, as from a damaged gear or bearing.
fn signal(fault: bool) -> Vec<f64> {
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    (0..LEN)
        .map(|n| {
            let t = n as f64 / FS;
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let noise = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            let mut v = 0.5 * (TAU * 1.5 * t).sin() + 0.2 * (TAU * 50.0 * t).sin() + 0.02 * noise;
            if fault {
                let dt = t % 0.08;
                v += 0.6 * (-300.0 * dt).exp() * (TAU * 180.0 * dt).sin();
            }
            v
        })
        .collect()
}
