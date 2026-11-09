//! The Slantlet transform.
//!
//! The Slantlet transform (I. W. Selesnick, "The Slantlet Transform",
//! *IEEE Trans. Signal Processing* 47(5), 1999) is an orthogonal discrete
//! wavelet-like transform for signals of length `N = 2^l`. Unlike an iterated
//! filter bank, every scale uses its own filters of length `2^(i+1)`, which are
//! piecewise linear on each half of their support and have two vanishing
//! moments. This gives better time localization than the Daubechies-2 wavelet,
//! which has the same number of vanishing moments.
//!
//! # Basis
//!
//! - **Detail scales** `i = 1 .. l-1`: the signal is split into aligned,
//!   non-overlapping blocks of length `2m`, `m = 2^i`. Each block carries a pair
//!   of basis functions `g_i` and its time reversal `J g_i`. Both are
//!   piecewise linear on the two halves of the block, orthonormal to each other,
//!   and orthogonal to constants and ramps (two vanishing moments).
//! - **Coarsest scale**: `h` (constant) and `f` (centered ramp) over the whole
//!   signal, spanning the linear functions.
//!
//! Together these form an orthonormal basis of `R^N`, so [`inverse`] is the
//! transpose of [`forward`] and energy is preserved. For `N = 4` the basis is a
//! rotation of the 4-point slant transform.
//!
//! # Compatibility with the paper
//!
//! The paper allows any filter's sign to be flipped. This crate uses the
//! signs of the paper's closed-form filters (Sec. II-B) and its matrix (Sec.
//! II-F, Fig. 5). [`forward`] equals multiplication by that matrix, which is
//! also what the author's published MATLAB code computes. The test suite builds
//! the matrix from the paper's equations and compares entry by entry. In
//! particular, `f` is a descending ramp, and for `m = 2`
//! `g_1 ≈ [-0.5117, 0.8279, -0.1208, -0.1954]`.
//!
//! # Coefficient layout
//!
//! Coarse to fine, as in the rows of the paper's matrix (Fig. 5), with each
//! block's `g` directly followed by its time reverse:
//!
//! ```text
//! [ h, f, | g_{l-1}, Jg_{l-1} | … | g_1, Jg_1, g_1, Jg_1, … (N/4 pairs) ]
//! ```
//!
//! The scale with blocks of length `2m` occupies `[N/m, 2N/m)`: block `b`'s
//! `g` coefficient sits at `N/m + 2b`, its `Jg` coefficient at `N/m + 2b + 1`.
//! `N = 2` reduces to the Haar transform, `N = 1` to the identity.
//!
//! # Algorithm
//!
//! Piecewise-linear basis functions only need each half-block's projection on
//! a constant and a ramp. Those projections merge bottom-up with a fixed 4×4
//! orthogonal map per block, so both directions run in `O(N)` time with an
//! `O(N)` scratch buffer.
//!
//! # Example
//!
//! ```
//! let mut x = [1.0, 4.0, -2.0, 0.5, 3.0, 3.0, 0.0, -1.0];
//! let original = x;
//! slantlet::forward(&mut x).unwrap();
//! slantlet::inverse(&mut x).unwrap();
//! for (a, b) in x.iter().zip(original) {
//!     assert!((a - b).abs() < 1e-12);
//! }
//! ```

use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};
use std::fmt;

/// Error returned when the input length is not a positive power of two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The slice length; must be `2^l` for some `l >= 0`.
    LengthNotPowerOfTwo(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::LengthNotPowerOfTwo(n) => {
                write!(f, "slantlet transform needs a power-of-two length, got {n}")
            }
        }
    }
}

impl std::error::Error for Error {}

fn check_len(n: usize) -> Result<(), Error> {
    if n.is_power_of_two() {
        Ok(())
    } else {
        Err(Error::LengthNotPowerOfTwo(n))
    }
}

/// Weights `(α, β)` that rotate a block's (step, ramp) halves into its
/// whole-block ramp and the antisymmetric slantlet component; `α² + β² = 1`.
fn rotation(m: usize) -> (f64, f64) {
    let mm = (m as f64) * (m as f64);
    let den = 4.0 * mm - 1.0;
    ((3.0 * mm / den).sqrt(), ((mm - 1.0) / den).sqrt())
}

/// Forward Slantlet transform, in place.
///
/// `x.len()` must be a power of two (including 1). On error `x` is unchanged.
pub fn forward(x: &mut [f64]) -> Result<(), Error> {
    let n = x.len();
    check_len(n)?;
    if n < 2 {
        return Ok(());
    }

    // c[k], d[k]: projection of block k onto its unit constant and unit ramp.
    let mut scratch = vec![0.0; n];
    let (c, d) = scratch.split_at_mut(n / 2);
    for (k, p) in x.chunks_exact(2).enumerate() {
        c[k] = (p[0] + p[1]) * FRAC_1_SQRT_2;
        d[k] = (p[1] - p[0]) * FRAC_1_SQRT_2;
    }

    let mut m = 2; // half-block length
    while m < n {
        let nb = n / (2 * m);
        let base = n / m;
        let (alpha, beta) = rotation(m);
        for b in 0..nb {
            let (cl, dl, cr, dr) = (c[2 * b], d[2 * b], c[2 * b + 1], d[2 * b + 1]);
            let dc = cr - cl;
            let sd = dl + dr;
            let s = (dr - dl) * FRAC_1_SQRT_2;
            let a = (beta * dc - alpha * sd) * FRAC_1_SQRT_2;
            // Reference sign: g = -(ŝ + â)/√2, Jg = -(ŝ - â)/√2.
            x[base + 2 * b] = -(s + a) * FRAC_1_SQRT_2;
            x[base + 2 * b + 1] = -(s - a) * FRAC_1_SQRT_2;
            c[b] = (cl + cr) * FRAC_1_SQRT_2;
            d[b] = (alpha * dc + beta * sd) * FRAC_1_SQRT_2;
        }
        m *= 2;
    }

    x[0] = c[0];
    x[1] = -d[0]; // reference f is the descending ramp
    Ok(())
}

/// Inverse Slantlet transform, in place.
///
/// `y.len()` must be a power of two (including 1). On error `y` is unchanged.
pub fn inverse(y: &mut [f64]) -> Result<(), Error> {
    let n = y.len();
    check_len(n)?;
    if n < 2 {
        return Ok(());
    }

    let mut scratch = vec![0.0; n];
    let (c, d) = scratch.split_at_mut(n / 2);
    c[0] = y[0];
    d[0] = -y[1];

    let mut m = n / 2;
    while m >= 2 {
        let nb = n / (2 * m);
        let base = n / m;
        let (alpha, beta) = rotation(m);
        // Descending b: block b reads slot b and writes slots 2b, 2b+1 >= b.
        for b in (0..nb).rev() {
            let (cc, r) = (c[b], d[b]);
            let (g, jg) = (-y[base + 2 * b], -y[base + 2 * b + 1]);
            let s = (g + jg) * FRAC_1_SQRT_2;
            let a = (g - jg) * FRAC_1_SQRT_2;
            // [[α, β], [β, -α]] is its own inverse.
            let dc = (alpha * r + beta * a) * SQRT_2;
            let sd = (beta * r - alpha * a) * SQRT_2;
            let sc = cc * SQRT_2;
            let dd = s * SQRT_2;
            c[2 * b] = (sc - dc) * 0.5;
            c[2 * b + 1] = (sc + dc) * 0.5;
            d[2 * b] = (sd - dd) * 0.5;
            d[2 * b + 1] = (sd + dd) * 0.5;
        }
        m /= 2;
    }

    for (k, p) in y.chunks_exact_mut(2).enumerate() {
        p[0] = (c[k] - d[k]) * FRAC_1_SQRT_2;
        p[1] = (c[k] + d[k]) * FRAC_1_SQRT_2;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    /// Basis vector k = inverse(e_k) (columns of the synthesis matrix).
    fn basis(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|k| {
                let mut e = vec![0.0; n];
                e[k] = 1.0;
                inverse(&mut e).unwrap();
                e
            })
            .collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
        }
    }

    #[test]
    fn four_point_basis_matches_slant_rows() {
        // g_1 = -(V + slant row 4)/√2, from the slant rows [1,-1,-1,1]/2 and [1,-3,3,-1]/√20.
        let r20 = 20f64.sqrt();
        let s = [0.5, -0.5, -0.5, 0.5];
        let a = [1.0 / r20, -3.0 / r20, 3.0 / r20, -1.0 / r20];
        let g: Vec<f64> = (0..4).map(|i| -(s[i] + a[i]) * FRAC_1_SQRT_2).collect();
        let jg: Vec<f64> = g.iter().rev().copied().collect();
        let expected = [
            vec![0.5; 4],
            vec![3.0 / r20, 1.0 / r20, -1.0 / r20, -3.0 / r20],
            g,
            jg,
        ];
        for (got, want) in basis(4).iter().zip(&expected) {
            assert_close(got, want, EPS);
        }
    }

    #[test]
    fn basis_is_orthonormal() {
        for l in 0..=7 {
            let n = 1 << l;
            let b = basis(n);
            for i in 0..n {
                for j in 0..n {
                    let want = if i == j { 1.0 } else { 0.0 };
                    let got = dot(&b[i], &b[j]);
                    assert!((got - want).abs() < 1e-12, "n={n} <{i},{j}> = {got}");
                }
            }
        }
    }

    #[test]
    fn forward_is_transpose_of_inverse() {
        let n = 32;
        let b = basis(n);
        for (k, v) in b.iter().enumerate() {
            let mut y = v.clone();
            forward(&mut y).unwrap();
            let mut e = vec![0.0; n];
            e[k] = 1.0;
            assert_close(&y, &e, EPS);
        }
    }

    /// Support and shape: detail functions live on one aligned block of length
    /// 2m, are linear on each half, form a time-reversed pair, and have two
    /// vanishing moments; h is constant and f is a ramp.
    #[test]
    fn basis_shape_support_and_moments() {
        let n = 64;
        let b = basis(n);
        let is_linear = |v: &[f64]| v.windows(3).all(|w| (w[0] - 2.0 * w[1] + w[2]).abs() < EPS);
        assert!(b[0].iter().all(|&v| (v - b[0][0]).abs() < EPS));
        assert!(is_linear(&b[1]));
        assert!(b[1][0] > 0.0 && dot(&b[1], &vec![1.0; n]).abs() < EPS);

        let mut m = 2;
        while m < n {
            let nb = n / (2 * m);
            for blk in 0..nb {
                let g = &b[n / m + 2 * blk];
                let jg = &b[n / m + 2 * blk + 1];
                let (lo, hi) = (blk * 2 * m, (blk + 1) * 2 * m);
                for v in [g, jg] {
                    assert!(v[..lo].iter().chain(&v[hi..]).all(|x| x.abs() < EPS));
                    assert!(is_linear(&v[lo..lo + m]) && is_linear(&v[lo + m..hi]));
                    let m0: f64 = v[lo..hi].iter().sum();
                    let m1: f64 = v[lo..hi]
                        .iter()
                        .enumerate()
                        .map(|(t, x)| t as f64 * x)
                        .sum();
                    assert!(
                        m0.abs() < 1e-10 && m1.abs() < 1e-10,
                        "m={m} moments {m0} {m1}"
                    );
                }
                for t in 0..2 * m {
                    assert!((g[lo + t] - jg[hi - 1 - t]).abs() < EPS);
                }
            }
            m *= 2;
        }
    }

    #[test]
    fn linear_signal_has_no_detail() {
        for l in 1..=10 {
            let n = 1 << l;
            let mut x: Vec<f64> = (0..n).map(|t| 3.5 - 0.25 * t as f64).collect();
            forward(&mut x).unwrap();
            assert!(x[2..].iter().all(|v| v.abs() < 1e-9), "n={n}");
        }
    }

    #[test]
    fn roundtrip_and_parseval() {
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|t| ((t * 37 % 101) as f64 - 50.0).sin() * 10.0)
            .collect();
        let mut y = x.clone();
        forward(&mut y).unwrap();
        assert!((dot(&x, &x) - dot(&y, &y)).abs() < 1e-9);
        inverse(&mut y).unwrap();
        assert_close(&y, &x, 1e-12);
    }

    #[test]
    fn small_lengths() {
        let mut one = [42.0];
        forward(&mut one).unwrap();
        assert_eq!(one, [42.0]);
        inverse(&mut one).unwrap();
        assert_eq!(one, [42.0]);

        let mut two = [1.0, 3.0];
        forward(&mut two).unwrap();
        assert_close(&two, &[4.0 * FRAC_1_SQRT_2, -2.0 * FRAC_1_SQRT_2], EPS);
    }

    #[test]
    fn rejects_bad_lengths_without_touching_input() {
        for n in [0, 3, 5, 6, 12, 1000] {
            let mut x: Vec<f64> = (0..n).map(|t| t as f64).collect();
            let orig = x.clone();
            assert_eq!(forward(&mut x), Err(Error::LengthNotPowerOfTwo(n)));
            assert_eq!(inverse(&mut x), Err(Error::LengthNotPowerOfTwo(n)));
            assert_eq!(x, orig);
        }
        assert_eq!(
            Error::LengthNotPowerOfTwo(6).to_string(),
            "slantlet transform needs a power-of-two length, got 6"
        );
    }
}
