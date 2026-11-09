//! Non-separable ("planar") 2-D slantlet transform on square images.
//!
//! Selesnick's paper (Sec. II-L) outlines a 2-D slantlet filterbank built from
//! separable filters without being separable itself, whose detail filters
//! annihilate planes `a·n1 + b·n2 + c` but not `n1·n2`, on square supports of
//! side `2^(i+1)`. The details were never published; this module constructs
//! such a basis from the 1-D one.
//!
//! # Construction
//!
//! Every aligned square block carries its projections on the unit constant and
//! the unit x- and y-ramps over the block (its best-fit plane). Four child
//! blocks of side `m` merge into a parent of side `2m`. With the 1-D half-block
//! functions — constant `Ĉ`, step `σ̂`, equal-slope local ramps `ρ̂`,
//! opposite-slope local ramps `ŝ`, and the slantlet pair built from `ŝ` and `â`
//! — the 12-dimensional space of quadrant-wise planar functions splits as
//! `span{Ĉ,σ̂}⊗span{Ĉ,σ̂} ⊕ span{ρ̂,ŝ}⊗span{Ĉ,σ̂} ⊕ span{Ĉ,σ̂}⊗span{ρ̂,ŝ}`
//! (x factor first). The parent plane takes 3 dimensions; the 9 details are
//!
//! | # | function (x ⊗ y) | meaning |
//! |---|---|---|
//! | 0, 1 | `g ⊗ Ĉ`, `Jg ⊗ Ĉ` | 1-D slantlet pair along x, constant in y |
//! | 2, 3 | `Ĉ ⊗ g`, `Ĉ ⊗ Jg` | 1-D slantlet pair along y, constant in x |
//! | 4 | `σ̂ ⊗ σ̂` | quadrant checkerboard |
//! | 5 | `ρ̂ ⊗ σ̂` | equal-slope x-ramps, opposite sign top/bottom |
//! | 6 | `ŝ ⊗ σ̂` | opposite-slope x-ramps, opposite sign top/bottom |
//! | 7 | `σ̂ ⊗ ρ̂` | equal-slope y-ramps, opposite sign left/right |
//! | 8 | `σ̂ ⊗ ŝ` | opposite-slope y-ramps, opposite sign left/right |
//!
//! All are products of 1-D functions and orthogonal to `1`, `n1`, `n2`. At
//! the finest level (2×2 pixels) only the checkerboard exists. Each merge is
//! three 2×2 Haar butterflies plus two of the 1-D slantlet rotations, so the
//! transform is `O(N²)` for an `N × N` image.
//!
//! The choice of the 9 detail functions within their span is a convention of
//! this crate (there is no published reference); it is pinned by tests.

use std::f64::consts::FRAC_1_SQRT_2;

use crate::{Error, check_dims, check_len, rotation};

/// Non-separable 2-D slantlet forward transform of a row-major `side × side`
/// image, in place.
///
/// The output is a flat coefficient vector, coarse to fine:
///
/// - `[0..3)`: the image's plane — constant, then x-ramp and y-ramp (both
///   descending, as `f` in 1-D).
/// - For block side `2m = side, side/2, …, 4`: with `t = side / (2m)` blocks
///   per row, block `(by, bx)` stores its 9 details (see the table in the
///   module docs) at `3t² + 9(by·t + bx)`.
/// - `[3·side²/4 ..)`: one checkerboard coefficient per 2×2 pixel block, in
///   row-major block order.
///
/// Detail coefficients annihilate planes. On an image constant along y,
/// details 0 and 1 equal `sqrt(2m)` times the 1-D [`forward`](crate::forward)
/// coefficients of a row, and all other details vanish.
///
/// `side` must be a power of two and `x.len() == side²`; otherwise an error is
/// returned and `x` is unchanged.
///
/// ```
/// let side = 8;
/// // A plane: all detail coefficients vanish.
/// let mut img: Vec<f64> = (0..side * side)
///     .map(|i| 1.0 + 0.5 * (i % side) as f64 - 0.25 * (i / side) as f64)
///     .collect();
/// slantlet::forward_2d_planar(&mut img, side)?;
/// assert!(img[3..].iter().all(|c| c.abs() < 1e-12));
/// slantlet::inverse_2d_planar(&mut img, side)?;
/// # Ok::<(), slantlet::Error>(())
/// ```
pub fn forward_2d_planar(x: &mut [f64], side: usize) -> Result<(), Error> {
    check_dims(x.len(), side, side)?;
    check_len(side)?;
    if side < 2 {
        return Ok(());
    }
    let n = side;
    let q = n * n / 4;
    let mut scratch = vec![0.0; 4 * q];
    let (c, rest) = scratch.split_at_mut(q);
    let (xr, rest) = rest.split_at_mut(q);
    let (yr, dd) = rest.split_at_mut(q);

    // 2×2 pixel blocks: plane (c, x-ramp, y-ramp) and checkerboard.
    let k = n / 2;
    for by in 0..k {
        for bx in 0..k {
            let i = 2 * by * n + 2 * bx;
            let h = haar([x[i], x[i + 1], x[i + n], x[i + n + 1]]);
            let b = by * k + bx;
            (c[b], xr[b], yr[b], dd[b]) = (h[0], h[1], h[2], h[3]);
        }
    }
    x[n * n - q..].copy_from_slice(dd);

    let mut m = 2; // child block side
    while m < n {
        let k = n / m; // children per row
        let t = k / 2; // parents per row
        let (alpha, beta) = rotation(m);
        // Ascending parents: parent p is stored at or before its first child.
        for p in 0..t * t {
            let ch = children(p, t);
            let hc = haar(ch.map(|i| c[i]));
            let hx = haar(ch.map(|i| xr[i]));
            let hy = haar(ch.map(|i| yr[i]));
            let ax = beta * hc[1] - alpha * hx[0];
            let ay = beta * hc[2] - alpha * hy[0];
            let (sx, sy) = (hx[1], hy[2]);
            let o = 3 * t * t + 9 * p;
            x[o..o + 9].copy_from_slice(&[
                -(sx + ax) * FRAC_1_SQRT_2,
                -(sx - ax) * FRAC_1_SQRT_2,
                -(sy + ay) * FRAC_1_SQRT_2,
                -(sy - ay) * FRAC_1_SQRT_2,
                hc[3],
                hx[2],
                hx[3],
                hy[1],
                hy[3],
            ]);
            c[p] = hc[0];
            xr[p] = alpha * hc[1] + beta * hx[0];
            yr[p] = alpha * hc[2] + beta * hy[0];
        }
        m *= 2;
    }

    x[0] = c[0];
    x[1] = -xr[0];
    x[2] = -yr[0];
    Ok(())
}

/// Inverse of [`forward_2d_planar`], in place.
pub fn inverse_2d_planar(y: &mut [f64], side: usize) -> Result<(), Error> {
    check_dims(y.len(), side, side)?;
    check_len(side)?;
    if side < 2 {
        return Ok(());
    }
    let n = side;
    let q = n * n / 4;
    let mut scratch = vec![0.0; 4 * q];
    let (c, rest) = scratch.split_at_mut(q);
    let (xr, rest) = rest.split_at_mut(q);
    let (yr, dd) = rest.split_at_mut(q);
    dd.copy_from_slice(&y[n * n - q..]);
    c[0] = y[0];
    xr[0] = -y[1];
    yr[0] = -y[2];

    let mut m = n / 2;
    while m >= 2 {
        let k = n / m;
        let t = k / 2;
        let (alpha, beta) = rotation(m);
        // Descending parents: children are stored at or after their parent.
        for p in (0..t * t).rev() {
            let o = 3 * t * t + 9 * p;
            let d = &y[o..o + 9];
            let sx = -(d[0] + d[1]) * FRAC_1_SQRT_2;
            let ax = -(d[0] - d[1]) * FRAC_1_SQRT_2;
            let sy = -(d[2] + d[3]) * FRAC_1_SQRT_2;
            let ay = -(d[2] - d[3]) * FRAC_1_SQRT_2;
            let (pc, px, py) = (c[p], xr[p], yr[p]);
            // [[α, β], [β, -α]] is its own inverse.
            let hc = [pc, alpha * px + beta * ax, alpha * py + beta * ay, d[4]];
            let hx = [beta * px - alpha * ax, sx, d[5], d[6]];
            let hy = [beta * py - alpha * ay, d[7], sy, d[8]];
            let (vc, vx, vy) = (unhaar(hc), unhaar(hx), unhaar(hy));
            for (j, &i) in children(p, t).iter().enumerate() {
                (c[i], xr[i], yr[i]) = (vc[j], vx[j], vy[j]);
            }
        }
        m /= 2;
    }

    let k = n / 2;
    for by in 0..k {
        for bx in 0..k {
            let b = by * k + bx;
            let v = unhaar([c[b], xr[b], yr[b], dd[b]]);
            let i = 2 * by * n + 2 * bx;
            (y[i], y[i + 1], y[i + n], y[i + n + 1]) = (v[0], v[1], v[2], v[3]);
        }
    }
    Ok(())
}

/// Indices of parent `p`'s children (top-left, top-right, bottom-left,
/// bottom-right) in a grid with `2t` blocks per row.
fn children(p: usize, t: usize) -> [usize; 4] {
    let (py, px) = (p / t, p % t);
    let k = 2 * t;
    let i = 2 * py * k + 2 * px;
    [i, i + 1, i + k, i + k + 1]
}

/// 2×2 Haar of `[top-left, top-right, bottom-left, bottom-right]`:
/// `[sum, x-difference, y-difference, checkerboard]`, orthonormal.
fn haar([a, b, c, d]: [f64; 4]) -> [f64; 4] {
    [
        0.5 * (a + b + c + d),
        0.5 * (-a + b - c + d),
        0.5 * (-a - b + c + d),
        0.5 * (a - b - c + d),
    ]
}

/// Inverse (= transpose) of [`haar`].
fn unhaar([s, dx, dy, dd]: [f64; 4]) -> [f64; 4] {
    [
        0.5 * (s - dx - dy + dd),
        0.5 * (s + dx - dy - dd),
        0.5 * (s - dx + dy - dd),
        0.5 * (s + dx + dy + dd),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn basis(n: usize) -> Vec<Vec<f64>> {
        (0..n * n)
            .map(|k| {
                let mut e = vec![0.0; n * n];
                e[k] = 1.0;
                inverse_2d_planar(&mut e, n).unwrap();
                e
            })
            .collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn basis_is_orthonormal_and_forward_is_its_transpose() {
        for n in [1, 2, 4, 8, 16] {
            let b = basis(n);
            for i in 0..n * n {
                for j in i..n * n {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((dot(&b[i], &b[j]) - want).abs() < EPS, "n={n} <{i},{j}>");
                }
                let mut y = b[i].clone();
                forward_2d_planar(&mut y, n).unwrap();
                for (j, v) in y.iter().enumerate() {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((v - want).abs() < EPS, "n={n} forward(b{i})[{j}] = {v}");
                }
            }
        }
    }

    /// Paper Sec. II-L: details annihilate a·n1 + b·n2 + c but not n1·n2, and
    /// each lives on one aligned square, planar on each quadrant.
    #[test]
    fn details_are_local_quadrant_planar_and_annihilate_planes() {
        let n = 16;
        let b = basis(n);
        let at = |v: &[f64], r: usize, c: usize| v[r * n + c];
        let mut saddle_seen = false;
        for (idx, v) in b.iter().enumerate().skip(3) {
            let m0: f64 = v.iter().sum();
            let m1: f64 = v.iter().enumerate().map(|(i, x)| (i % n) as f64 * x).sum();
            let m2: f64 = v.iter().enumerate().map(|(i, x)| (i / n) as f64 * x).sum();
            assert!(
                m0.abs() < 1e-10 && m1.abs() < 1e-10 && m2.abs() < 1e-10,
                "#{idx}"
            );
            let m12: f64 = v
                .iter()
                .enumerate()
                .map(|(i, x)| ((i % n) * (i / n)) as f64 * x)
                .sum();
            saddle_seen |= m12.abs() > 1e-6;

            // Support: bounding box of nonzeros is an aligned square of side 2^j.
            let nz: Vec<usize> = (0..n * n).filter(|&i| v[i].abs() > EPS).collect();
            let (r0, r1) = (
                nz.iter().map(|i| i / n).min().unwrap(),
                nz.iter().map(|i| i / n).max().unwrap(),
            );
            let (c0, c1) = (
                nz.iter().map(|i| i % n).min().unwrap(),
                nz.iter().map(|i| i % n).max().unwrap(),
            );
            let s = (r1 - r0 + 1).max(c1 - c0 + 1).next_power_of_two();
            let (r0, c0) = (r0 / s * s, c0 / s * s);
            assert!(
                r1 < r0 + s && c1 < c0 + s,
                "#{idx} not on an aligned square"
            );
            // Planar on each quadrant: all second differences vanish there.
            let h = (s / 2).max(1);
            for qr in [r0, r0 + h] {
                for qc in [c0, c0 + h] {
                    for r in qr..qr + h {
                        for c in qc..qc + h {
                            if c + 2 < qc + h {
                                let d = at(v, r, c) - 2.0 * at(v, r, c + 1) + at(v, r, c + 2);
                                assert!(d.abs() < EPS, "#{idx} not planar in x");
                            }
                            if r + 2 < qr + h {
                                let d = at(v, r, c) - 2.0 * at(v, r + 1, c) + at(v, r + 2, c);
                                assert!(d.abs() < EPS, "#{idx} not planar in y");
                            }
                            if c + 1 < qc + h && r + 1 < qr + h {
                                let d = at(v, r, c) - at(v, r, c + 1) - at(v, r + 1, c)
                                    + at(v, r + 1, c + 1);
                                assert!(d.abs() < EPS, "#{idx} has a twist term");
                            }
                        }
                    }
                }
            }
        }
        assert!(saddle_seen, "some detail must respond to n1·n2");
    }

    /// An image constant along y reduces to the 1-D transform of a row.
    #[test]
    fn y_constant_image_reduces_to_1d() {
        let n = 32;
        let row: Vec<f64> = (0..n).map(|i| ((i * 7 % 13) as f64).sqrt()).collect();
        let mut img: Vec<f64> = (0..n * n).map(|i| row[i % n]).collect();
        forward_2d_planar(&mut img, n).unwrap();
        let mut one = row.clone();
        crate::forward(&mut one).unwrap();

        let nf = n as f64;
        assert!((img[0] - nf.sqrt() * one[0]).abs() < 1e-10);
        assert!((img[1] - nf.sqrt() * one[1]).abs() < 1e-10);
        assert!(img[2].abs() < 1e-10);
        let mut m = n / 2;
        while m >= 2 {
            let t = n / (2 * m);
            for by in 0..t {
                for bx in 0..t {
                    let o = 3 * t * t + 9 * (by * t + bx);
                    let scale = ((2 * m) as f64).sqrt();
                    assert!(
                        (img[o] - scale * one[n / m + 2 * bx]).abs() < 1e-10,
                        "m={m}"
                    );
                    assert!((img[o + 1] - scale * one[n / m + 2 * bx + 1]).abs() < 1e-10);
                    assert!(img[o + 2..o + 9].iter().all(|v| v.abs() < 1e-10), "m={m}");
                }
            }
            m /= 2;
        }
        assert!(img[3 * n * n / 4..].iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn rejects_bad_shapes_without_touching_input() {
        let mut x: Vec<f64> = (0..36).map(f64::from).collect();
        let orig = x.clone();
        assert_eq!(
            forward_2d_planar(&mut x, 6),
            Err(Error::LengthNotPowerOfTwo(6))
        );
        assert_eq!(
            inverse_2d_planar(&mut x, 6),
            Err(Error::LengthNotPowerOfTwo(6))
        );
        let mismatch = Error::DimensionMismatch {
            len: 36,
            width: 4,
            height: 4,
        };
        assert_eq!(forward_2d_planar(&mut x, 4), Err(mismatch));
        assert_eq!(x, orig);
    }
}
