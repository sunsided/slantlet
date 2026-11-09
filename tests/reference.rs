//! Cross-check against the slantlet matrix as defined in the paper.
//!
//! I. W. Selesnick, "The Slantlet Transform", IEEE Trans. Signal Processing
//! 47(5):1304–1313, 1999. `g_filter` and `hf_filters` implement the closed-form
//! filter coefficients of Sec. II-B (p. 1307); `paper_matrix` assembles the
//! periodized 2^l × 2^l matrix of Sec. II-F, eqs. (9)–(11), with the row order
//! of Fig. 5. This is a dense O(N²) construction, independent of the crate's
//! O(N) algorithm. The golden values in `golden/` pin both.
//!
//! Note: eq. (12) prints the ramp slope as `2·sqrt(m / (m(m²−1)))`; the folded
//! filter of eq. (11) actually gives `2·sqrt(3 / (m(m²−1)))`, which is the
//! unit-norm value (checked by `paper_eq12_typo`).

mod golden;

/// `p(n)` for `n < m`, `q(n - m)` for `m <= n < 2m`, with `p`, `q` linear.
fn piecewise(m: usize, p0: f64, p1: f64, q0: f64, q1: f64) -> Vec<f64> {
    let first = (0..m).map(|k| p0 + p1 * k as f64);
    let second = (0..m).map(|k| q0 + q1 * k as f64);
    first.chain(second).collect()
}

/// `g_i(n)`, length 2^(i+1) (Sec. II-B, parameters a_{0,0} … a_{1,1}).
fn g_filter(i: u32) -> Vec<f64> {
    let m = f64::from(1u32 << i);
    let s1 = 6.0 * (m / ((m * m - 1.0) * (4.0 * m * m - 1.0))).sqrt();
    let t1 = 2.0 * (3.0 / (m * (m * m - 1.0))).sqrt();
    let s0 = -s1 * (m - 1.0) / 2.0;
    let t0 = ((m + 1.0) * s1 / 3.0 - m * t1) * (m - 1.0) / (2.0 * m);
    let a00 = (s0 + t0) / 2.0;
    let a10 = (s0 - t0) / 2.0;
    let a01 = (s1 + t1) / 2.0;
    let a11 = (s1 - t1) / 2.0;
    piecewise(1 << i, a00, a01, a10, a11)
}

/// `(h_i(n), f_i(n))`, length 2^(i+1) (Sec. II-B, parameters b_{·,·}, c_{·,·}).
fn hf_filters(i: u32) -> (Vec<f64>, Vec<f64>) {
    let m = f64::from(1u32 << i);
    let u = 1.0 / m.sqrt();
    let v = ((2.0 * m * m + 1.0) / 3.0).sqrt();
    let b00 = u * (v + 1.0) / (2.0 * m);
    let b10 = u - b00;
    let b01 = u / m;
    let b11 = -b01;
    let q = (3.0 / (m * (m * m - 1.0))).sqrt() / m;
    let c01 = q * (v - m);
    let c11 = -q * (v + m);
    let c10 = c11 * (v + 1.0 - 2.0 * m) / 2.0;
    let c00 = c01 * (v + 1.0) / 2.0;
    let len = 1 << i;
    (
        piecewise(len, b00, b01, b10, b11),
        piecewise(len, c00, c01, c10, c11),
    )
}

/// The periodized 2^l × 2^l slantlet matrix, row-major (Sec. II-F, Fig. 5):
/// rows h_l and f_l folded onto N samples (eqs. 9, 11), then for i = l-1 down
/// to 1 and each shift k: g_i(n - 2^(i+1)k) followed by its time reverse.
fn paper_matrix(l: u32) -> Vec<Vec<f64>> {
    let n = 1usize << l;
    let mut t = vec![vec![0.0; n]; n];
    let (h, f) = hf_filters(l);
    for j in 0..n {
        t[0][j] = h[j] + h[j + n];
        t[1][j] = f[j] + f[j + n];
    }
    let mut row = 2;
    for i in (1..l).rev() {
        let g = g_filter(i);
        let len = g.len();
        for k in 0..n / len {
            for (j, &gj) in g.iter().enumerate() {
                t[row][len * k + j] = gj;
                t[row + 1][len * k + len - 1 - j] = gj;
            }
            row += 2;
        }
    }
    t
}

#[test]
fn forward_matches_reference_matrix() {
    for l in 1..=9 {
        let n = 1usize << l;
        let t = paper_matrix(l);
        for j in 0..n {
            let mut col = vec![0.0; n];
            col[j] = 1.0;
            slantlet::forward(&mut col).unwrap();
            for i in 0..n {
                assert!(
                    (col[i] - t[i][j]).abs() < 1e-12,
                    "N={n} T[{i}][{j}]: crate {} vs reference {}",
                    col[i],
                    t[i][j]
                );
            }
        }
    }
}

#[test]
fn inverse_matches_reference_transpose() {
    for l in 1..=9 {
        let n = 1usize << l;
        let t = paper_matrix(l);
        for i in 0..n {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            slantlet::inverse(&mut row).unwrap();
            for j in 0..n {
                assert!((row[j] - t[i][j]).abs() < 1e-12, "N={n} row {i} col {j}");
            }
        }
    }
}

/// Eq. (12) as printed is not unit norm; the slope 2·sqrt(3/(m(m²−1))) is.
#[test]
fn paper_eq12_typo() {
    for l in 2..=8 {
        let m = f64::from(1u32 << l);
        let c = (3.0 * (m - 1.0) / (m * (m + 1.0))).sqrt();
        let printed = 2.0 * (m / (m * (m * m - 1.0))).sqrt();
        let fixed = 2.0 * (3.0 / (m * (m * m - 1.0))).sqrt();
        let norm2 = |slope: f64| {
            (0..1u32 << l)
                .map(|n| (c - slope * f64::from(n)).powi(2))
                .sum::<f64>()
        };
        assert!((norm2(fixed) - 1.0).abs() < 1e-12);
        assert!((norm2(printed) - 1.0).abs() > 0.1);
        let row = &paper_matrix(l)[1];
        for (n, v) in row.iter().enumerate() {
            assert!((v - (c - fixed * n as f64)).abs() < 1e-12);
        }
    }
}

/// The paper's 4-point example: rows h, f, g_1, reversed g_1.
#[test]
fn reference_n4_values() {
    let t = paper_matrix(2);
    let r20 = 20f64.sqrt();
    let want_f = [3.0 / r20, 1.0 / r20, -1.0 / r20, -3.0 / r20];
    for j in 0..4 {
        assert!((t[0][j] - 0.5).abs() < 1e-15);
        assert!((t[1][j] - want_f[j]).abs() < 1e-15);
    }
    // Closed form: -([1,-1,-1,1]/2 + [1,-3,3,-1]/√20)/√2.
    let g = [
        -0.511_667_273_601_692_7,
        0.827_895_039_618_530_7,
        -0.120_788_258_431_983_1,
        -0.195_439_507_584_854_8,
    ];
    for j in 0..4 {
        assert!((t[2][j] - g[j]).abs() < 1e-12, "g[{j}] = {}", t[2][j]);
        assert!((t[3][j] - g[3 - j]).abs() < 1e-12);
    }
}

/// Guard against a broken reference: the paper matrix itself must be orthonormal, and
/// every row except h must annihilate constants; every g row also ramps.
#[test]
fn reference_matrix_is_a_valid_slantlet_basis() {
    for l in 1..=8 {
        let n = 1usize << l;
        let t = paper_matrix(l);
        for i in 0..n {
            for k in 0..n {
                let d: f64 = t[i].iter().zip(&t[k]).map(|(a, b)| a * b).sum();
                let want = if i == k { 1.0 } else { 0.0 };
                assert!((d - want).abs() < 1e-12, "N={n} <{i},{k}> = {d}");
            }
            let m0: f64 = t[i].iter().sum();
            let m1: f64 = t[i].iter().enumerate().map(|(j, v)| j as f64 * v).sum();
            if i >= 1 {
                assert!(m0.abs() < 1e-10, "N={n} row {i} sum {m0}");
            }
            if i >= 2 {
                assert!(m1.abs() < 1e-9, "N={n} row {i} first moment {m1}");
            }
        }
    }
}

/// Signals of awkward magnitude still agree with the reference matrix product.
#[test]
fn forward_matches_reference_on_signal() {
    let l = 10;
    let n = 1usize << l;
    let t = paper_matrix(l);
    let x: Vec<f64> = (0..n)
        .map(|j| ((j as f64) / 3.0).sin().sin() * 1e3)
        .collect();
    let want: Vec<f64> = t
        .iter()
        .map(|r| r.iter().zip(&x).map(|(a, b)| a * b).sum())
        .collect();
    let mut got = x.clone();
    slantlet::forward(&mut got).unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-9, "{g} vs {w}");
    }
}

fn assert_matrix<const N: usize>(got: &[Vec<f64>], want: &[[f64; N]; N], what: &str) {
    for i in 0..N {
        for j in 0..N {
            assert!(
                (got[i][j] - want[i][j]).abs() < 1e-14,
                "{what} N={N} [{i}][{j}]: {} vs golden {}",
                got[i][j],
                want[i][j]
            );
        }
    }
}

/// Matrix of `forward`: row i, column j = forward(e_j)[i].
fn crate_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut col = vec![0.0; n];
        col[j] = 1.0;
        slantlet::forward(&mut col).unwrap();
        for i in 0..n {
            t[i][j] = col[i];
        }
    }
    t
}

#[test]
fn golden_pins_crate() {
    assert_matrix(&crate_matrix(4), &golden::MATRIX_4, "crate");
    assert_matrix(&crate_matrix(8), &golden::MATRIX_8, "crate");
    assert_matrix(&crate_matrix(16), &golden::MATRIX_16, "crate");
    let mut y = golden::sig64_input();
    slantlet::forward(&mut y).unwrap();
    for (i, (a, b)) in y.iter().zip(golden::SIG64_COEFFS).enumerate() {
        assert!((a - b).abs() < 1e-13, "sig64[{i}]: {a} vs golden {b}");
    }
}

#[test]
fn golden_pins_reference() {
    assert_matrix(&paper_matrix(2), &golden::MATRIX_4, "reference");
    assert_matrix(&paper_matrix(3), &golden::MATRIX_8, "reference");
    assert_matrix(&paper_matrix(4), &golden::MATRIX_16, "reference");
}
