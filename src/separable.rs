//! Separable 2-D slantlet transform: the 1-D transform along every row, then
//! along every column.

use crate::{Error, apply_pieces, check_dims, check_len, forward_with, inverse_with, scratch_len};

/// Separable 2-D forward transform of a row-major `width × height` array, in
/// place.
///
/// Every row is replaced by its [`forward`](crate::forward) transform, then
/// every column. The result is the tensor product of two 1-D slantlet bases:
/// coefficient `(r, c)` pairs basis function `r` of the column transform with
/// basis function `c` of the row transform, so the top-left corner holds the
/// coarse (constant and ramp) terms. Orthogonal; inverted by [`inverse_2d`].
///
/// This is the standard separable extension, not the non-separable 2-D
/// slantlet filterbank of the paper's Sec. II-L; for that see
/// [`forward_2d_planar`](crate::forward_2d_planar).
///
/// `width` and `height` must be powers of two and `x.len()` must equal
/// `width * height`; otherwise an error is returned and `x` is unchanged.
/// For other sizes use [`forward_2d_frames`].
pub fn forward_2d(x: &mut [f64], width: usize, height: usize) -> Result<(), Error> {
    check_dims(x.len(), width, height)?;
    check_len(width)?;
    check_len(height)?;
    apply(x, width, height, width, height, forward_with);
    Ok(())
}

/// Inverse of [`forward_2d`], in place.
pub fn inverse_2d(y: &mut [f64], width: usize, height: usize) -> Result<(), Error> {
    check_dims(y.len(), width, height)?;
    check_len(width)?;
    check_len(height)?;
    apply(y, width, height, width, height, inverse_with);
    Ok(())
}

/// Separable 2-D forward transform of a row-major `width × height` array of
/// any size, in place.
///
/// Each row, then each column, is transformed with
/// [`forward_frames`](crate::forward_frames) using `frame_len`, so both axes
/// are tiled into `frame_len` frames plus power-of-two remainder pieces.
/// Orthogonal and lossless; inverted by [`inverse_2d_frames`] with the same
/// `frame_len`.
///
/// Returns an error, leaving `x` unchanged, if `x.len() != width * height` or
/// `frame_len` is not a power of two.
///
/// ```
/// let (w, h) = (37, 19);
/// let mut img: Vec<f64> = (0..w * h).map(|i| (i % 7) as f64).collect();
/// let original = img.clone();
/// slantlet::forward_2d_frames(&mut img, w, h, 16)?;
/// slantlet::inverse_2d_frames(&mut img, w, h, 16)?;
/// assert!(img.iter().zip(&original).all(|(a, b)| (a - b).abs() < 1e-12));
/// # Ok::<(), slantlet::Error>(())
/// ```
pub fn forward_2d_frames(
    x: &mut [f64],
    width: usize,
    height: usize,
    frame_len: usize,
) -> Result<(), Error> {
    check_dims(x.len(), width, height)?;
    check_len(frame_len)?;
    apply(x, width, height, frame_len, frame_len, forward_with);
    Ok(())
}

/// Inverse of [`forward_2d_frames`] with the same `frame_len`, in place.
pub fn inverse_2d_frames(
    y: &mut [f64],
    width: usize,
    height: usize,
    frame_len: usize,
) -> Result<(), Error> {
    check_dims(y.len(), width, height)?;
    check_len(frame_len)?;
    apply(y, width, height, frame_len, frame_len, inverse_with);
    Ok(())
}

/// Applies the 1-D piece transform `f` along rows (frames of `frame_w`) and
/// columns (frames of `frame_h`). The two passes commute.
fn apply(
    x: &mut [f64],
    width: usize,
    height: usize,
    frame_w: usize,
    frame_h: usize,
    f: fn(&mut [f64], &mut [f64]),
) {
    if x.is_empty() {
        return;
    }
    #[cfg(feature = "rayon")]
    if crate::parallel(x.len(), crate::par_min::SEPARABLE) {
        apply_parallel(x, width, height, frame_w, frame_h, f);
        return;
    }
    let mut scratch = vec![0.0; scratch_len(frame_w.min(width).max(frame_h.min(height)))];
    for row in x.chunks_exact_mut(width) {
        apply_pieces(row, frame_w, f, &mut scratch);
    }
    // Columns in batches, so gathering and scattering read and write whole
    // row segments instead of striding through memory once per column.
    let batch = COLUMN_BATCH.min(width);
    let mut cols = vec![0.0; batch * height];
    for c0 in (0..width).step_by(batch) {
        let b = batch.min(width - c0);
        for (r, row) in x.chunks_exact(width).enumerate() {
            for (j, &v) in row[c0..c0 + b].iter().enumerate() {
                cols[j * height + r] = v;
            }
        }
        for col in cols[..b * height].chunks_exact_mut(height) {
            apply_pieces(col, frame_h, f, &mut scratch);
        }
        for (r, row) in x.chunks_exact_mut(width).enumerate() {
            for (j, v) in row[c0..c0 + b].iter_mut().enumerate() {
                *v = cols[j * height + r];
            }
        }
    }
}

/// [`apply`] on all cores: rows in parallel, then columns through a
/// transposed copy so every task reads and writes contiguous memory.
#[cfg(feature = "rayon")]
fn apply_parallel(
    x: &mut [f64],
    width: usize,
    height: usize,
    frame_w: usize,
    frame_h: usize,
    f: fn(&mut [f64], &mut [f64]),
) {
    use rayon::prelude::*;
    x.par_chunks_mut(width).for_each_init(
        || vec![0.0; scratch_len(frame_w.min(width))],
        |s, row| apply_pieces(row, frame_w, f, s),
    );
    // t holds the columns of x as rows; each task gathers COLUMN_BATCH of them.
    let mut t = vec![0.0; width * height];
    let xs = &*x;
    t.par_chunks_mut(COLUMN_BATCH * height)
        .enumerate()
        .for_each_init(
            || vec![0.0; scratch_len(frame_h.min(height))],
            |s, (i, cols)| {
                let c0 = i * COLUMN_BATCH;
                for (r, row) in xs.chunks_exact(width).enumerate() {
                    for (j, &v) in row[c0..c0 + cols.len() / height].iter().enumerate() {
                        cols[j * height + r] = v;
                    }
                }
                for col in cols.chunks_exact_mut(height) {
                    apply_pieces(col, frame_h, f, s);
                }
            },
        );
    // Scatter back COLUMN_BATCH rows per task, reading contiguous runs of t.
    x.par_chunks_mut(COLUMN_BATCH * width)
        .enumerate()
        .for_each(|(i, rows)| {
            let r0 = i * COLUMN_BATCH;
            let nr = rows.len() / width;
            for (c, col) in t.chunks_exact(height).enumerate() {
                for (k, &v) in col[r0..r0 + nr].iter().enumerate() {
                    rows[k * width + c] = v;
                }
            }
        });
}

/// Columns gathered per pass: 8 `f64` = one 64-byte cache line per row
/// segment. Measured best of 4–128 on the BENCH.md machine.
const COLUMN_BATCH: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense 1-D slantlet matrix (row i, column j = forward(e_j)[i]).
    fn matrix(n: usize) -> Vec<Vec<f64>> {
        let mut t = vec![vec![0.0; n]; n];
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            crate::forward(&mut e).unwrap();
            for i in 0..n {
                t[i][j] = e[i];
            }
        }
        t
    }

    /// forward_2d(X) = T_h · X · T_wᵀ for X stored row-major.
    #[test]
    fn equals_kronecker_product() {
        for (w, h) in [(1, 1), (4, 8), (8, 2), (1, 4), (16, 16)] {
            let x: Vec<f64> = (0..w * h).map(|i| ((i * 13 % 29) as f64).sin()).collect();
            let (tw, th) = (matrix(w), matrix(h));
            let mut want = vec![0.0; w * h];
            for r in 0..h {
                for c in 0..w {
                    let mut acc = 0.0;
                    for i in 0..h {
                        for j in 0..w {
                            acc += th[r][i] * x[i * w + j] * tw[c][j];
                        }
                    }
                    want[r * w + c] = acc;
                }
            }
            let mut got = x.clone();
            forward_2d(&mut got, w, h).unwrap();
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-12, "{w}x{h}: {a} vs {b}");
            }
            inverse_2d(&mut got, w, h).unwrap();
            for (a, b) in got.iter().zip(&x) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn frames_handle_any_size() {
        for (w, h, frame) in [(37, 19, 16), (5, 3, 1), (64, 33, 8), (0, 7, 4), (9, 1, 64)] {
            let x: Vec<f64> = (0..w * h).map(|i| (i as f64 * 0.31).cos() * 3.0).collect();
            let mut y = x.clone();
            forward_2d_frames(&mut y, w, h, frame).unwrap();
            let e0: f64 = x.iter().map(|v| v * v).sum();
            let e1: f64 = y.iter().map(|v| v * v).sum();
            assert!((e0 - e1).abs() < 1e-10 * (1.0 + e0), "{w}x{h}");
            inverse_2d_frames(&mut y, w, h, frame).unwrap();
            for (a, b) in y.iter().zip(&x) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        // Large enough for the parallel path (feature `rayon`): must match the
        // per-row / per-column definition bit for bit.
        let (w, h) = (300, 200);
        let x: Vec<f64> = (0..w * h).map(|i| ((i * 31 % 113) as f64).sin()).collect();
        let mut want = x.clone();
        for row in want.chunks_exact_mut(w) {
            crate::forward_frames(row, 64).unwrap();
        }
        for c in 0..w {
            let mut col: Vec<f64> = (0..h).map(|r| want[r * w + c]).collect();
            crate::forward_frames(&mut col, 64).unwrap();
            for (r, v) in col.into_iter().enumerate() {
                want[r * w + c] = v;
            }
        }
        let mut got = x.clone();
        forward_2d_frames(&mut got, w, h, 64).unwrap();
        assert_eq!(got, want);
        inverse_2d_frames(&mut got, w, h, 64).unwrap();
        assert!(got.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-12));

        // Power-of-two sizes with frames covering each axis equal forward_2d.
        let x: Vec<f64> = (0..32 * 8).map(|i| (i % 11) as f64).collect();
        let (mut a, mut b) = (x.clone(), x);
        forward_2d(&mut a, 32, 8).unwrap();
        forward_2d_frames(&mut b, 32, 8, 32).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_bad_shapes_without_touching_input() {
        let mut x: Vec<f64> = (0..12).map(f64::from).collect();
        let orig = x.clone();
        let mismatch = Error::DimensionMismatch {
            len: 12,
            width: 4,
            height: 4,
        };
        assert_eq!(forward_2d(&mut x, 4, 4), Err(mismatch));
        assert_eq!(inverse_2d(&mut x, 4, 4), Err(mismatch));
        assert_eq!(forward_2d(&mut x, 4, 3), Err(Error::LengthNotPowerOfTwo(3)));
        assert_eq!(
            forward_2d_frames(&mut x, 4, 3, 6),
            Err(Error::LengthNotPowerOfTwo(6))
        );
        assert_eq!(
            forward_2d_frames(&mut x, usize::MAX, 2, 4),
            Err(Error::DimensionMismatch {
                len: 12,
                width: usize::MAX,
                height: 2
            })
        );
        assert_eq!(x, orig);
    }
}
