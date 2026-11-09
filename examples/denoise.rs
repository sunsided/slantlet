//! Denoising by hard thresholding, in 1-D and 2-D: signal → transform →
//! drop small coefficients → inverse transform.
//!
//! Slantlet details annihilate ramps (1-D) and planes (planar 2-D), so a
//! piecewise-linear signal is described by few coefficients, while white
//! noise spreads evenly over all of them. The transform is orthonormal, so
//! the noise keeps its standard deviation σ in every coefficient, and the
//! universal threshold σ·sqrt(2·ln N) removes almost all of it. This is the
//! experiment of Fig. 6 in Selesnick (1999).
//!
//! Writes `docs/denoise-1d.svg` and `docs/denoise-2d.png` (or into the
//! directory given as the first argument):
//!
//! ```sh
//! cargo run --example denoise
//! ```

use std::{fmt::Write as _, fs, path::PathBuf};

const SIGMA_1D: f64 = 0.1;
const SIGMA_2D: f64 = 0.08;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "docs".into()));
    fs::create_dir_all(&dir)?;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    // 1-D: steps and ramps, 1024 samples.
    let clean = signal_1d(1024);
    let noisy: Vec<f64> = clean.iter().map(|v| v + SIGMA_1D * rng.gauss()).collect();
    let mut coeffs = noisy.clone();
    slantlet::forward(&mut coeffs)?;
    let t1 = threshold(SIGMA_1D, clean.len());
    let mut denoised = coeffs.clone();
    let kept = hard_threshold(&mut denoised, 2, t1); // keep h and f
    slantlet::inverse(&mut denoised)?;
    let (r_noisy, r_den) = (rmse(&noisy, &clean), rmse(&denoised, &clean));
    println!(
        "1-D: RMSE noisy {r_noisy:.4}, denoised {r_den:.4}; kept {kept} of {} coefficients",
        clean.len()
    );
    let svg = svg_1d(&clean, &noisy, &coeffs, &denoised, t1, kept, r_noisy, r_den);
    fs::write(dir.join("denoise-1d.svg"), svg)?;

    // 2-D: piecewise-planar image, planar (non-separable) transform.
    let n = 128;
    let img = image_2d(n);
    let noisy2: Vec<f64> = img.iter().map(|v| v + SIGMA_2D * rng.gauss()).collect();
    let mut c2 = noisy2.clone();
    slantlet::forward_2d_planar(&mut c2, n)?;
    let t2 = threshold(SIGMA_2D, n * n);
    let mut den2 = c2.clone();
    let kept2 = hard_threshold(&mut den2, 3, t2); // keep the image's plane
    slantlet::inverse_2d_planar(&mut den2, n)?;
    let spun = cycle_spin(&noisy2, n, 8, t2)?;
    let (r2_noisy, r2_den, r2_spun) = (rmse(&noisy2, &img), rmse(&den2, &img), rmse(&spun, &img));
    println!(
        "2-D: RMSE noisy {r2_noisy:.4}, denoised {r2_den:.4} (kept {kept2} of {}), cycle-spun {r2_spun:.4}",
        n * n
    );
    let panels = [
        gray(&img),
        gray(&noisy2),
        energy_pyramid(&c2, n, SIGMA_2D),
        gray(&den2),
        gray(&spun),
    ];
    fs::write(dir.join("denoise-2d.png"), png_panels(&panels, n, 2))?;
    println!(
        "wrote {} and {}",
        dir.join("denoise-1d.svg").display(),
        dir.join("denoise-2d.png").display()
    );
    Ok(())
}

/// Translation-invariant denoising (Coifman & Donoho): denoise every
/// circular shift by `0..shifts` pixels in x and y, shift back and average.
/// This removes the block artifacts of thresholding a non-redundant basis.
fn cycle_spin(img: &[f64], n: usize, shifts: usize, t: f64) -> Result<Vec<f64>, slantlet::Error> {
    let mut sum = vec![0.0; n * n];
    for dy in 0..shifts {
        for dx in 0..shifts {
            let at = |x: usize, y: usize| ((y + dy) % n) * n + (x + dx) % n;
            let mut c: Vec<f64> = (0..n * n).map(|i| img[at(i % n, i / n)]).collect();
            slantlet::forward_2d_planar(&mut c, n)?;
            hard_threshold(&mut c, 3, t);
            slantlet::inverse_2d_planar(&mut c, n)?;
            for (i, v) in c.iter().enumerate() {
                sum[at(i % n, i / n)] += v;
            }
        }
    }
    let k = (shifts * shifts) as f64;
    Ok(sum.into_iter().map(|v| v / k).collect())
}

/// Universal threshold for `len` coefficients with noise level `sigma`.
fn threshold(sigma: f64, len: usize) -> f64 {
    sigma * (2.0 * (len as f64).ln()).sqrt()
}

/// Zeroes detail coefficients below `t`, keeping the first `coarse` (the
/// global constant and ramp terms, which carry the signal's mean and slope);
/// returns how many coefficients remain non-zero.
fn hard_threshold(c: &mut [f64], coarse: usize, t: f64) -> usize {
    for v in &mut c[coarse..] {
        if v.abs() < t {
            *v = 0.0;
        }
    }
    c.iter().filter(|v| **v != 0.0).count()
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

/// Steps and ramps in the spirit of the paper's Fig. 6.
fn signal_1d(n: usize) -> Vec<f64> {
    // (start, value at start, slope per sample) for each piece.
    let pieces: [(usize, f64, f64); 9] = [
        (0, 0.0, 0.0),
        (140, -1.2, 0.0),
        (230, 1.7, 0.0),
        (300, -1.5, 0.008),
        (380, -0.3, 0.0),
        (470, 0.9, -0.012),
        (560, 0.4, 0.0),
        (700, -0.8, 0.006),
        (860, 1.5, -0.004),
    ];
    (0..n)
        .map(|i| {
            let &(s, v, k) = pieces.iter().rev().find(|p| p.0 <= i).unwrap();
            v + k * (i - s) as f64
        })
        .collect()
}

/// Piecewise-planar test image in [0, 1]: a tilted background, a tilted
/// rotated square, a flat disk and a ramped triangle.
fn image_2d(n: usize) -> Vec<f64> {
    let f = n as f64;
    (0..n * n)
        .map(|i| {
            let (x, y) = ((i % n) as f64 / f, (i / n) as f64 / f);
            let (u, v) = (x - 0.33, y - 0.35);
            let (ru, rv) = (0.8 * u + 0.6 * v, -0.6 * u + 0.8 * v);
            if (x - 0.72).powi(2) + (y - 0.3).powi(2) < 0.025 {
                0.9
            } else if ru.abs() < 0.17 && rv.abs() < 0.17 {
                0.05 + 0.75 * (ru + 0.17)
            } else if y > 0.55 && x > 0.45 && (y - 0.55) > 0.9 * (x - 0.45) - 0.05 && x < 0.95 {
                0.95 - 0.9 * (y - 0.55)
            } else {
                0.5 + 0.2 * x - 0.15 * y
            }
        })
        .collect()
}

/// Detail energy per block and level, on a log scale, laid out as a pyramid:
/// the map with `t × t` blocks sits at offset `(t, t)`, coarse top left,
/// finest (2×2 checkerboards, 64×64 map) bottom right.
fn energy_pyramid(c: &[f64], n: usize, sigma: f64) -> Vec<u8> {
    let mut out = vec![0u8; n * n];
    let scale = |e: f64| ((1.0 + e / (sigma * sigma)).log10() / 3.0).min(1.0);
    let mut put = |t: usize, b: usize, e: f64| {
        let (by, bx) = (b / t, b % t);
        out[(t + by) * n + t + bx] = (255.0 * scale(e)) as u8;
    };
    let mut t = 1; // blocks per row at this level
    while t < n / 2 {
        for b in 0..t * t {
            let o = 3 * t * t + 9 * b;
            put(t, b, c[o..o + 9].iter().map(|v| v * v).sum());
        }
        t *= 2;
    }
    for b in 0..t * t {
        put(t, b, c[3 * n * n / 4 + b].powi(2));
    }
    out
}

fn gray(v: &[f64]) -> Vec<u8> {
    v.iter()
        .map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn svg_1d(
    clean: &[f64],
    noisy: &[f64],
    coeffs: &[f64],
    denoised: &[f64],
    t: f64,
    kept: usize,
    r_noisy: f64,
    r_den: f64,
) -> String {
    let (w, ph, pad, left) = (900.0, 190.0, 34.0, 44.0);
    let n = clean.len();
    let h = 3.0 * (ph + pad) + pad;
    let x = |i: usize| left + (w - left - 12.0) * i as f64 / (n - 1) as f64;
    let mut s = String::new();
    let _ = write!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="system-ui, sans-serif" font-size="13">
<rect width="100%" height="100%" fill="#ffffff"/>
"##
    );
    let poly = |s: &mut String, v: &[f64], y0: f64, lo: f64, hi: f64, style: &str| {
        let pts: Vec<String> = v
            .iter()
            .enumerate()
            .map(|(i, &y)| format!("{:.1},{:.1}", x(i), y0 + ph * (hi - y) / (hi - lo)))
            .collect();
        let _ = writeln!(
            s,
            r#"<polyline fill="none" {style} points="{}"/>"#,
            pts.join(" ")
        );
    };
    let frame = |s: &mut String, y0: f64, title: &str, lo: f64, hi: f64| {
        let _ = writeln!(
            s,
            r##"<rect x="{left}" y="{y0}" width="{}" height="{ph}" fill="none" stroke="#c8c8c8"/>
<text x="{left}" y="{}" fill="#222">{title}</text>
<text x="{}" y="{}" fill="#777" text-anchor="end">{hi}</text>
<text x="{}" y="{}" fill="#777" text-anchor="end">{lo}</text>"##,
            w - left - 12.0,
            y0 - 8.0,
            left - 6.0,
            y0 + 10.0,
            left - 6.0,
            y0 + ph
        );
    };

    // (a) clean and noisy signal.
    let y0 = pad;
    frame(
        &mut s,
        y0,
        &format!("Noisy input (σ = {SIGMA_1D}, RMSE {r_noisy:.3}) and clean signal"),
        -2.0,
        2.0,
    );
    poly(
        &mut s,
        noisy,
        y0,
        -2.0,
        2.0,
        r##"stroke="#9aa5b1" stroke-width="1""##,
    );
    poly(
        &mut s,
        clean,
        y0,
        -2.0,
        2.0,
        r##"stroke="#1f6feb" stroke-width="1.6""##,
    );

    // (b) coefficients, threshold band and scale boundaries.
    let y0 = 2.0 * pad + ph;
    let lim = coeffs[2..]
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .ceil();
    frame(
        &mut s,
        y0,
        &format!("Slantlet coefficients: coarse → fine; kept {kept} of {n} (|c| ≥ {t:.2})"),
        -lim,
        lim,
    );
    let yc = |v: f64| y0 + ph * (lim - v.clamp(-lim, lim)) / (2.0 * lim);
    let _ = writeln!(
        s,
        r##"<rect x="{left}" y="{:.1}" width="{}" height="{:.1}" fill="#f0b429" fill-opacity="0.18"/>"##,
        yc(t),
        w - left - 12.0,
        yc(-t) - yc(t)
    );
    let mut m = n / 2; // scale boundaries at N/m
    while m >= 2 {
        let _ = writeln!(
            s,
            r##"<line x1="{0:.1}" x2="{0:.1}" y1="{y0}" y2="{1}" stroke="#d0d0d0" stroke-dasharray="3 3"/>"##,
            x(n / m),
            y0 + ph
        );
        m /= 2;
    }
    for (i, &v) in coeffs.iter().enumerate() {
        let colour = if i < 2 || v.abs() >= t {
            "#cf222e"
        } else {
            "#9aa5b1"
        };
        let _ = writeln!(
            s,
            r#"<line x1="{0:.1}" x2="{0:.1}" y1="{1:.1}" y2="{2:.1}" stroke="{colour}" stroke-width="1"/>"#,
            x(i),
            yc(0.0),
            yc(v)
        );
    }

    // (c) denoised result.
    let y0 = 3.0 * pad + 2.0 * ph;
    frame(
        &mut s,
        y0,
        &format!("After thresholding and inverse transform (RMSE {r_den:.3})"),
        -2.0,
        2.0,
    );
    poly(
        &mut s,
        clean,
        y0,
        -2.0,
        2.0,
        r##"stroke="#1f6feb" stroke-width="1.2" stroke-dasharray="4 3""##,
    );
    poly(
        &mut s,
        denoised,
        y0,
        -2.0,
        2.0,
        r##"stroke="#111" stroke-width="1.4""##,
    );
    s.push_str("</svg>\n");
    s
}

/// Grayscale `n × n` panels side by side, each scaled `zoom`× (nearest
/// neighbour), separated by white gaps, as an 8-bit PNG.
fn png_panels(panels: &[Vec<u8>], n: usize, zoom: usize) -> Vec<u8> {
    let gap = 12;
    let side = n * zoom;
    let (w, h) = (panels.len() * side + (panels.len() - 1) * gap, side);
    let mut raw = Vec::with_capacity((w + 1) * h);
    for y in 0..h {
        raw.push(0); // filter: none
        for (p, panel) in panels.iter().enumerate() {
            if p > 0 {
                raw.extend(std::iter::repeat_n(255, gap));
            }
            for x in 0..side {
                raw.push(panel[(y / zoom) * n + x / zoom]);
            }
        }
    }
    png_gray(w, h, &raw)
}

/// Minimal PNG writer: 8-bit grayscale, zlib stream of stored (uncompressed)
/// deflate blocks, so no compression library is needed.
fn png_gray(w: usize, h: usize, filtered: &[u8]) -> Vec<u8> {
    let mut z = vec![0x78, 0x01];
    let mut blocks = filtered.chunks(65_535).peekable();
    while let Some(b) = blocks.next() {
        z.push(u8::from(blocks.peek().is_none()));
        let len = b.len() as u16;
        z.extend(len.to_le_bytes());
        z.extend((!len).to_le_bytes());
        z.extend(b);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in filtered {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    z.extend(((b << 16) | a).to_be_bytes());

    let mut ihdr = Vec::new();
    ihdr.extend((w as u32).to_be_bytes());
    ihdr.extend((h as u32).to_be_bytes());
    ihdr.extend([8, 0, 0, 0, 0]); // 8 bit, grayscale, deflate, no filter set, no interlace

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    for (kind, data) in [(b"IHDR", ihdr), (b"IDAT", z), (b"IEND", Vec::new())] {
        png.extend((data.len() as u32).to_be_bytes());
        let start = png.len();
        png.extend(kind);
        png.extend(&data);
        let crc = crc32(&png[start..]);
        png.extend(crc.to_be_bytes());
    }
    png
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// xorshift64 with Box–Muller, so the figures are reproducible.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}
