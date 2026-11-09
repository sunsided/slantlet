use proptest::prelude::*;

fn signal() -> impl Strategy<Value = Vec<f64>> {
    (0u32..=12).prop_flat_map(|l| prop::collection::vec(-1e6f64..1e6, 1usize << l))
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn tol(v: &[f64]) -> f64 {
    1e-10 * (1.0 + norm(v))
}

proptest! {
    #[test]
    fn roundtrip(x in signal()) {
        let mut y = x.clone();
        slantlet::forward(&mut y).unwrap();
        slantlet::inverse(&mut y).unwrap();
        let t = tol(&x);
        for (a, b) in y.iter().zip(&x) {
            prop_assert!((a - b).abs() <= t, "{a} vs {b}");
        }
    }

    #[test]
    fn inverse_then_forward(x in signal()) {
        let mut y = x.clone();
        slantlet::inverse(&mut y).unwrap();
        slantlet::forward(&mut y).unwrap();
        let t = tol(&x);
        for (a, b) in y.iter().zip(&x) {
            prop_assert!((a - b).abs() <= t);
        }
    }

    #[test]
    fn preserves_energy(x in signal()) {
        let mut y = x.clone();
        slantlet::forward(&mut y).unwrap();
        prop_assert!((norm(&x) - norm(&y)).abs() <= tol(&x));
    }

    #[test]
    fn is_linear(
        (x, z) in (0u32..=10).prop_flat_map(|l| {
            let n = 1usize << l;
            (prop::collection::vec(-1e3f64..1e3, n), prop::collection::vec(-1e3f64..1e3, n))
        }),
        a in -10f64..10.0,
        b in -10f64..10.0,
    ) {
        let mut mix: Vec<f64> = x.iter().zip(&z).map(|(p, q)| a * p + b * q).collect();
        let (mut tx, mut tz) = (x.clone(), z.clone());
        slantlet::forward(&mut mix).unwrap();
        slantlet::forward(&mut tx).unwrap();
        slantlet::forward(&mut tz).unwrap();
        let t = 1e-9 * (1.0 + norm(&x) + norm(&z)) * (1.0 + a.abs() + b.abs());
        for i in 0..mix.len() {
            prop_assert!((mix[i] - (a * tx[i] + b * tz[i])).abs() <= t);
        }
    }

    #[test]
    fn annihilates_linear_signals(l in 1u32..=12, c0 in -1e3f64..1e3, c1 in -1e3f64..1e3) {
        let n = 1usize << l;
        let mut x: Vec<f64> = (0..n).map(|t| c0 + c1 * t as f64).collect();
        let t = 1e-12 * (1.0 + norm(&x));
        slantlet::forward(&mut x).unwrap();
        for v in &x[2..] {
            prop_assert!(v.abs() <= t, "detail {v} > {t}");
        }
    }

    #[test]
    fn rejects_non_power_of_two(x in prop::collection::vec(any::<f64>(), 0..300)) {
        prop_assume!(!x.len().is_power_of_two());
        let mut y = x.clone();
        let err = Err(slantlet::Error::LengthNotPowerOfTwo(x.len()));
        prop_assert_eq!(slantlet::forward(&mut y), err);
        prop_assert_eq!(slantlet::inverse(&mut y), err);
        prop_assert!(y.iter().zip(&x).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

proptest! {
    #[test]
    fn frames_roundtrip_and_energy(
        x in prop::collection::vec(-1e6f64..1e6, 0..2000),
        k in 0u32..=11,
    ) {
        let frame = 1usize << k;
        let mut y = x.clone();
        slantlet::forward_frames(&mut y, frame).unwrap();
        let t = tol(&x);
        prop_assert!((norm(&x) - norm(&y)).abs() <= t);
        slantlet::inverse_frames(&mut y, frame).unwrap();
        for (a, b) in y.iter().zip(&x) {
            prop_assert!((a - b).abs() <= t, "{a} vs {b}");
        }
    }

    /// A signal made of whole frames transforms like each frame on its own,
    /// so it can be streamed chunk by chunk.
    #[test]
    fn frames_stream_chunkwise(
        x in prop::collection::vec(-1e3f64..1e3, 0..1500),
        k in 0u32..=8,
        chunk_frames in 1usize..5,
    ) {
        let frame = 1usize << k;
        let mut whole = x.clone();
        slantlet::forward_frames(&mut whole, frame).unwrap();
        let mut streamed = x.clone();
        for chunk in streamed.chunks_mut(frame * chunk_frames) {
            slantlet::forward_frames(chunk, frame).unwrap();
        }
        // Only the final partial chunk may split differently; compare the rest.
        let full = x.len() / (frame * chunk_frames) * frame * chunk_frames;
        prop_assert_eq!(&whole[..full], &streamed[..full]);
    }
}

fn image() -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    (0usize..40, 0usize..40)
        .prop_flat_map(|(w, h)| (prop::collection::vec(-1e6f64..1e6, w * h), Just(w), Just(h)))
}

proptest! {
    #[test]
    fn separable_2d_roundtrip_any_size((x, w, h) in image(), k in 0u32..=6) {
        let mut y = x.clone();
        slantlet::forward_2d_frames(&mut y, w, h, 1 << k).unwrap();
        let t = tol(&x);
        prop_assert!((norm(&x) - norm(&y)).abs() <= t);
        slantlet::inverse_2d_frames(&mut y, w, h, 1 << k).unwrap();
        for (a, b) in y.iter().zip(&x) {
            prop_assert!((a - b).abs() <= t);
        }
    }

    #[test]
    fn planar_2d_roundtrip(
        x in (0u32..=6).prop_flat_map(|l| prop::collection::vec(-1e6f64..1e6, 1usize << (2 * l))),
    ) {
        let side = x.len().isqrt();
        let mut y = x.clone();
        slantlet::forward_2d_planar(&mut y, side).unwrap();
        let t = tol(&x);
        prop_assert!((norm(&x) - norm(&y)).abs() <= t);
        slantlet::inverse_2d_planar(&mut y, side).unwrap();
        for (a, b) in y.iter().zip(&x) {
            prop_assert!((a - b).abs() <= t);
        }
    }

    /// Planes a + b·n1 + c·n2 leave only the three coarse coefficients.
    #[test]
    fn planar_2d_annihilates_planes(l in 1u32..=6, a in -1e3f64..1e3, b in -1e3f64..1e3, c in -1e3f64..1e3) {
        let side = 1usize << l;
        let mut x: Vec<f64> = (0..side * side)
            .map(|i| a + b * (i % side) as f64 + c * (i / side) as f64)
            .collect();
        let t = 1e-12 * (1.0 + norm(&x));
        slantlet::forward_2d_planar(&mut x, side).unwrap();
        for v in &x[3.min(x.len())..] {
            prop_assert!(v.abs() <= t);
        }
    }
}
