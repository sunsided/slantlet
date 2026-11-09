#![no_main]
//! Arbitrary bit patterns (NaN, inf, subnormals) and any length: never panic;
//! error exactly for invalid shapes (non-power-of-two lengths, frame lengths
//! or sides; length not width × height), leaving the input untouched.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let x: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let ok = x.len().is_power_of_two();

    for f in [slantlet::forward, slantlet::inverse] {
        let mut y = x.clone();
        let r = f(&mut y);
        assert_eq!(r.is_ok(), ok);
        if !ok {
            assert!(y.iter().zip(&x).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    // Frame length from the first two bytes: any value, power of two or not.
    let frame = data
        .get(..2)
        .map_or(0, |b| usize::from(u16::from_le_bytes([b[0], b[1]])));
    for f in [slantlet::forward_frames, slantlet::inverse_frames] {
        let mut y = x.clone();
        let r = f(&mut y, frame);
        assert_eq!(r.is_ok(), frame.is_power_of_two());
        if r.is_err() {
            assert!(y.iter().zip(&x).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    // 2-D shapes from the next bytes: any width, height and side.
    let byte = |i: usize| usize::from(data.get(i).copied().unwrap_or(0));
    let (w, h) = (byte(2) % 33, byte(3) % 33);
    let dims_ok = w * h == x.len();
    let separable_ok = dims_ok && w.is_power_of_two() && h.is_power_of_two();
    let planar_ok = x.len() == w * w && w.is_power_of_two();
    let cases: [(&dyn Fn(&mut [f64]) -> Result<(), slantlet::Error>, bool); 6] = [
        (&|v| slantlet::forward_2d(v, w, h), separable_ok),
        (&|v| slantlet::inverse_2d(v, w, h), separable_ok),
        (
            &|v| slantlet::forward_2d_frames(v, w, h, frame),
            dims_ok && frame.is_power_of_two(),
        ),
        (
            &|v| slantlet::inverse_2d_frames(v, w, h, frame),
            dims_ok && frame.is_power_of_two(),
        ),
        (&|v| slantlet::forward_2d_planar(v, w), planar_ok),
        (&|v| slantlet::inverse_2d_planar(v, w), planar_ok),
    ];
    for (f, ok) in cases {
        let mut y = x.clone();
        let r = f(&mut y);
        assert_eq!(r.is_ok(), ok);
        if r.is_err() {
            assert!(y.iter().zip(&x).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
});
