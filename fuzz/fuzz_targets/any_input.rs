#![no_main]
//! Arbitrary bit patterns (NaN, inf, subnormals) and any length: never panic;
//! error exactly for non-power-of-two lengths, leaving the input untouched.

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
});
