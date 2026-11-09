# slantlet

The Slantlet transform (Selesnick, 1999) in Rust: an orthogonal discrete
wavelet-like transform. Its basis functions are piecewise linear, have two
vanishing moments, and have shorter support than the Daubechies-2 wavelet.

- Forward and inverse transforms work in place in `O(N)`, for any power-of-two length.
- `forward_frames` / `inverse_frames` handle any length (or a stream) frame by
  frame, losslessly and without padding.
- 2-D: separable (`forward_2d`, any size via `forward_2d_frames`) and a
  non-separable, plane-annihilating variant (`forward_2d_planar`) following
  the paper's outline (Sec. II-L); its exact detail basis is this crate's own
  choice, as the paper never published one.
- The basis is orthonormal, so energy is preserved and `inverse` is the transpose of `forward`.
- Linear signals (`a + b·n`) produce zero detail coefficients.
- Coefficients match the paper's slantlet matrix (Sec. II-F, Fig. 5) exactly.
- O(N) and fast: roughly 500–800 million samples/s on a laptop CPU; see
  [BENCH.md](BENCH.md) for numbers and the hardware they were measured on.
- No dependencies.

```rust
let mut x = [1.0, 4.0, -2.0, 0.5, 3.0, 3.0, 0.0, -1.0];
slantlet::forward(&mut x)?;   // x = [h, f, g₂, Jg₂, g₁, Jg₁, g₁, Jg₁]
slantlet::inverse(&mut x)?;   // back to the signal
# Ok::<(), slantlet::Error>(())
```

See the crate docs for the basis definition and coefficient layout.

## Applications

Short, piecewise-linear basis functions localize transients well, which makes
slantlet subband energies useful features for vibration-based fault
diagnosis. Alobaidy et al. ([*Sci. Rep.* 15, 34145, 2025][robot]) use them with
a small neural network to detect joint faults in an industrial robot arm
from tip-accelerometer data.

[`examples/fault_features.rs`](examples/fault_features.rs) shows that
feature-extraction step on synthetic data. It uses the paper's recording
length (3973 samples), splits each recording into power-of-two frames, and
reports the relative energy per scale; gear-like impacts raise the
finest-scale share by more than an order of magnitude:

```sh
cargo run --example fault_features
```

[robot]: https://doi.org/10.1038/s41598-025-09272-9

## Testing

With [go-task](https://taskfile.dev):

```sh
task test        # unit, property, reference-matrix and doc tests
task fuzz:smoke  # 10 s per fuzz target (nightly + cargo-fuzz)
task ci          # fmt check, clippy, docs, tests, fuzz smoke run
```

`tests/reference.rs` builds the slantlet matrix from the paper's closed-form
filters and checks that `forward` reproduces it entry by entry. Golden values in
`tests/golden/` pin both the crate and that reference.

## Reference

I. W. Selesnick, "The Slantlet Transform", *IEEE Transactions on Signal
Processing*, 47(5):1304–1313, 1999.

## License

Licensed under the [European Union Public Licence v. 1.2](LICENSE) (EUPL-1.2).
