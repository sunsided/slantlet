# slantlet

The Slantlet transform (Selesnick, 1999) in Rust: an orthogonal discrete
wavelet-like transform. Its basis functions are piecewise linear, have two
vanishing moments, and have shorter support than the Daubechies-2 wavelet.

- Forward and inverse transforms work in place in `O(N)`, for any power-of-two length.
- The basis is orthonormal, so energy is preserved and `inverse` is the transpose of `forward`.
- Linear signals (`a + b·n`) produce zero detail coefficients.
- Coefficients match the paper's slantlet matrix (Sec. II-F, Fig. 5) exactly.
- No dependencies.

```rust
let mut x = [1.0, 4.0, -2.0, 0.5, 3.0, 3.0, 0.0, -1.0];
slantlet::forward(&mut x)?;   // x = [h, f, g₂, Jg₂, g₁, Jg₁, g₁, Jg₁]
slantlet::inverse(&mut x)?;   // back to the signal
# Ok::<(), slantlet::Error>(())
```

See the crate docs for the basis definition and coefficient layout.

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
