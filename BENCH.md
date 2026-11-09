# Benchmarks

Throughput of every public transform, measured with
[Criterion](https://github.com/criterion-rs/criterion.rs) (`benches/transforms.rs`).
Numbers only hold for the machine below: compare runs on the same hardware,
and treat them as a rough guide anywhere else.

## Machine

| | |
|---|---|
| System | HP EliteBook 645 14 inch G10 Notebook PC (laptop) |
| CPU | AMD Ryzen 7 PRO 7730U (Zen 3), 8 cores / 16 threads, 0.41–4.55 GHz |
| Caches | L1d 32 KiB/core, L2 512 KiB/core, L3 16 MiB shared |
| Memory | 62 GiB |
| OS | Ubuntu 24.04.5 LTS, Linux 7.0.0-38-generic x86_64 |
| Power | On AC, `performance` governor and energy preference, boost enabled |
| Load | Desktop session running; load average ≈ 1.3 during the run |
| Toolchain | rustc 1.99.1, `cargo bench` default release profile, no `target-cpu=native` |
| Date | 2026-11-09 |

Benchmarks run single-threaded; the crate does not parallelise. A laptop's
clock depends on temperature, so expect a few percent of run-to-run variation.

## Results

Time is Criterion's point estimate with its 95 % confidence interval.
Throughput counts input samples (pixels for 2-D).

### 1-D

| Benchmark | Time | 95 % CI | Throughput |
|---|---|---|---|
| `forward`, N = 256 | 429.6 ns | 426.9–433.2 ns | 596 Melem/s |
| `forward`, N = 4096 | 5.178 µs | 5.157–5.205 µs | 791 Melem/s |
| `forward`, N = 65536 | 96.31 µs | 92.80–99.78 µs | 680 Melem/s |
| `forward`, N = 1048576 | 2.034 ms | 2.000–2.074 ms | 516 Melem/s |
| `inverse`, N = 256 | 454.7 ns | 452.9–457.0 ns | 563 Melem/s |
| `inverse`, N = 4096 | 6.047 µs | 6.018–6.090 µs | 677 Melem/s |
| `inverse`, N = 65536 | 107.7 µs | 104.6–111.1 µs | 609 Melem/s |
| `inverse`, N = 1048576 | 2.060 ms | 2.040–2.082 ms | 509 Melem/s |
| `forward_frames`, 3973 samples, frame 256 | 5.543 µs | 5.523–5.571 µs | 717 Melem/s |
| `forward_frames`, 10⁶ samples, frame 1024 | 1.266 ms | 1.260–1.274 ms | 790 Melem/s |
| `forward_frames`, 10⁶ samples, frame 65536 | 1.258 ms | 1.245–1.271 ms | 795 Melem/s |

### 2-D

| Benchmark | Time | 95 % CI | Throughput |
|---|---|---|---|
| `forward_2d` (separable), 256×256 | 240.6 µs | 239.8–241.3 µs | 272 Melem/s |
| `forward_2d_planar`, 256×256 | 76.78 µs | 76.28–77.21 µs | 854 Melem/s |
| `inverse_2d_planar`, 256×256 | 79.59 µs | 79.35–79.81 µs | 823 Melem/s |
| `forward_2d` (separable), 1024×1024 | 4.110 ms | 4.080–4.142 ms | 255 Melem/s |
| `forward_2d_planar`, 1024×1024 | 1.791 ms | 1.777–1.809 ms | 585 Melem/s |
| `inverse_2d_planar`, 1024×1024 | 2.012 ms | 1.955–2.075 ms | 521 Melem/s |

## Reading the numbers

- **About 1.3–2 ns per sample** in 1-D, as expected for an O(N) transform with
  a handful of flops per sample. Timings include the scratch allocation each
  call makes.
- **Large single transforms are memory-bound.** At N = 2²⁰ the signal and
  scratch buffer take 16 MiB together, the whole L3. Splitting the same
  amount of data into frames (`forward_frames`, 10⁶ samples) keeps each frame
  in cache and is about 1.5× faster. For long signals, frames are the faster
  choice whenever the coarsest scale you need fits in a frame.
- **Separable 2-D is limited by the column pass.** Columns are strided in a
  row-major image. They are copied out 8 at a time (one 64-byte cache line per
  row segment). That batch size was the best of 4–128 here, and lifted
  1024×1024 from 92 to 255 Melem/s compared with copying one column at a time.
- **Planar 2-D is 2.3–3.1× faster than separable** because each pixel is touched
  once per level, in blocks, with no column pass.

## Reproducing

```sh
task bench                                  # or: cargo bench --bench transforms
cargo bench --bench transforms -- 2d/       # filter by benchmark name
```

When updating this file, rerun the full suite on otherwise idle hardware and
update the machine table together with the results.
