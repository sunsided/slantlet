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
| GPU | Integrated AMD Radeon Graphics (Vega, "RENOIR"), Vulkan via Mesa 25.2.8 RADV; shares system memory |
| OS | Ubuntu 24.04.5 LTS, Linux 7.0.0-38-generic x86_64 |
| Power | On AC, `performance` governor and energy preference, boost enabled |
| Load | Desktop session running; load average 0.2–1.3 at the start of each run, up to ≈ 6 at the end of the parallel run |
| Toolchain | rustc 1.99.1, `cargo bench` default release profile, no `target-cpu=native` |
| Date | 2026-11-09 |

A laptop's clock depends on temperature, so expect a few percent of
run-to-run variation, and more in the parallel column.

## Results

Time is Criterion's point estimate, with its 95 % confidence interval, for
the default single-threaded build. Throughput counts input samples (pixels
for 2-D). The last column is the same benchmark built with
`--features rayon`. Below each parallel threshold the two builds run
identical code, so differences there are noise.

### 1-D

| Benchmark | Time | 95 % CI | Throughput | With `rayon` |
|---|---|---|---|---|
| `forward`, N = 256 | 330.3 ns | 329.2–331.4 ns | 775 Melem/s | 757 Melem/s |
| `forward`, N = 4096 | 3.692 µs | 3.676–3.710 µs | 1.11 Gelem/s | 1.10 Gelem/s |
| `forward`, N = 65536 | 67.18 µs | 65.81–68.63 µs | 975 Melem/s | 956 Melem/s |
| `forward`, N = 2¹⁷ | 163.8 µs | 162.9–164.8 µs | 800 Melem/s | 788 Melem/s |
| `forward`, N = 2¹⁸ | 333.6 µs | 332.3–335.0 µs | 786 Melem/s | 798 Melem/s |
| `forward`, N = 2¹⁹ | 638.3 µs | 635.5–641.3 µs | 821 Melem/s | 826 Melem/s |
| `forward`, N = 2²⁰ | 1.625 ms | 1.603–1.652 ms | 645 Melem/s | **888 Melem/s** |
| `inverse`, N = 256 | 359.8 ns | 359.0–360.7 ns | 711 Melem/s | 714 Melem/s |
| `inverse`, N = 4096 | 4.436 µs | 4.424–4.447 µs | 923 Melem/s | 916 Melem/s |
| `inverse`, N = 65536 | 85.82 µs | 82.67–89.21 µs | 764 Melem/s | 846 Melem/s |
| `inverse`, N = 2¹⁷ | 205.4 µs | 201.8–209.0 µs | 638 Melem/s | 750 Melem/s |
| `inverse`, N = 2¹⁸ | 375.5 µs | 365.7–385.4 µs | 698 Melem/s | 671 Melem/s |
| `inverse`, N = 2¹⁹ | 691.6 µs | 679.3–706.6 µs | 758 Melem/s | **1.20 Gelem/s** |
| `inverse`, N = 2²⁰ | 1.690 ms | 1.681–1.700 ms | 621 Melem/s | **1.18 Gelem/s** |
| `forward_frames`, 3973 samples, frame 256 | 4.161 µs | 4.153–4.170 µs | 955 Melem/s | 949 Melem/s |
| `forward_frames`, 10⁶ samples, frame 1024 | 864.6 µs | 859.5–871.7 µs | 1.16 Gelem/s | **4.31 Gelem/s** |
| `forward_frames`, 10⁶ samples, frame 65536 | 1.010 ms | 0.974–1.050 ms | 990 Melem/s | **2.52 Gelem/s** |

### 2-D

Measured in a separate run on the same day, after planar 2-D was parallelised.

| Benchmark | Time | 95 % CI | Throughput | With `rayon` |
|---|---|---|---|---|
| `forward_2d` (separable), 256×256 | 196.7 µs | 195.5–198.4 µs | 333 Melem/s | 343 Melem/s |
| `forward_2d_planar`, 256×256 | 79.52 µs | 79.09–80.01 µs | 824 Melem/s | 589 Melem/s |
| `inverse_2d_planar`, 256×256 | 111.2 µs | 110.7–111.7 µs | 590 Melem/s | 577 Melem/s |
| `forward_2d` (separable), 512×512 | 786.2 µs | 780.8–793.8 µs | 333 Melem/s | **533 Melem/s** |
| `forward_2d_planar`, 512×512 | 384.1 µs | 381.5–386.9 µs | 683 Melem/s | **767 Melem/s** |
| `inverse_2d_planar`, 512×512 | 438.7 µs | 437.1–440.5 µs | 598 Melem/s | **852 Melem/s** |
| `forward_2d` (separable), 1024×1024 | 3.425 ms | 3.352–3.511 ms | 306 Melem/s | **564 Melem/s** |
| `forward_2d_planar`, 1024×1024 | 1.856 ms | 1.833–1.882 ms | 565 Melem/s | **954 Melem/s** |
| `inverse_2d_planar`, 1024×1024 | 1.896 ms | 1.881–1.912 ms | 553 Melem/s | **1.02 Gelem/s** |

256×256 is below both 2-D parallel thresholds, so its two columns run the
same code.

### GPU (`--features wgpu`, `f32`)

Frame-wise forward transform, 1024-sample frames, on the integrated GPU.
"Resident" means the data already lives in a GPU buffer: it times
`encode_forward`, submit, and waiting for completion. "Host" includes the
upload and the download of the results.

| Benchmark | Time | Throughput |
|---|---|---|
| GPU, resident, 2²⁰ samples | 481.2 µs | 2.18 Gelem/s |
| GPU, host round trip, 2²⁰ samples | 1.624 ms | 646 Melem/s |
| CPU `forward_frames` (f64, 1 thread), 2²⁰ samples | 903.9 µs | 1.16 Gelem/s |
| GPU, resident, 2²⁴ samples | 6.163 ms | 2.72 Gelem/s |
| GPU, host round trip, 2²⁴ samples | 35.79 ms | 469 Melem/s |
| CPU `forward_frames` (f64, 1 thread), 2²⁴ samples | 23.09 ms | 727 Melem/s |

## Reading the numbers

- **About 0.9–1.6 ns per sample** in 1-D, single-threaded, for an O(N)
  transform with a handful of flops per sample. Timings include the scratch
  allocation each call makes.
- **Cache blocking** runs all levels inside each 2048-sample chunk while it
  is in L1/L2 (see the README). It made `forward` 22–42 % faster and
  `forward_frames` about 1.5× faster than the previous level-by-level loop.
  Block sizes of 256–4096 pairs were measured; 1024 was the best overall.
- **AVX2 was tried and dropped.** An AVX2 build of the hot loops (dispatched at
  runtime, without FMA, so bit-identical) ran within noise of the
  default SSE2 build. Forward was about 3 % slower, inverse 5–10 % faster. The
  interleaved pair loads need shuffles that eat the extra lanes.
- **Parallelism (`rayon`)** pays off where work is independent and large:
  frame-wise transforms (3.7× at 10⁶ samples), separable 2-D at 1024² (2×),
  and single transforms from 2¹⁹ samples on (inverse 1.6–1.9×, forward
  1.4×). A single forward transform is limited by its sequential streaming
  and coarse levels, and by memory bandwidth. The thresholds in `src/lib.rs`
  (`par_min`) come from these measurements; below them the sequential code
  runs.
- **Planar 2-D runs in bands when parallel.** Each band of 32 image rows runs
  every level whose blocks fit inside it on its own thread, then the coarse
  levels run on the whole image. This gives 1.3–1.8× from 512×512 up; band
  heights 16–128 were measured, 32 was best. Single-threaded, the band
  layout was slower, so the sequential path keeps one band over the whole
  image. The refactor still costs it 0–5 % (inverse up to ≈ 6 %, measured
  A/B against the previous code).
- **Separable 2-D is limited by the column pass.** Columns are strided in a
  row-major image. They are copied out 8 at a time, which is one 64-byte cache
  line per row segment and the best of the batch sizes 4–128 measured.
- **The GPU wins only on resident data.** On data already in GPU memory, the
  integrated GPU runs 2.2–2.7 Gelem/s, 1.9–3.7× a single CPU thread. Copying
  data from the host and back costs more than the transform: the round trip
  is slower than the CPU. The `rayon` CPU build (4.3 Gelem/s on frames)
  beats this integrated GPU outright. A discrete GPU, or a pipeline that
  keeps data on the GPU, changes that balance.

## Reproducing

```sh
task bench                                            # or: cargo bench --bench transforms
cargo bench --bench transforms --features rayon       # parallel column
cargo bench --bench transforms --features wgpu -- gpu/   # GPU (needs an adapter)
cargo bench --bench transforms -- 2d/                 # filter by name
```

When updating this file, rerun the full suite on otherwise idle hardware and
update the machine table together with the results.
