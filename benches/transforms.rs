//! Throughput of every public transform. Results and hardware: see BENCH.md.
//!
//! Each benchmark transforms the same buffer in place repeatedly. The
//! transforms are orthogonal, so values stay bounded, and the timing includes
//! the per-call scratch allocation that real callers pay.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn signal(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (i as f64 * 0.013).sin() + 0.1 * ((i * 7919 % 101) as f64 - 50.0))
        .collect()
}

fn one_d(c: &mut Criterion) {
    for (name, f) in [
        ("forward", slantlet::forward as fn(&mut [f64]) -> _),
        ("inverse", slantlet::inverse),
    ] {
        let mut g = c.benchmark_group(format!("1d/{name}"));
        for l in [8, 12, 16, 17, 18, 19, 20] {
            let n = 1usize << l;
            let mut x = signal(n);
            g.throughput(Throughput::Elements(n as u64));
            g.bench_function(BenchmarkId::from_parameter(n), |b| {
                b.iter(|| f(black_box(&mut x)).unwrap())
            });
        }
        g.finish();
    }
}

fn frames(c: &mut Criterion) {
    let mut g = c.benchmark_group("1d/forward_frames");
    // 3973 samples: one robot-arm recording (Alobaidy et al. 2025).
    for (len, frame) in [(3973, 256), (1_000_000, 1024), (1_000_000, 65536)] {
        let mut x = signal(len);
        g.throughput(Throughput::Elements(len as u64));
        g.bench_function(BenchmarkId::from_parameter(format!("{len}@{frame}")), |b| {
            b.iter(|| slantlet::forward_frames(black_box(&mut x), frame).unwrap())
        });
    }
    g.finish();
}

fn two_d(c: &mut Criterion) {
    let mut g = c.benchmark_group("2d");
    for side in [256usize, 1024] {
        let mut x = signal(side * side);
        g.throughput(Throughput::Elements((side * side) as u64));
        g.bench_function(
            BenchmarkId::new("forward_2d", format!("{side}x{side}")),
            |b| b.iter(|| slantlet::forward_2d(black_box(&mut x), side, side).unwrap()),
        );
        g.bench_function(
            BenchmarkId::new("forward_2d_planar", format!("{side}x{side}")),
            |b| b.iter(|| slantlet::forward_2d_planar(black_box(&mut x), side).unwrap()),
        );
        g.bench_function(
            BenchmarkId::new("inverse_2d_planar", format!("{side}x{side}")),
            |b| b.iter(|| slantlet::inverse_2d_planar(black_box(&mut x), side).unwrap()),
        );
    }
    g.finish();
}

/// GPU frame-wise transform (feature `wgpu`), with and without host copies.
#[cfg(feature = "wgpu")]
fn gpu(c: &mut Criterion) {
    use wgpu::util::DeviceExt;
    let Ok(gpu) = slantlet::gpu::Gpu::new() else {
        eprintln!("no GPU adapter; skipping GPU benchmarks");
        return;
    };
    let mut g = c.benchmark_group("gpu");
    for (len, frame) in [(1usize << 20, 1024), (1 << 24, 1024)] {
        let x: Vec<f32> = signal(len).into_iter().map(|v| v as f32).collect();
        g.throughput(Throughput::Elements(len as u64));
        let mut host = x.clone();
        g.bench_function(
            BenchmarkId::new("forward_frames_host", format!("{len}@{frame}")),
            |b| b.iter(|| gpu.forward_frames(black_box(&mut host), frame).unwrap()),
        );
        let buf = gpu
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&x),
                usage: wgpu::BufferUsages::STORAGE,
            });
        g.bench_function(
            BenchmarkId::new("forward_resident", format!("{len}@{frame}")),
            |b| {
                b.iter(|| {
                    let mut enc = gpu.device().create_command_encoder(&Default::default());
                    gpu.encode_forward(&mut enc, &buf, frame, len / frame)
                        .unwrap();
                    gpu.queue().submit([enc.finish()]);
                    gpu.device()
                        .poll(wgpu::PollType::wait_indefinitely())
                        .unwrap();
                })
            },
        );
        let mut cpu: Vec<f64> = signal(len);
        g.bench_function(
            BenchmarkId::new("cpu_forward_frames_f64", format!("{len}@{frame}")),
            |b| b.iter(|| slantlet::forward_frames(black_box(&mut cpu), frame).unwrap()),
        );
    }
    g.finish();
}

#[cfg(not(feature = "wgpu"))]
fn gpu(_: &mut Criterion) {}

criterion_group!(benches, one_d, frames, two_d, gpu);
criterion_main!(benches);
