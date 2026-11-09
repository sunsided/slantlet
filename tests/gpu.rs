//! GPU transform against the f64 CPU transform. Skips (with a note) when no
//! adapter is available, unless `SLANTLET_REQUIRE_GPU` is set (as in CI, which
//! provides a software Vulkan driver, lavapipe).
#![cfg(feature = "wgpu")]

use std::sync::OnceLock;

use slantlet::gpu::{Gpu, GpuError, MAX_FRAME_LEN};

fn gpu() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) if std::env::var_os("SLANTLET_REQUIRE_GPU").is_some() => {
            panic!("SLANTLET_REQUIRE_GPU is set but no GPU is usable: {e}")
        }
        Err(e) => {
            eprintln!("skipping GPU tests: {e}");
            None
        }
    })
    .as_ref()
}

fn signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 37 % 101) as f32 * 0.29).sin() * 3.0 + 0.01 * i as f32)
        .collect()
}

/// Max error relative to the signal's largest magnitude; f32 rounding.
fn assert_close_f64(got: &[f32], want: &[f64], what: &str) {
    let scale = want.iter().fold(1e-30f64, |m, v| m.max(v.abs()));
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs() / scale;
        assert!(err < 1e-5, "{what}[{i}]: gpu {g} vs cpu {w} (rel {err:e})");
    }
}

#[test]
fn matches_cpu_for_every_frame_length() {
    let Some(gpu) = gpu() else { return };
    for k in 0..=MAX_FRAME_LEN.trailing_zeros() {
        let frame = 1usize << k;
        let len = 3 * frame + frame / 2 + 1; // full frames plus a tail
        let x = signal(len);
        let mut want: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        slantlet::forward_frames(&mut want, frame).unwrap();

        let mut y = x.clone();
        gpu.forward_frames(&mut y, frame).unwrap();
        assert_close_f64(&y, &want, &format!("forward frame={frame}"));

        gpu.inverse_frames(&mut y, frame).unwrap();
        let orig: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        assert_close_f64(&y, &orig, &format!("roundtrip frame={frame}"));
    }
}

/// More frames than one dispatch dimension holds (65535).
#[test]
fn many_frames_use_two_dimensional_dispatch() {
    let Some(gpu) = gpu() else { return };
    let frame = 4;
    let x = signal(frame * 70_000);
    let mut want: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    slantlet::forward_frames(&mut want, frame).unwrap();
    let mut y = x.clone();
    gpu.forward_frames(&mut y, frame).unwrap();
    assert_close_f64(&y, &want, "forward");
}

/// Data already on the GPU: encode, submit, read back.
#[test]
fn encode_on_resident_buffer() {
    use wgpu::util::DeviceExt;
    let Some(gpu) = gpu() else { return };
    let (frame, frames) = (1024, 8);
    let x = signal(frame * frames);
    let buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let out = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buf.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device().create_command_encoder(&Default::default());
    gpu.encode_forward(&mut enc, &buf, frame, frames).unwrap();
    gpu.encode_inverse(&mut enc, &buf, frame, frames).unwrap();
    gpu.encode_forward(&mut enc, &buf, frame, frames).unwrap();
    enc.copy_buffer_to_buffer(&buf, 0, &out, 0, buf.size());
    gpu.queue().submit([enc.finish()]);
    out.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let got: Vec<f32> = bytemuck::cast_slice(&out.get_mapped_range(..).unwrap()).to_vec();

    let mut want: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    slantlet::forward_frames(&mut want, frame).unwrap();
    assert_close_f64(&got, &want, "encoded");
}

#[test]
fn rejects_bad_frames_without_touching_input() {
    let Some(gpu) = gpu() else { return };
    let mut x = signal(100);
    let orig = x.clone();
    assert!(matches!(
        gpu.forward_frames(&mut x, 6),
        Err(GpuError::Shape(_))
    ));
    assert!(matches!(
        gpu.forward_frames(&mut x, 2 * MAX_FRAME_LEN),
        Err(GpuError::FrameTooLong(_))
    ));
    assert_eq!(x, orig);
    // A buffer smaller than frames * frame_len is rejected.
    let small = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device().create_command_encoder(&Default::default());
    assert!(matches!(
        gpu.encode_forward(&mut enc, &small, 16, 2),
        Err(GpuError::Shape(_))
    ));
}
