//! Frame-wise slantlet transform as a WGSL compute shader (feature `wgpu`).
//!
//! One workgroup transforms one frame entirely in workgroup memory, so a
//! single dispatch handles any number of frames. The GPU computes in `f32`
//! (WGSL has no `f64`), so results match the `f64` CPU transform to about
//! `1e-6` relative, not bit for bit.
//!
//! Use [`Gpu::forward_frames`] / [`Gpu::inverse_frames`] for data in host
//! memory (upload, dispatch, download), or [`Gpu::encode_forward`] /
//! [`Gpu::encode_inverse`] to transform a `wgpu::Buffer` that already lives
//! on the GPU as part of a larger pipeline. Copying data to and from the GPU
//! usually costs more than the transform itself; see BENCH.md.

use std::{fmt, sync::mpsc};

use wgpu::util::DeviceExt;

use crate::Error;

/// Longest frame the shader handles: its `(c, d)` pairs fill the 16 KiB of
/// workgroup memory every WebGPU device provides.
pub const MAX_FRAME_LEN: usize = 4096;

const SHADER: &str = include_str!("slantlet.wgsl");
const WORKGROUPS_PER_DIM: usize = 65_535;

/// Errors from the GPU transform.
#[derive(Debug)]
#[non_exhaustive]
pub enum GpuError {
    /// No suitable GPU adapter was found.
    NoAdapter(wgpu::RequestAdapterError),
    /// The adapter refused to create a device.
    Device(wgpu::RequestDeviceError),
    /// Invalid frame length or buffer size; nothing was transformed.
    Shape(Error),
    /// `frame_len` exceeds [`MAX_FRAME_LEN`].
    FrameTooLong(usize),
    /// Reading results back from the GPU failed.
    Readback(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoAdapter(e) => write!(f, "no GPU adapter: {e}"),
            GpuError::Device(e) => write!(f, "cannot create GPU device: {e}"),
            GpuError::Shape(e) => e.fmt(f),
            GpuError::FrameTooLong(n) => {
                write!(
                    f,
                    "GPU frames hold at most {MAX_FRAME_LEN} samples, got {n}"
                )
            }
            GpuError::Readback(e) => write!(f, "GPU readback failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<Error> for GpuError {
    fn from(e: Error) -> Self {
        GpuError::Shape(e)
    }
}

/// Compiled slantlet compute pipelines on one `wgpu` device.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    forward: wgpu::ComputePipeline,
    inverse: wgpu::ComputePipeline,
    rot: wgpu::Buffer,
}

impl Gpu {
    /// Opens the default adapter (honouring `WGPU_ADAPTER_NAME`,
    /// `WGPU_BACKEND` and `WGPU_POWER_PREF`) and builds the pipelines.
    pub fn new() -> Result<Self, GpuError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .map_err(GpuError::NoAdapter)?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(GpuError::Device)?;
        Ok(Self::from_device(device, queue))
    }

    /// Builds the pipelines on an existing device, e.g. one shared with a
    /// renderer. The device needs default limits.
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slantlet"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slantlet"),
            entries: &[
                storage(0, false),
                storage(1, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slantlet"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (forward, inverse) = (pipeline("forward"), pipeline("inverse"));

        // (α, β) for children of length m = 2, 4, …, MAX_FRAME_LEN / 2, in f64.
        let rot: Vec<f32> = (1..MAX_FRAME_LEN.trailing_zeros())
            .flat_map(|j| {
                let (a, b) = crate::rotation(1 << j);
                [a as f32, b as f32]
            })
            .collect();
        let rot = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("slantlet rotations"),
            contents: bytemuck::cast_slice(&rot),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            device,
            queue,
            layout,
            forward,
            inverse,
            rot,
        }
    }

    /// The device the pipelines live on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue used by [`forward_frames`](Self::forward_frames) and
    /// [`inverse_frames`](Self::inverse_frames).
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// GPU version of [`forward_frames`](crate::forward_frames) for `f32`
    /// data in host memory.
    ///
    /// Full frames run on the GPU; a shorter remainder is transformed on the
    /// CPU exactly as `forward_frames` would. `frame_len` must be a power of
    /// two no larger than [`MAX_FRAME_LEN`]; on error `x` is unchanged.
    pub fn forward_frames(&self, x: &mut [f32], frame_len: usize) -> Result<(), GpuError> {
        self.run_frames(x, frame_len, &self.forward, crate::forward_frames)
    }

    /// Inverse of [`forward_frames`](Self::forward_frames) with the same
    /// `frame_len`.
    pub fn inverse_frames(&self, y: &mut [f32], frame_len: usize) -> Result<(), GpuError> {
        self.run_frames(y, frame_len, &self.inverse, crate::inverse_frames)
    }

    /// Records a forward transform of the first `frames * frame_len` `f32`
    /// values of `data` (which needs `STORAGE` usage) into `encoder`, in
    /// place. Nothing runs until the encoder is submitted.
    pub fn encode_forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        data: &wgpu::Buffer,
        frame_len: usize,
        frames: usize,
    ) -> Result<(), GpuError> {
        self.encode(encoder, &self.forward, data, frame_len, frames)
    }

    /// Inverse of [`encode_forward`](Self::encode_forward).
    pub fn encode_inverse(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        data: &wgpu::Buffer,
        frame_len: usize,
        frames: usize,
    ) -> Result<(), GpuError> {
        self.encode(encoder, &self.inverse, data, frame_len, frames)
    }

    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        data: &wgpu::Buffer,
        frame_len: usize,
        frames: usize,
    ) -> Result<(), GpuError> {
        check_frame_len(frame_len)?;
        let len = frames
            .checked_mul(frame_len)
            .filter(|&l| l as u64 * 4 <= data.size());
        let (Some(_), Ok(frame_len32), Ok(frames32)) =
            (len, u32::try_from(frame_len), u32::try_from(frames))
        else {
            return Err(Error::DimensionMismatch {
                len: data.size() as usize / 4,
                width: frame_len,
                height: frames,
            }
            .into());
        };
        if frames == 0 || frame_len < 2 {
            return Ok(()); // nothing to do; a 1-sample frame is the identity
        }
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("slantlet params"),
                contents: bytemuck::cast_slice(&[frame_len32, frames32, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slantlet"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.rot.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let gx = frames.min(WORKGROUPS_PER_DIM);
        let gy = frames.div_ceil(gx);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("slantlet"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(gx as u32, gy as u32, 1);
        Ok(())
    }

    fn run_frames(
        &self,
        x: &mut [f32],
        frame_len: usize,
        pipeline: &wgpu::ComputePipeline,
        cpu: fn(&mut [f64], usize) -> Result<(), Error>,
    ) -> Result<(), GpuError> {
        check_frame_len(frame_len)?;
        let full = x.len() / frame_len * frame_len;
        let (frames, tail) = x.split_at_mut(full);

        // Largest batch of whole frames one storage binding can hold.
        let max_bytes = self.device.limits().max_storage_buffer_binding_size as usize;
        let batch = (max_bytes / 4 / frame_len).min(WORKGROUPS_PER_DIM * WORKGROUPS_PER_DIM);
        for chunk in frames.chunks_mut(batch.max(1) * frame_len) {
            self.run_chunk(chunk, frame_len, pipeline)?;
        }

        if !tail.is_empty() {
            let mut t: Vec<f64> = tail.iter().map(|&v| f64::from(v)).collect();
            cpu(&mut t, frame_len)?;
            for (dst, src) in tail.iter_mut().zip(t) {
                *dst = src as f32;
            }
        }
        Ok(())
    }

    fn run_chunk(
        &self,
        chunk: &mut [f32],
        frame_len: usize,
        pipeline: &wgpu::ComputePipeline,
    ) -> Result<(), GpuError> {
        let bytes = (chunk.len() * 4) as u64;
        let data = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("slantlet data"),
                contents: bytemuck::cast_slice(chunk),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slantlet readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(
            &mut encoder,
            pipeline,
            &data,
            frame_len,
            chunk.len() / frame_len,
        )?;
        encoder.copy_buffer_to_buffer(&data, 0, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);

        let (tx, rx) = mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Readback(e.to_string()))?;
        rx.recv()
            .map_err(|e| GpuError::Readback(e.to_string()))?
            .map_err(|e| GpuError::Readback(e.to_string()))?;
        {
            let view = staging
                .get_mapped_range(..)
                .map_err(|e| GpuError::Readback(e.to_string()))?;
            chunk.copy_from_slice(bytemuck::cast_slice(&view));
        }
        staging.unmap();
        Ok(())
    }
}

fn check_frame_len(frame_len: usize) -> Result<(), GpuError> {
    crate::check_len(frame_len)?;
    if frame_len > MAX_FRAME_LEN {
        return Err(GpuError::FrameTooLong(frame_len));
    }
    Ok(())
}
