// Slantlet transform of independent power-of-two frames, one workgroup per
// frame. Mirrors `merge_level` / `split_level` in src/lib.rs, in f32.
//
// Each frame's (c, d) pairs live in workgroup memory: 2 × 2048 f32 = 16 KiB,
// exactly the WebGPU default limit, so frames hold up to 4096 samples. Every
// level reads into registers, synchronises, then writes, so threads never
// race on the shared arrays.

struct Params {
    frame_len: u32,
    frames: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
// (α, β) per level, index log2(m) - 1 for children of length m; f64-exact on the host.
@group(0) @binding(1) var<storage, read> rot: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;
const MAX_PAIRS: u32 = 2048u;
const PER: u32 = 4u; // MAX_PAIRS / 2 / WG parents per thread and level
const R: f32 = 0.70710678118654752; // 1/√2
const S2: f32 = 1.41421356237309505; // √2

var<workgroup> c: array<f32, MAX_PAIRS>;
var<workgroup> d: array<f32, MAX_PAIRS>;

@compute @workgroup_size(256)
fn forward(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let f = wg.y * nwg.x + wg.x;
    if f >= params.frames {
        return;
    }
    let n = params.frame_len;
    let base = f * n;
    let pairs = n / 2u;

    for (var k = t; k < pairs; k += WG) {
        let a = data[base + 2u * k];
        let b = data[base + 2u * k + 1u];
        c[k] = (a + b) * R;
        d[k] = (b - a) * R;
    }
    // All samples of the frame are consumed before details overwrite them.
    workgroupBarrier();

    var len = pairs;
    var m = 2u;
    var j = 0u;
    loop {
        if len <= 1u {
            break;
        }
        let nb = len / 2u;
        let r = rot[j];
        var pc: array<f32, PER>;
        var pd: array<f32, PER>;
        for (var i = 0u; i < PER; i++) {
            let b = t + i * WG;
            if b < nb {
                let cl = c[2u * b];
                let cr = c[2u * b + 1u];
                let dl = d[2u * b];
                let dr = d[2u * b + 1u];
                let dc = cr - cl;
                let sd = dl + dr;
                let s = (dr - dl) * R;
                let a = (r.y * dc - r.x * sd) * R;
                let o = base + n / m + 2u * b;
                data[o] = -(s + a) * R;
                data[o + 1u] = -(s - a) * R;
                pc[i] = (cl + cr) * R;
                pd[i] = (r.x * dc + r.y * sd) * R;
            }
        }
        workgroupBarrier();
        for (var i = 0u; i < PER; i++) {
            let b = t + i * WG;
            if b < nb {
                c[b] = pc[i];
                d[b] = pd[i];
            }
        }
        workgroupBarrier();
        len = nb;
        m = 2u * m;
        j++;
    }

    if t == 0u {
        data[base] = c[0];
        data[base + 1u] = -d[0];
    }
}

@compute @workgroup_size(256)
fn inverse(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let f = wg.y * nwg.x + wg.x;
    if f >= params.frames {
        return;
    }
    let n = params.frame_len;
    let base = f * n;
    let pairs = n / 2u;

    if t == 0u {
        c[0] = data[base];
        d[0] = -data[base + 1u];
    }
    workgroupBarrier();

    var len = 1u;
    var m = pairs;
    loop {
        if len >= pairs {
            break;
        }
        let r = rot[countTrailingZeros(m) - 1u];
        var c0: array<f32, PER>;
        var c1: array<f32, PER>;
        var d0: array<f32, PER>;
        var d1: array<f32, PER>;
        for (var i = 0u; i < PER; i++) {
            let b = t + i * WG;
            if b < len {
                let o = base + n / m + 2u * b;
                let g = -data[o];
                let jg = -data[o + 1u];
                let s = (g + jg) * R;
                let a = (g - jg) * R;
                let dc = (r.x * d[b] + r.y * a) * S2;
                let sd = (r.y * d[b] - r.x * a) * S2;
                let sc = c[b] * S2;
                let dd = s * S2;
                c0[i] = (sc - dc) * 0.5;
                c1[i] = (sc + dc) * 0.5;
                d0[i] = (sd - dd) * 0.5;
                d1[i] = (sd + dd) * 0.5;
            }
        }
        workgroupBarrier();
        for (var i = 0u; i < PER; i++) {
            let b = t + i * WG;
            if b < len {
                c[2u * b] = c0[i];
                c[2u * b + 1u] = c1[i];
                d[2u * b] = d0[i];
                d[2u * b + 1u] = d1[i];
            }
        }
        workgroupBarrier();
        len = 2u * len;
        m = m / 2u;
    }

    // Every detail has been read (last barrier); now write the samples.
    for (var k = t; k < pairs; k += WG) {
        data[base + 2u * k] = (c[k] - d[k]) * R;
        data[base + 2u * k + 1u] = (c[k] + d[k]) * R;
    }
}
