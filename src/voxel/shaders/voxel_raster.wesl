// Import Bevy's built-in view uniform layout to ensure absolute memory layout alignment
import bevy_render::view::View;

@group(0) @binding(0) var<uniform> view: View;

@group(1) @binding(0) var texture_array: texture_2d_array<f32>;
@group(1) @binding(1) var texture_sampler: sampler;
@group(2) @binding(4) var<storage, read> material_buffer: array<u32>;

fn sample_material_id(slot: u32, coord: vec3<i32>) -> u32 {
    let ts = vis_uniforms.texture_size;
    let c = vec3<u32>(clamp(coord, vec3<i32>(0), vec3<i32>(i32(ts) - 1)));
    let elem = slot * ts * ts * ts + flatten_sdf_idx(c, ts);
    let word = material_buffer[elem / 4u];
    return (word >> ((elem % 4u) * 8u)) & 0xFFu;
}

struct FragBlend {
    id_a: u32,
    id_b: u32,
    w: f32, // weight of id_a, 0..1
}

fn fragment_material_blend(slot: u32, world_pos: vec3<f32>) -> FragBlend {
    let origin = chunk_meta[slot].chunk_world_origin;
    let local = (world_pos - origin) / vis_uniforms.voxel_size;
    let base = floor(local);
    let ib = vec3<i32>(base);

    // Pass 1: fetch the 8 corner ids only. No float math yet.
    var corner_ids: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let o = vec3<i32>(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
        corner_ids[i] = sample_material_id(slot, ib + o);
    }

    // Early out: all 8 corners agree, which is true for almost every pixel
    // away from a material border. Skips weights, dedupe and both ranking loops.
    let first = corner_ids[0];
    var all_same = true;
    for (var i = 1u; i < 8u; i++) {
        all_same = all_same && (corner_ids[i] == first);
    }
    if (all_same) {
        return FragBlend(first, first, 1.0);
    }

    // Pass 2: mixed cell. Trilinear corner weights, merged per unique id.
    let f = local - base;

    var ids: array<u32, 8>;
    var ws: array<f32, 8>;
    var n = 0u;

    for (var i = 0u; i < 8u; i++) {
        let wx = select(1.0 - f.x, f.x, (i & 1u) == 1u);
        let wy = select(1.0 - f.y, f.y, ((i >> 1u) & 1u) == 1u);
        let wz = select(1.0 - f.z, f.z, ((i >> 2u) & 1u) == 1u);
        let w = wx * wy * wz;
        let id = corner_ids[i];

        var found = false;
        for (var j = 0u; j < n; j++) {
            if (ids[j] == id) { ws[j] += w; found = true; break; }
        }
        if (!found) { ids[n] = id; ws[n] = w; n += 1u; }
    }

    var best = 0u;
    for (var j = 1u; j < n; j++) { if (ws[j] > ws[best]) { best = j; } }

    var second = best;
    var second_w = -1.0;
    for (var j = 0u; j < n; j++) {
        if (j != best && ws[j] > second_w) { second_w = ws[j]; second = j; }
    }

    return FragBlend(ids[best], ids[second], ws[best] / (ws[best] + second_w));
}

struct MaterialProperties {
    texture_scale: f32,
    blend_sharpness: f32,
    roughness_override: f32,
    metallic_override: f32,
}

@group(1) @binding(2) var<storage, read> material_properties: array<MaterialProperties>;

// --- Group 2: per-arena visibility data, fragment-stage only. ---
struct ChunkMeta {
    chunk_world_origin: vec3<f32>,
    active_list_pos: u32,
}

struct VisBatchUniforms {
    cell_count: u32,
    texture_size: u32,
    wg_per_chunk_z: u32, // unused here — reusing pass3's uniform buffer as-is
    voxel_size: f32,
}

@group(2) @binding(0) var<storage, read> chunk_meta: array<ChunkMeta>;
@group(2) @binding(1) var<storage, read> visibility_mask_buffer: array<u32>;
@group(2) @binding(2) var<storage, read> chunk_has_mask_buffer: array<u32>;
@group(2) @binding(3) var<uniform> vis_uniforms: VisBatchUniforms;

struct VertexInput {
    @location(0) position: vec4<f32>, // .xyz = world position, .w = bitcast-packed material_id_a (low byte) + blend weight (next byte)
    @location(1) normal: vec4<f32>,   // .xyz = normal, .w = bitcast-packed material_id_b (low byte) + real_slot (next 15 bits)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) @interpolate(flat) material_id_a: u32,
    @location(3) @interpolate(flat) material_id_b: u32,
    @location(4) blend_weight: f32,
    // NEW — flat-interpolated (every vertex in a triangle is from the same
    // chunk, so this is constant across the triangle regardless of
    // interpolation mode; flat avoids any float-precision surprises).
    @location(5) @interpolate(flat) real_slot: u32,
};

// Width of the transition band, measured in |normal| units. Smaller = harder,
// narrower seams. 0.15 is a moderate band; try 0.05-0.25.
const TRIPLANAR_BLEND_WIDTH: f32 = 0.15;

fn triplanar_weights(N: vec3<f32>, sharpness: f32) -> vec3<f32> {
    let a = abs(N);
    let m = max(a.x, max(a.y, a.z));

    // Per-material blend_sharpness now also narrows the band (1.0 = default width).
    let width = max(TRIPLANAR_BLEND_WIDTH / max(sharpness, 1.0), 0.02);

    // The dominant axis gets 1.0. Other axes get 0 until they come within
    // `width` of the dominant one, then ramp up linearly.
    var w = clamp((a - vec3<f32>(m - width)) / width, vec3<f32>(0.0), vec3<f32>(1.0));

    // Optional: smooth the ramp so the seam has no visible kink.
    w = w * w * (3.0 - 2.0 * w);

    let total = w.x + w.y + w.z; // always >= 1 because the dominant axis is 1
    return w / total;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.world_position = input.position.xyz;
    out.world_normal = input.normal.xyz;

    let packed_a = bitcast<u32>(input.position.w);
    let packed_b = bitcast<u32>(input.normal.w);

    out.material_id_a = packed_a & 0xFFu;
    let weight_u8 = (packed_a >> 8u) & 0xFFu;
    out.blend_weight = f32(weight_u8) / 255.0; // 0.0 = fully id_b, 1.0 = fully id_a

    out.material_id_b = packed_b & 0xFFu;
    out.real_slot = (packed_b >> 8u) & 0x7FFFu; // see pack_material_b in pass3

    out.clip_position = view.clip_from_world * vec4<f32>(input.position.xyz, 1.0);

    return out;
}

// --- Visibility sampling, ported from pass1/pass3's identical logic ---
fn flatten_sdf_idx(coord: vec3<u32>, texture_size: u32) -> u32 {
    return coord.z * texture_size * texture_size + coord.y * texture_size + coord.x;
}

fn mask_words_per_chunk() -> u32 {
    let elems = vis_uniforms.texture_size * vis_uniforms.texture_size * vis_uniforms.texture_size;
    return (elems + 31u) / 32u;
}

fn chunk_has_mask(slot: u32) -> bool {
    let word = chunk_has_mask_buffer[slot / 32u];
    return ((word >> (slot % 32u)) & 1u) != 0u;
}

fn sample_visibility(slot: u32, coord: vec3<u32>) -> bool {
    let local_bit = flatten_sdf_idx(coord, vis_uniforms.texture_size);
    let word_idx = slot * mask_words_per_chunk() + local_bit / 32u;
    let word = visibility_mask_buffer[word_idx];
    return ((word >> (local_bit % 32u)) & 1u) != 0u;
}

// Nearest-voxel sample at the fragment's actual interpolated world
// position, converted back to this chunk's local texture-space coords.
// This is what makes the discard boundary follow the visibility grid at
// per-pixel resolution instead of being frozen per-triangle at vertex time.
fn fragment_is_visible(slot: u32, world_pos: vec3<f32>) -> bool {
    if (!chunk_has_mask(slot)) {
        return true; // uniformly-visible chunk — see pass1's identical comment
    }
    let origin = chunk_meta[slot].chunk_world_origin;
    let local = (world_pos - origin) / vis_uniforms.voxel_size;
    let max_coord = f32(vis_uniforms.texture_size) - 1.0;
    let rounded = clamp(round(local), vec3<f32>(0.0), vec3<f32>(max_coord));
    return sample_visibility(slot, vec3<u32>(rounded));
}
// Weights below this are dropped (and the rest renormalized). Your
// triplanar_weights already returns exact 0 for non-dominant axes outside the
// blend band, so most pixels end up with a single projection.
const MIN_PROJ_WEIGHT: f32 = 0.01;

fn sample_triplanar(
    layer: i32,
    p: vec3<f32>,
    ddx_p: vec3<f32>,
    ddy_p: vec3<f32>,
    inv_scale: f32,
    N: vec3<f32>,
    sharpness: f32,
) -> vec4<f32> {
    var w = triplanar_weights(N, sharpness);
    w = select(vec3<f32>(0.0), w, w > vec3<f32>(MIN_PROJ_WEIGHT));
    w = w / (w.x + w.y + w.z); // dominant axis is always kept, so the sum is > 0

    var c = vec4<f32>(0.0);
    if (w.x > 0.0) {
        c += w.x * textureSampleGrad(texture_array, texture_sampler,
            p.yz * inv_scale, layer, ddx_p.yz * inv_scale, ddy_p.yz * inv_scale);
    }
    if (w.y > 0.0) {
        c += w.y * textureSampleGrad(texture_array, texture_sampler,
            p.xz * inv_scale, layer, ddx_p.xz * inv_scale, ddy_p.xz * inv_scale);
    }
    if (w.z > 0.0) {
        c += w.z * textureSampleGrad(texture_array, texture_sampler,
            p.xy * inv_scale, layer, ddx_p.xy * inv_scale, ddy_p.xy * inv_scale);
    }
    return c;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Derivatives must be taken in uniform control flow, so do it before the
    // discard and before any branching on per-pixel data.
    let ddx_p = dpdx(in.world_position);
    let ddy_p = dpdy(in.world_position);

    if (!fragment_is_visible(in.real_slot, in.world_position)) {
        discard;
    }

    let N = normalize(in.world_normal);
    let blend = fragment_material_blend(in.real_slot, in.world_position);

    // Decide which materials are actually needed.
    let t = smoothstep(0.35, 0.65, blend.w);
    let same = blend.id_a == blend.id_b;
    let need_a = same || t > 0.001;
    let need_b = !same && t < 0.999;

    var color = vec4<f32>(0.0);
    var color_a = vec4<f32>(0.0);
    var color_b = vec4<f32>(0.0);

    if (need_a) {
        let mat_a = material_properties[blend.id_a];
        color_a = sample_triplanar(
            i32(blend.id_a), in.world_position, ddx_p, ddy_p,
            1.0 / max(mat_a.texture_scale, 0.0001), N, mat_a.blend_sharpness);
    }
    if (need_b) {
        let mat_b = material_properties[blend.id_b];
        color_b = sample_triplanar(
            i32(blend.id_b), in.world_position, ddx_p, ddy_p,
            1.0 / max(mat_b.texture_scale, 0.0001), N, mat_b.blend_sharpness);
    }

    if (need_a && need_b) {
        color = mix(color_b, color_a, t);
    } else if (need_a) {
        color = color_a;
    } else {
        color = color_b;
    }

    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.2));
    let diffuse = max(dot(N, light_dir), 0.0);
    let ambient = 0.2;
    let lighting = diffuse + ambient;

    return vec4<f32>(color.rgb * lighting, 1.0);
}
