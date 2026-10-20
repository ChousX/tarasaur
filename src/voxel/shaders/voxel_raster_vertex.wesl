// Shared by every layer's raster shader: the vertex stage over pass3's
// vertices, the per-arena chunk data (group 3), and the per-pixel
// visibility and material lookups a fragment stage can make from it.
// Imported as `tarasaur::voxel::shaders::voxel_raster_vertex`.
//
// Groups 0 and 1 are Bevy's mesh view bind groups (view, lights, shadows,
// fog, light probes), bound by the raster pass; 2 is the layer's own
// material, declared by its shader; 3 the per-arena chunk data below.
//
// A layer's shader wraps `voxel_vertex` as its entry point:
//
//     @vertex
//     fn vs_main(input: VertexInput) -> VertexOutput { return voxel_vertex(input); }

import tarasaur::voxel::shaders::voxel_common::{BatchUniforms, ChunkMeta, flatten_sdf_idx};
import bevy_pbr::render::mesh_view_bindings::view;

// --- Group 3: per-arena chunk data, fragment-stage only. ---
@group(3) @binding(0) var<storage, read> chunk_meta: array<ChunkMeta>;
@group(3) @binding(1) var<storage, read> visibility_mask_buffer: array<u32>;
@group(3) @binding(2) var<storage, read> chunk_has_mask_buffer: array<u32>;
@group(3) @binding(3) var<uniform> vis_uniforms: BatchUniforms; // pass3's, reused
@group(3) @binding(4) var<storage, read> material_buffer: array<u32>;

struct VertexInput {
    @location(0) position: vec4<f32>, // .xyz = world position, .w = bitcast-packed material_id_a (low byte) + blend weight (next byte), uniform flag (bit 16), macro noise (bits 17-22)
    @location(1) normal: vec4<f32>,   // .xyz = normal, .w = bitcast-packed material_id_b (low byte) + real_slot (next 15 bits)
};

struct VertexOutput {
    // Invariant: the depth prepass and the shading pass must compute the
    // exact same depth, or shading would fail its depth test.
    @builtin(position) @invariant clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) @interpolate(flat) material_id_a: u32,
    @location(3) @interpolate(flat) material_id_b: u32,
    @location(4) blend_weight: f32,
    // Flat-interpolated (every vertex in a triangle is from the same
    // chunk, so this is constant across the triangle regardless of
    // interpolation mode; flat avoids any float-precision surprises).
    @location(5) @interpolate(flat) real_slot: u32,
    // (1, id) at a vertex whose material neighbourhood is all `id`, else
    // (0, 0); see uniform_material_near in pass3. Interpolated, so x is 1
    // only where every vertex of the triangle is uniform.
    @location(6) uniform_mat: vec2<f32>,
    // Large-scale noise for `macro_variation`, -1..1 (from pass3's macro_noise).
    @location(7) macro_n: f32,
};

// The vertex stage: unpacks pass3's vertex and projects it.
fn voxel_vertex(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.world_position = input.position.xyz;
    out.world_normal = input.normal.xyz;

    let packed_a = bitcast<u32>(input.position.w);
    let packed_b = bitcast<u32>(input.normal.w);

    out.material_id_a = packed_a & 0xFFu;
    let uniform_flag = f32((packed_a >> 16u) & 1u);
    out.uniform_mat = vec2<f32>(uniform_flag, uniform_flag * f32(out.material_id_a));
    out.macro_n = f32((packed_a >> 17u) & 0x3Fu) * (2.0 / 63.0) - 1.0;
    let weight_u8 = (packed_a >> 8u) & 0xFFu;
    out.blend_weight = f32(weight_u8) / 255.0; // 0.0 = fully id_b, 1.0 = fully id_a

    out.material_id_b = packed_b & 0xFFu;
    out.real_slot = (packed_b >> 8u) & 0x7FFFu; // see pack_material_b in pass3

    out.clip_position = view.clip_from_world * vec4<f32>(input.position.xyz, 1.0);

    return out;
}

// --- Material ids ---

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

// The two material ids around the fragment and the weight of the first,
// from the 8 voxels of the cell it lies in.
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

// --- Visibility ---

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
        return true; // uniformly visible chunk: no mask uploaded
    }
    let origin = chunk_meta[slot].chunk_world_origin;
    let local = (world_pos - origin) / vis_uniforms.voxel_size;
    let max_coord = f32(vis_uniforms.texture_size) - 1.0;
    let rounded = clamp(round(local), vec3<f32>(0.0), vec3<f32>(max_coord));
    return sample_visibility(slot, vec3<u32>(rounded));
}
