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

import tarasaur::voxel::shaders::voxel_common::{
    BatchUniforms, ChunkMeta, MATERIAL_BAND, flatten_sdf_idx, material_blend_weight, trilinear8,
};
import bevy_pbr::render::mesh_view_bindings::view;

// --- Group 3: per-arena chunk data, fragment-stage only. ---
@group(3) @binding(0) var<storage, read> chunk_meta: array<ChunkMeta>;
@group(3) @binding(1) var<storage, read> visibility_mask_buffer: array<u32>;
@group(3) @binding(2) var<storage, read> chunk_has_mask_buffer: array<u32>;
@group(3) @binding(3) var<uniform> vis_uniforms: BatchUniforms; // pass3's, reused
@group(3) @binding(4) var<storage, read> material_buffer: array<u32>;
// [slot] -> which of its chunk's surfaces the slot holds (SubSurface).
@group(3) @binding(5) var<storage, read> slot_sub_buffer: array<u32>;

// Which of its chunk's surfaces `slot` holds; 0 for most layers.
fn sub_surface(slot: u32) -> u32 {
    return slot_sub_buffer[slot];
}

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

// Material bytes are planar within a slot: every id, then, where the arena
// carries them, every boundary distance (see field/material.rs). An id word
// still holds four ids, as it did before distances existed.
fn material_elem(slot: u32, coord: vec3<i32>) -> vec2<u32> {
    let ts = vis_uniforms.texture_size;
    let c = vec3<u32>(clamp(coord, vec3<i32>(0), vec3<i32>(i32(ts) - 1)));
    let elems = ts * ts * ts;
    return vec2<u32>(slot * elems * vis_uniforms.material_stride, flatten_sdf_idx(c, ts));
}

fn material_byte(byte_idx: u32) -> u32 {
    return (material_buffer[byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu;
}

fn sample_material_id(slot: u32, coord: vec3<i32>) -> u32 {
    let e = material_elem(slot, coord);
    return material_byte(e.x + e.y);
}

// The boundary distance at `coord`, in world units. Stored in world units
// precisely so this stage needs nothing it does not have: there are no seam
// ranks in group 3, so a voxel-unit distance could not be read across an
// LOD seam here at all. An arena that carries none reads as "no boundary
// near", which makes the blend fall back to weighting the corners.
fn sample_material_distance(slot: u32, coord: vec3<i32>) -> f32 {
    if (vis_uniforms.material_stride < 2u) {
        return MATERIAL_BAND;
    }
    let ts = vis_uniforms.texture_size;
    let e = material_elem(slot, coord);
    return f32(material_byte(e.x + ts * ts * ts + e.y)) / 255.0 * MATERIAL_BAND;
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

    // The 8 corner ids. Scalars rather than an array: this runs per pixel,
    // and an indexable local array sits in scratch memory.
    let i0 = sample_material_id(slot, ib + vec3<i32>(0, 0, 0));
    let i1 = sample_material_id(slot, ib + vec3<i32>(1, 0, 0));
    let i2 = sample_material_id(slot, ib + vec3<i32>(0, 1, 0));
    let i3 = sample_material_id(slot, ib + vec3<i32>(1, 1, 0));
    let i4 = sample_material_id(slot, ib + vec3<i32>(0, 0, 1));
    let i5 = sample_material_id(slot, ib + vec3<i32>(1, 0, 1));
    let i6 = sample_material_id(slot, ib + vec3<i32>(0, 1, 1));
    let i7 = sample_material_id(slot, ib + vec3<i32>(1, 1, 1));

    // Early out: all 8 agree, which is true for almost every pixel away
    // from a material border.
    if (i0 == i1 && i0 == i2 && i0 == i3 && i0 == i4 && i0 == i5 && i0 == i6 && i0 == i7) {
        return FragBlend(i0, i0, 1.0);
    }

    // A mixed cell. Each voxel carries the distance to the nearest material
    // boundary, so give every label present its own signed field over the
    // cell (+d where a corner holds that label, -d elsewhere), interpolate,
    // and take the largest: the multiphase level-set rule, which puts the
    // boundary back inside the voxel and stays defined where three meet.
    let f = local - base;
    let d0 = sample_material_distance(slot, ib + vec3<i32>(0, 0, 0));
    let d1 = sample_material_distance(slot, ib + vec3<i32>(1, 0, 0));
    let d2 = sample_material_distance(slot, ib + vec3<i32>(0, 1, 0));
    let d3 = sample_material_distance(slot, ib + vec3<i32>(1, 1, 0));
    let d4 = sample_material_distance(slot, ib + vec3<i32>(0, 0, 1));
    let d5 = sample_material_distance(slot, ib + vec3<i32>(1, 0, 1));
    let d6 = sample_material_distance(slot, ib + vec3<i32>(0, 1, 1));
    let d7 = sample_material_distance(slot, ib + vec3<i32>(1, 1, 1));

    // Two labels is the overwhelming case: the second field is the negation
    // of the first, so one interpolation answers both and no list is built.
    // The first corner id that differs from i0: assigned last wins, so the
    // lowest-numbered differing corner does.
    var second_id = i0;
    second_id = select(second_id, i7, i7 != i0);
    second_id = select(second_id, i6, i6 != i0);
    second_id = select(second_id, i5, i5 != i0);
    second_id = select(second_id, i4, i4 != i0);
    second_id = select(second_id, i3, i3 != i0);
    second_id = select(second_id, i2, i2 != i0);
    second_id = select(second_id, i1, i1 != i0);
    var only_two = true;
    only_two = only_two && (i1 == i0 || i1 == second_id);
    only_two = only_two && (i2 == i0 || i2 == second_id);
    only_two = only_two && (i3 == i0 || i3 == second_id);
    only_two = only_two && (i4 == i0 || i4 == second_id);
    only_two = only_two && (i5 == i0 || i5 == second_id);
    only_two = only_two && (i6 == i0 || i6 == second_id);
    only_two = only_two && (i7 == i0 || i7 == second_id);
    if (only_two) {
        let v = trilinear8(
            d0, select(-d1, d1, i1 == i0),
            select(-d2, d2, i2 == i0), select(-d3, d3, i3 == i0),
            select(-d4, d4, i4 == i0), select(-d5, d5, i5 == i0),
            select(-d6, d6, i6 == i0), select(-d7, d7, i7 == i0),
            f,
        );
        return FragBlend(i0, second_id, material_blend_weight(v, -v));
    }

    // Three or more: rank the labels' fields and take the top two.
    var ids: array<u32, 8>;
    var n = 0u;
    let corner = array<u32, 8>(i0, i1, i2, i3, i4, i5, i6, i7);
    for (var i = 0u; i < 8u; i++) {
        var found = false;
        for (var j = 0u; j < n; j++) {
            found = found || (ids[j] == corner[i]);
        }
        if (!found) { ids[n] = corner[i]; n += 1u; }
    }
    var best = 0u;
    var best_phi = -1.0e30;
    var second = 0u;
    var second_phi = -1.0e30;
    for (var j = 0u; j < n; j++) {
        let l = ids[j];
        let v = trilinear8(
            select(-d0, d0, i0 == l), select(-d1, d1, i1 == l),
            select(-d2, d2, i2 == l), select(-d3, d3, i3 == l),
            select(-d4, d4, i4 == l), select(-d5, d5, i5 == l),
            select(-d6, d6, i6 == l), select(-d7, d7, i7 == l),
            f,
        );
        if (v > best_phi) {
            second_phi = best_phi;
            second = best;
            best_phi = v;
            best = j;
        } else if (v > second_phi) {
            second_phi = v;
            second = j;
        }
    }
    return FragBlend(ids[best], ids[second], material_blend_weight(best_phi, second_phi));
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
