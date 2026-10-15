// Surface nets, second stage: for each active cell from pass1, places its
// vertex, computes its normal and material blend, stitches LOD seams, and
// emits quads into the slot's fixed vertex/index range.

import tarasaur::voxel::shaders::voxel_common::{
    BatchUniforms, ChunkMeta, apron_unit_shift, decode_sdf, flatten_sdf_idx,
    rank_log2_size, seam_rank,
};

struct Vertex {
    position: vec4<f32>, // .xyz = world position, .w = bitcast-packed material_id_a (low byte) + blend weight (next byte)
    normal: vec4<f32>,   // .xyz = normal, .w = bitcast-packed material_id_b (low byte) + real_slot (next 15 bits)
}

struct IndirectDrawArgs {
    index_count: atomic<u32>,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<storage, read> sdf_buffer: array<u32>;
// pass1's per-dispatch maps: [vertex] -> cell and [cell] -> vertex.
@group(0) @binding(1) var<storage, read> active_cells: array<u32>;
@group(0) @binding(2) var<storage, read> cell_vertex: array<u32>;
@group(0) @binding(3) var<storage, read_write> vertex_buffer: array<Vertex>;
@group(0) @binding(4) var<storage, read_write> final_index_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> indirect_args: array<IndirectDrawArgs>;
@group(0) @binding(6) var<uniform> uniforms: BatchUniforms;
@group(0) @binding(7) var<storage, read> chunk_meta: array<ChunkMeta>;
// Per slot, CPU-written: each slot owns [base, base + count) vertices and
// 18x that range of indices, so re-meshing one chunk never moves another.
@group(0) @binding(8) var<storage, read> slot_vertex_base: array<u32>;
@group(0) @binding(9) var<storage, read> slot_vertex_count: array<u32>;
// Range-guard telemetry: [trips, slot, gpu_count, cpu_count]; see map_guard_readbacks.
@group(0) @binding(12) var<storage, read_write> guard_debug: array<atomic<u32>, 4>;
@group(0) @binding(13) var<storage, read> chunk_counts: array<u32>; // [chunk] -> active cells, from pass1
@group(0) @binding(10) var<storage, read> active_slot_map: array<u32>;
@group(0) @binding(11) var<storage, read> material_buffer: array<u32>;

fn get_cell_index(local_coord: vec3<u32>, cell_offset: u32) -> u32 {
    return cell_offset + local_coord.x
        + (local_coord.y * uniforms.cell_count)
        + (local_coord.z * uniforms.cell_count * uniforms.cell_count);
}

fn sdf_offset_for(slot: u32) -> u32 {
    return slot * uniforms.texture_size * uniforms.texture_size * uniforms.texture_size;
}

fn sample_material_id(material_base: u32, elem_idx: u32) -> u32 {
    let word_idx = (material_base + elem_idx) / 4u;
    let byte_shift = ((material_base + elem_idx) % 4u) * 8u;
    let word = material_buffer[word_idx];
    return (word >> byte_shift) & 0xFFu;
}

fn cell_offset_for(chunk_idx: u32) -> u32 {
    return chunk_idx * uniforms.cell_count * uniforms.cell_count * uniforms.cell_count;
}

// --- LOD seams ---
// Apron voxels from a neighbour at another LOD hold that neighbour's voxel
// units. Along each apron axis the second layer sits one step past the
// boundary, a step being a voxel of the coarser of this chunk and the face
// neighbour across that axis (see fill_apron_with). The functions below
// turn texels back into this chunk's units and positions. Set from
// ChunkMeta at the top of cs_main.
var<private> seam_ranks: vec2<u32>;
var<private> own_log2: i32;
// Per axis, log2 of this chunk's voxels per apron step.
var<private> face_shift: vec3<i32>;
// Cells with every coordinate in [seam_reach, n) can't be shared across a
// seam (see seam_cell); 0 when all 27 chunks around share one LOD.
var<private> seam_reach: u32;

fn init_seams(ranks: vec2<u32>) {
    seam_ranks = ranks;
    let own = seam_rank(ranks, vec3<i32>(0));
    own_log2 = rank_log2_size(own);
    let axes = array<vec3<i32>, 3>(vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, 0, 1));
    for (var a = 0u; a < 3u; a++) {
        face_shift[a] = max(own_log2 - seam_log2(axes[a]), 0);
    }
    // All 27 the same rank: 2 bits per entry, 16 in the first word, 11 in
    // the second.
    if (all(ranks == own * vec2<u32>(0x55555555u, 0x155555u))) {
        seam_reach = 0u;
        return;
    }
    // A shared cell starts at 0 on some axis and spans one lattice step of
    // the coarsest chunk around, at most.
    var coarsest = own_log2;
    for (var i = 0u; i < 27u; i++) {
        let r = (ranks[i / 16u] >> (2u * (i % 16u))) & 3u;
        coarsest = min(coarsest, rank_log2_size(r));
    }
    seam_reach = 1u << u32(own_log2 - coarsest);
}

// log2 of the voxels per edge of the chunk at `offset` (-1..1 per axis).
// A chunk the apron can't read counts as this chunk's LOD (see
// seam_lod_ranks).
fn seam_log2(offset: vec3<i32>) -> i32 {
    return rank_log2_size(seam_rank(seam_ranks, offset));
}

// log2(own size / source size) for the voxels at texel `t`: > 0 inside a
// coarser neighbour's apron, < 0 inside a finer one's, else 0.
fn texel_lod_shift(t: vec3<u32>) -> i32 {
    return apron_unit_shift(seam_ranks, uniforms.cell_count - 1u, t);
}

// Where texel `t` sits, in this chunk's voxels.
fn texel_pos(t: vec3<u32>) -> vec3<f32> {
    let n = uniforms.cell_count - 1u;
    let far = f32(n) + ldexp(vec3<f32>(1.0), face_shift);
    return select(vec3<f32>(t), far, t > vec3<u32>(n));
}

// Inverse of texel_pos, for continuous positions.
fn texel_coord(p: vec3<f32>) -> vec3<f32> {
    let n = vec3<f32>(f32(uniforms.cell_count - 1u));
    return select(p, n + ldexp(p - n, -face_shift), p > n);
}

// SDF at texel `t`, in this chunk's voxels.
fn sdf_at(sdf_base: u32, t: vec3<u32>) -> f32 {
    let i = sdf_base + flatten_sdf_idx(t, uniforms.texture_size);
    return ldexp(decode_sdf(sdf_buffer[i >> 1u], i), texel_lod_shift(t));
}

fn sample_sdf(sdf_base: u32, coord: vec3<i32>) -> f32 {
    let max_coord = i32(uniforms.texture_size) - 1;
    let clamped = clamp(coord, vec3<i32>(0), vec3<i32>(max_coord));
    return sdf_at(sdf_base, vec3<u32>(clamped));
}

// Trilinear at `pos`, in this chunk's voxels.
fn sample_sdf_trilinear(sdf_base: u32, pos: vec3<f32>) -> f32 {
    let p = texel_coord(pos);
    let ip = vec3<i32>(floor(p));
    let f = fract(p);
    let c000 = sample_sdf(sdf_base, ip + vec3<i32>(0, 0, 0));
    let c100 = sample_sdf(sdf_base, ip + vec3<i32>(1, 0, 0));
    let c010 = sample_sdf(sdf_base, ip + vec3<i32>(0, 1, 0));
    let c110 = sample_sdf(sdf_base, ip + vec3<i32>(1, 1, 0));
    let c001 = sample_sdf(sdf_base, ip + vec3<i32>(0, 0, 1));
    let c101 = sample_sdf(sdf_base, ip + vec3<i32>(1, 0, 1));
    let c011 = sample_sdf(sdf_base, ip + vec3<i32>(0, 1, 1));
    let c111 = sample_sdf(sdf_base, ip + vec3<i32>(1, 1, 1));
    let x0 = mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y);
    let x1 = mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y);
    return mix(x0, x1, f.z);
}

fn compute_normal(sdf_base: u32, pos: vec3<f32>) -> vec3<f32> {
    let h = 0.1;
    let dx = sample_sdf_trilinear(sdf_base, pos + vec3<f32>(h, 0.0, 0.0)) - sample_sdf_trilinear(sdf_base, pos - vec3<f32>(h, 0.0, 0.0));
    let dy = sample_sdf_trilinear(sdf_base, pos + vec3<f32>(0.0, h, 0.0)) - sample_sdf_trilinear(sdf_base, pos - vec3<f32>(0.0, h, 0.0));
    let dz = sample_sdf_trilinear(sdf_base, pos + vec3<f32>(0.0, 0.0, h)) - sample_sdf_trilinear(sdf_base, pos - vec3<f32>(0.0, 0.0, h));
    let norm = vec3<f32>(dx, dy, dz);
    let len = length(norm);
    if (len > 0.00001) {
        return normalize(norm);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

struct MaterialBlend {
    id_a: u32,
    id_b: u32,
    weight_u8: u32, // weight of id_a, 0-255; id_b implicitly gets (255 - weight_u8)
}

// Corner i of the cell is texel `lo + corners[i] * step`; `frac` is the
// vertex's position within it, 0..1 per axis.
fn tally_corner_materials(material_base: u32, lo: vec3<u32>, step: vec3<u32>, frac: vec3<f32>) -> MaterialBlend {
    let corners = array<vec3<u32>, 8>(
        vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u),
        vec3<u32>(0u, 1u, 0u), vec3<u32>(1u, 1u, 0u),
        vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u),
        vec3<u32>(0u, 1u, 1u), vec3<u32>(1u, 1u, 1u)
    );

    var corner_weights: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let c = corners[i];
        let wx = select(1.0 - frac.x, frac.x, c.x == 1u);
        let wy = select(1.0 - frac.y, frac.y, c.y == 1u);
        let wz = select(1.0 - frac.z, frac.z, c.z == 1u);
        corner_weights[i] = wx * wy * wz;
    }

    var uniq_ids: array<u32, 8>;
    var uniq_weights: array<f32, 8>;
    var uniq_count = 0u;

    for (var i = 0u; i < 8u; i++) {
        let mat_id = sample_material_id(material_base, flatten_sdf_idx(lo + corners[i] * step, uniforms.texture_size));
        let w = corner_weights[i];

        var found = false;
        for (var j = 0u; j < uniq_count; j++) {
            if (uniq_ids[j] == mat_id) {
                uniq_weights[j] = uniq_weights[j] + w;
                found = true;
                break;
            }
        }
        if (!found) {
            uniq_ids[uniq_count] = mat_id;
            uniq_weights[uniq_count] = w;
            uniq_count = uniq_count + 1u;
        }
    }

    var best_idx = 0u;
    var best_w = -1.0;
    for (var j = 0u; j < uniq_count; j++) {
        if (uniq_weights[j] > best_w) {
            best_w = uniq_weights[j];
            best_idx = j;
        }
    }

    var second_idx = best_idx;
    var second_w = -1.0;
    for (var j = 0u; j < uniq_count; j++) {
        if (j != best_idx && uniq_weights[j] > second_w) {
            second_w = uniq_weights[j];
            second_idx = j;
        }
    }

    let id_a = uniq_ids[best_idx];
    var id_b = id_a;
    var weight_u8 = 255u;
    if (second_w >= 0.0 && (best_w + second_w) > 0.00001) {
        id_b = uniq_ids[second_idx];
        let norm_a = best_w / (best_w + second_w);
        weight_u8 = u32(clamp(norm_a * 255.0, 0.0, 255.0));
    }

    // Canonical order: lower id is always "a". Weight is always "amount of the
    // lower id", so every vertex touching the same pair agrees on direction.
    if (id_b < id_a) {
        return MaterialBlend(id_b, id_a, 255u - weight_u8);
    }
    return MaterialBlend(id_a, id_b, weight_u8);
}

fn pack_material_a(id_a: u32, weight_u8: u32) -> f32 {
    let packed = (id_a & 0xFFu) | ((weight_u8 & 0xFFu) << 8u) | 0x3F800000u;
    return bitcast<f32>(packed);
}

// real_slot occupies bits 8-22 (15 bits). id_b keeps bits 0-7. The fragment
// shader unpacks real_slot for its per-pixel visibility discard test.
fn pack_material_b(id_b: u32, real_slot: u32) -> f32 {
    let packed = (id_b & 0xFFu) | ((real_slot & 0x7FFFu) << 8u) | 0x3F800000u;
    return bitcast<f32>(packed);
}

// Surface-nets vertex of the cell whose corner i is texel
// `lo + corners[i] * step`, in this chunk's voxels; .w = crossing edges.
fn cell_surface_vertex(sdf_base: u32, lo: vec3<u32>, step: vec3<u32>) -> vec4<f32> {
    let corners = array<vec3<u32>, 8>(
        vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u),
        vec3<u32>(0u, 1u, 0u), vec3<u32>(1u, 1u, 0u),
        vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u),
        vec3<u32>(0u, 1u, 1u), vec3<u32>(1u, 1u, 1u)
    );
    let edges = array<vec2<u32>, 12>(
        vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(4u, 5u), vec2<u32>(6u, 7u),
        vec2<u32>(0u, 2u), vec2<u32>(1u, 3u), vec2<u32>(4u, 6u), vec2<u32>(5u, 7u),
        vec2<u32>(0u, 4u), vec2<u32>(1u, 5u), vec2<u32>(2u, 6u), vec2<u32>(3u, 7u)
    );

    var sdfs: array<f32, 8>;
    var pos: array<vec3<f32>, 8>;
    for (var i = 0u; i < 8u; i++) {
        let t = lo + corners[i] * step;
        sdfs[i] = sdf_at(sdf_base, t);
        pos[i] = texel_pos(t);
    }

    var vert_pos = vec3<f32>(0.0);
    var edge_count = 0.0;
    for (var i = 0u; i < 12u; i++) {
        let c0 = edges[i].x;
        let c1 = edges[i].y;
        let v0 = sdfs[c0];
        let v1 = sdfs[c1];
        if ((v0 <= 0.0) != (v1 <= 0.0)) {
            // Prevent divide-by-zero or step artifacts
            let diff = v1 - v0;
            let t = select(0.5, -v0 / diff, abs(diff) > 0.00001);

            vert_pos += mix(pos[c0], pos[c1], clamp(t, 0.0, 1.0));
            edge_count += 1.0;
        }
    }

    if (edge_count > 0.0) {
        return vec4<f32>(vert_pos / edge_count, edge_count);
    }
    return vec4<f32>((pos[0] + pos[7]) * 0.5, 0.0);
}

// --- Shared seam cells ---
// A cell next to a coarser chunk takes the vertex of the coarse cell it
// lies in, so every chunk that meshes a quad touching that spot uses the
// same vertex and the meshes meet. Each chunk works it out in the frame of
// the cell's owner O (the chunk whose voxels it lies in: this one, or the
// +neighbour whose apron it is in), from the LODs of the chunks around O.
//
// "Below" O are the chunks at O + d, d in {-1, 0}^3 \ 0; one touches a
// lattice cell of O if the cell starts at 0 on every axis where d = -1.

// The 7 offsets d to O's below-neighbours.
fn below_offset(i: u32) -> vec3<i32> {
    return -vec3<i32>(vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u));
}

// The coarsest touching below-neighbour of the O cell starting at 0 on the
// axes in `zero`, as log2(O size / its size), and 0 if none is coarser.
fn seam_shift(o: vec3<i32>, o_log2: i32, zero: vec3<bool>) -> i32 {
    var s = 0;
    for (var i = 1u; i < 8u; i++) {
        let d = below_offset(i);
        if (any((d < vec3<i32>(0)) & !zero)) {
            continue;
        }
        s = max(s, o_log2 - seam_log2(o + d));
    }
    return s;
}

// The coarsest below-neighbour, as log2(O size / its size), whose first
// lattice cell holds the O voxel `id_o`; 0 if none is coarser.
fn seam_depth_shift(o: vec3<i32>, o_log2: i32, id_o: vec3<u32>) -> i32 {
    var s = 0;
    for (var i = 1u; i < 8u; i++) {
        let d = below_offset(i);
        let sd = o_log2 - seam_log2(o + d);
        if (sd <= 0) {
            continue;
        }
        let inside = id_o < vec3<u32>(1u << u32(sd));
        if (all(inside | (d == vec3<i32>(0)))) {
            s = max(s, sd);
        }
    }
    return s;
}

// Axes the shared cell (lattice 2^s, starting at 0 on `zero`) is flattened
// on: onto its low face along b when a touching below-neighbour across b
// can't see one lattice step past that face. Its apron along b steps by
// its own voxel or its face neighbour's, whichever is coarser (see
// fill_apron_with), so it sees the step only if that is the lattice's. The
// face it flattens onto lies at the boundary every one of them can read.
fn seam_collapse(o: vec3<i32>, o_log2: i32, zero: vec3<bool>, s: i32) -> vec3<bool> {
    var collapse = vec3<bool>(false);
    for (var i = 1u; i < 8u; i++) {
        let d = below_offset(i);
        if (any((d < vec3<i32>(0)) & !zero)) {
            continue;
        }
        let l = seam_log2(o + d);
        for (var b = 0u; b < 3u; b++) {
            if (d[b] == 0) {
                continue;
            }
            var across = d;
            across[b] = 0;
            let seen = o_log2 - min(l, seam_log2(o + across));
            if (seen != s) {
                collapse[b] = true;
            }
        }
    }
    return collapse;
}

struct SeamCell {
    lo: vec3<u32>,
    step: vec3<u32>,
    vert: vec4<f32>, // .w = 0: none, use the cell's own corners
}

// The shared vertex for cell `id`, in this chunk's texels (see above).
fn seam_cell(sdf_base: u32, id: vec3<u32>) -> SeamCell {
    var out = SeamCell(id, vec3<u32>(1u), vec4<f32>(0.0));
    let n = uniforms.cell_count - 1u;
    if (all((id >= vec3<u32>(seam_reach)) & (id < vec3<u32>(n)))) {
        return out;
    }
    let o = vec3<i32>(id >= vec3<u32>(n));
    let o_log2 = seam_log2(o);
    // > 0: O is coarser than this chunk.
    let up = own_log2 - o_log2;
    let up_u = u32(max(up, 0));
    let down_u = u32(max(-up, 0));
    // The O voxel this cell starts at (the lowest, if it spans several).
    var id_o = (id >> vec3<u32>(up_u)) << vec3<u32>(down_u);
    id_o = select(id_o, vec3<u32>(0u), o == vec3<i32>(1));

    // The lattice: the coarsest below-neighbour whose first lattice cell
    // holds this one, grown until the cell touches no coarser chunk. Every
    // cell inside that first lattice cell snaps, not just the row against
    // the neighbour: a coarse vertex can sit up to a whole coarse cell deep,
    // past the next row's fine vertices, and the strip between them folded
    // back over itself and z-fought.
    var s = seam_depth_shift(o, o_log2, id_o);
    var lo = (id_o >> vec3<u32>(u32(s))) << vec3<u32>(u32(s));
    for (var k = 0u; k < 3u; k++) {
        let s2 = seam_shift(o, o_log2, lo == vec3<u32>(0u));
        if (s2 == s) {
            break;
        }
        s = s2;
        lo = (id_o >> vec3<u32>(u32(s))) << vec3<u32>(u32(s));
    }
    let collapse = seam_collapse(o, o_log2, lo == vec3<u32>(0u), s);
    if (s == 0 && up == 0 && !any(collapse)) {
        return out; // an ordinary cell of our own
    }

    // To our texels. Along O's apron axes our two layers are its boundary
    // and one of our apron steps past it, which must be the lattice step
    // unless flattened.
    var lo_t = vec3<u32>(0u);
    var step = vec3<u32>(0u);
    let r = 1u << u32(s);
    for (var a = 0u; a < 3u; a++) {
        if (o[a] == 1) {
            lo_t[a] = n;
            step[a] = 1u;
            if (face_shift[a] - own_log2 != s - o_log2 && !collapse[a]) {
                return out;
            }
        } else {
            lo_t[a] = (lo[a] << up_u) >> down_u;
            step[a] = (r << up_u) >> down_u;
        }
    }
    step = select(step, vec3<u32>(0u), collapse);
    out.lo = lo_t;
    out.step = step;
    out.vert = cell_surface_vertex(sdf_base, lo_t, step);
    // The surface misses the shared cell (a feature finer than it, or
    // flattened away): every chunk sharing it still needs one vertex, so
    // not our own cell's but a point near the surface from its corners.
    if (out.vert.w == 0.0) {
        out.vert = vec4<f32>(relaxed_vertex(sdf_base, lo_t, step), 1.0);
    }
    return out;
}

// One Newton step from the centre of the cell whose corner i is texel
// `lo + corners[i] * step` toward the surface, along the cell's trilinear
// gradient there, clamped to the cell. Flat (step 0) axes stay put.
fn relaxed_vertex(sdf_base: u32, lo: vec3<u32>, step: vec3<u32>) -> vec3<f32> {
    var p_lo = vec3<f32>(0.0);
    var p_hi = vec3<f32>(0.0);
    var f_lo = vec3<f32>(0.0);
    var f_hi = vec3<f32>(0.0);
    var f_sum = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let c = vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
        let t = lo + c * step;
        let f = sdf_at(sdf_base, t);
        let p = texel_pos(t);
        let hi = c == vec3<u32>(1u);
        p_hi += select(vec3<f32>(0.0), p, hi);
        p_lo += select(p, vec3<f32>(0.0), hi);
        f_hi += select(vec3<f32>(0.0), vec3<f32>(f), hi);
        f_lo += select(vec3<f32>(f), vec3<f32>(0.0), hi);
        f_sum += f;
    }
    // Four corners on each side per axis.
    let lo_face = p_lo * 0.25;
    let hi_face = p_hi * 0.25;
    let span = hi_face - lo_face;
    let grad = select(vec3<f32>(0.0), (f_hi - f_lo) * 0.25 / span, span > vec3<f32>(0.0));
    let centre = (lo_face + hi_face) * 0.5;
    let g2 = dot(grad, grad);
    var p = centre;
    if (g2 > 1e-12) {
        p = centre - grad * (f_sum * 0.125 / g2);
    }
    return clamp(p, lo_face, hi_face);
}

// An `axis` edge at id[axis] == n runs from the +axis neighbour's first
// voxel layer to its second: that neighbour emits it as its own
// id[axis] == 0 edge (snapped to our grid if it is finer). Emitting it here
// too drew every such quad twice, from two chunks whose vertices and
// normals differ slightly, so they z-fought; across a LOD seam the normals
// differ a lot and it showed.
fn owns_edge(id: vec3<u32>, axis: u32) -> bool {
    return id[axis] < uniforms.cell_count - 1u;
}

// One invocation per active cell (pass1's list), not per cell: only a few
// percent of a chunk's cells carry surface. Dispatched as x = vertex within
// the chunk, y = chunk; x is sized from the CPU's per-chunk count.
@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vertex = global_id.x;
    let chunk_idx = global_id.y;

    // chunk_idx = position in this dispatch. real_slot indexes every
    // per-slot buffer (sdf, materials, chunk_meta, ranges, indirect_args).
    let real_slot = active_slot_map[chunk_idx];
    let cell_base = cell_offset_for(chunk_idx); // active_cells/cell_vertex are per-dispatch scratch

    // The range was sized from a CPU count of the same test pass1 uses. If
    // they ever disagreed, writing on would spill into another chunk's
    // range, so emit nothing instead.
    let gpu_count = chunk_counts[chunk_idx];
    if (gpu_count > slot_vertex_count[real_slot]) {
        if (vertex == 0u) {
            atomicAdd(&guard_debug[0], 1u);
            atomicStore(&guard_debug[1], real_slot);
            atomicStore(&guard_debug[2], gpu_count);
            atomicStore(&guard_debug[3], slot_vertex_count[real_slot]);
        }
        return;
    }
    if (vertex >= gpu_count) {
        return;
    }

    let cell = active_cells[cell_base + vertex];
    let cc = uniforms.cell_count;
    let id = vec3<u32>(cell % cc, (cell / cc) % cc, cell / (cc * cc));
    let cmeta = chunk_meta[real_slot];
    init_seams(cmeta.seam_ranks);
    let sdf_base = sdf_offset_for(real_slot);
    let vbase = slot_vertex_base[real_slot];
    let idx_base = vbase * 18u;

    // --- PART 1: THIS CELL'S VERTEX ---
    {
        let vert_idx = vbase + vertex;

        // Next to a coarser chunk, the coarse cell's vertex (see seam_cell);
        // else, or if the surface misses that cell (a feature finer than
        // the coarse grid), our own cell's.
        let seam = seam_cell(sdf_base, id);
        var lo = seam.lo;
        var step = seam.step;
        var vert = seam.vert;
        if (vert.w == 0.0) {
            lo = id;
            step = vec3<u32>(1u);
            vert = cell_surface_vertex(sdf_base, lo, step);
        }
        let vert_pos = vert.xyz;

        let normal = compute_normal(sdf_base, vert_pos);
        let world_pos = cmeta.chunk_world_origin + (vert_pos * uniforms.voxel_size);

        let p_lo = texel_pos(lo);
        let span = texel_pos(lo + step) - p_lo; // 0 on flattened axes
        let frac = select(vec3<f32>(0.0), (vert_pos - p_lo) / span, span > vec3<f32>(0.0));
        let blend = tally_corner_materials(sdf_base, lo, step, frac);

        vertex_buffer[vert_idx] = Vertex(
            vec4<f32>(world_pos, pack_material_a(blend.id_a, blend.weight_u8)),
            vec4<f32>(normal, pack_material_b(blend.id_b, real_slot))
        );
    }

    // --- PART 2: INDEX GENERATION FOR ACTIVE EDGES ---
    let sdf_curr = sample_sdf(sdf_base, vec3<i32>(id));
    let curr_inside = sdf_curr <= 0.0;

    // EDGE X AXIS
    let id_x = id + vec3<u32>(1u, 0u, 0u);
    let sdf_x = sample_sdf(sdf_base, vec3<i32>(id_x));
    if (curr_inside != (sdf_x <= 0.0)) {
        if (id.y > 0u && id.z > 0u && owns_edge(id, 0u)) {
            let idx_0 = get_cell_index(id, cell_base);
            let idx_1 = get_cell_index(id - vec3<u32>(0u, 1u, 0u), cell_base);
            let idx_2 = get_cell_index(id - vec3<u32>(0u, 1u, 1u), cell_base);
            let idx_3 = get_cell_index(id - vec3<u32>(0u, 0u, 1u), cell_base);

            let v0 = vbase + cell_vertex[idx_0];
            let v1 = vbase + cell_vertex[idx_1];
            let v2 = vbase + cell_vertex[idx_2];
            let v3 = vbase + cell_vertex[idx_3];

            let base_idx = idx_base + atomicAdd(&indirect_args[real_slot].index_count, 6u);
            if (curr_inside) {
                final_index_buffer[base_idx + 0u] = v0;
                final_index_buffer[base_idx + 1u] = v1;
                final_index_buffer[base_idx + 2u] = v2;
                final_index_buffer[base_idx + 3u] = v0;
                final_index_buffer[base_idx + 4u] = v2;
                final_index_buffer[base_idx + 5u] = v3;
            } else {
                final_index_buffer[base_idx + 0u] = v0;
                final_index_buffer[base_idx + 1u] = v2;
                final_index_buffer[base_idx + 2u] = v1;
                final_index_buffer[base_idx + 3u] = v0;
                final_index_buffer[base_idx + 4u] = v3;
                final_index_buffer[base_idx + 5u] = v2;
            }
        }
    }

    // EDGE Y AXIS
    let id_y = id + vec3<u32>(0u, 1u, 0u);
    let sdf_y = sample_sdf(sdf_base, vec3<i32>(id_y));
    if (curr_inside != (sdf_y <= 0.0)) {
        if (id.x > 0u && id.z > 0u && owns_edge(id, 1u)) {
            let idx_0 = get_cell_index(id, cell_base);
            let idx_1 = get_cell_index(id - vec3<u32>(0u, 0u, 1u), cell_base);
            let idx_2 = get_cell_index(id - vec3<u32>(1u, 0u, 1u), cell_base);
            let idx_3 = get_cell_index(id - vec3<u32>(1u, 0u, 0u), cell_base);

            let v0 = vbase + cell_vertex[idx_0];
            let v1 = vbase + cell_vertex[idx_1];
            let v2 = vbase + cell_vertex[idx_2];
            let v3 = vbase + cell_vertex[idx_3];

            let base_idx = idx_base + atomicAdd(&indirect_args[real_slot].index_count, 6u);
            if (curr_inside) {
                final_index_buffer[base_idx + 0u] = v0;
                final_index_buffer[base_idx + 1u] = v1;
                final_index_buffer[base_idx + 2u] = v2;
                final_index_buffer[base_idx + 3u] = v0;
                final_index_buffer[base_idx + 4u] = v2;
                final_index_buffer[base_idx + 5u] = v3;
            } else {
                final_index_buffer[base_idx + 0u] = v0;
                final_index_buffer[base_idx + 1u] = v2;
                final_index_buffer[base_idx + 2u] = v1;
                final_index_buffer[base_idx + 3u] = v0;
                final_index_buffer[base_idx + 4u] = v3;
                final_index_buffer[base_idx + 5u] = v2;
            }
        }
    }

    // EDGE Z AXIS
    let id_z = id + vec3<u32>(0u, 0u, 1u);
    let sdf_z = sample_sdf(sdf_base, vec3<i32>(id_z));
    if (curr_inside != (sdf_z <= 0.0)) {
        if (id.x > 0u && id.y > 0u && owns_edge(id, 2u)) {
            let idx_0 = get_cell_index(id, cell_base);
            let idx_1 = get_cell_index(id - vec3<u32>(1u, 0u, 0u), cell_base);
            let idx_2 = get_cell_index(id - vec3<u32>(1u, 1u, 0u), cell_base);
            let idx_3 = get_cell_index(id - vec3<u32>(0u, 1u, 0u), cell_base);

            let v0 = vbase + cell_vertex[idx_0];
            let v1 = vbase + cell_vertex[idx_1];
            let v2 = vbase + cell_vertex[idx_2];
            let v3 = vbase + cell_vertex[idx_3];

            let base_idx = idx_base + atomicAdd(&indirect_args[real_slot].index_count, 6u);
            if (curr_inside) {
                final_index_buffer[base_idx + 0u] = v0;
                final_index_buffer[base_idx + 1u] = v1;
                final_index_buffer[base_idx + 2u] = v2;
                final_index_buffer[base_idx + 3u] = v0;
                final_index_buffer[base_idx + 4u] = v2;
                final_index_buffer[base_idx + 5u] = v3;
            } else {
                final_index_buffer[base_idx + 0u] = v0;
                final_index_buffer[base_idx + 1u] = v2;
                final_index_buffer[base_idx + 2u] = v1;
                final_index_buffer[base_idx + 3u] = v0;
                final_index_buffer[base_idx + 4u] = v3;
                final_index_buffer[base_idx + 5u] = v2;
            }
        }
    }
}
