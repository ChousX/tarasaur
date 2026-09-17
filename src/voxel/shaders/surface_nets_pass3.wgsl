struct Vertex {
    position: vec4<f32>, // .xyz = world position, .w = bitcast-packed material_id_a (low byte) + blend weight (next byte)
    normal: vec4<f32>,   // .xyz = normal, .w = bitcast-packed material_id_b (low byte)
}

struct IndirectDrawArgs {
    index_count: atomic<u32>,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct BatchUniforms {
    cell_count: u32,
    texture_size: u32,
    wg_per_chunk_z: u32,
    voxel_size: f32, // per-arena constant — every chunk here shares one LOD
}

// Shrunk from 6 fields to 2: sdf_offset, cell_offset, and voxel_size were
// all pure functions of slot/arena constants and moved out (see
// sdf_offset_for/cell_offset_for below and uniforms.voxel_size). Only
// chunk_world_origin (per-chunk) and active_list_pos (per-chunk-per-frame)
// remain — the only two fields that actually can't be derived.
struct ChunkMeta {
    chunk_world_origin: vec3<f32>,
    active_list_pos: u32,
}

@group(0) @binding(0) var<storage, read> sdf_buffer: array<f32>;
@group(0) @binding(1) var<storage, read> flags_buffer: array<u32>;
@group(0) @binding(2) var<storage, read> compacted_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> vertex_buffer: array<Vertex>;
@group(0) @binding(4) var<storage, read_write> final_index_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> indirect_args: array<IndirectDrawArgs>;
@group(0) @binding(6) var<uniform> uniforms: BatchUniforms;
@group(0) @binding(7) var<storage, read> chunk_meta: array<ChunkMeta>;
@group(0) @binding(8) var<storage, read> chunk_vertex_base: array<u32>;
@group(0) @binding(9) var<storage, read> chunk_index_base: array<u32>;
@group(0) @binding(10) var<storage, read> active_slot_map: array<u32>;
@group(0) @binding(11) var<storage, read> material_buffer: array<u32>;

fn get_cell_index(local_coord: vec3<u32>, cell_offset: u32) -> u32 {
    return cell_offset + local_coord.x
        + (local_coord.y * uniforms.cell_count)
        + (local_coord.z * uniforms.cell_count * uniforms.cell_count);
}

fn index_buffer_base(cell_offset: u32) -> u32 {
    return cell_offset * 18u;
}

// Pure functions of slot + per-arena constants — replaces ChunkMeta's old
// sdf_offset / cell_offset fields.
fn sdf_offset_for(slot: u32) -> u32 {
    return slot * uniforms.texture_size * uniforms.texture_size * uniforms.texture_size;
}

// material_base is a byte-granular element offset (same units as
// sdf_offset_for), always a multiple of 4 — see arena sizing invariant.
// Unpacks one material id (0-255) from its packed u32 word.
fn sample_material_id(material_base: u32, elem_idx: u32) -> u32 {
    let word_idx = (material_base + elem_idx) / 4u;
    let byte_shift = ((material_base + elem_idx) % 4u) * 8u;
    let word = material_buffer[word_idx];
    return (word >> byte_shift) & 0xFFu;
}

fn cell_offset_for(slot: u32) -> u32 {
    return slot * uniforms.cell_count * uniforms.cell_count * uniforms.cell_count;
}

fn flatten_sdf_idx(coord: vec3<u32>, texture_size: u32) -> u32 {
    return coord.z * texture_size * texture_size + coord.y * texture_size + coord.x;
}

fn sample_sdf(sdf_base: u32, coord: vec3<i32>) -> f32 {
    let max_coord = i32(uniforms.texture_size) - 1;
    let clamped = clamp(coord, vec3<i32>(0), vec3<i32>(max_coord));
    let idx = u32(clamped.z) * uniforms.texture_size * uniforms.texture_size
        + u32(clamped.y) * uniforms.texture_size + u32(clamped.x);
    return sdf_buffer[sdf_base + idx];
}

fn sample_sdf_trilinear(sdf_base: u32, p: vec3<f32>) -> f32 {
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

// Samples material at the 8 cell corners, weights each by the same
// trilinear shape used to place vert_pos itself, and collapses to the
// top-2 ids by accumulated weight — the rest are discarded and the pair
// renormalized (design: 2-material blend per vertex, not N-way).
fn tally_corner_materials(material_base: u32, id: vec3<u32>, frac: vec3<f32>) -> MaterialBlend {
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

    // At most 8 distinct ids possible (one per corner) — fixed-size
    // arrays, no dynamic allocation needed.
    var uniq_ids: array<u32, 8>;
    var uniq_weights: array<f32, 8>;
    var uniq_count = 0u;

    for (var i = 0u; i < 8u; i++) {
        let mat_id = sample_material_id(material_base, flatten_sdf_idx(id + corners[i], uniforms.texture_size));
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
    // Only one distinct id among the 8 corners (common in solid interior
    // cells) — no second material to blend against. id_b duplicates id_a
    // and weight_u8 saturates to 255, which the fragment shader's blend
    // (a future step) will handle correctly with zero special-casing:
    // blending a material with itself at full weight is a no-op.
    if (second_w >= 0.0 && (best_w + second_w) > 0.00001) {
        id_b = uniq_ids[second_idx];
        let norm_a = best_w / (best_w + second_w);
        weight_u8 = u32(clamp(norm_a * 255.0, 0.0, 255.0));
    }

    return MaterialBlend(id_a, id_b, weight_u8);
}

fn pack_material_a(id_a: u32, weight_u8: u32) -> f32 {
    let packed = (id_a & 0xFFu) | ((weight_u8 & 0xFFu) << 8u) | 0x3F800000u;
    return bitcast<f32>(packed);
}

fn pack_material_b(id_b: u32) -> f32 {
    return bitcast<f32>((id_b & 0xFFu) | 0x3F800000u);
}

@compute @workgroup_size(8, 8, 8)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let chunk_idx = wg_id.z / uniforms.wg_per_chunk_z;
    let local_wg_z = wg_id.z % uniforms.wg_per_chunk_z;
    let id = vec3<u32>(global_id.x, global_id.y, local_wg_z * 8u + local_id.z);

    if (id.x >= uniforms.cell_count || id.y >= uniforms.cell_count || id.z >= uniforms.cell_count) {
        return;
    }

    let real_slot = active_slot_map[chunk_idx];
    let cmeta = chunk_meta[real_slot];
    let sdf_base = sdf_offset_for(real_slot);
    let cell_base = cell_offset_for(real_slot);
    let cell_idx = get_cell_index(id, cell_base);
    let idx_base = chunk_index_base[cmeta.active_list_pos];

    // --- PART 1: VERTEX GENERATION FOR ACTIVE DUAL CELLS ---
    if (flags_buffer[cell_idx] == 1u) {
        let vert_idx = chunk_vertex_base[cmeta.active_list_pos] + compacted_offsets[cell_idx];

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
        for (var i = 0u; i < 8u; i++) {
            sdfs[i] = sample_sdf(sdf_base, vec3<i32>(id + corners[i]));
        }

        var vert_pos = vec3<f32>(0.0);
        var edge_count = 0.0;
        for (var i = 0u; i < 12u; i++) {
            let c0 = edges[i].x;
            let c1 = edges[i].y;
            let v0 = sdfs[c0];
            let v1 = sdfs[c1];
            let v0_inside = v0 <= 0.0;
            let v1_inside = v1 <= 0.0;
            if (v0_inside != v1_inside) {
                let p0 = vec3<f32>(id + corners[c0]);
                let p1 = vec3<f32>(id + corners[c1]);
                let t = -v0 / (v1 - v0);
                vert_pos += mix(p0, p1, clamp(t, 0.0, 1.0));
                edge_count += 1.0;
            }
        }

        if (edge_count > 0.0) {
            vert_pos = vert_pos / edge_count;
        } else {
            vert_pos = vec3<f32>(id) + vec3<f32>(0.5);
        }

        let normal = compute_normal(sdf_base, vert_pos);
        let world_pos = cmeta.chunk_world_origin + (vert_pos * uniforms.voxel_size);

        let frac = vert_pos - vec3<f32>(id);
        let blend = tally_corner_materials(sdf_base, id, frac);
        vertex_buffer[vert_idx] = Vertex(
            vec4<f32>(world_pos, pack_material_a(blend.id_a, blend.weight_u8)),
            vec4<f32>(normal, pack_material_b(blend.id_b))
        );
    }

    // --- PART 2: INDEX GENERATION FOR ACTIVE EDGES ---
    let sdf_curr = sample_sdf(sdf_base, vec3<i32>(id));
    let curr_inside = sdf_curr <= 0.0;

    // EDGE X AXIS
    let id_x = id + vec3<u32>(1u, 0u, 0u);
    let sdf_x = sample_sdf(sdf_base, vec3<i32>(id_x));
    if (curr_inside != (sdf_x <= 0.0)) {
        if (id.y > 0u && id.z > 0u) {
            let idx_0 = get_cell_index(id, cell_base);
            let idx_1 = get_cell_index(id - vec3<u32>(0u, 1u, 0u), cell_base);
            let idx_2 = get_cell_index(id - vec3<u32>(0u, 1u, 1u), cell_base);
            let idx_3 = get_cell_index(id - vec3<u32>(0u, 0u, 1u), cell_base);

            let vbase = chunk_vertex_base[cmeta.active_list_pos];
            let v0 = vbase + compacted_offsets[idx_0];
            let v1 = vbase + compacted_offsets[idx_1];
            let v2 = vbase + compacted_offsets[idx_2];
            let v3 = vbase + compacted_offsets[idx_3];

            let base_idx = idx_base + atomicAdd(&indirect_args[chunk_idx].index_count, 6u);
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
        if (id.x > 0u && id.z > 0u) {
            let idx_0 = get_cell_index(id, cell_base);
            let idx_1 = get_cell_index(id - vec3<u32>(0u, 0u, 1u), cell_base);
            let idx_2 = get_cell_index(id - vec3<u32>(1u, 0u, 1u), cell_base);
            let idx_3 = get_cell_index(id - vec3<u32>(1u, 0u, 0u), cell_base);

            let vbase = chunk_vertex_base[cmeta.active_list_pos];
            let v0 = vbase + compacted_offsets[idx_0];
            let v1 = vbase + compacted_offsets[idx_1];
            let v2 = vbase + compacted_offsets[idx_2];
            let v3 = vbase + compacted_offsets[idx_3];

            let base_idx = idx_base + atomicAdd(&indirect_args[chunk_idx].index_count, 6u);

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
        if (id.x > 0u && id.y > 0u) {
            let idx_0 = get_cell_index(id, cell_base);
            let idx_1 = get_cell_index(id - vec3<u32>(1u, 0u, 0u), cell_base);
            let idx_2 = get_cell_index(id - vec3<u32>(1u, 1u, 0u), cell_base);
            let idx_3 = get_cell_index(id - vec3<u32>(0u, 1u, 0u), cell_base);

            let vbase = chunk_vertex_base[cmeta.active_list_pos];
            let v0 = vbase + compacted_offsets[idx_0];
            let v1 = vbase + compacted_offsets[idx_1];
            let v2 = vbase + compacted_offsets[idx_2];
            let v3 = vbase + compacted_offsets[idx_3];

            let base_idx = idx_base + atomicAdd(&indirect_args[chunk_idx].index_count, 6u);

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
