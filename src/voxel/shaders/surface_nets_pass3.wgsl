struct Vertex {
    position: vec4<f32>,
    normal: vec4<f32>,
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
    _pad0: u32,
}

struct ChunkMeta {
    chunk_world_origin: vec3<f32>,
    voxel_size: f32,
    sdf_offset: u32,
    cell_offset: u32,
    active_list_pos: u32,
    _pad: u32,
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

fn get_cell_index(local_coord: vec3<u32>, cell_offset: u32) -> u32 {
    return cell_offset + local_coord.x
        + (local_coord.y * uniforms.cell_count)
        + (local_coord.z * uniforms.cell_count * uniforms.cell_count);
}

// Per-chunk base offset into final_index_buffer. Each chunk gets a fixed-size
// span of total_cells * 18 u32 slots (same stride used to size the arena's
// index_buffer), matching the 18-indices-per-cell budget from the original
// per-chunk buffer sizing.
fn index_buffer_base(cell_offset: u32) -> u32 {
    return cell_offset * 18u;
}

fn sample_sdf(cmeta: ChunkMeta, coord: vec3<i32>) -> f32 {
    let max_coord = i32(uniforms.texture_size) - 1;
    let clamped = clamp(coord, vec3<i32>(0), vec3<i32>(max_coord));
    let idx = u32(clamped.z) * uniforms.texture_size * uniforms.texture_size
        + u32(clamped.y) * uniforms.texture_size + u32(clamped.x);
    return sdf_buffer[cmeta.sdf_offset + idx];
}

fn sample_sdf_trilinear(cmeta: ChunkMeta, p: vec3<f32>) -> f32 {
    let ip = vec3<i32>(floor(p));
    let f = fract(p);
    let c000 = sample_sdf(cmeta, ip + vec3<i32>(0, 0, 0));
    let c100 = sample_sdf(cmeta, ip + vec3<i32>(1, 0, 0));
    let c010 = sample_sdf(cmeta, ip + vec3<i32>(0, 1, 0));
    let c110 = sample_sdf(cmeta, ip + vec3<i32>(1, 1, 0));
    let c001 = sample_sdf(cmeta, ip + vec3<i32>(0, 0, 1));
    let c101 = sample_sdf(cmeta, ip + vec3<i32>(1, 0, 1));
    let c011 = sample_sdf(cmeta, ip + vec3<i32>(0, 1, 1));
    let c111 = sample_sdf(cmeta, ip + vec3<i32>(1, 1, 1));
    let x0 = mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y);
    let x1 = mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y);
    return mix(x0, x1, f.z);
}

fn compute_normal(cmeta: ChunkMeta, pos: vec3<f32>) -> vec3<f32> {
    let h = 0.1;
    let dx = sample_sdf_trilinear(cmeta, pos + vec3<f32>(h, 0.0, 0.0)) - sample_sdf_trilinear(cmeta, pos - vec3<f32>(h, 0.0, 0.0));
    let dy = sample_sdf_trilinear(cmeta, pos + vec3<f32>(0.0, h, 0.0)) - sample_sdf_trilinear(cmeta, pos - vec3<f32>(0.0, h, 0.0));
    let dz = sample_sdf_trilinear(cmeta, pos + vec3<f32>(0.0, 0.0, h)) - sample_sdf_trilinear(cmeta, pos - vec3<f32>(0.0, 0.0, h));
    let norm = vec3<f32>(dx, dy, dz);
    let len = length(norm);
    if (len > 0.00001) {
        return normalize(norm);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
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

    let cmeta = chunk_meta[chunk_idx];
    let cell_idx = get_cell_index(id, cmeta.cell_offset);
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
            sdfs[i] = sample_sdf(cmeta, vec3<i32>(id + corners[i]));
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

        let normal = compute_normal(cmeta, vert_pos);
        let world_pos = cmeta.chunk_world_origin + (vert_pos * cmeta.voxel_size);

        vertex_buffer[vert_idx] = Vertex(
            vec4<f32>(world_pos, 1.0),
            vec4<f32>(normal, 0.0)
        );
    }

    // --- PART 2: INDEX GENERATION FOR ACTIVE EDGES ---
    let sdf_curr = sample_sdf(cmeta, vec3<i32>(id));
    let curr_inside = sdf_curr <= 0.0;

    // EDGE X AXIS
    let id_x = id + vec3<u32>(1u, 0u, 0u);
    let sdf_x = sample_sdf(cmeta, vec3<i32>(id_x));
    if (curr_inside != (sdf_x <= 0.0)) {
        if (id.y > 0u && id.z > 0u) {
            let idx_0 = get_cell_index(id, cmeta.cell_offset);
            let idx_1 = get_cell_index(id - vec3<u32>(0u, 1u, 0u), cmeta.cell_offset);
            let idx_2 = get_cell_index(id - vec3<u32>(0u, 1u, 1u), cmeta.cell_offset);
            let idx_3 = get_cell_index(id - vec3<u32>(0u, 0u, 1u), cmeta.cell_offset);

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
    let sdf_y = sample_sdf(cmeta, vec3<i32>(id_y));
    if (curr_inside != (sdf_y <= 0.0)) {
        if (id.x > 0u && id.z > 0u) {
            let idx_0 = get_cell_index(id, cmeta.cell_offset);
            let idx_1 = get_cell_index(id - vec3<u32>(0u, 0u, 1u), cmeta.cell_offset);
            let idx_2 = get_cell_index(id - vec3<u32>(1u, 0u, 1u), cmeta.cell_offset);
            let idx_3 = get_cell_index(id - vec3<u32>(1u, 0u, 0u), cmeta.cell_offset);

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
    let sdf_z = sample_sdf(cmeta, vec3<i32>(id_z));
    if (curr_inside != (sdf_z <= 0.0)) {
        if (id.x > 0u && id.y > 0u) {
            let idx_0 = get_cell_index(id, cmeta.cell_offset);
            let idx_1 = get_cell_index(id - vec3<u32>(1u, 0u, 0u), cmeta.cell_offset);
            let idx_2 = get_cell_index(id - vec3<u32>(1u, 1u, 0u), cmeta.cell_offset);
            let idx_3 = get_cell_index(id - vec3<u32>(0u, 1u, 0u), cmeta.cell_offset);

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
