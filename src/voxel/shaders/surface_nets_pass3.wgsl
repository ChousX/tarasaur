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

struct Uniforms {
    cell_count: u32,
    texture_size: u32,
    voxel_size: f32,
    chunk_world_origin: vec3<f32>,
}

@group(0) @binding(0) var sdf_volume: texture_storage_3d<r32float, read>;
@group(0) @binding(1) var<storage, read> flags_buffer: array<u32>;
@group(0) @binding(2) var<storage, read> compacted_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> vertex_buffer: array<Vertex>;
@group(0) @binding(4) var<storage, read_write> final_index_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> indirect_args: IndirectDrawArgs;
@group(0) @binding(6) var<uniform> uniforms: Uniforms;

fn get_cell_index(coord: vec3<u32>) -> u32 {
    return coord.x + (coord.y * uniforms.cell_count) + (coord.z * uniforms.cell_count * uniforms.cell_count);
}

fn sample_sdf(coord: vec3<i32>) -> f32 {
    let max_coord = i32(uniforms.texture_size) - 1;
    let clamped = clamp(coord, vec3<i32>(0), vec3<i32>(max_coord));
    return textureLoad(sdf_volume, clamped).x;
}

// Computes normal using central differences on the SDF volume
fn compute_normal(pos: vec3<f32>) -> vec3<f32> {
    let ipos = vec3<i32>(pos);
    let dx = sample_sdf(ipos + vec3<i32>(1, 0, 0)) - sample_sdf(ipos - vec3<i32>(1, 0, 0));
    let dy = sample_sdf(ipos + vec3<i32>(0, 1, 0)) - sample_sdf(ipos - vec3<i32>(0, 1, 0));
    let dz = sample_sdf(ipos + vec3<i32>(0, 0, 1)) - sample_sdf(ipos - vec3<i32>(0, 0, 1));
    
    let norm = vec3<f32>(dx, dy, dz);
    let len = length(norm);
    if (len > 0.00001) {
        return normalize(norm);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

@compute @workgroup_size(8, 8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // 1. GRID BOUNDS
    if (id.x >= uniforms.cell_count || id.y >= uniforms.cell_count || id.z >= uniforms.cell_count) {
    return;
}

    let cell_idx = get_cell_index(id);

    // --- PART 1: VERTEX GENERATION FOR ACTIVE DUAL CELLS ---
    if (flags_buffer[cell_idx] == 1u) {
        let vert_idx = compacted_offsets[cell_idx];

        let corners = array<vec3<u32>, 8>(
            vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u),
            vec3<u32>(0u, 1u, 0u), vec3<u32>(1u, 1u, 0u),
            vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u),
            vec3<u32>(0u, 1u, 1u), vec3<u32>(1u, 1u, 1u)
        );

        let edges = array<vec2<u32>, 12>(
            vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(4u, 5u), vec2<u32>(6u, 7u), // X edges
            vec2<u32>(0u, 2u), vec2<u32>(1u, 3u), vec2<u32>(4u, 6u), vec2<u32>(5u, 7u), // Y edges
            vec2<u32>(0u, 4u), vec2<u32>(1u, 5u), vec2<u32>(2u, 6u), vec2<u32>(3u, 7u)  // Z edges
        );

        var sdfs: array<f32, 8>;
        for (var i = 0u; i < 8u; i++) {
            sdfs[i] = textureLoad(sdf_volume, vec3<i32>(id + corners[i])).x;
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

        let normal = compute_normal(vert_pos);

        let world_pos = uniforms.chunk_world_origin + (vert_pos * uniforms.voxel_size);

        vertex_buffer[vert_idx] = Vertex(
            vec4<f32>(world_pos, 1.0),
            vec4<f32>(normal, 0.0)
        );
    }

    // --- PART 2: INDEX GENERATION FOR ACTIVE EDGES ---
    let sdf_curr = textureLoad(sdf_volume, vec3<i32>(id)).x;
    let curr_inside = sdf_curr <= 0.0;

    // EDGE X AXIS
    let id_x = id + vec3<u32>(1u, 0u, 0u);
    let sdf_x = textureLoad(sdf_volume, vec3<i32>(id_x)).x;
    if (curr_inside != (sdf_x <= 0.0)) {
        if (id.y > 0u && id.z > 0u) {
            let idx_0 = get_cell_index(id);
            let idx_1 = get_cell_index(id - vec3<u32>(0u, 1u, 0u));
            let idx_2 = get_cell_index(id - vec3<u32>(0u, 1u, 1u));
            let idx_3 = get_cell_index(id - vec3<u32>(0u, 0u, 1u));

            let v0 = compacted_offsets[idx_0];
            let v1 = compacted_offsets[idx_1];
            let v2 = compacted_offsets[idx_2];
            let v3 = compacted_offsets[idx_3];

            let base_idx = atomicAdd(&indirect_args.index_count, 6u);

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
    let sdf_y = textureLoad(sdf_volume, vec3<i32>(id_y)).x;
    if (curr_inside != (sdf_y <= 0.0)) {
        if (id.x > 0u && id.z > 0u) {
            let idx_0 = get_cell_index(id);
            let idx_1 = get_cell_index(id - vec3<u32>(0u, 0u, 1u));
            let idx_2 = get_cell_index(id - vec3<u32>(1u, 0u, 1u));
            let idx_3 = get_cell_index(id - vec3<u32>(1u, 0u, 0u));

            let v0 = compacted_offsets[idx_0];
            let v1 = compacted_offsets[idx_1];
            let v2 = compacted_offsets[idx_2];
            let v3 = compacted_offsets[idx_3];

            let base_idx = atomicAdd(&indirect_args.index_count, 6u);

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
    let sdf_z = textureLoad(sdf_volume, vec3<i32>(id_z)).x;
    if (curr_inside != (sdf_z <= 0.0)) {
        if (id.x > 0u && id.y > 0u) {
            let idx_0 = get_cell_index(id);
            let idx_1 = get_cell_index(id - vec3<u32>(1u, 0u, 0u));
            let idx_2 = get_cell_index(id - vec3<u32>(1u, 1u, 0u));
            let idx_3 = get_cell_index(id - vec3<u32>(0u, 1u, 0u));

            let v0 = compacted_offsets[idx_0];
            let v1 = compacted_offsets[idx_1];
            let v2 = compacted_offsets[idx_2];
            let v3 = compacted_offsets[idx_3];

            let base_idx = atomicAdd(&indirect_args.index_count, 6u);

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
