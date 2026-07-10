@group(0) @binding(0) var sdf_volume: texture_storage_3d<r32float, read>;
@group(0) @binding(1) var<storage, read> flags_buffer: array<u32>;
@group(0) @binding(2) var<storage, read> compacted_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> final_index_buffer: array<u32>;
@group(0) @binding(4) var<storage, read_write> indirect_args: IndirectDrawArgs;

struct IndirectDrawArgs {
    index_count: atomic<u32>,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

const CHUNK_SIZE: u32 = 32u; // Matches uniform sizing limits

fn get_cell_index(coord: vec3<u32>) -> u32 {
    return coord.x + (coord.y * CHUNK_SIZE) + (coord.z * CHUNK_SIZE * CHUNK_SIZE);
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Avoid boundary spill on lookups (require 1 extra cell padding for edge calculation)
    if (id.x >= CHUNK_SIZE - 1u || id.y >= CHUNK_SIZE - 1u || id.z >= CHUNK_SIZE - 1u) {
        return;
    }

    let cell_idx = get_cell_index(id);
    let sdf_curr = textureLoad(sdf_volume, vec3<i32>(id)).x;

    // We evaluate three edges extending from the current cell: X, Y, and Z
    
    // --- EDGE X AXIS ---
    let id_x = id + vec3<u32>(1u, 0u, 0u);
    let sdf_x = textureLoad(sdf_volume, vec3<i32>(id_x)).x;
    if ((sdf_curr < 0.0 && sdf_x >= 0.0) || (sdf_curr >= 0.0 && sdf_x < 0.0)) {
        if (id.y > 0u && id.z > 0u) {
            // Retrieve compacted vertex indices of the 4 cells surrounding this edge
            let v0 = compacted_offsets[get_cell_index(id)];
            let v1 = compacted_offsets[get_cell_index(id - vec3<u32>(0u, 1u, 0u))];
            let v2 = compacted_offsets[get_cell_index(id - vec3<u32>(0u, 1u, 1u))];
            let v3 = compacted_offsets[get_cell_index(id - vec3<u32>(0u, 0u, 1u))];

            // Only emit geometry if all adjacent cells successfully created vertices
            if (flags_buffer[get_cell_index(id)] == 1u &&
                flags_buffer[get_cell_index(id - vec3<u32>(0u, 1u, 0u))] == 1u &&
                flags_buffer[get_cell_index(id - vec3<u32>(0u, 1u, 1u))] == 1u &&
                flags_buffer[get_cell_index(id - vec3<u32>(0u, 0u, 1u))] == 1u) {
                
                // Allocate dynamic chunk slot in index array atomically
                let base_idx = atomicAdd(&indirect_args.index_count, 6u);
                
                // Wind counter-clockwise depending on transition sign direction
                if (sdf_curr < 0.0) {
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

    // --- EDGE Y AXIS ---
    let id_y = id + vec3<u32>(0u, 1u, 0u);
    let sdf_y = textureLoad(sdf_volume, vec3<i32>(id_y)).x;
    if ((sdf_curr < 0.0 && sdf_y >= 0.0) || (sdf_curr >= 0.0 && sdf_y < 0.0)) {
        if (id.x > 0u && id.z > 0u) {
            let v0 = compacted_offsets[get_cell_index(id)];
            let v1 = compacted_offsets[get_cell_index(id - vec3<u32>(0u, 0u, 1u))];
            let v2 = compacted_offsets[get_cell_index(id - vec3<u32>(1u, 0u, 1u))];
            let v3 = compacted_offsets[get_cell_index(id - vec3<u32>(1u, 0u, 0u))];

            if (flags_buffer[get_cell_index(id)] == 1u &&
                flags_buffer[get_cell_index(id - vec3<u32>(0u, 0u, 1u))] == 1u &&
                flags_buffer[get_cell_index(id - vec3<u32>(1u, 0u, 1u))] == 1u &&
                flags_buffer[get_cell_index(id - vec3<u32>(1u, 0u, 0u))] == 1u) {
                
                let base_idx = atomicAdd(&indirect_args.index_count, 6u);
                
                if (sdf_curr < 0.0) {
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

    // --- EDGE Z AXIS ---
    let id_z = id + vec3<u32>(0u, 0u, 1u);
    let sdf_z = textureLoad(sdf_volume, vec3<i32>(id_z)).x;
    if ((sdf_curr < 0.0 && sdf_z >= 0.0) || (sdf_curr >= 0.0 && sdf_z < 0.0)) {
        if (id.x > 0u && id.y > 0u) {
            let v0 = compacted_offsets[get_cell_index(id)];
            let v1 = compacted_offsets[get_cell_index(id - vec3<u32>(1u, 0u, 0u))];
            let v2 = compacted_offsets[get_cell_index(id - vec3<u32>(1u, 1u, 0u))];
            let v3 = compacted_offsets[get_cell_index(id - vec3<u32>(0u, 1u, 0u))];

            if (flags_buffer[get_cell_index(id)] == 1u &&
                flags_buffer[get_cell_index(id - vec3<u32>(1u, 0u, 0u))] == 1u &&
                flags_buffer[get_cell_index(id - vec3<u32>(1u, 1u, 0u))] == 1u &&
                flags_buffer[get_cell_index(id - vec3<u32>(0u, 1u, 0u))] == 1u) {
                
                let base_idx = atomicAdd(&indirect_args.index_count, 6u);
                
                if (sdf_curr < 0.0) {
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
}
