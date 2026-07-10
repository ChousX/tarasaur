// --- Bindings From Milestone 1 Layout ---
@group(0) @binding(0) var sdf_texture: texture_storage_3d<r32float, read>;
@group(0) @binding(1) var jfa_texture: texture_3d<u32>;

struct Vertex {
    position: vec4<f32>,
    normal: vec4<f32>,
}

struct ScatteredVertexBuffer {
    // Atomic allocation tracker for stream compaction tracking
    vertex_count: atomic<u32>,
    // Rigid, 1:1 spatial buffer matching flattened cell index mapping
    data: array<Vertex>,
}

@group(0) @binding(2) var<storage, read_write> vertex_buffer: ScatteredVertexBuffer;

// Flatten 3D local cell coordinate into a linear, predictable 1D index
fn flatten_cell_idx(coord: vec3<u32>, size: u32) -> u32 {
    return coord.z * size * size + coord.y * size + coord.x;
}

// Unpack your CPU-generated PackedCoord (10 bits per axis)
fn unpack_jfa_coord(packed: u32) -> vec3<f32> {
    let x = packed & 0x3FFu;
    let y = (packed >> 10u) & 0x3FFu;
    let z = (packed >> 20u) & 0x3FFu;
    return vec3<f32>(f32(x), f32(y), f32(z));
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let size = num_workgroups.x * 4u; // Derived grid scale bounds
    
    // Surface Nets samples dual cell corners; clamp execution cleanly inside grid boundaries
    if (global_id.x >= size - 1u || global_id.y >= size - 1u || global_id.z >= size - 1u) {
        return;
    }

    // 1. Identify the 8 corners of the current dual cell voxel space
    let offsets = array<vec3<u32>, 8>(
        vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u),
        vec3<u32>(0u, 1u, 0u), vec3<u32>(1u, 1u, 0u),
        vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u),
        vec3<u32>(0u, 1u, 1u), vec3<u32>(1u, 1u, 1u)
    );

    var corner_sdfs: array<f32, 8>;
    var signs: array<bool, 8>;
    var inside_count = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let pos = global_id + offsets[i];
        let val = textureLoad(sdf_texture, vec3<i32>(pos)).x;
        corner_sdfs[i] = val;
        signs[i] = val < 0.0;
        if (signs[i]) { inside_count = inside_count + 1u; }
    }

    // Isolate crossing state: No surface crossing if cell is fully inside or fully outside
    if (inside_count == 0u || inside_count == 8u) {
        return;
    }

    // 2. Linearly interpolate along the 12 spatial edges to calculate the zero-crossings
    var edge_crossings = vec3<f32>(0.0);
    var crossing_count = 0.0;

    // Fixed edge configuration mapping corner pairs
    let edge_start = array<u32, 12>(0u,4u,0u,2u,0u,1u,1u,5u,2u,6u,4u,5u);
    let edge_end   = array<u32, 12>(1u,5u,2u,3u,4u,5u,3u,7u,6u,7u,6u,7u);

    for (var e = 0u; e < 12u; e = e + 1u) {
        let i0 = edge_start[e];
        let i1 = edge_end[e];
        
        if (signs[i0] != signs[i1]) {
            let v0 = corner_sdfs[i0];
            let v1 = corner_sdfs[i1];
            
            // Intersection parameter (zero-crossing target)
            let t = v0 / (v0 - v1);
            
            let p0 = vec3<f32>(global_id + offsets[i0]);
            let p1 = vec3<f32>(global_id + offsets[i1]);
            
            edge_crossings = edge_crossings + mix(p0, p1, t);
            crossing_count = crossing_count + 1.0;
        }
    }

    // Generate mass-point center within the cell bounding box
    let vertex_pos = edge_crossings / crossing_count;

    // 3. Resolve precision surface normal vector using the CPU-packed JFA texture
    let packed_jfa = textureLoad(jfa_texture, vec3<i32>(global_id), 0).x;
    var normal_vector = vec3<f32>(0.0, 1.0, 0.0);
    
    if (packed_jfa != 0xFFFFFFFFu) {
        let seed_coord = unpack_jfa_coord(packed_jfa);
        let to_seed = vertex_pos - seed_coord;
        if (length(to_seed) > 0.001) {
            normal_vector = normalize(to_seed);
            // Orient face vectors outwards depending on density field gradient signs
            if (corner_sdfs[0] < 0.0) {
                normal_vector = -normal_vector;
            }
        }
    }

    // 4. Atomically commit allocation tracking metrics and write to scattered layout
    let output_idx = flatten_cell_idx(global_id, size);
    let _discard = atomicAdd(&vertex_buffer.vertex_count, 1u);

    // .w = 1.0 marks this specific scattered index slot as valid/occupied for stream compaction
    vertex_buffer.data[output_idx].position = vec4<f32>(vertex_pos, 1.0);
    vertex_buffer.data[output_idx].normal = vec4<f32>(normal_vector, 0.0);
}
