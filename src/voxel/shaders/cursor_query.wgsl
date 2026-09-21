struct RayQuery {
    origin: vec3<f32>,
    max_distance: f32,
    direction: vec3<f32>,
    user_id: u32,
};

struct HitResult {
    hit_pos_world: vec3<f32>,
    hit_distance: f32,
    hit_normal: vec3<f32>,
    did_hit: u32,
    voxel_coord: vec3<u32>,
    chunk_slot: u32,
};

struct ChunkMetaGpu {
    chunk_world_origin: vec3<f32>,
    active_list_pos: u32,
};

struct QueryUniforms {
    query_count: u32,
    chunk_size: f32,
    voxel_size: f32,
    texture_size: u32,
    lookup_capacity: u32,
    max_dda_steps: u32,
    max_sphere_steps: u32,
    surface_epsilon: f32,
};

@group(0) @binding(0) var<storage, read> queries: array<RayQuery>;
@group(0) @binding(1) var<storage, read_write> hits: array<HitResult>;
@group(0) @binding(2) var<uniform> params: QueryUniforms;
@group(0) @binding(3) var<storage, read> chunk_lookup: array<vec4<u32>>; // key_x,key_y,key_z,slot
@group(0) @binding(4) var<storage, read> sdf_buffer: array<f32>;
@group(0) @binding(5) var<storage, read> chunk_meta: array<ChunkMetaGpu>;

const EMPTY_SLOT: u32 = 0xFFFFFFFFu;

fn hash_chunk(x: i32, y: i32, z: i32) -> u32 {
    var h = bitcast<u32>(x) * 0x9E3779B1u;
    h = h ^ (bitcast<u32>(y) * 0x85EBCA77u);
    h = h ^ (bitcast<u32>(z) * 0xC2B2AE3Du);
    return h;
}

fn lookup_chunk_slot(cx: i32, cy: i32, cz: i32) -> u32 {
    let cap = params.lookup_capacity;
    var idx = hash_chunk(cx, cy, cz) % cap;
    for (var probe = 0u; probe < cap; probe = probe + 1u) {
        let e = chunk_lookup[idx];
        if (e.w == EMPTY_SLOT) { return EMPTY_SLOT; }
        if (bitcast<i32>(e.x) == cx && bitcast<i32>(e.y) == cy && bitcast<i32>(e.z) == cz) {
            return e.w;
        }
        idx = (idx + 1u) % cap;
    }
    return EMPTY_SLOT;
}

fn sdf_index(base: u32, x: u32, y: u32, z: u32, s: u32) -> u32 {
    return base + (z * s + y) * s + x;
}

fn sample_sdf_trilinear(slot: u32, local_pos: vec3<f32>) -> f32 {
    let s = params.texture_size;
    let sf = f32(s);
    let p = clamp(local_pos, vec3<f32>(0.0), vec3<f32>(sf - 1.0));
    let p0 = floor(p);
    let t = p - p0;
    let x0 = u32(p0.x); let y0 = u32(p0.y); let z0 = u32(p0.z);
    let x1 = min(x0 + 1u, s - 1u);
    let y1 = min(y0 + 1u, s - 1u);
    let z1 = min(z0 + 1u, s - 1u);
    let base = slot * s * s * s;

    let c000 = sdf_buffer[sdf_index(base, x0, y0, z0, s)];
    let c100 = sdf_buffer[sdf_index(base, x1, y0, z0, s)];
    let c010 = sdf_buffer[sdf_index(base, x0, y1, z0, s)];
    let c110 = sdf_buffer[sdf_index(base, x1, y1, z0, s)];
    let c001 = sdf_buffer[sdf_index(base, x0, y0, z1, s)];
    let c101 = sdf_buffer[sdf_index(base, x1, y0, z1, s)];
    let c011 = sdf_buffer[sdf_index(base, x0, y1, z1, s)];
    let c111 = sdf_buffer[sdf_index(base, x1, y1, z1, s)];

    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z);
}

fn compute_normal(slot: u32, local_pos: vec3<f32>) -> vec3<f32> {
    let e = 1.0;
    let dx = sample_sdf_trilinear(slot, local_pos + vec3<f32>(e, 0.0, 0.0))
           - sample_sdf_trilinear(slot, local_pos - vec3<f32>(e, 0.0, 0.0));
    let dy = sample_sdf_trilinear(slot, local_pos + vec3<f32>(0.0, e, 0.0))
           - sample_sdf_trilinear(slot, local_pos - vec3<f32>(0.0, e, 0.0));
    let dz = sample_sdf_trilinear(slot, local_pos + vec3<f32>(0.0, 0.0, e))
           - sample_sdf_trilinear(slot, local_pos - vec3<f32>(0.0, 0.0, e));
    return normalize(vec3<f32>(dx, dy, dz));
}

// Phase 2: sphere-trace inside one chunk. On surface hit, did_hit=1.
// On leaving the chunk's own (unpadded) footprint without a hit,
// did_hit stays 0 and hit_distance carries the exit t so Phase 1 can resume.
fn sphere_trace_chunk(slot: u32, chunk_origin: vec3<f32>, ray_origin: vec3<f32>, dir: vec3<f32>, t_enter: f32, t_max: f32) -> HitResult {
    var result: HitResult;
    result.did_hit = 0u;
    result.chunk_slot = slot;
    var t = t_enter;
    let voxel_size = params.voxel_size;
    let cell_count_f = f32(params.texture_size) - 2.0; // PADDING = 2, see systems.rs

    for (var i = 0u; i < params.max_sphere_steps; i = i + 1u) {
        let world_pos = ray_origin + dir * t;
        let local_pos = (world_pos - chunk_origin) / voxel_size + vec3<f32>(1.0);

        if (any(local_pos < vec3<f32>(0.0)) || any(local_pos > vec3<f32>(cell_count_f))) {
            result.hit_distance = t;
            return result;
        }

        let d = sample_sdf_trilinear(slot, local_pos);
        if (d < params.surface_epsilon) {
            result.did_hit = 1u;
            result.hit_pos_world = world_pos;
            result.hit_distance = t;
            result.hit_normal = compute_normal(slot, local_pos);
            result.voxel_coord = vec3<u32>(local_pos);
            return result;
        }

        t = t + max(d * voxel_size, voxel_size * 0.1);
        if (t > t_max) {
            result.hit_distance = t_max;
            return result;
        }
    }
    result.hit_distance = t;
    return result;
}

@compute @workgroup_size(1, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.query_count) { return; }

    let q = queries[qi];
    var result: HitResult;
    result.did_hit = 0u;
    result.chunk_slot = EMPTY_SLOT;
    result.hit_distance = q.max_distance;

    let dir = normalize(q.direction);
    let chunk_size = params.chunk_size;

    var cx = i32(floor(q.origin.x / chunk_size));
    var cy = i32(floor(q.origin.y / chunk_size));
    var cz = i32(floor(q.origin.z / chunk_size));

    let step_x = select(-1, 1, dir.x >= 0.0);
    let step_y = select(-1, 1, dir.y >= 0.0);
    let step_z = select(-1, 1, dir.z >= 0.0);

    let inv_dx = select(1e30, 1.0 / dir.x, abs(dir.x) > 1e-8);
    let inv_dy = select(1e30, 1.0 / dir.y, abs(dir.y) > 1e-8);
    let inv_dz = select(1e30, 1.0 / dir.z, abs(dir.z) > 1e-8);

    var t_max_x = (f32(cx + select(0, 1, step_x > 0)) * chunk_size - q.origin.x) * inv_dx;
    var t_max_y = (f32(cy + select(0, 1, step_y > 0)) * chunk_size - q.origin.y) * inv_dy;
    var t_max_z = (f32(cz + select(0, 1, step_z > 0)) * chunk_size - q.origin.z) * inv_dz;

    let t_delta_x = chunk_size * abs(inv_dx);
    let t_delta_y = chunk_size * abs(inv_dy);
    let t_delta_z = chunk_size * abs(inv_dz);

    var t = 0.0;
    for (var step_i = 0u; step_i < params.max_dda_steps; step_i = step_i + 1u) {
        if (t > q.max_distance) { break; }

        let slot = lookup_chunk_slot(cx, cy, cz);
        if (slot != EMPTY_SLOT) {
            let chunk_info = chunk_meta[slot];
            let hit = sphere_trace_chunk(slot, chunk_info.chunk_world_origin, q.origin, dir, t, q.max_distance);
            if (hit.did_hit == 1u) {
                hits[qi] = hit;
                return;
            }
            t = hit.hit_distance;
        }

        if (t_max_x < t_max_y && t_max_x < t_max_z) {
            cx = cx + step_x; t = t_max_x; t_max_x = t_max_x + t_delta_x;
        } else if (t_max_y < t_max_z) {
            cy = cy + step_y; t = t_max_y; t_max_y = t_max_y + t_delta_y;
        } else {
            cz = cz + step_z; t = t_max_z; t_max_z = t_max_z + t_delta_z;
        }
    }
    hits[qi] = result;
}
