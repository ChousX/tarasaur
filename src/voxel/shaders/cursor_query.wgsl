// GPU ray queries, one invocation per ray: DDA across chunks via a hashed
// chunk lookup table, sphere tracing inside each resident chunk.
// See voxel/query.rs.

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
    lod_ranks: u32,
};

struct QueryUniforms {
    query_count: u32,
    chunk_size: f32,
    lookup_capacity: u32,
    max_dda_steps: u32,
    max_sphere_steps: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    lod_voxel_size: vec4<f32>,
    lod_texture_size: vec4<u32>,
    lod_surface_epsilon: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> queries: array<RayQuery>;
@group(0) @binding(1) var<storage, read_write> hits: array<HitResult>;
@group(0) @binding(2) var<uniform> params: QueryUniforms;
@group(0) @binding(3) var<storage, read> chunk_lookup: array<vec4<u32>>; // x,y,z,packed (see below)

@group(0) @binding(4)  var<storage, read> sdf_buffer_0: array<u32>;
@group(0) @binding(5)  var<storage, read> chunk_meta_0: array<ChunkMetaGpu>;
@group(0) @binding(6)  var<storage, read> sdf_buffer_1: array<u32>;
@group(0) @binding(7)  var<storage, read> chunk_meta_1: array<ChunkMetaGpu>;
@group(0) @binding(8)  var<storage, read> sdf_buffer_2: array<u32>;
@group(0) @binding(9)  var<storage, read> chunk_meta_2: array<ChunkMetaGpu>;
@group(0) @binding(10) var<storage, read> sdf_buffer_3: array<u32>;
@group(0) @binding(11) var<storage, read> chunk_meta_3: array<ChunkMetaGpu>;

// The query runs once per arena index; the bindings above are arena
// `arena` of each LOD, and only chunks stored there are traced.
struct QueryPass {
    arena: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(12) var<uniform> query_pass: QueryPass;

const EMPTY_SLOT: u32 = 0xFFFFFFFFu;
// Lookup entry: rank << 28 | arena << 20 | slot (see LOOKUP_ARENA_SHIFT).
const LOD_RANK_SHIFT: u32 = 28u;
const ARENA_SHIFT: u32 = 20u;
const ARENA_MASK: u32 = 0xFFu;
const SLOT_MASK: u32 = 0xFFFFFu;

fn hash_chunk(x: i32, y: i32, z: i32) -> u32 {
    var h = bitcast<u32>(x) * 0x9E3779B1u;
    h = h ^ (bitcast<u32>(y) * 0x85EBCA77u);
    h = h ^ (bitcast<u32>(z) * 0xC2B2AE3Du);
    return h;
}

fn lookup_chunk(cx: i32, cy: i32, cz: i32) -> vec3<u32> { // (lod_rank, arena, slot), slot==EMPTY_SLOT if absent
    let cap = params.lookup_capacity;
    var idx = hash_chunk(cx, cy, cz) % cap;
    for (var probe = 0u; probe < cap; probe = probe + 1u) {
        let e = chunk_lookup[idx];
        if (e.w == EMPTY_SLOT) { return vec3<u32>(0u, 0u, EMPTY_SLOT); }
        if (bitcast<i32>(e.x) == cx && bitcast<i32>(e.y) == cy && bitcast<i32>(e.z) == cz) {
            return vec3<u32>(e.w >> LOD_RANK_SHIFT, (e.w >> ARENA_SHIFT) & ARENA_MASK, e.w & SLOT_MASK);
        }
        idx = (idx + 1u) % cap;
    }
    return vec3<u32>(0u, 0u, EMPTY_SLOT);
}

// snorm16 SDF, 2 voxels per word (see quantize_sdf in field/sdf.rs).
const SDF_BAND: f32 = 8.0; // keep in sync with field::sdf::SDF_BAND

fn read_sdf(rank: u32, idx: u32) -> f32 {
    var word: u32;
    if (rank == 0u) { word = sdf_buffer_0[idx >> 1u]; }
    else if (rank == 1u) { word = sdf_buffer_1[idx >> 1u]; }
    else if (rank == 2u) { word = sdf_buffer_2[idx >> 1u]; }
    else { word = sdf_buffer_3[idx >> 1u]; }
    let v = unpack2x16snorm(word);
    return select(v.x, v.y, (idx & 1u) == 1u) * SDF_BAND;
}

fn sdf_index(base: u32, x: u32, y: u32, z: u32, s: u32) -> u32 {
    return base + (z * s + y) * s + x;
}

// Apron voxels from a neighbour at another LOD hold that neighbour's voxel
// units (see seam_lod_ranks in field/apron.rs): factor to this chunk's.
fn apron_scale(lod_ranks: u32, n: u32, x: u32, y: u32, z: u32) -> f32 {
    let bits = u32(x >= n) | (u32(y >= n) << 1u) | (u32(z >= n) << 2u);
    if (bits == 0u) {
        return 1.0;
    }
    let log2_sizes = array<i32, 4>(2, 4, 5, 6); // LOD sizes 4, 16, 32, 64
    let region = array<u32, 8>(0u, 0u, 1u, 3u, 2u, 4u, 5u, 6u); // -> NEIGHBORS_MASK index
    let source = (lod_ranks >> (2u + 2u * region[bits])) & 3u;
    return ldexp(1.0, log2_sizes[lod_ranks & 3u] - log2_sizes[source]);
}

// Trilinear sample, parameterized over which buffer/texture_size to hit —
// one function per bound buffer since WGSL can't index storage-buffer
// bindings dynamically without a binding_array.
fn sample_sdf(rank: u32, slot: u32, texture_size: u32, local_pos: vec3<f32>) -> f32 {
    let sf = f32(texture_size);
    let p = clamp(local_pos, vec3<f32>(0.0), vec3<f32>(sf - 1.0));
    let p0 = floor(p);
    let t = p - p0;
    let x0 = u32(p0.x); let y0 = u32(p0.y); let z0 = u32(p0.z);
    let x1 = min(x0 + 1u, texture_size - 1u);
    let y1 = min(y0 + 1u, texture_size - 1u);
    let z1 = min(z0 + 1u, texture_size - 1u);
    let s = texture_size;
    let base = slot * s * s * s;
    let ranks = chunk_meta_at(rank, slot).lod_ranks;
    let n = s - 2u;

    let c000 = read_sdf(rank, sdf_index(base, x0, y0, z0, s)) * apron_scale(ranks, n, x0, y0, z0);
    let c100 = read_sdf(rank, sdf_index(base, x1, y0, z0, s)) * apron_scale(ranks, n, x1, y0, z0);
    let c010 = read_sdf(rank, sdf_index(base, x0, y1, z0, s)) * apron_scale(ranks, n, x0, y1, z0);
    let c110 = read_sdf(rank, sdf_index(base, x1, y1, z0, s)) * apron_scale(ranks, n, x1, y1, z0);
    let c001 = read_sdf(rank, sdf_index(base, x0, y0, z1, s)) * apron_scale(ranks, n, x0, y0, z1);
    let c101 = read_sdf(rank, sdf_index(base, x1, y0, z1, s)) * apron_scale(ranks, n, x1, y0, z1);
    let c011 = read_sdf(rank, sdf_index(base, x0, y1, z1, s)) * apron_scale(ranks, n, x0, y1, z1);
    let c111 = read_sdf(rank, sdf_index(base, x1, y1, z1, s)) * apron_scale(ranks, n, x1, y1, z1);

    let c00 = mix(c000, c100, t.x); let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x); let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y); let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z);
}

fn compute_normal(rank: u32, slot: u32, texture_size: u32, local_pos: vec3<f32>) -> vec3<f32> {
    let e = 1.0;
    let dx = sample_sdf(rank, slot, texture_size, local_pos + vec3<f32>(e, 0.0, 0.0))
           - sample_sdf(rank, slot, texture_size, local_pos - vec3<f32>(e, 0.0, 0.0));
    let dy = sample_sdf(rank, slot, texture_size, local_pos + vec3<f32>(0.0, e, 0.0))
           - sample_sdf(rank, slot, texture_size, local_pos - vec3<f32>(0.0, e, 0.0));
    let dz = sample_sdf(rank, slot, texture_size, local_pos + vec3<f32>(0.0, 0.0, e))
           - sample_sdf(rank, slot, texture_size, local_pos - vec3<f32>(0.0, 0.0, e));
    return normalize(vec3<f32>(dx, dy, dz));
}

fn chunk_meta_at(rank: u32, slot: u32) -> ChunkMetaGpu {
    if (rank == 0u) { return chunk_meta_0[slot]; }
    if (rank == 1u) { return chunk_meta_1[slot]; }
    if (rank == 2u) { return chunk_meta_2[slot]; }
    return chunk_meta_3[slot];
}

fn sphere_trace_chunk(rank: u32, slot: u32, chunk_origin: vec3<f32>, ray_origin: vec3<f32>, dir: vec3<f32>, t_enter: f32, t_max: f32) -> HitResult {
    var result: HitResult;
    result.did_hit = 0u;
    result.chunk_slot = slot;
    var t = t_enter;
    let texture_size = select(select(select(params.lod_texture_size.x, params.lod_texture_size.y, rank == 1u), params.lod_texture_size.z, rank == 2u), params.lod_texture_size.w, rank == 3u);
    let voxel_size = select(select(select(params.lod_voxel_size.x, params.lod_voxel_size.y, rank == 1u), params.lod_voxel_size.z, rank == 2u), params.lod_voxel_size.w, rank == 3u);

    let size_f = f32(texture_size) - 2.0;   // chunk spans local [0, size_f]
    let tol = 1e-3;

    // The last sample's t, while it was outside.
    var outside_t = -1.0;
    for (var i = 0u; i < params.max_sphere_steps; i = i + 1u) {
        let world_pos = ray_origin + dir * t;
        let local_pos = (world_pos - chunk_origin) / voxel_size;   // mesh has no +1 offset

        if (any(local_pos < vec3<f32>(-tol)) || any(local_pos > vec3<f32>(size_f + tol))) {
            result.hit_distance = t;
            return result;
        }

        let d = sample_sdf(rank, slot, texture_size, local_pos);   // d is in voxels
        if (d < 0.05) {
            var hit_t = t;
            // Stepped from outside to inside: find the crossing between them.
            if (d <= 0.0 && outside_t >= 0.0) {
                var lo = outside_t;
                var hi = t;
                for (var k = 0u; k < 10u; k = k + 1u) {
                    let mid = 0.5 * (lo + hi);
                    let mid_local = (ray_origin + dir * mid - chunk_origin) / voxel_size;
                    if (sample_sdf(rank, slot, texture_size, mid_local) > 0.0) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                hit_t = hi;
            }
            let hit_local = (ray_origin + dir * hit_t - chunk_origin) / voxel_size;
            result.did_hit = 1u;
            result.hit_pos_world = ray_origin + dir * hit_t;
            result.hit_distance = hit_t;
            result.hit_normal = compute_normal(rank, slot, texture_size, hit_local);
            result.voxel_coord = vec3<u32>(hit_local);
            return result;
        }
        outside_t = t;

        // Edited distances (jump flood) can overestimate near thin
        // features, so cap the step or the ray could hop over one.
        t = t + clamp(d * 0.9, 0.1, 1.5) * voxel_size;
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

        let found = lookup_chunk(cx, cy, cz);
        let slot = found.z;
        // Chunks in other arenas are traced by their own pass.
        if (slot != EMPTY_SLOT && found.y == query_pass.arena) {
            let rank = found.x;
            let cmeta = chunk_meta_at(rank, slot);
            let hit = sphere_trace_chunk(rank, slot, cmeta.chunk_world_origin, q.origin, dir, t, q.max_distance);
            if (hit.did_hit == 1u) {
                // Pass 0 writes first; a later pass only keeps a closer hit.
                let prev = hits[qi];
                if (query_pass.arena == 0u || prev.did_hit == 0u || hit.hit_distance < prev.hit_distance) {
                    hits[qi] = hit;
                }
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
    if (query_pass.arena == 0u) {
        hits[qi] = result;
    }
}
