// Shared by the voxel shaders: how a chunk's padded SDF is stored, the
// per-arena uniforms and per-chunk metadata, and the LOD-rank helpers seam
// stitching uses. Imported as `tarasaur::voxel::shaders::voxel_common`.

// SDF values are snorm16 over +/-SDF_BAND, 2 voxels per u32 word
// (see quantize_sdf in field/sdf.rs).
const SDF_BAND: f32 = 8.0; // keep in sync with field::sdf::SDF_BAND

// Voxel `idx` of a padded SDF, from the word holding it.
fn decode_sdf(word: u32, idx: u32) -> f32 {
    let v = unpack2x16snorm(word);
    return select(v.x, v.y, (idx & 1u) == 1u) * SDF_BAND;
}

// Index of `coord` in a `texture_size`^3 padded volume, x fastest.
fn flatten_sdf_idx(coord: vec3<u32>, texture_size: u32) -> u32 {
    return coord.z * texture_size * texture_size + coord.y * texture_size + coord.x;
}

// One arena's sizes (see make_batch_uniform_buffer in voxel/util.rs).
struct BatchUniforms {
    cell_count: u32,       // cells per chunk edge: voxels + 1
    texture_size: u32,     // padded voxels per edge: voxels + 2
    wg_per_chunk_z: u32,   // pass1's z workgroups per chunk
    voxel_size: f32,       // world units, constant within an arena (one LOD)
    material_stride: u32,  // material bytes per voxel: 2 with distances, else 1
    // Scalars, not a vec3: a vec3 would align to 16 and make the struct 48.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Per slot (see ChunkMeta in voxel/arena.rs).
struct ChunkMeta {
    chunk_world_origin: vec3<f32>,
    seam_ranks: vec2<u32>, // see seam_lod_ranks in field/apron.rs
}

// log2 of a LOD's voxels per chunk edge, by rank: 4, 16, 32, 64. No
// lookup table: indexing a local array per texel read was slow.
fn rank_log2_size(rank: u32) -> i32 {
    let r = i32(rank & 3u);
    return select(r + 3, 2, r == 0);
}

// --- Material boundary distances ---
// Each voxel keeps its palette id and the distance to the nearest material
// boundary, so a blend can put that boundary back inside the voxel instead
// of on the grid. Keep both constants in step with field/material.rs.

// Distances are stored in WORLD units and saturate here (one Lowest
// voxel). World units mean the value needs no rescale when it is read
// across an LOD seam -- the fragment stage has no seam ranks to rescale
// with -- and that it stride-downsamples to a coarser LOD unchanged.
const MATERIAL_BAND: f32 = 2.5;
// Half-width of the blend between two materials, in world units. Constant
// in world space, so the boundary neither moves nor changes width when a
// chunk switches LOD.
const MATERIAL_BLEND_WORLD: f32 = 0.25;

// Trilinear interpolation of 8 cell corners, corner i being (i&1, i>>1&1,
// i>>2&1), at `f` within the cell.
fn trilinear8(v: array<f32, 8>, f: vec3<f32>) -> f32 {
    var c = v;
    let x00 = mix(c[0], c[1], f.x);
    let x10 = mix(c[2], c[3], f.x);
    let x01 = mix(c[4], c[5], f.x);
    let x11 = mix(c[6], c[7], f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// The winning label's share, from the top two interpolated label fields.
// They are equal on the boundary, and their half-difference is the signed
// distance to it, so this is a world-constant ramp across it.
fn material_blend_weight(best: f32, second: f32) -> f32 {
    return clamp(0.5 + (best - second) / (4.0 * MATERIAL_BLEND_WORLD), 0.0, 1.0);
}

// LOD rank of the chunk at `offset` (-1..1 per axis; 0 = this chunk).
fn seam_rank(seam_ranks: vec2<u32>, offset: vec3<i32>) -> u32 {
    let k = u32(offset.x + 1 + 3 * (offset.y + 1) + 9 * (offset.z + 1));
    return (seam_ranks[k / 16u] >> (2u * (k % 16u))) & 3u;
}

// log2(own size / source size) for the values at texel `t` of an
// `n`-voxel chunk: apron texels hold their source neighbour's voxel units.
fn apron_unit_shift(seam_ranks: vec2<u32>, n: u32, t: vec3<u32>) -> i32 {
    if (all(t < vec3<u32>(n))) {
        return 0; // our own voxels: by far the most reads
    }
    let source = seam_rank(seam_ranks, vec3<i32>(t >= vec3<u32>(n)));
    return rank_log2_size(seam_rank(seam_ranks, vec3<i32>(0))) - rank_log2_size(source);
}
