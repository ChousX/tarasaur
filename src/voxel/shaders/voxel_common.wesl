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
    cell_count: u32,     // cells per chunk edge: voxels + 1
    texture_size: u32,   // padded voxels per edge: voxels + 2
    wg_per_chunk_z: u32, // pass1's z workgroups per chunk
    voxel_size: f32,     // world units, constant within an arena (one LOD)
}

// Per slot (see ChunkMeta in voxel/arena.rs).
struct ChunkMeta {
    chunk_world_origin: vec3<f32>,
    seam_ranks: vec2<u32>, // see seam_lod_ranks in field/apron.rs
}

// log2 of a LOD's voxels per chunk edge, by rank: 4, 16, 32, 64.
fn rank_log2_size(rank: u32) -> i32 {
    let log2_sizes = array<i32, 4>(2, 4, 5, 6);
    return log2_sizes[rank & 3u];
}

// LOD rank of the chunk at `offset` (-1..1 per axis; 0 = this chunk).
fn seam_rank(seam_ranks: vec2<u32>, offset: vec3<i32>) -> u32 {
    let k = u32(offset.x + 1 + 3 * (offset.y + 1) + 9 * (offset.z + 1));
    return (seam_ranks[k / 16u] >> (2u * (k % 16u))) & 3u;
}

// log2(own size / source size) for the values at texel `t` of an
// `n`-voxel chunk: apron texels hold their source neighbour's voxel units.
fn apron_unit_shift(seam_ranks: vec2<u32>, n: u32, t: vec3<u32>) -> i32 {
    let source = seam_rank(seam_ranks, vec3<i32>(t >= vec3<u32>(n)));
    return rank_log2_size(seam_rank(seam_ranks, vec3<i32>(0))) - rank_log2_size(source);
}
