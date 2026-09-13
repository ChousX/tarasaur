struct BatchCompactionUniforms {
    chunk_size: u32,        // cell_count, e.g. 32 — fixed per LOD bucket
    total_cells: u32,       // chunk_size^3
    blocks_per_chunk: u32,  // ceil(total_cells / (WORKGROUP_SIZE * 2))
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> uniforms: BatchCompactionUniforms;
@group(0) @binding(1) var<storage, read_write> cell_flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> compacted_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(4) var<storage, read_write> chunk_active_counts: array<u32>;

/// Phase C: one thread per active chunk. Reads the same cell_offset convention
/// as scan_workgroup/resolve_block_offsets (chunk_idx * total_cells) and writes
/// this chunk's total active-cell count — compacted_offsets is exclusive, so add
/// the last cell's own flag to get the true total.
@compute @workgroup_size(1, 1, 1)
fn write_chunk_active_count(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let chunk_idx = wg_id.x;
    let cell_offset = chunk_idx * uniforms.total_cells;
    let last = cell_offset + uniforms.total_cells - 1u;
    chunk_active_counts[chunk_idx] = compacted_offsets[last] + cell_flags[last];
}

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> shared_data: array<u32, WORKGROUP_SIZE * 2u>;

/// Phase A: Up-Sweep (Reduction) & Down-Sweep Workgroup Scan, batched across chunks.
/// Dispatched as (blocks_per_chunk * active_chunk_count, 1, 1); each chunk's
/// local block index is wg_id.x % blocks_per_chunk, chunk index is wg_id.x / blocks_per_chunk.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn scan_workgroup(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let thid = local_id.x;
    let chunk_idx = wg_id.x / uniforms.blocks_per_chunk;
    let local_bid = wg_id.x % uniforms.blocks_per_chunk;

    let cell_offset = chunk_idx * uniforms.total_cells;
    let block_offset = chunk_idx * uniforms.blocks_per_chunk;

    let local_idx_a = local_bid * (WORKGROUP_SIZE * 2u) + thid;
    let local_idx_b = local_idx_a + WORKGROUP_SIZE;
    let idx_a = cell_offset + local_idx_a;
    let idx_b = cell_offset + local_idx_b;

    shared_data[thid] = select(0u, cell_flags[idx_a], local_idx_a < uniforms.total_cells);
    shared_data[thid + WORKGROUP_SIZE] = select(0u, cell_flags[idx_b], local_idx_b < uniforms.total_cells);

    var offset = 1u;

    // 1. Up-Sweep Phase
    for (var d = WORKGROUP_SIZE; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (thid < d) {
            let ai = offset * (2u * thid + 1u) - 1u;
            let bi = offset * (2u * thid + 2u) - 1u;
            shared_data[bi] += shared_data[ai];
        }
        offset *= 2u;
    }

    if (thid == 0u) {
        let last_idx = WORKGROUP_SIZE * 2u - 1u;
        let global_block_idx = block_offset + local_bid;
        if (global_block_idx < arrayLength(&block_sums)) {
            block_sums[global_block_idx] = shared_data[last_idx];
        }
        shared_data[last_idx] = 0u;
    }

    // 2. Down-Sweep Phase
    for (var d = 1u; d <= WORKGROUP_SIZE; d *= 2u) {
        offset >>= 1u;
        workgroupBarrier();
        if (thid < d) {
            let ai = offset * (2u * thid + 1u) - 1u;
            let bi = offset * (2u * thid + 2u) - 1u;
            let t = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] += t;
        }
    }
    workgroupBarrier();

    if (local_idx_a < uniforms.total_cells) {
        compacted_offsets[idx_a] = shared_data[thid];
    }
    if (local_idx_b < uniforms.total_cells) {
        compacted_offsets[idx_b] = shared_data[thid + WORKGROUP_SIZE];
    }
}

/// Phase B: Global Block Offset Resolve, batched. Same wg_id.x -> (chunk_idx, local_bid)
/// split as scan_workgroup; block_sums lookup uses the per-chunk block_offset so each
/// chunk's blocks only see their own chunk's prefix sums.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn resolve_block_offsets(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let thid = local_id.x;
    let chunk_idx = wg_id.x / uniforms.blocks_per_chunk;
    let local_bid = wg_id.x % uniforms.blocks_per_chunk;

    let cell_offset = chunk_idx * uniforms.total_cells;
    let block_offset = chunk_idx * uniforms.blocks_per_chunk;

    let block_modifier = block_sums[block_offset + local_bid];

    let local_idx_a = local_bid * (WORKGROUP_SIZE * 2u) + thid;
    let local_idx_b = local_idx_a + WORKGROUP_SIZE;
    let idx_a = cell_offset + local_idx_a;
    let idx_b = cell_offset + local_idx_b;

    if (local_idx_a < uniforms.total_cells) { compacted_offsets[idx_a] += block_modifier; }
    if (local_idx_b < uniforms.total_cells) { compacted_offsets[idx_b] += block_modifier; }
}

/// Phase A.5: Turns per-block totals into exclusive prefix sums, one chunk's
/// worth of blocks per workgroup invocation. Dispatched as (active_chunk_count, 1, 1) —
/// each invocation still does its scan serially since blocks_per_chunk is small,
/// but now scans only its own chunk's slice of block_sums rather than the whole buffer.
@compute @workgroup_size(1, 1, 1)
fn scan_block_sums(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let chunk_idx = wg_id.x;
    let block_offset = chunk_idx * uniforms.blocks_per_chunk;

    var running_total = 0u;
    for (var i = 0u; i < uniforms.blocks_per_chunk; i = i + 1u) {
        let global_i = block_offset + i;
        if (global_i >= arrayLength(&block_sums)) {
            break;
        }
        let block_total = block_sums[global_i];
        block_sums[global_i] = running_total;
        running_total += block_total;
    }
}
