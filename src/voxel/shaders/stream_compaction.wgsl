struct CompactionUniforms {
    chunk_size: u32,       // e.g., 32
    total_cells: u32,      // chunk_size^3 (e.g., 32768)
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> uniforms: CompactionUniforms;
@group(0) @binding(1) var<storage, read_write> cell_flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> compacted_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> shared_data: array<u32, WORKGROUP_SIZE * 2u>;

/// Phase A: Up-Sweep (Reduction) & Down-Sweep Workgroup Scan
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn scan_workgroup(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let thid = local_id.x;
    let bid = wg_id.x;
    
    let idx_a = bid * (WORKGROUP_SIZE * 2u) + thid;
    let idx_b = idx_a + WORKGROUP_SIZE;

    shared_data[thid] = select(0u, cell_flags[idx_a], idx_a < uniforms.total_cells);
    shared_data[thid + WORKGROUP_SIZE] = select(0u, cell_flags[idx_b], idx_b < uniforms.total_cells);

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
        if (bid < arrayLength(&block_sums)) {
            block_sums[bid] = shared_data[last_idx];
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

    if (idx_a < uniforms.total_cells) {
        compacted_offsets[idx_a] = shared_data[thid];
    }
    if (idx_b < uniforms.total_cells) {
        compacted_offsets[idx_b] = shared_data[thid + WORKGROUP_SIZE];
    }
}

/// Phase B: Global Block Offset Resolve (Optimized O(1) per thread lookup)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn resolve_block_offsets(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let thid = local_id.x;
    let bid = wg_id.x;
    
    // If block sums were pre-scanned on CPU or via a secondary reduction pass, 
    // load the exclusive block offset directly instead of an O(N^2) loop.
    let block_modifier = block_sums[bid];

    let idx_a = bid * (WORKGROUP_SIZE * 2u) + thid;
    let idx_b = idx_a + WORKGROUP_SIZE;

    if (idx_a < uniforms.total_cells) { compacted_offsets[idx_a] += block_modifier; }
    if (idx_b < uniforms.total_cells) { compacted_offsets[idx_b] += block_modifier; }
}

/// Phase A.5: Turns per-block totals into exclusive prefix sums across blocks.
/// Single workgroup, single thread — num_blocks is small enough that a serial
/// scan here is effectively free, and it removes any cross-workgroup sync hazard.
@compute @workgroup_size(1, 1, 1)
fn scan_block_sums() {
    let num_blocks = arrayLength(&block_sums);
    var running_total = 0u;
    for (var i = 0u; i < num_blocks; i = i + 1u) {
        let block_total = block_sums[i];
        block_sums[i] = running_total;
        running_total += block_total;
    }
}
