struct IndirectDrawArgs {
    index_count: atomic<u32>,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct ChunkBasesUniforms {
    active_count: u32,
    budget_cells_per_chunk: u32,
    total_budget_cells: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: ChunkBasesUniforms;
@group(0) @binding(1) var<storage, read> chunk_active_counts: array<u32>;
@group(0) @binding(2) var<storage, read> active_slot_map: array<u32>;
@group(0) @binding(3) var<storage, read_write> chunk_vertex_base: array<u32>;
@group(0) @binding(4) var<storage, read_write> chunk_index_base: array<u32>;
@group(0) @binding(5) var<storage, read_write> indirect_args: array<IndirectDrawArgs>;
@group(0) @binding(6) var<storage, read_write> overflow_flag: array<atomic<u32>>;

// Single workgroup, serial. Cheap even at hundreds of active chunks (this is
// a tiny prefix sum, not the O(cells) scan stream_compaction does).
@compute @workgroup_size(1, 1, 1)
fn cs_main() {
    var running: u32 = 0u;
    for (var i = 0u; i < u.active_count; i = i + 1u) {
        let count = chunk_active_counts[i];
        let slot = active_slot_map[i];
        if (running + count > u.total_budget_cells) {
            // Dropped chunk: sentinel base makes pass3 early-out, and it
            // consumes no budget.
            atomicStore(&overflow_flag[0], 1u);
            chunk_vertex_base[i] = 0xFFFFFFFFu;
            chunk_index_base[i] = 0u;
            indirect_args[slot].first_index = 0u;
            atomicStore(&indirect_args[slot].index_count, 0u);
        } else {
            chunk_vertex_base[i] = running;
            chunk_index_base[i] = running * 18u;
            indirect_args[slot].first_index = running * 18u;
            running = running + count;
        }
    }
}
