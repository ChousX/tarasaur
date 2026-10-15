// Packs the draw arguments of one view's visible chunks, so a single
// multi_draw_indexed_indirect covers a whole arena. See voxel/draw.rs.

const WORKGROUP_SIZE: u32 = 64u; // keep in sync with draw.rs
const ARGS_WORDS: u32 = 5u;      // DrawIndexedIndirectArgs

@group(0) @binding(0) var<storage, read> visible: array<u32>;       // [count, slot, slot, ...]
@group(0) @binding(1) var<storage, read> slot_args: array<u32>;     // per slot, written by pass3
@group(0) @binding(2) var<storage, read_write> packed_args: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= visible[0]) {
        return;
    }
    let src = visible[i + 1u] * ARGS_WORDS;
    let dst = i * ARGS_WORDS;
    for (var k = 0u; k < ARGS_WORDS; k++) {
        packed_args[dst + k] = slot_args[src + k];
    }
}
