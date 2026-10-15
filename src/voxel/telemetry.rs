//! Debug readback of pass3's range guard.

use std::sync::atomic::Ordering;

use crate::num::usize_from;

use bevy::{prelude::*, render::render_resource::MapMode};

use crate::voxel::arena::VoxelChunkArenaSet;

/// Logs pass3's range guard (a chunk skipped because the GPU counted more
/// cells than its CPU-sized range). Should never fire; costs one small
/// readback per arena per frame, so on in debug builds only.
#[derive(Resource)]
pub struct GuardTelemetry(pub bool);

impl Default for GuardTelemetry {
    fn default() -> Self {
        Self(cfg!(debug_assertions))
    }
}

pub(super) const GUARD_FREE: u8 = 0;
pub(super) const GUARD_COPIED: u8 = 1;
const GUARD_MAPPED: u8 = 2;

pub fn map_guard_readbacks(telemetry: Res<GuardTelemetry>, arena_set: Res<VoxelChunkArenaSet>) {
    if !telemetry.0 {
        return;
    }
    for (rank, arenas) in arena_set.arenas.iter().enumerate() {
        for (arena_idx, arena) in arenas.iter().enumerate() {
            let state = arena.guard_state.clone();
            if state
                .compare_exchange(
                    GUARD_COPIED,
                    GUARD_MAPPED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }
            let buf = arena.guard_readback_buffer.clone();
            let chunk_pos_of_slot = arena.chunk_pos_of_slot.clone();
            buf.clone().slice(..).map_async(MapMode::Read, move |result| {
                if result.is_ok() {
                    // The view must be dropped before `unmap`; it is, at the
                    // end of this statement.
                    let v = buf.slice(..).get_mapped_range().ok().and_then(|data| {
                        let bytes = data.get(..16)?;
                        bytemuck::try_pod_read_unaligned::<[u32; 4]>(bytes).ok()
                    });
                    buf.unmap();
                    if let Some(v) = v.filter(|v| v[0] != 0) {
                        let pos = chunk_pos_of_slot.get(usize_from(v[1])).copied().flatten();
                        warn!(
                            "[guard] LOD rank {rank} arena {arena_idx}: {} chunk(s) skipped; last slot {} at {:?}: gpu_count {} > cpu_count {}",
                            v[0], v[1], pos, v[2], v[3]
                        );
                    }
                }
                state.store(GUARD_FREE, Ordering::Release);
            });
        }
    }
}
