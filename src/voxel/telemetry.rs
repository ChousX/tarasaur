//! Debug readback of pass3's range guard and the draw arguments it wrote.

use std::sync::atomic::Ordering;

use crate::num::usize_from;

use bevy::{prelude::*, render::render_resource::MapMode};

use crate::voxel::{
    arena::{GUARD_BYTES, INDICES_PER_CELL, VoxelChunkArenaSet},
    types::DrawIndexedIndirectArgs,
};

/// Logs pass3's range guard and any drawn slot with bad indirect args.
///
/// The guard fires when the GPU counted more cells than a chunk's CPU-sized
/// range; the args are bad when they don't fit the slot's vertex range (a bad
/// count can hang the GPU). Should never fire; costs a readback per re-meshed
/// arena per frame, so on in debug builds only.
#[derive(Resource)]
pub struct GuardTelemetry(pub bool);

// Not derivable in debug builds (it defaults to on there); clippy only sees
// the release value.
#[allow(clippy::derivable_impls)]
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
            // Unchanged since the copy: uploads run before the dispatch.
            let drawn: Vec<(u32, (u32, u32))> = arena
                .active_slots
                .iter()
                .filter_map(|&s| Some((s, *arena.slot_range.get(usize_from(s))?)))
                .collect();
            buf.clone().slice(..).map_async(MapMode::Read, move |result| {
                if result.is_ok() {
                    // The view must be dropped before `unmap`; it is, at the
                    // end of this statement.
                    let (v, bad_args) = buf
                        .slice(..)
                        .get_mapped_range()
                        .ok()
                        .and_then(|data| {
                            let (guard, args) = data.split_at_checked(usize_from(GUARD_BYTES))?;
                            let v = bytemuck::try_pod_read_unaligned::<[u32; 4]>(guard).ok();
                            Some((v, check_args(args, &drawn)))
                        })
                        .unwrap_or_default();
                    buf.unmap();
                    if let Some(&(slot, range, args)) = bad_args.first() {
                        let pos = chunk_pos_of_slot.get(usize_from(slot)).copied().flatten();
                        error!(
                            "[guard] LOD rank {rank} arena {arena_idx}: {} drawn slot(s) with bad indirect args; first: slot {slot} at {pos:?}, range (base, cap) {range:?}, args {args:?}",
                            bad_args.len()
                        );
                    }
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

/// Drawn slots whose args don't fit their `(base, capacity)` vertex range:
/// first index off the range's start, or more indices than it has room for.
fn check_args(
    args: &[u8],
    drawn: &[(u32, (u32, u32))],
) -> Vec<(u32, (u32, u32), DrawIndexedIndirectArgs)> {
    let size = std::mem::size_of::<DrawIndexedIndirectArgs>();
    drawn
        .iter()
        .filter_map(|&(slot, range)| {
            let at = usize_from(slot).saturating_mul(size);
            let a: DrawIndexedIndirectArgs =
                bytemuck::try_pod_read_unaligned(args.get(at..at.saturating_add(size))?).ok()?;
            let (base, cap) = range;
            let ok = a.instance_count == 1
                && a.base_vertex == 0
                && a.first_instance == 0
                && a.first_index == base.wrapping_mul(INDICES_PER_CELL)
                && a.index_count <= cap.saturating_mul(INDICES_PER_CELL)
                && a.index_count.is_multiple_of(6);
            (!ok).then_some((slot, range, a))
        })
        .collect()
}
