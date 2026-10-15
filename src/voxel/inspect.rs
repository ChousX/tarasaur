//! A read-only view of the GPU arenas, for tests and tools that check what
//! the render world holds. Use it from render-world systems.

use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    render::{render_resource::Buffer, sync_world::MainEntity},
};

use crate::{field::LOD, voxel::arena::VoxelChunkArenaSet};

/// Where a resident chunk's material ids live on the GPU.
pub struct MaterialSlot {
    /// The arena's material buffer (one `u8` id per voxel, `COPY_SRC`).
    pub buffer: Buffer,
    /// Byte offset of the chunk's padded volume in `buffer`.
    pub offset: u64,
    /// Edge of the padded volume: the chunk's voxels plus the apron.
    pub padded_size: u32,
}

/// Where a resident chunk's visibility mask lives on the GPU.
pub struct MaskSlot {
    /// The arena's mask buffer: one bit per padded voxel, 32 per `u32`,
    /// x fastest (`COPY_SRC`).
    pub buffer: Buffer,
    /// Byte offset of the chunk's mask words in `buffer`.
    pub offset: u64,
    /// Bytes of mask words per chunk (a multiple of 4).
    pub bytes: u64,
    /// The arena's per-slot "has a mask" bits, 32 per `u32` (`COPY_SRC`).
    pub has_mask_buffer: Buffer,
    /// Byte offset of the `u32` holding this slot's "has a mask" bit.
    pub has_mask_offset: u64,
    /// That bit's index within the word.
    pub has_mask_bit: u32,
    /// Edge of the padded volume: the chunk's voxels plus the apron.
    pub padded_size: u32,
}

/// Render-world system param: which arenas exist and which chunks they hold.
#[derive(SystemParam)]
pub struct ArenaInspector<'w> {
    arenas: Res<'w, VoxelChunkArenaSet>,
}

impl ArenaInspector<'_> {
    /// Arenas per LOD, indexed by `LOD::rank()`.
    #[must_use]
    pub fn arena_counts(&self) -> [usize; LOD::COUNT] {
        self.arenas.arenas.each_ref().map(Vec::len)
    }

    /// Every chunk holding a slot: (chunk position, LOD, arena index).
    pub fn resident_chunks(&self) -> impl Iterator<Item = (IVec3, LOD, usize)> + '_ {
        LOD::ALL
            .iter()
            .zip(&self.arenas.arenas)
            .flat_map(|(&lod, arenas)| {
                arenas.iter().enumerate().flat_map(move |(index, arena)| {
                    arena
                        .slot_of_chunk_pos
                        .keys()
                        .map(move |&pos| (pos, lod, index))
                })
            })
    }

    /// The material slot of `chunk` (a main-world entity) at `lod`, if it
    /// holds one.
    #[must_use]
    pub fn material_slot(&self, lod: LOD, chunk: Entity) -> Option<MaterialSlot> {
        let found = self.arenas.find_slot(lod, MainEntity::from(chunk))?;
        let arena = self
            .arenas
            .arenas
            .get(VoxelChunkArenaSet::lod_index(lod))?
            .get(found.arena_idx)?;
        let bytes = u64::from(arena.sdf_elems_per_chunk);
        Some(MaterialSlot {
            buffer: arena.material_buffer.clone(),
            offset: u64::from(found.slot).saturating_mul(bytes),
            padded_size: arena.texture_size,
        })
    }

    /// The visibility-mask slot of `chunk` (a main-world entity) at `lod`,
    /// if it holds one.
    #[must_use]
    pub fn mask_slot(&self, lod: LOD, chunk: Entity) -> Option<MaskSlot> {
        let found = self.arenas.find_slot(lod, MainEntity::from(chunk))?;
        let arena = self
            .arenas
            .arenas
            .get(VoxelChunkArenaSet::lod_index(lod))?
            .get(found.arena_idx)?;
        // Same layout as `prepare_mask_upload`: a word-aligned region per slot.
        let bytes = u64::from(arena.sdf_elems_per_chunk)
            .div_ceil(32)
            .saturating_mul(4);
        let slot = u64::from(found.slot);
        Some(MaskSlot {
            buffer: arena.visibility_mask_buffer.clone(),
            offset: slot.saturating_mul(bytes),
            bytes,
            has_mask_buffer: arena.chunk_has_mask_buffer.clone(),
            has_mask_offset: (slot / 32).saturating_mul(4),
            has_mask_bit: found.slot % 32,
            padded_size: arena.texture_size,
        })
    }
}
