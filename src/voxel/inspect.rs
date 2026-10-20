//! A read-only view of the GPU arenas, for tests and tools that check what
//! the render world holds. Use it from render-world systems.

use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    render::{render_resource::Buffer, sync_world::MainEntity},
};

use crate::{
    field::LOD,
    num::usize_from,
    voxel::{
        arena::VoxelChunkArenaSet,
        layer::{LayerIndex, VoxelLayers},
    },
};

/// Where a resident chunk's material ids live on the GPU.
pub struct MaterialSlot {
    /// The arena's material buffer (one `u8` id per voxel, `COPY_SRC`).
    pub buffer: Buffer,
    /// Byte offset of the chunk's padded volume in `buffer`.
    pub offset: u64,
    /// Edge of the padded volume: the chunk's voxels plus the apron.
    pub padded_size: u32,
}

/// One arena's mesh output, and where each of its chunks' meshes lie in it.
pub struct MeshArena {
    pub lod: LOD,
    /// Two `vec4<f32>` per vertex: world position (.w packed material), then
    /// normal (.w packed material and slot) (`COPY_SRC`).
    pub vertex_buffer: Buffer,
    /// `u32` indices into `vertex_buffer`, absolute (`COPY_SRC`).
    pub index_buffer: Buffer,
    /// `DrawIndexedIndirectArgs` per slot; `index_count` comes first
    /// (`COPY_SRC`).
    pub indirect_args_buffer: Buffer,
    /// `[slot] -> vertex count`, a `u32` per slot (`COPY_SRC`).
    pub vertex_count_buffer: Buffer,
    /// Padded SDF volumes, `padded_size`^3 snorm16 values (over
    /// +/-`SDF_BAND`) per slot, x fastest (`COPY_SRC`).
    pub sdf_buffer: Buffer,
    /// `ChunkMeta` per slot, 32 bytes: world origin, then the seam ranks at
    /// byte 16 (`COPY_SRC`).
    pub chunk_meta_buffer: Buffer,
    /// Edge of the padded volume: the chunk's voxels plus the apron.
    pub padded_size: u32,
    /// Every chunk holding a slot here: (chunk position, slot, first vertex).
    /// Its indices start at `first vertex * INDICES_PER_CELL`.
    pub slots: Vec<(IVec3, u32, u32)>,
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

/// Render-world system param: which arenas the terrain layer has and which
/// chunks they hold.
#[derive(SystemParam)]
pub struct ArenaInspector<'w> {
    layers: Res<'w, VoxelLayers>,
}

/// The terrain layer's arenas, or none before it is registered.
static NO_ARENAS: [Vec<super::arena::VoxelChunkArena>; LOD::COUNT] =
    [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

impl ArenaInspector<'_> {
    fn terrain(&self) -> Option<&VoxelChunkArenaSet> {
        self.layers.set(LayerIndex::TERRAIN)
    }

    fn arenas(&self) -> &[Vec<super::arena::VoxelChunkArena>; LOD::COUNT] {
        self.terrain().map_or(&NO_ARENAS, |set| &set.arenas)
    }

    /// Arenas per LOD, indexed by `LOD::rank()`.
    #[must_use]
    pub fn arena_counts(&self) -> [usize; LOD::COUNT] {
        self.arenas().each_ref().map(Vec::len)
    }

    /// Every chunk holding a slot: (chunk position, LOD, arena index).
    pub fn resident_chunks(&self) -> impl Iterator<Item = (IVec3, LOD, usize)> + '_ {
        LOD::ALL
            .iter()
            .zip(self.arenas())
            .flat_map(|(&lod, arenas)| {
                arenas.iter().enumerate().flat_map(move |(index, arena)| {
                    arena
                        .slot_of_chunk_pos
                        .keys()
                        .map(move |&pos| (pos, lod, index))
                })
            })
    }

    /// Every arena's mesh output, for reading the meshes back.
    #[must_use]
    pub fn mesh_arenas(&self) -> Vec<MeshArena> {
        LOD::ALL
            .iter()
            .zip(self.arenas())
            .flat_map(|(&lod, arenas)| {
                arenas.iter().map(move |arena| MeshArena {
                    lod,
                    vertex_buffer: arena.final_vertex_buffer.clone(),
                    index_buffer: arena.index_buffer.clone(),
                    indirect_args_buffer: arena.indirect_args_buffer.clone(),
                    vertex_count_buffer: arena.slot_vertex_count_buffer.clone(),
                    sdf_buffer: arena.sdf_buffer.clone(),
                    chunk_meta_buffer: arena.chunk_meta_buffer.clone(),
                    padded_size: arena.texture_size,
                    slots: arena
                        .slot_of_chunk_pos
                        .iter()
                        .filter_map(|(&pos, &slot)| {
                            let (base, _) = *arena.slot_range.get(usize_from(slot))?;
                            Some((pos, slot, base))
                        })
                        .collect(),
                })
            })
            .collect()
    }

    /// The material slot of `chunk` (a main-world entity) at `lod`, if it
    /// holds one.
    #[must_use]
    pub fn material_slot(&self, lod: LOD, chunk: Entity) -> Option<MaterialSlot> {
        let set = self.terrain()?;
        let found = set.find_first(lod, MainEntity::from(chunk))?;
        let arena = set
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
        let set = self.terrain()?;
        let found = set.find_first(lod, MainEntity::from(chunk))?;
        let arena = set
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
