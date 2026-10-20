//! Tarasaur: a GPU-meshed, editable smooth-voxel terrain engine for Bevy.
//!
//! Data flows one way, main world -> render world -> GPU:
//!
//! 1. [`chunk`]: `ChunkLoader`s (usually on the camera) decide which chunks
//!    exist and at what `TargetLOD`. Chunks are entities on a grid of
//!    `CHUNK_SIZE` world units.
//! 2. [`storage::loading`]: off-thread tasks fill each chunk's fields from a
//!    save (downsampled) or the registered generators, then install them
//!    together with the chunk's `LOD`.
//! 3. [`field`]: per-chunk voxel grids. `SDF` (shape), `VisibilityField`
//!    (optional: distances to the edge of what is shown) and
//!    `MaterialField<M>` (opt-in). Edits go through `WorldEditor` / CSG
//!    messages, only at `MaxEditLod`.
//! 4. `voxel::extract`: changed chunks get padded volumes (own voxels plus
//!    a 2-voxel apron from neighbours, the shape clipped by the visibility)
//!    built in parallel and handed to the render world.
//! 5. [`voxel`]: volumes are packed into GPU arenas and meshed with surface
//!    nets in compute shaders, culled per view and drawn with one
//!    multi-draw per arena, triplanar-textured from a [`texture_palette`].
//!
//! Persistence ([`storage::persistence`]) saves edited max-LOD chunks to
//! `./saves/chunks/`. Cursor ray queries ([`raycast::cursor`]) run on the CPU
//! SDF or on the GPU arenas.

#![recursion_limit = "256"]

pub mod chunk;
pub mod field;
pub mod grid;
pub mod num;
mod plugin;
pub mod prelude;
pub mod raycast;
pub mod storage;
pub mod texture_palette;
pub mod voxel;

pub use plugin::{AppVoxelFieldExt, MaterialFieldPlugin, TarasaurPlugin, TerrainLayer};
pub use voxel::{AppSurfaceLayerExt, LayerRaster, SurfaceLayer};
