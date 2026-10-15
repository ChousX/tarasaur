//! Tarasaur: a GPU-meshed, editable smooth-voxel terrain engine for Bevy.
//!
//! Data flows one way, main world -> render world -> GPU:
//!
//! 1. [`chunk`]: `ChunkLoader`s (usually on the camera) decide which chunks
//!    exist and at what `TargetLOD`. Chunks are entities on a grid of
//!    `CHUNK_SIZE` world units.
//! 2. [`field::loading`]: off-thread tasks fill each chunk's fields from a
//!    save (downsampled) or the registered generators, then install them
//!    together with the chunk's `LOD`.
//! 3. [`field`]: per-chunk voxel grids. `SDF` (shape), `VisibilityField`
//!    (optional mask) and `MaterialField<M>` (opt-in). Edits go through
//!    `WorldEditor` / CSG messages, only at `MaxEditLod`.
//! 4. [`voxel::extract`]: changed chunks get padded volumes (own voxels plus
//!    a 2-voxel apron from neighbours) built in parallel and handed to the
//!    render world.
//! 5. [`voxel`]: volumes are packed into GPU arenas and meshed with surface
//!    nets in compute shaders, culled per view and drawn with one
//!    multi-draw per arena, triplanar-textured from a [`texture_palette`].
//!
//! Persistence ([`field::persistence`]) saves edited max-LOD chunks to
//! `./saves/chunks/`. Cursor ray queries ([`voxel::cursor`]) run on the CPU
//! SDF or on the GPU arenas.

#![recursion_limit = "256"]

pub mod chunk;
pub mod field;
pub mod grid;
pub mod num;
mod plugin;
pub mod prelude;
pub mod texture_palette;
pub mod voxel;

pub use chunk::*;
pub use field::*;
pub use plugin::TarasaurPlugin;
