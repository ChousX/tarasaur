//! Common imports: `use tarasaur::prelude::*;`.
//!
//! Everything here also has one canonical path in its module; the prelude
//! only gathers what most apps touch.

pub use crate::plugin::{AppVoxelFieldExt, MaterialFieldPlugin, TarasaurPlugin};

// Chunks and streaming
pub use crate::chunk::{CHUNK_SIZE, ChunkLoader, ChunkManager, ShowChunkBounds};
pub use crate::storage::generator::ChunkGeneratorRegistry;

// Fields and editing
pub use crate::field::{
    Field, GpuChannel, LOD, MaterialField, MaxEditLod, SDF, TargetLOD, VisibilityField, VoxelField,
    VoxelMaterial,
    editor::{EditMode, SdfCsgSphere, SdfSphereStamp, WorldEditor},
    ops::{AccumulateExt, BlendExt},
};
pub use crate::storage::persistence::{AutosaveIntervalSecs, SaveChunkMessage, SaveDirectory};

// Rendering, palettes and ray casts
pub use crate::raycast::{
    cursor::{CursorBrush, CursorHit, CursorQueryMode},
    terrain_ray::TerrainRay,
};
pub use crate::texture_palette::{
    PaletteBuilder, PaletteMaterial, TexturePalette, plugin::ActivePalette,
};
pub use crate::voxel::VoxelDepthPrepass;
