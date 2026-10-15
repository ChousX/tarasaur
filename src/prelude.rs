//! Common imports: `use tarasaur::prelude::*;`.
//!
//! Everything here also has one canonical path in its module; the prelude
//! only gathers what most apps touch.

pub use crate::plugin::TarasaurPlugin;

// Chunks and streaming
pub use crate::chunk::{CHUNK_SIZE, ChunkLoader, ChunkManager, ShowChunkBounds};
pub use crate::field::generator::ChunkGeneratorRegistry;

// Fields and editing
pub use crate::field::{
    Field, LOD, MaterialField, MaterialFieldPlugin, MaxEditLod, SDF, TargetLOD, VisibilityField,
    VoxelMaterial,
    editor::{EditMode, SdfCsgSphere, SdfSphereStamp, WorldEditor},
    ops::{AccumulateExt, BlendExt},
    persistence::{AutosaveIntervalSecs, RegisterSaveableFieldExt, SaveChunkMessage},
};

// Rendering, palettes and ray casts
pub use crate::texture_palette::{
    PaletteBuilder, PaletteMaterial, TexturePalette, plugin::ActivePalette,
};
pub use crate::voxel::{
    VoxelDepthPrepass,
    cursor::{CursorBrush, CursorHit, CursorQueryMode},
    terrain_ray::TerrainRay,
};
