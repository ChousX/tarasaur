//! Common imports: `use tarasaur::prelude::*;`.
//!
//! Everything here also has one canonical path in its module; the prelude
//! only gathers what most apps touch.

pub use crate::plugin::{AppVoxelFieldExt, MaterialFieldPlugin, TarasaurPlugin, TerrainLayer};

// Chunks and streaming
pub use crate::chunk::{CHUNK_SIZE, ChunkLoader, ChunkManager, ShowChunkBounds};
pub use crate::storage::generator::ChunkGeneratorRegistry;

// Fields and editing
pub use crate::field::{
    DistanceField, Field, GpuChannel, LOD, MaterialField, MaxEditLod, SDF, TargetLOD,
    VisibilityField, VoxelField, VoxelMaterial,
    editor::{
        EditMode, MaterialPaint, SdfCsgSphere, SdfSculptOp, SdfSculptSphere, SdfSphereStamp,
        VisibilityCsgBox, VisibilityCsgSphere, WorldEditor,
    },
    ops::{AccumulateExt, BlendExt, PlaneMode, Sweep},
};
pub use crate::storage::persistence::{AutosaveIntervalSecs, SaveChunkMessage, SaveDirectory};

// Rendering, palettes and ray casts
pub use crate::raycast::{
    collider::{
        HitTarget, Intersect, LayerCollider, RayCollider, RayColliderExt, RayHit, Union, cast,
        cast_filtered,
    },
    cursor::{
        BrushShape, BrushStroke, BrushTool, CursorBrush, CursorFilter, CursorHit,
        CursorQueryMode, FlattenPlane, Stroke,
    },
    terrain_ray::TerrainRay,
};
pub use crate::texture_palette::{
    PaletteBuilder, PaletteMaterial, TexturePalette, plugin::ActivePalette,
};
pub use crate::voxel::{
    AppSurfaceLayerExt, ClipChannel, LayerMaterial, LayerPass, LayerRaster, LayerResidency,
    NoChannel, PaletteRaster, SubSurface, SurfaceLayer, VoxelDepthPrepass,
};
