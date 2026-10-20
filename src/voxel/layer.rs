//! Surface layers: each a shape field meshed into its own arenas and drawn
//! with its own look.
//!
//! Layer 0 is the terrain (`SDF`, drawn by `PaletteRaster`); the upload,
//! meshing, culling and raster systems loop over every registered layer.

use std::{any::type_name, marker::PhantomData};

use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    render::{
        render_resource::{BindGroup, BindGroupLayout, BindGroupLayoutEntry, BlendState, Face},
        renderer::RenderDevice,
    },
};

use crate::{
    field::{Field, LOD, MaterialField, VisibilityField, VoxelField, VoxelMaterial},
    plugin::AppVoxelFieldExt,
    voxel::{
        add_channel_upload,
        arena::{VoxelChunkArena, VoxelChunkArenaSet},
        extract::{NoClip, Surfaces},
    },
};

/// Which registered layer: an index into [`VoxelLayers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerIndex(pub usize);

impl LayerIndex {
    /// The terrain, registered by `TarasaurPlugin`.
    pub const TERRAIN: Self = Self(0);
}

/// Which chunks a layer holds arena slots for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LayerResidency {
    /// Every loaded chunk. The layer's shape field decides per chunk whether
    /// it sends data (`VoxelField::should_extract`); a chunk whose volume
    /// meshes to nothing holds no slot either way.
    #[default]
    Loaded,
    /// Only chunks where the terrain has a surface and that its
    /// `VisibilityField` doesn't hide entirely (unless leased by
    /// `ForceResident`). The terrain's own, and right for a layer that only
    /// exists on it.
    Terrain,
}

/// Which of a layer's surfaces in a chunk a slot holds. 0 for most layers;
/// a layer whose field stores several owners per voxel meshes one surface
/// per owner (see `SurfaceLayer::surfaces`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubSurface(pub u8);

impl SubSurface {
    /// Every surface of a chunk: what an extraction with no data stands
    /// for when the whole field is gone. Never a surface's own id.
    pub const ALL: Self = Self(u8::MAX);
}

/// A meshed surface: a shape field, its optional material and clip, and
/// its look. Register one with `app.add_surface_layer::<L>()`.
///
/// Every const has a default. The terrain is `TerrainLayer`: `SDF` clipped
/// by `VisibilityField`, `PaletteRaster`, `Terrain` residency, queryable.
pub trait SurfaceLayer: Send + Sync + 'static {
    /// The field whose zero crossing is meshed. Its `GpuChannel` must be
    /// `Shape`.
    type Shape: VoxelField + Field<<Self::Shape as VoxelField>::Value>;
    /// Per-voxel ids blended into vertices and read per fragment
    /// (`MaterialField<M>`), or `NoChannel`.
    type Material: SideChannel;
    /// A distance field that clips the shape (`VisibilityField`), or
    /// `NoChannel`. Only where both are inside is meshed, so the shape is
    /// cut where the clip is positive and the cut is closed by a surface
    /// along the clip's zero crossing. Folded into the shape's padded
    /// volume at extraction; the shape must hold `f32` distances, or
    /// have `SURFACES`.
    type Clip: ClipChannel;
    /// A per-voxel discard mask (`VisibilityField` with the
    /// `visibility-mask` feature), or `NoChannel`: the mesh is left whole
    /// and the fragment shader discards hidden voxels, with hard,
    /// voxel-aligned edges and no cap. The older way; `Clip` is the
    /// default.
    type Mask: SideChannel;
    /// How the meshes are drawn.
    type Raster: LayerRaster;

    /// Which chunks hold a slot.
    const RESIDENCY: LayerResidency = LayerResidency::Loaded;

    /// Coarsest LOD this layer is uploaded at. A chunk at a finer LOD is
    /// resampled first. `None` = the chunk's own LOD. A smooth, coarse
    /// surface at `Some(LOD::Low)` costs 1/64 of the arena memory of
    /// `High`.
    const UPLOAD_LOD: Option<LOD> = None;

    /// Whether the GPU ray queries and the cursor trace this layer. One
    /// layer may be queryable; the terrain is.
    const QUERYABLE: bool = false;

    /// How the field becomes surfaces, when it isn't its own: from a
    /// chunk's voxels (`lod.volume()` of them, x fastest, at `lod`), the
    /// signed distances of each surface in voxels of `lod`, negative
    /// inside, under its own `SubSurface` id. It runs on the chunk and on
    /// each neighbour the apron reads, each in its own LOD, so a shift by
    /// world units must be scaled by `lod.voxel_size()`.
    ///
    /// `None` (the default) meshes the field's own values through
    /// `Shape::gpu_bytes`, without copying neighbours: an SDF-like field is
    /// one surface, itself.
    const SURFACES: Option<Surfaces<Self::Shape>> = None;
}

/// A layer's optional material or mask field.
pub trait SideChannel: Send + Sync + 'static {
    /// Registers the field (if any) and its upload into `layer`, with the
    /// layer's residency and upload LOD.
    fn register(app: &mut App, layer: LayerIndex, residency: LayerResidency, cap: Option<LOD>);
}

/// A layer without a material (or a mask, or a clip).
pub struct NoChannel;

impl SideChannel for NoChannel {
    fn register(_: &mut App, _: LayerIndex, _: LayerResidency, _: Option<LOD>) {}
}

impl<M: VoxelMaterial> SideChannel for MaterialField<M> {
    fn register(app: &mut App, layer: LayerIndex, residency: LayerResidency, cap: Option<LOD>) {
        app.add_voxel_field::<Self>();
        add_channel_upload::<Self>(app, layer, residency, cap);
    }
}

/// With the `visibility-mask` feature the field is a `GpuChannel::Mask`
/// (hidden = distance > 0) and can be a layer's `Mask`.
#[cfg(feature = "visibility-mask")]
impl SideChannel for VisibilityField {
    fn register(app: &mut App, layer: LayerIndex, residency: LayerResidency, cap: Option<LOD>) {
        app.add_voxel_field::<Self>();
        add_channel_upload::<Self>(app, layer, residency, cap);
    }
}

/// A layer's optional clip field (see `SurfaceLayer::Clip`).
pub trait ClipChannel: Send + Sync + 'static {
    /// The field the shape's extraction reads: `f32` distances in voxels
    /// of the chunk's LOD, negative where the shape is kept. `NoClip` (a
    /// component no chunk has) for a layer without one.
    type Field: VoxelField<Elem = f32>;
    /// Registers the field (if any) for editing, saving and loading.
    fn register(app: &mut App);
}

impl ClipChannel for NoChannel {
    type Field = NoClip;
    fn register(_: &mut App) {}
}

impl ClipChannel for VisibilityField {
    type Field = Self;
    fn register(app: &mut App) {
        app.add_voxel_field::<Self>();
    }
}

/// Which of Bevy's passes a layer draws in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerPass {
    /// After `main_opaque_pass_3d`, writing depth; may use the depth
    /// prepass.
    Opaque,
    /// After `main_transparent_pass_3d`, back to front per chunk, testing
    /// depth without writing it.
    Transparent,
}

/// How a layer's meshes are drawn.
///
/// The vertex format is fixed: pass3's `Vertex` (32 bytes: world position +
/// packed material a, normal + packed material b and slot), and the shared
/// vertex stage lives in the `voxel_raster_vertex` shader library, so a
/// raster supplies the fragment side: its shader, and the group 2 bind
/// group that shader reads.
///
/// Every const has a default, and the defaults are the terrain's: a raster
/// that only supplies `shader`, `material_layout_entries` and `material`
/// draws opaque, unblended and back-face culled.
pub trait LayerRaster: Send + Sync + 'static {
    const PASS: LayerPass = LayerPass::Opaque;
    /// `None` is `BlendState::REPLACE`.
    const BLEND: Option<BlendState> = None;
    const CULL: Option<Face> = Some(Face::Back);

    /// The shader. It must define `vs_main` (usually a one-line wrapper
    /// around the library's `voxel_vertex`) and `fs_main`; an opaque layer
    /// also `fs_prepass`, for the depth prepass.
    fn shader() -> Handle<Shader>;

    /// Group 2's layout. The palette raster binds its texture arrays; a
    /// zone raster might bind a small uniform of colors.
    fn material_layout_entries() -> Vec<BindGroupLayoutEntry>;

    /// Builds group 2's bind group, from the render world. Called every
    /// frame in `RenderSystems::Prepare` with the material built last time
    /// (`None` the first time); return `None` to keep it. Nothing draws
    /// until it has returned `Some` once.
    fn material(
        world: &mut World,
        layout: &BindGroupLayout,
        current: Option<&LayerMaterial>,
    ) -> Option<LayerMaterial>;
}

/// A layer's group 2 bind group and what its raster pipelines specialize
/// on. The `arm`/`normal` bits belong to the palette shader; other rasters
/// leave them `false`.
pub struct LayerMaterial {
    pub bind_group: BindGroup,
    /// Whether `bind_group` holds a real ARM texture (selects `VOXEL_ARM`).
    pub has_arm: bool,
    /// Whether `bind_group` holds a real normal map (selects `VOXEL_NORMAL`).
    pub has_normal: bool,
    /// Distance from the camera past which no material samples its normal
    /// map; infinite without one. Chunks entirely past it are drawn without
    /// `VOXEL_NORMAL`, which looks the same.
    pub normals_end: f32,
}

impl LayerMaterial {
    /// A material with nothing to specialize on.
    #[must_use]
    pub const fn plain(bind_group: BindGroup) -> Self {
        Self {
            bind_group,
            has_arm: false,
            has_normal: false,
            normals_end: f32::INFINITY,
        }
    }
}

/// A [`LayerRaster`] with its type erased, so the draw and raster systems
/// need no generics.
#[derive(Clone)]
pub struct LayerDesc {
    pub pass: LayerPass,
    pub blend: Option<BlendState>,
    pub cull: Option<Face>,
    pub shader: fn() -> Handle<Shader>,
    pub material_layout_entries: fn() -> Vec<BindGroupLayoutEntry>,
    pub material: fn(&mut World, &BindGroupLayout, Option<&LayerMaterial>) -> Option<LayerMaterial>,
    /// The raster's type name, for labels.
    pub label: &'static str,
}

impl LayerDesc {
    #[must_use]
    pub fn of<R: LayerRaster>() -> Self {
        Self {
            pass: R::PASS,
            blend: R::BLEND,
            cull: R::CULL,
            shader: R::shader,
            material_layout_entries: R::material_layout_entries,
            material: R::material,
            label: type_name::<R>(),
        }
    }
}

/// One layer in the render world: its arenas, its look, and the material
/// its look is drawn with.
pub struct Layer {
    pub set: VoxelChunkArenaSet,
    pub raster: LayerDesc,
    pub residency: LayerResidency,
    /// Group 2's layout, created once the render device exists.
    pub material_layout: Option<BindGroupLayout>,
    /// Built by `LayerRaster::material`; `None` until it has been.
    pub material: Option<LayerMaterial>,
}

impl Layer {
    /// Distance past which this layer's chunks need no normal maps.
    #[must_use]
    pub fn normals_end(&self) -> f32 {
        self.material
            .as_ref()
            .map_or(f32::INFINITY, |m| m.normals_end)
    }
}

/// Every layer, in registration order. Render world.
#[derive(Resource, Default)]
pub struct VoxelLayers {
    layers: Vec<Layer>,
    /// The layer the GPU ray queries and the cursor trace.
    queryable: Option<LayerIndex>,
}

impl VoxelLayers {
    /// Registers a layer and returns its index.
    pub fn add(
        &mut self,
        raster: LayerDesc,
        residency: LayerResidency,
        queryable: bool,
    ) -> LayerIndex {
        self.layers.push(Layer {
            set: VoxelChunkArenaSet::default(),
            raster,
            residency,
            material_layout: None,
            material: None,
        });
        let index = LayerIndex(self.layers.len().saturating_sub(1));
        if queryable {
            if let Some(existing) = self.queryable {
                warn!(
                    "layer {} is queryable; layer {} stays the one the GPU queries trace",
                    index.0, existing.0
                );
            } else {
                self.queryable = Some(index);
            }
        }
        index
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    #[must_use]
    pub fn get(&self, layer: LayerIndex) -> Option<&Layer> {
        self.layers.get(layer.0)
    }

    pub fn get_mut(&mut self, layer: LayerIndex) -> Option<&mut Layer> {
        self.layers.get_mut(layer.0)
    }

    /// A layer's arenas.
    #[must_use]
    pub fn set(&self, layer: LayerIndex) -> Option<&VoxelChunkArenaSet> {
        self.get(layer).map(|l| &l.set)
    }

    pub fn set_mut(&mut self, layer: LayerIndex) -> Option<&mut VoxelChunkArenaSet> {
        self.get_mut(layer).map(|l| &mut l.set)
    }

    pub fn iter(&self) -> impl Iterator<Item = (LayerIndex, &Layer)> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, l)| (LayerIndex(i), l))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (LayerIndex, &mut Layer)> {
        self.layers
            .iter_mut()
            .enumerate()
            .map(|(i, l)| (LayerIndex(i), l))
    }

    pub fn sets_mut(&mut self) -> impl Iterator<Item = &mut VoxelChunkArenaSet> {
        self.layers.iter_mut().map(|l| &mut l.set)
    }

    /// Every arena of every layer.
    pub fn arenas(&self) -> impl Iterator<Item = &VoxelChunkArena> {
        self.layers.iter().flat_map(|l| l.set.iter())
    }

    pub fn arenas_mut(&mut self) -> impl Iterator<Item = &mut VoxelChunkArena> {
        self.layers.iter_mut().flat_map(|l| l.set.iter_mut())
    }

    /// The arenas the GPU ray queries and the cursor trace.
    #[must_use]
    pub fn queryable(&self) -> Option<&VoxelChunkArenaSet> {
        self.set(self.queryable?)
    }
}

/// The layer field `F` is uploaded into. Render world, one per uploaded
/// field type.
#[derive(Resource)]
pub struct FieldLayer<F> {
    pub layer: LayerIndex,
    _field: PhantomData<F>,
}

impl<F> FieldLayer<F> {
    #[must_use]
    pub const fn new(layer: LayerIndex) -> Self {
        Self {
            layer,
            _field: PhantomData,
        }
    }
}

/// System param: the arenas of the layer field `F` is uploaded into.
#[derive(SystemParam)]
pub struct LayerOf<'w, F: Send + Sync + 'static> {
    layers: ResMut<'w, VoxelLayers>,
    field: Res<'w, FieldLayer<F>>,
}

impl<F: Send + Sync + 'static> LayerOf<'_, F> {
    pub fn set(&mut self) -> Option<&mut VoxelChunkArenaSet> {
        self.layers.set_mut(self.field.layer)
    }
}

/// Runs in `RenderSystems::Prepare`: gives every layer whose raster has
/// something new (a palette that finished loading, a changed color table)
/// its material for this frame.
pub fn update_layer_materials(world: &mut World) {
    world.resource_scope::<VoxelLayers, _>(|world, mut layers| {
        for (_, layer) in layers.iter_mut() {
            let Some(layout) = &layer.material_layout else {
                continue;
            };
            if let Some(material) = (layer.raster.material)(world, layout, layer.material.as_ref())
            {
                layer.material = Some(material);
            }
        }
    });
}

/// Runs once the render device exists (`VoxelRenderPlugin::finish`): creates
/// every registered layer's group 2 layout. Layers registered later are not
/// drawn.
pub fn create_layer_layouts(world: &mut World) {
    world.resource_scope::<VoxelLayers, _>(|world, mut layers| {
        let device = world.resource::<RenderDevice>();
        for (_, layer) in layers.iter_mut() {
            if layer.material_layout.is_none() {
                layer.material_layout = Some(device.create_bind_group_layout(
                    Some(layer.raster.label),
                    &(layer.raster.material_layout_entries)(),
                ));
            }
        }
    });
}
