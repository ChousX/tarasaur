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

use crate::voxel::arena::{VoxelChunkArena, VoxelChunkArenaSet};

/// Which registered layer: an index into [`VoxelLayers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerIndex(pub usize);

impl LayerIndex {
    /// The terrain, registered by `TarasaurPlugin`.
    pub const TERRAIN: Self = Self(0);
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
}

impl VoxelLayers {
    /// Registers a layer and returns its index.
    pub fn add(&mut self, raster: LayerDesc) -> LayerIndex {
        self.layers.push(Layer {
            set: VoxelChunkArenaSet::default(),
            raster,
            material_layout: None,
            material: None,
        });
        LayerIndex(self.layers.len().saturating_sub(1))
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

    /// The arenas the GPU ray queries and the cursor trace: the terrain's.
    #[must_use]
    pub fn queryable(&self) -> Option<&VoxelChunkArenaSet> {
        self.set(LayerIndex::TERRAIN)
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
