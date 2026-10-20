//! Surface layers: each a shape field meshed into its own arenas. Layer 0
//! is the terrain (`SDF`); the upload, meshing, culling and raster systems
//! loop over every registered layer.

use std::marker::PhantomData;

use bevy::{ecs::system::SystemParam, prelude::*};

use crate::voxel::arena::{VoxelChunkArena, VoxelChunkArenaSet};

/// Which registered layer: an index into [`VoxelLayers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerIndex(pub usize);

impl LayerIndex {
    /// The terrain, registered by `TarasaurPlugin`.
    pub const TERRAIN: Self = Self(0);
}

/// Every layer's arenas, in registration order. Render world.
#[derive(Resource, Default)]
pub struct VoxelLayers {
    sets: Vec<VoxelChunkArenaSet>,
}

impl VoxelLayers {
    /// Registers a layer and returns its index.
    pub fn add(&mut self) -> LayerIndex {
        self.sets.push(VoxelChunkArenaSet::default());
        LayerIndex(self.sets.len().saturating_sub(1))
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.sets.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    #[must_use]
    pub fn get(&self, layer: LayerIndex) -> Option<&VoxelChunkArenaSet> {
        self.sets.get(layer.0)
    }

    pub fn get_mut(&mut self, layer: LayerIndex) -> Option<&mut VoxelChunkArenaSet> {
        self.sets.get_mut(layer.0)
    }

    pub fn iter(&self) -> impl Iterator<Item = (LayerIndex, &VoxelChunkArenaSet)> {
        self.sets
            .iter()
            .enumerate()
            .map(|(i, s)| (LayerIndex(i), s))
    }

    pub fn sets_mut(&mut self) -> impl Iterator<Item = &mut VoxelChunkArenaSet> {
        self.sets.iter_mut()
    }

    /// Every arena of every layer.
    pub fn arenas(&self) -> impl Iterator<Item = &VoxelChunkArena> {
        self.sets.iter().flat_map(VoxelChunkArenaSet::iter)
    }

    pub fn arenas_mut(&mut self) -> impl Iterator<Item = &mut VoxelChunkArena> {
        self.sets.iter_mut().flat_map(VoxelChunkArenaSet::iter_mut)
    }

    /// The layer the GPU ray queries and the cursor trace: the terrain.
    #[must_use]
    pub fn queryable(&self) -> Option<&VoxelChunkArenaSet> {
        self.get(LayerIndex::TERRAIN)
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
        self.layers.get_mut(self.field.layer)
    }
}
