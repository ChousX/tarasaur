//! Edit application and SDF upkeep.
//!
//! Edit systems split each world-space edit across the chunks it overlaps
//! and retry the ones not yet editable (`apply_edits`). Value edits leave the
//! SDF as signs, so `spawn_sdf_reinit_tasks` / `resolve_sdf_reinit_tasks`
//! rebuild real distances with an async jump flood; CSG edits write exact
//! distances and skip that.

use std::collections::{HashMap, hash_map::Entry};

use super::{
    EditableChunks, Field, SDF, VoxelField,
    editor::{EditFieldMessage, EditMode},
    ops::{AccumulateExt, BlendExt, ShapeBounds, ShapeScale},
};
use crate::{
    chunk::CHUNK_SIZE,
    chunk::{ChunkManager, overlapping_chunks},
    field::{
        DirtyField, MaterialField, MaxEditLod, VoxelMaterial,
        editor::{
            MaterialPaint, MaterialPaintVoxels, SdfCsgSphere, SdfSculptOp, SdfSculptSphere,
            SdfSphereStamp, WorldEditor,
        },
        ops::{FieldSphereOps, ShapeEditOps},
    },
    num::{f32_from_u32, usize_from},
    voxel::resident::ForceResident,
};
use bevy::{
    ecs::message::MessageReader,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

const MAX_DEFER_FRAMES: u32 = 600;
const MAX_DEFERRED: usize = 4096;

/// One world-space edit queued against one chunk.
pub struct DeferredEdit<E> {
    chunk_pos: IVec3,
    edit: E,
    age: u32,
}

/// Shared retry loop for the edit systems. Runs last frame's retries (oldest
/// first), then `new` edits — `(center, half_extent, edit)` — against every
/// chunk they overlap. `apply` gets the field in grid space: the chunk's
/// world origin and voxel size. An edit whose chunk isn't editable yet (see
/// `EditableChunks`) is retried for `MAX_DEFER_FRAMES`, then handed to
/// `on_drop`; one whose chunk isn't loaded is skipped.
fn apply_edits<F, V, E: Copy>(
    deferred: &mut Vec<DeferredEdit<E>>,
    new: impl IntoIterator<Item = (Vec3, Vec3, E)>,
    chunk_manager: &ChunkManager,
    editable: &mut EditableChunks<F>,
    mut apply: impl FnMut(Entity, &mut F, Vec3, f32, E),
    mut on_drop: impl FnMut(IVec3),
) where
    F: Field<V> + VoxelField,
    V: Copy + Default,
{
    let mut work = std::mem::take(deferred);
    for (center, half_extent, edit) in new {
        work.extend(
            overlapping_chunks(center, half_extent).map(|chunk_pos| DeferredEdit {
                chunk_pos,
                edit,
                age: 0,
            }),
        );
    }
    for mut e in work {
        let Some(entity) = chunk_manager.get_chunk(&e.chunk_pos) else {
            continue; // chunk not loaded / gone: nothing to edit
        };
        if let Some(mut field) = editable.get_mut(entity) {
            let origin = e.chunk_pos.as_vec3() * CHUNK_SIZE;
            let voxel_size = CHUNK_SIZE / f32_from_u32(field.size().x);
            apply(entity, &mut field, origin, voxel_size, e.edit);
        } else {
            e.age = e.age.saturating_add(1); // waiting for the chunk to reach max LOD
            if e.age < MAX_DEFER_FRAMES {
                deferred.push(e);
            } else {
                on_drop(e.chunk_pos);
            }
        }
    }
    if deferred.len() > MAX_DEFERRED {
        let excess = deferred.len().saturating_sub(MAX_DEFERRED);
        deferred.drain(..excess); // drop the oldest
    }
}

/// One queued shape edit: (center, shape, value, mode).
type ShapeEdit<S, V> = (Vec3, S, V, EditMode<V>);

pub fn process_shape_edits<F, S, V>(
    mut commands: Commands,
    mut events: MessageReader<EditFieldMessage<F, S, V>>,
    mut editable: EditableChunks<F>,
    chunk_manager: Res<ChunkManager>,
    mut deferred: Local<Vec<DeferredEdit<ShapeEdit<S, V>>>>,
) where
    F: Field<V> + VoxelField + ShapeEditOps<V, S>,
    S: Primitive3d + Copy + Send + Sync + 'static + ShapeBounds + ShapeScale,
    V: Copy + Default + Send + Sync + 'static + AccumulateExt + BlendExt,
{
    let new = events.read().map(|ev| {
        (
            ev.center,
            ev.shape.half_extent(),
            (ev.center, ev.shape, ev.val, ev.mode),
        )
    });
    apply_edits(
        &mut deferred,
        new,
        &chunk_manager,
        &mut editable,
        |entity, field, origin, voxel_size, (center, shape, val, mode)| {
            let grid_center = (center - origin) / voxel_size;
            let grid_shape = shape.scaled_by(1.0 / voxel_size);
            match mode {
                EditMode::Absolute => field.fill(grid_center, grid_shape, val),
                EditMode::Accumulate { delta } => field.accumulate(grid_center, grid_shape, delta),
                EditMode::Blend { rate } => field.blend(grid_center, grid_shape, val, rate),
            }
            commands
                .entity(entity)
                .insert(DirtyField::<F, V>::default());
        },
        |_| {},
    );
}

pub fn process_sdf_sphere_stamps(
    mut commands: Commands,
    mut events: MessageReader<SdfSphereStamp>,
    mut editable: EditableChunks<SDF>,
    chunk_manager: Res<ChunkManager>,
    mut deferred: Local<Vec<DeferredEdit<SdfSphereStamp>>>,
) {
    let new = events
        .read()
        .map(|&s| (s.center, Vec3::splat(s.bound_radius), s));
    apply_edits(
        &mut deferred,
        new,
        &chunk_manager,
        &mut editable,
        |entity, sdf, origin, voxel_size, stamp| {
            sdf.stamp_sdf_sphere(
                (stamp.center - origin) / voxel_size,
                stamp.bound_radius / voxel_size,
                stamp.sdf_radius / voxel_size,
            );
            commands
                .entity(entity)
                .insert(DirtyField::<SDF, f32>::default());
        },
        |_| {},
    );
}

/// In-flight async JFA recompute for one chunk's SDF.
///
/// `started_version` is the `SDF::version` this task's input was cloned at — used by
/// `resolve_sdf_reinit_tasks` to detect whether a newer edit landed while
/// the task was running.
#[derive(Component)]
pub struct SdfReinitTask {
    task: Task<Box<[f32]>>,
    started_version: u64,
}

/// Dirty SDF chunks with no reinit in flight.
type NeedsReinit = (With<DirtyField<SDF, f32>>, Without<SdfReinitTask>);

/// Spawns a background JFA task for every dirty SDF chunk that doesn't
/// already have one in flight.
///
/// The `Without<SdfReinitTask>` filter is
/// what prevents re-spawning a second task each frame while the sphere
/// is still resizing — `DirtyField` stays set (see resolve, below) but no
/// new task starts until the current one resolves.
pub fn spawn_sdf_reinit_tasks(mut commands: Commands, query: Query<(Entity, &SDF), NeedsReinit>) {
    for (entity, sdf) in query.iter() {
        let (data, size, started_version) = sdf.reinit_input();

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut data = data.into_vec();
            crate::field::sdf::compute_sdf_distances(&mut data, size);
            data.into_boxed_slice()
        });

        commands.entity(entity).insert(SdfReinitTask {
            task,
            started_version,
        });
    }
}

/// Runs every frame; non-blocking poll of every in-flight reinit task.
pub fn resolve_sdf_reinit_tasks(
    mut commands: Commands,
    mut query: Query<(Entity, &mut SDF, &mut SdfReinitTask)>,
) {
    for (entity, mut sdf, mut pending) in query.iter_mut() {
        let Some(data) = block_on(poll_once(&mut pending.task)) else {
            continue;
        };

        if sdf.version != pending.started_version {
            // A newer edit landed while this task was computing — its
            // input is stale and applying it would erase that edit.
            // Drop the result; DirtyField is still set, so next frame's
            // spawn_sdf_reinit_tasks picks up the *current* data instead.
            commands.entity(entity).remove::<SdfReinitTask>();
            continue;
        }

        sdf.apply_reinit_result(data);
        commands
            .entity(entity)
            .remove::<SdfReinitTask>()
            .remove::<DirtyField<SDF, f32>>();
    }
}

/// (entity, SDF edited, visibility edited).
#[cfg(debug_assertions)]
type EditedFields = (
    Entity,
    Has<DirtyField<SDF, f32>>,
    Has<DirtyField<crate::field::VisibilityField, bool>>,
);

/// Chunks with both fields, at least one edited, and no reinit in flight.
#[cfg(debug_assertions)]
type DesyncCandidate = (
    With<SDF>,
    With<crate::field::VisibilityField>,
    Without<SdfReinitTask>,
    Or<(
        With<DirtyField<SDF, f32>>,
        With<DirtyField<crate::field::VisibilityField, bool>>,
    )>,
);

/// Debug builds only: warns when an edit cycle changed a chunk's SDF or its
/// visibility but not both. Often deliberate (hiding a region), so it's a
/// development aid, and release builds don't pay for it.
#[cfg(debug_assertions)]
pub fn detect_topology_desync(
    query: Query<EditedFields, DesyncCandidate>,
    positions: Query<&crate::chunk::ChunkPosition>,
) {
    for (entity, sdf_dirty, vis_dirty) in query.iter() {
        if sdf_dirty != vis_dirty {
            let pos = positions.get(entity).map(|p| p.0);
            warn!(
                "[topology_desync] chunk {:?} (entity {:?}): SDF edited={}, VisibilityField edited={} — \
                 these usually need to change together (a solid-sphere edit that only touched one field \
                 will render geometry that doesn't match its visibility mask, or vice versa)",
                pos, entity, sdf_dirty, vis_dirty
            );
        }
    }
}

/// Ends the edit cycle for a field whose `DirtyField` needs no follow-up.
///
/// That's every field but `SDF`, which keeps its marker until the jump flood
/// lands. Runs in `FieldSet::Reinit`, after every edit of the cycle.
pub fn clear_dirty<F, V>(mut commands: Commands, query: Query<Entity, With<DirtyField<F, V>>>)
where
    F: Field<V>,
    V: Copy + Default + Send + Sync + 'static,
{
    for entity in query.iter() {
        commands.entity(entity).remove::<DirtyField<F, V>>();
    }
}

const SDF_CSG_REACH_VOXELS: f32 = 4.0;

/// A chunk's faces as (axis, positive): -x, +x, -y, +y, -z, +z.
const FACES: [(usize, bool); 6] = [
    (0, false),
    (0, true),
    (1, false),
    (1, true),
    (2, false),
    (2, true),
];

/// A face's voxel at `layer` along `axis` and (a, b) along the other two
/// axes, lower axis first.
const fn face_voxel(axis: usize, layer: u32, a: u32, b: u32) -> UVec3 {
    match axis {
        0 => UVec3::new(layer, a, b),
        1 => UVec3::new(a, layer, b),
        _ => UVec3::new(a, b, layer),
    }
}

/// The voxel layer just past each face of a chunk, copied from its face
/// neighbours: what a gradient on the chunk's border reads (see
/// `FieldSphereOps::csg_sphere`). Gathered for every chunk before any is
/// edited, so they all step from the same state. A face whose neighbour
/// isn't loaded at the same LOD is `None`.
struct Halo {
    size: u32,
    /// In `FACES` order, each indexed (a, b) with a fastest.
    faces: [Option<Vec<f32>>; 6],
}

impl Halo {
    fn gather(pos: IVec3, chunks: &ChunkManager, editable: &EditableChunks<SDF>) -> Option<Self> {
        let size = chunks
            .get_chunk(&pos)
            .and_then(|e| editable.get(e))?
            .size()
            .x;
        let last = size.saturating_sub(1);
        let faces = FACES.map(|(axis, positive)| {
            let step = IVec3::AXES.get(axis).copied().unwrap_or(IVec3::X);
            let neighbour = if positive {
                pos.saturating_add(step)
            } else {
                pos.saturating_sub(step)
            };
            let sdf = chunks
                .get_chunk(&neighbour)
                .and_then(|e| editable.get(e))
                .filter(|sdf| sdf.size().x == size)?;
            // The neighbour's layer that touches this chunk.
            let layer = if positive { 0 } else { last };
            Some(
                (0..size)
                    .flat_map(|b| (0..size).map(move |a| face_voxel(axis, layer, a, b)))
                    .map(|v| sdf.get(v.x, v.y, v.z))
                    .collect(),
            )
        });
        Some(Self { size, faces })
    }

    /// The value one voxel past this chunk's face from `v`, on it.
    fn get(&self, v: UVec3, axis: usize, positive: bool) -> Option<f32> {
        let face = FACES.iter().position(|&f| f == (axis, positive))?;
        let (a, b) = match axis {
            0 => (v.y, v.z),
            1 => (v.x, v.z),
            _ => (v.x, v.y),
        };
        let i = usize_from(b)
            .checked_mul(usize_from(self.size))?
            .checked_add(usize_from(a))?;
        self.faces.get(face)?.as_ref()?.get(i).copied()
    }
}

pub fn process_sdf_csg_spheres(
    mut events: MessageReader<SdfCsgSphere>,
    mut editable: EditableChunks<SDF>,
    chunk_manager: Res<ChunkManager>,
    max: Res<MaxEditLod>,
    mut deferred: Local<Vec<DeferredEdit<SdfCsgSphere>>>,
    mut resident: ResMut<ForceResident>,
    mut paint: MessageWriter<MaterialPaintVoxels>,
) {
    let reach_world = SDF_CSG_REACH_VOXELS * max.0.voxel_size();
    let new: Vec<_> = events
        .read()
        .map(|&op| {
            resident.touch_sphere(op.center, op.radius + reach_world);
            (op.center, Vec3::splat(op.radius + reach_world), op)
        })
        .collect();
    // Gradual edits read their neighbours' borders (see `Halo`).
    let mut halos = HashMap::new();
    let gradual = new
        .iter()
        .filter(|(_, _, op)| op.max_step.is_some())
        .flat_map(|&(center, half, _)| overlapping_chunks(center, half))
        .chain(
            deferred
                .iter()
                .filter(|d| d.edit.max_step.is_some())
                .map(|d| d.chunk_pos),
        );
    for pos in gradual {
        let Some(entity) = chunk_manager.get_chunk(&pos) else {
            continue;
        };
        if let Entry::Vacant(slot) = halos.entry(entity)
            && let Some(halo) = Halo::gather(pos, &chunk_manager, &editable)
        {
            slot.insert(halo);
        }
    }
    apply_edits(
        &mut deferred,
        new,
        &chunk_manager,
        &mut editable,
        |chunk, sdf, origin, voxel_size, op| {
            // No DirtyField: CSG output is already a valid distance field.
            // `set` bumps version/inside_count, which drives re-extract.
            let filled = sdf.csg_sphere(
                (op.center - origin) / voxel_size,
                op.radius / voxel_size,
                SDF_CSG_REACH_VOXELS,
                op.subtract,
                op.max_step.map(|s| s / voxel_size),
                |v, axis, positive| halos.get(&chunk)?.get(v, axis, positive),
            );
            if let Some(material) = op.paint
                && !filled.is_empty()
            {
                paint.write(MaterialPaintVoxels {
                    chunk,
                    voxels: sdf.with_air_neighbours(&filled),
                    material,
                });
            }
        },
        |chunk_pos| warn!("[csg] chunk {chunk_pos:?} never reached MaxEditLod, edit dropped"),
    );
}

pub fn process_sdf_sculpts(
    mut events: MessageReader<SdfSculptSphere>,
    mut editable: EditableChunks<SDF>,
    chunk_manager: Res<ChunkManager>,
    mut deferred: Local<Vec<DeferredEdit<SdfSculptSphere>>>,
    mut resident: ResMut<ForceResident>,
) {
    let new = events.read().map(|&op| {
        resident.touch_sphere(op.center, op.radius);
        (op.center, Vec3::splat(op.radius), op)
    });
    apply_edits(
        &mut deferred,
        new,
        &chunk_manager,
        &mut editable,
        |_, sdf, origin, voxel_size, op| {
            // Like CSG: the output is still a distance field, so no DirtyField.
            let center = (op.center - origin) / voxel_size;
            let radius = op.radius / voxel_size;
            match op.op {
                SdfSculptOp::Smooth => sdf.smooth_sphere(center, radius, op.amount),
                SdfSculptOp::Flatten { point, normal } => sdf.flatten_sphere(
                    center,
                    radius,
                    (point - origin) / voxel_size,
                    normal,
                    op.amount,
                ),
            }
        },
        |chunk_pos| warn!("[sculpt] chunk {chunk_pos:?} never reached MaxEditLod, edit dropped"),
    );
}

/// Applies `MaterialPaintVoxels` to `MaterialField<M>`. The chunk was just
/// edited, so it's at max LOD; one whose field isn't ready is skipped.
pub fn apply_material_paint_voxels<M: VoxelMaterial>(
    mut paints: MessageReader<MaterialPaintVoxels>,
    mut editable: EditableChunks<MaterialField<M>>,
) {
    for p in paints.read() {
        let Some(mut field) = editable.get_mut(p.chunk) else {
            continue;
        };
        let material = M::from_id(p.material);
        for v in &p.voxels {
            field.set(v.x, v.y, v.z, material);
        }
    }
}

/// Turns id-based `MaterialPaint`s into `MaterialField<M>` edits.
pub fn forward_material_paint<M: VoxelMaterial>(
    mut paints: MessageReader<MaterialPaint>,
    mut editor: WorldEditor<MaterialField<M>, M>,
) {
    for p in paints.read() {
        editor.fill_sphere(p.center, p.radius, M::from_id(p.material));
    }
}
