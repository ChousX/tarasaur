//! Edit application and distance-field upkeep.
//!
//! Edit systems split each world-space edit across the chunks it overlaps
//! and retry the ones not yet editable (`apply_edits`). Value edits leave a
//! distance field (`SDF`, `VisibilityField`) as signs, so
//! `spawn_reinit_tasks` / `resolve_reinit_tasks` rebuild real distances
//! with an async jump flood; CSG edits write exact distances and skip that.

use std::{
    collections::{HashMap, hash_map::Entry},
    marker::PhantomData,
};

use super::{
    DistanceField, EditableChunks, Field, SDF, VoxelField,
    editor::{EditFieldMessage, EditMode},
    ops::{AccumulateExt, BlendExt, CsgDelta, ShapeBounds, ShapeScale},
};
use crate::{
    chunk::CHUNK_SIZE,
    chunk::{ChunkManager, overlapping_chunks},
    field::{
        DirtyField, MaterialField, MaxEditLod, VoxelMaterial,
        editor::{
            CsgMessage, CsgOp, MaterialPaint, MaterialPaintVoxels, SdfSculptOp,
            SdfSculptSphere, SdfSphereStamp, WorldEditor,
        },
        ops::{FieldSphereOps, ShapeEditOps},
    },
    num::f32_from_u32,
    voxel::resident::ForceResident,
};
use bevy::{
    ecs::{message::MessageReader, system::SystemParam},
    prelude::*,
    tasks::{AsyncComputeTaskPool, ComputeTaskPool, Task, block_on, poll_once},
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

/// In-flight async JFA recompute for one chunk's distance field `F`.
///
/// `started_version` is the field's version this task's input was cloned
/// at; `resolve_reinit_tasks` uses it to detect whether a newer edit landed
/// while the task was running.
#[derive(Component)]
pub struct ReinitTask<F> {
    task: Task<Box<[f32]>>,
    started_version: u64,
    _field: PhantomData<F>,
}

/// The `SDF`'s reinit: what loading and saving wait for.
pub type SdfReinitTask = ReinitTask<SDF>;

/// Dirty chunks of `F` with no reinit in flight.
type NeedsReinit<F> = (With<DirtyField<F, f32>>, Without<ReinitTask<F>>);

/// Spawns a background JFA task for every dirty chunk of `F` that doesn't
/// already have one in flight.
///
/// The `Without<ReinitTask>` filter is what prevents re-spawning a second
/// task each frame while the sphere is still resizing: `DirtyField` stays
/// set (see resolve, below) but no new task starts until the current one
/// resolves.
pub fn spawn_reinit_tasks<F: DistanceField>(
    mut commands: Commands,
    query: Query<(Entity, &F), NeedsReinit<F>>,
) {
    for (entity, field) in query.iter() {
        let (data, size, started_version) = field.reinit_input();

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut data = data.into_vec();
            crate::field::sdf::compute_sdf_distances(&mut data, size);
            data.into_boxed_slice()
        });

        commands.entity(entity).insert(ReinitTask::<F> {
            task,
            started_version,
            _field: PhantomData,
        });
    }
}

/// Runs every frame; non-blocking poll of every in-flight reinit task.
pub fn resolve_reinit_tasks<F: DistanceField>(
    mut commands: Commands,
    mut query: Query<(Entity, &mut F, &mut ReinitTask<F>)>,
) {
    for (entity, mut field, mut pending) in query.iter_mut() {
        let Some(data) = block_on(poll_once(&mut pending.task)) else {
            continue;
        };

        if field.version() != pending.started_version {
            // A newer edit landed while this task was computing: its input
            // is stale and applying it would erase that edit. Drop the
            // result; DirtyField is still set, so next frame's
            // spawn_reinit_tasks picks up the *current* data instead.
            commands.entity(entity).remove::<ReinitTask<F>>();
            continue;
        }

        field.apply_reinit_result(data);
        commands
            .entity(entity)
            .remove::<ReinitTask<F>>()
            .remove::<DirtyField<F, f32>>();
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

/// The value one voxel past the field's face from `v` (on it), read from
/// the face neighbour in `fields`: what a gradient on the chunk's border
/// reads (see `FieldSphereOps::csg_sphere`). `None` if that neighbour
/// isn't loaded at the same LOD.
fn beyond_face<F: Field<f32>>(
    fields: &HashMap<IVec3, &F>,
    chunk_pos: IVec3,
    size: u32,
    v: UVec3,
    axis: usize,
    positive: bool,
) -> Option<f32> {
    let step = IVec3::AXES.get(axis).copied().unwrap_or(IVec3::X);
    let neighbour = if positive {
        chunk_pos.saturating_add(step)
    } else {
        chunk_pos.saturating_sub(step)
    };
    let sdf = fields.get(&neighbour).filter(|sdf| sdf.size().x == size)?;
    // The neighbour's layer that touches this chunk.
    let mut n = v;
    n[axis] = if positive { 0 } else { size.saturating_sub(1) };
    Some(sdf.get(n.x, n.y, n.z))
}

/// Queues `ops` (each with its `reinit_after`) against every chunk they
/// overlap (their shape plus `reach_world`), leasing those chunks resident
/// meanwhile.
fn queue_csg_edits(
    queued: &mut Vec<DeferredEdit<(CsgOp, bool)>>,
    ops: impl Iterator<Item = (CsgOp, bool)>,
    reach_world: f32,
    resident: &mut ForceResident,
) {
    for (op, reinit) in ops {
        let (mid, half) = op.shape.bounds(reach_world);
        resident.touch_box(mid, half);
        queued.extend(overlapping_chunks(mid, half).map(|chunk_pos| DeferredEdit {
            chunk_pos,
            edit: (op, reinit),
            age: 0,
        }));
    }
}

/// One chunk's share of a CSG edit, ready to work out.
struct CsgWork {
    entity: Entity,
    chunk_pos: IVec3,
    op: CsgOp,
    /// See `CsgMessage::reinit_after`.
    reinit: bool,
}

/// The editable edits of `queued` in waves, each chunk at most once per
/// wave, so a chunk hit by several edits takes them in order. The rest
/// wait in `deferred` (for `MAX_DEFER_FRAMES`, then they're dropped).
fn csg_waves<F: DistanceField>(
    queued: Vec<DeferredEdit<(CsgOp, bool)>>,
    chunk_manager: &ChunkManager,
    editable: &mut EditableChunks<F>,
    deferred: &mut Vec<DeferredEdit<(CsgOp, bool)>>,
) -> Vec<Vec<CsgWork>> {
    let mut waves: Vec<Vec<CsgWork>> = Vec::new();
    let mut wave_of: HashMap<Entity, usize> = HashMap::new();
    for mut e in queued {
        let Some(entity) = chunk_manager.get_chunk(&e.chunk_pos) else {
            continue; // chunk not loaded / gone: nothing to edit
        };
        if editable.get_mut(entity).is_some() {
            let wave = wave_of.entry(entity).or_insert(0);
            if waves.len() <= *wave {
                waves.push(Vec::new());
            }
            if let Some(w) = waves.get_mut(*wave) {
                w.push(CsgWork {
                    entity,
                    chunk_pos: e.chunk_pos,
                    op: e.edit.0,
                    reinit: e.edit.1,
                });
            }
            *wave = wave.saturating_add(1);
        } else {
            e.age = e.age.saturating_add(1); // waiting for the chunk to reach max LOD
            if e.age < MAX_DEFER_FRAMES {
                deferred.push(e);
            } else {
                warn!(
                    "[csg] chunk {:?} never reached MaxEditLod, edit dropped",
                    e.chunk_pos
                );
            }
        }
    }
    if deferred.len() > MAX_DEFERRED {
        let excess = deferred.len().saturating_sub(MAX_DEFERRED);
        deferred.drain(..excess); // drop the oldest
    }
    waves
}

/// What `process_csg_edits` writes besides the field.
#[derive(SystemParam)]
pub struct CsgOutputs<'w, 's> {
    commands: Commands<'w, 's>,
    resident: ResMut<'w, ForceResident>,
    paint: MessageWriter<'w, MaterialPaintVoxels>,
}

/// Applies `M`'s CSG edits to its distance field (`SdfCsgSphere` to the
/// `SDF`, `VisibilityCsgSphere` and `VisibilityCsgBox` to the
/// `VisibilityField`): every chunk an edit overlaps, worked out in parallel
/// from one state, then written.
pub fn process_csg_edits<M: CsgMessage>(
    mut events: MessageReader<M>,
    mut editable: EditableChunks<M::Field>,
    chunk_manager: Res<ChunkManager>,
    max: Res<MaxEditLod>,
    mut deferred: Local<Vec<DeferredEdit<(CsgOp, bool)>>>,
    mut out: CsgOutputs,
) {
    let reach_world = SDF_CSG_REACH_VOXELS * max.0.voxel_size();
    // Like `apply_edits`: last frame's retries first, then the new edits on
    // every chunk they overlap.
    let mut queued = std::mem::take(&mut *deferred);
    let ops = events.read().map(|m| (m.csg(), m.reinit_after()));
    queue_csg_edits(&mut queued, ops, reach_world, &mut out.resident);

    let waves = csg_waves(queued, &chunk_manager, &mut editable, &mut deferred);
    for wave in waves {
        // Every chunk of the wave is worked out from the same state, reading
        // its neighbours' borders directly, on every core; then all are
        // written. No DirtyField unless the message asks: CSG output is a
        // valid distance field. `set` bumps version/inside_count, which
        // drives re-extract.
        let deltas: Vec<CsgDelta> = {
            let mut fields: HashMap<IVec3, &M::Field> = HashMap::new();
            for w in &wave {
                for offset in [IVec3::ZERO, IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y]
                    .into_iter()
                    .chain([IVec3::Z, IVec3::NEG_Z])
                {
                    let pos = w.chunk_pos.saturating_add(offset);
                    if let Entry::Vacant(slot) = fields.entry(pos)
                        && let Some(sdf) =
                            chunk_manager.get_chunk(&pos).and_then(|e| editable.get(e))
                    {
                        slot.insert(sdf);
                    }
                }
            }
            let fields = &fields;
            ComputeTaskPool::get().scope(|s| {
                for w in &wave {
                    s.spawn(async move {
                        let Some(sdf) = fields.get(&w.chunk_pos) else {
                            return CsgDelta::default();
                        };
                        let size = sdf.size().x;
                        let origin = w.chunk_pos.as_vec3() * CHUNK_SIZE;
                        let voxel_size = CHUNK_SIZE / f32_from_u32(size);
                        sdf.csg_shape_delta(
                            w.op.shape.to_grid(origin, voxel_size),
                            SDF_CSG_REACH_VOXELS,
                            w.op.subtract,
                            w.op.max_step.map(|s| s / voxel_size),
                            |v, axis, positive| {
                                beyond_face(fields, w.chunk_pos, size, v, axis, positive)
                            },
                        )
                    });
                }
            })
        };
        for (w, delta) in wave.iter().zip(deltas) {
            let Some(mut sdf) = editable.get_mut(w.entity) else {
                continue;
            };
            let filled = sdf.apply_csg_delta(delta);
            if w.reinit {
                out.commands
                    .entity(w.entity)
                    .insert(DirtyField::<M::Field, f32>::default());
            }
            if let Some(material) = w.op.paint
                && !filled.is_empty()
            {
                out.paint.write(MaterialPaintVoxels {
                    chunk: w.entity,
                    voxels: sdf.with_air_neighbours(&filled),
                    material,
                });
            }
        }
    }
}

pub fn process_sdf_sculpts(
    mut events: MessageReader<SdfSculptSphere>,
    mut editable: EditableChunks<SDF>,
    chunk_manager: Res<ChunkManager>,
    mut deferred: Local<Vec<DeferredEdit<SdfSculptSphere>>>,
    mut resident: ResMut<ForceResident>,
) {
    let new = events.read().map(|&op| {
        let sweep = op.sweep();
        let (mid, half) = sweep.bounds(op.radius);
        resident.touch_sphere(mid, sweep.from.distance(sweep.to).mul_add(0.5, op.radius));
        (mid, half, op)
    });
    apply_edits(
        &mut deferred,
        new,
        &chunk_manager,
        &mut editable,
        |_, sdf, origin, voxel_size, op| {
            // Like CSG: the output is still a distance field, so no DirtyField.
            let at = op.sweep().to_grid(origin, voxel_size);
            let radius = op.radius / voxel_size;
            match op.op {
                SdfSculptOp::Smooth => sdf.smooth_sphere(at, radius, op.amount),
                SdfSculptOp::Flatten {
                    point,
                    normal,
                    mode,
                } => sdf.flatten_sphere(
                    at,
                    radius,
                    (point - origin) / voxel_size,
                    normal,
                    op.amount,
                    mode,
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
