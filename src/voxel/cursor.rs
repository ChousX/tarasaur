use bevy::{ecs::message::MessageWriter, prelude::*, window::PrimaryWindow};

use crate::{
    chunk::update_cursor_focus,
    editor::SdfCsgSphere,
    voxel::{
        query::{PendingVoxelQueries, VoxelQueryResults},
        terrain_ray::TerrainRay,
        types::RayQuery,
    },
};

const MAX_RAY_DISTANCE: f32 = 100.0;
const BRUSH_RADIUS: f32 = 2.0;

/// Which backend answers "what terrain is under the cursor?".
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CursorQueryMode {
    /// CPU ray first; fall back to the GPU result if the CPU finds nothing.
    #[default]
    Auto,
    /// GPU readback only (2-3 frames late, arena-0 chunks only).
    Gpu,
    /// CPU SDF raycast only. No GPU query is submitted.
    Cpu,
}

impl CursorQueryMode {
    fn next(self) -> Self {
        match self {
            Self::Auto => Self::Cpu,
            Self::Cpu => Self::Gpu,
            Self::Gpu => Self::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HitSource {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug)]
pub struct CursorSurfaceHit {
    pub pos: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub source: HitSource,
}

/// The single resolved cursor hit. Gameplay and chunk focus read this and
/// never need to know which backend produced it.
#[derive(Resource, Default)]
pub struct CursorHit(pub Option<CursorSurfaceHit>);

pub fn cycle_cursor_mode(keys: Res<ButtonInput<KeyCode>>, mut mode: ResMut<CursorQueryMode>) {
    if keys.just_pressed(KeyCode::F8) {
        *mode = mode.next();
        info!("[cursor] query mode -> {:?}", *mode);
    }
}

pub fn resolve_cursor_hit(
    mode: Res<CursorQueryMode>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    terrain: TerrainRay,
    gpu: Res<VoxelQueryResults>,
    mut pending: ResMut<PendingVoxelQueries>,
    mut out: ResMut<CursorHit>,
) {
    let ray = windows
        .single()
        .ok()
        .and_then(|w| w.cursor_position())
        .and_then(|cursor| {
            let (cam, cam_t) = cameras.iter().find(|(cam, _)| cam.is_active)?;
            cam.viewport_to_world(cam_t, cursor).ok()
        });

    let Some(ray) = ray else {
        out.0 = None;
        return;
    };
    let dir: Vec3 = *ray.direction;

    // Submit the GPU query only when it can be used.
    if *mode != CursorQueryMode::Cpu {
        pending.0.push(RayQuery {
            origin: ray.origin.into(),
            max_distance: MAX_RAY_DISTANCE,
            direction: dir.into(),
            user_id: 0,
        });
    }

    let cpu_hit = (*mode != CursorQueryMode::Gpu)
        .then(|| terrain.cast(ray.origin, dir, MAX_RAY_DISTANCE))
        .flatten()
        .map(|h| CursorSurfaceHit {
            pos: h.pos,
            normal: h.normal,
            distance: h.distance,
            source: HitSource::Cpu,
        });

    let gpu_hit = || {
        gpu.0
            .first()
            .filter(|h| h.did_hit != 0)
            .map(|h| CursorSurfaceHit {
                pos: Vec3::from(h.hit_pos_world),
                normal: Vec3::from(h.hit_normal),
                distance: h.hit_distance,
                source: HitSource::Gpu,
            })
    };

    out.0 = match *mode {
        CursorQueryMode::Cpu => cpu_hit,
        CursorQueryMode::Gpu => gpu_hit(),
        CursorQueryMode::Auto => cpu_hit.or_else(gpu_hit),
    };
}

/// Left mouse digs, right mouse adds. CSG is idempotent, so holding the
/// button every frame doesn't accumulate.
pub fn cursor_brush(
    mouse: Res<ButtonInput<MouseButton>>,
    hit: Res<CursorHit>,
    terrain: TerrainRay,
    mut out: MessageWriter<SdfCsgSphere>,
) {
    let Some(h) = hit.0 else { return };
    let dig = mouse.pressed(MouseButton::Left);
    let add = mouse.pressed(MouseButton::Right);
    if !dig && !add {
        return;
    }

    // A coarse-LOD hit gets a bigger brush so the edit doesn't land in air.
    let vs = terrain.voxel_size_at(h.pos).unwrap_or(0.3);
    let radius = BRUSH_RADIUS.max(vs * 1.5);

    if dig {
        out.write(SdfCsgSphere {
            center: h.pos,
            radius,
            subtract: true,
        });
    } else {
        out.write(SdfCsgSphere {
            center: h.pos + h.normal * radius * 0.5,
            radius,
            subtract: false,
        });
    }
}

pub struct CursorPlugin;
impl Plugin for CursorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CursorQueryMode>()
            .init_resource::<CursorHit>()
            .add_systems(
                Update,
                (
                    cycle_cursor_mode,
                    resolve_cursor_hit
                        .after(cycle_cursor_mode)
                        .before(update_cursor_focus),
                    cursor_brush.after(resolve_cursor_hit),
                ),
            );
    }
}
