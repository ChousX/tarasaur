//! The terrain point under the mouse (`CursorHit`) and a default CSG brush
//! (left digs, right adds). F8 cycles the query backend.

use bevy::{
    ecs::message::MessageWriter,
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};

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
    /// Picks the backend for you. For a single cursor ray that is the CPU:
    /// it answers the same frame, sees every LOD, and the main world always
    /// holds the SDF, so the GPU can't hit anything the CPU misses. Falling
    /// back to the GPU would only add a 2-3 frame old hit (a ghost hit right
    /// after the cursor leaves the terrain).
    #[default]
    Auto,
    /// GPU readback only (2-3 frames late).
    Gpu,
    /// CPU SDF raycast only. No GPU query is submitted.
    Cpu,
}

impl CursorQueryMode {
    const fn next(self) -> Self {
        match self {
            Self::Auto => Self::Cpu,
            Self::Cpu => Self::Gpu,
            Self::Gpu => Self::Auto,
        }
    }

    /// The backend this mode actually runs.
    const fn backend(self) -> HitSource {
        match self {
            Self::Auto | Self::Cpu => HitSource::Cpu,
            Self::Gpu => HitSource::Gpu,
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

/// Toggles the built-in mouse brush (`cursor_brush`). Turn it off when the
/// app does its own editing from `CursorHit`.
#[derive(Resource)]
pub struct CursorBrush {
    pub enabled: bool,
}

impl Default for CursorBrush {
    fn default() -> Self {
        Self { enabled: true }
    }
}

pub fn cycle_cursor_mode(keys: Res<ButtonInput<KeyCode>>, mut mode: ResMut<CursorQueryMode>) {
    if keys.just_pressed(KeyCode::F8) {
        *mode = mode.next();
        info!("[cursor] query mode -> {:?}", *mode);
    }
}

pub fn resolve_cursor_hit(
    mode: Res<CursorQueryMode>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    terrain: TerrainRay,
    gpu: Res<VoxelQueryResults>,
    mut pending: ResMut<PendingVoxelQueries>,
    mut out: ResMut<CursorHit>,
) {
    let ray = windows
        .single()
        .ok()
        .and_then(|(w, opts)| {
            // With a grabbed/locked cursor (mouse-look), aim from the screen centre.
            if opts.grab_mode == CursorGrabMode::None {
                w.cursor_position()
            } else {
                Some(w.size() / 2.0)
            }
        })
        .and_then(|cursor| {
            let (cam, cam_t) = cameras.iter().find(|(cam, _)| cam.is_active)?;
            cam.viewport_to_world(cam_t, cursor).ok()
        });

    let Some(ray) = ray else {
        out.0 = None;
        return;
    };
    let dir: Vec3 = *ray.direction;

    out.0 = match mode.backend() {
        HitSource::Cpu => {
            terrain
                .cast(ray.origin, dir, MAX_RAY_DISTANCE)
                .map(|h| CursorSurfaceHit {
                    pos: h.pos,
                    normal: h.normal,
                    distance: h.distance,
                    source: HitSource::Cpu,
                })
        }
        HitSource::Gpu => {
            pending.0.push(RayQuery {
                origin: ray.origin.into(),
                max_distance: MAX_RAY_DISTANCE,
                direction: dir.into(),
                user_id: 0,
            });
            gpu.0
                .first()
                .filter(|h| h.did_hit != 0)
                .map(|h| CursorSurfaceHit {
                    pos: Vec3::from(h.hit_pos_world),
                    normal: Vec3::from(h.hit_normal),
                    distance: h.hit_distance,
                    source: HitSource::Gpu,
                })
        }
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
            .init_resource::<CursorBrush>()
            .add_systems(
                Update,
                (
                    cycle_cursor_mode,
                    resolve_cursor_hit
                        .after(cycle_cursor_mode)
                        .before(update_cursor_focus),
                    cursor_brush
                        .after(resolve_cursor_hit)
                        .run_if(|b: Res<CursorBrush>| b.enabled),
                ),
            );
    }
}
