//! The terrain point under the mouse (`CursorHit`), the chunk focus it
//! drives, and the default brush (`CursorBrush`: dig, build, smooth,
//! flatten, paint). F8 cycles the query backend.

use bevy::{
    ecs::{message::MessageWriter, system::SystemParam},
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};

use crate::{
    chunk::{CursorFocusChunk, FOCUS_RADIUS, chebyshev, world_pos_to_chunk_pos},
    field::editor::{MaterialPaint, SdfCsgSphere, SdfSculptOp, SdfSculptSphere},
    raycast::{
        gpu::{PendingVoxelQueries, RayQuery, VoxelQueryResults},
        terrain_ray::TerrainRay,
    },
};

const MAX_RAY_DISTANCE: f32 = 100.0;

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

/// What the built-in brush does while the left mouse button is held.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrushTool {
    /// Carves the sphere out (exact CSG).
    #[default]
    Dig,
    /// Adds a sphere on the surface (exact CSG), painted with the brush's
    /// material when `paint_on_build` is set.
    Build,
    /// Rounds off bumps and creases.
    Smooth,
    /// Pulls terrain toward a plane through the point where the stroke
    /// started (see `FlattenPlane`): fills
    /// dips, shaves bumps.
    Flatten,
    /// Sets the material inside the sphere; the shape is untouched.
    Paint,
}

impl BrushTool {
    pub const ALL: [Self; 5] = [
        Self::Dig,
        Self::Build,
        Self::Smooth,
        Self::Flatten,
        Self::Paint,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dig => "Dig",
            Self::Build => "Build",
            Self::Smooth => "Smooth",
            Self::Flatten => "Flatten",
            Self::Paint => "Paint",
        }
    }

    /// Dig and Build swap; the rest have no opposite.
    #[must_use]
    pub const fn inverted(self) -> Self {
        match self {
            Self::Dig => Self::Build,
            Self::Build => Self::Dig,
            other => other,
        }
    }

    const fn color(self) -> Color {
        match self {
            Self::Dig => Color::srgb(1.0, 0.35, 0.3),
            Self::Build => Color::srgb(0.4, 1.0, 0.4),
            Self::Smooth => Color::srgb(0.4, 0.7, 1.0),
            Self::Flatten => Color::srgb(1.0, 0.9, 0.3),
            Self::Paint => Color::srgb(1.0, 0.4, 1.0),
        }
    }
}

/// The plane `BrushTool::Flatten` pulls toward, through the point where
/// the stroke started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlattenPlane {
    /// Horizontal: pads, floors, roads.
    #[default]
    Level,
    /// Along the surface under the cursor: ramps.
    Slope,
}

impl FlattenPlane {
    #[must_use]
    pub const fn toggled(self) -> Self {
        match self {
            Self::Level => Self::Slope,
            Self::Slope => Self::Level,
        }
    }
}

/// The built-in mouse brush (`cursor_brush`).
///
/// Hold left mouse to apply `tool` at the `CursorHit`; hold Ctrl as well to
/// swap Dig and Build. Change the fields at runtime to switch tools, or set
/// `enabled: false` and edit from `CursorHit` yourself.
#[derive(Resource, Clone, Debug)]
pub struct CursorBrush {
    pub enabled: bool,
    pub tool: BrushTool,
    /// World units. Grown to 1.5 voxels on a coarse-LOD hit so the edit
    /// doesn't vanish between voxels.
    pub radius: f32,
    /// How fast the brush works, at its center (slower toward the rim).
    /// Dig and Build: world units per second the surface moves; they stop
    /// at the sphere either way. Smooth and Flatten: the fraction of the way
    /// voxels move per second.
    pub strength: f32,
    /// Palette id that Paint (and Build, with `paint_on_build`) writes.
    /// Needs a `MaterialFieldPlugin`; without one painting does nothing.
    pub material: u8,
    pub paint_on_build: bool,
    /// The plane Flatten pulls toward.
    pub flatten: FlattenPlane,
    /// Draw the brush sphere at the cursor.
    pub show_gizmo: bool,
}

impl Default for CursorBrush {
    fn default() -> Self {
        Self {
            enabled: true,
            tool: BrushTool::default(),
            radius: 2.0,
            strength: 3.0,
            material: 0,
            paint_on_build: true,
            flatten: FlattenPlane::default(),
            show_gizmo: true,
        }
    }
}

impl CursorBrush {
    /// The tool in effect this frame: `tool`, with Ctrl swapping Dig and Build.
    #[must_use]
    pub fn active_tool(&self, keys: &ButtonInput<KeyCode>) -> BrushTool {
        if keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]) {
            self.tool.inverted()
        } else {
            self.tool
        }
    }

    /// The unit normal of the plane Flatten would use, starting at `hit`.
    fn flatten_normal(&self, hit: &CursorSurfaceHit) -> Vec3 {
        match self.flatten {
            FlattenPlane::Level => Vec3::Y,
            FlattenPlane::Slope => hit.normal.normalize_or(Vec3::Y),
        }
    }

    /// The sphere `tool` edits for `hit`: (center, radius) in world space.
    /// Build sits half a radius out of the surface so it adds material.
    fn sphere(&self, tool: BrushTool, hit: &CursorSurfaceHit, terrain: &TerrainRay) -> (Vec3, f32) {
        let vs = terrain.voxel_size_at(hit.pos).unwrap_or(0.3);
        let radius = self.radius.max(vs * 1.5);
        let center = if tool == BrushTool::Build {
            hit.pos + hit.normal * radius * 0.5
        } else {
            hit.pos
        };
        (center, radius)
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

/// The input `cursor_brush` reads.
#[derive(SystemParam)]
pub struct BrushInput<'w> {
    mouse: Res<'w, ButtonInput<MouseButton>>,
    keys: Res<'w, ButtonInput<KeyCode>>,
    time: Res<'w, Time>,
}

/// Everything `cursor_brush` writes.
#[derive(SystemParam)]
pub struct BrushWriters<'w> {
    csg: MessageWriter<'w, SdfCsgSphere>,
    sculpt: MessageWriter<'w, SdfSculptSphere>,
    paint: MessageWriter<'w, MaterialPaint>,
}

/// Applies `CursorBrush` while left mouse is held.
///
/// Dig, Build, Smooth and Flatten are gradual, scaled by the frame time;
/// Dig, Build and Paint stop at the brush sphere, so holding them in place
/// doesn't pile up.
pub fn cursor_brush(
    brush: Res<CursorBrush>,
    input: BrushInput,
    hit: Res<CursorHit>,
    terrain: TerrainRay,
    // Flatten's plane (point, normal), fixed when the stroke starts.
    mut plane: Local<Option<(Vec3, Vec3)>>,
    mut out: BrushWriters,
) {
    let tool = brush.active_tool(&input.keys);
    let held = input.mouse.pressed(MouseButton::Left);
    if !held || tool != BrushTool::Flatten {
        *plane = None;
    }
    let Some(h) = hit.0 else { return };
    if !held {
        return;
    }

    let (center, radius) = brush.sphere(tool, &h, &terrain);
    let dt = input.time.delta_secs();
    let amount = (brush.strength * dt).clamp(0.0, 1.0);
    match tool {
        BrushTool::Dig | BrushTool::Build => {
            let build = tool == BrushTool::Build;
            out.csg.write(SdfCsgSphere {
                center,
                radius,
                subtract: !build,
                max_step: Some(brush.strength.max(0.0) * dt),
                // Only what the build fills, as it fills it.
                paint: (build && brush.paint_on_build).then_some(brush.material),
            });
        }
        BrushTool::Smooth => {
            out.sculpt.write(SdfSculptSphere {
                center,
                radius,
                amount,
                op: SdfSculptOp::Smooth,
            });
        }
        BrushTool::Flatten => {
            let (point, normal) = *plane.get_or_insert((h.pos, brush.flatten_normal(&h)));
            out.sculpt.write(SdfSculptSphere {
                center,
                radius,
                amount,
                op: SdfSculptOp::Flatten { point, normal },
            });
        }
        BrushTool::Paint => {
            out.paint.write(MaterialPaint {
                center,
                radius,
                material: brush.material,
            });
        }
    }
}

/// Outlines the sphere `cursor_brush` would edit, colored by tool. Does
/// nothing without `GizmoPlugin`.
pub fn draw_cursor_brush(
    brush: Res<CursorBrush>,
    keys: Res<ButtonInput<KeyCode>>,
    hit: Res<CursorHit>,
    terrain: TerrainRay,
    gizmos: Option<Gizmos>,
) {
    let (Some(mut gizmos), Some(h)) = (gizmos, hit.0) else {
        return;
    };
    if !brush.show_gizmo {
        return;
    }
    let tool = brush.active_tool(&keys);
    let (center, radius) = brush.sphere(tool, &h, &terrain);
    let color = tool.color();
    gizmos.sphere(Isometry3d::from_translation(center), radius, color);
    if tool == BrushTool::Flatten {
        // The plane a stroke started here would flatten toward.
        let rotation = Quat::from_rotation_arc(Vec3::Z, brush.flatten_normal(&h));
        gizmos.circle(Isometry3d::new(h.pos, rotation), radius, color);
    }
}

/// Keeps `CursorFocusChunk` pointing at the chunk under the cursor.
///
/// Backend-agnostic: reads the resolved `CursorHit` (CPU, GPU or Auto, see
/// `CursorQueryMode`). Sticky: the focus only moves once the hit leaves the
/// current `FOCUS_RADIUS` window, and a miss keeps the last focus. Otherwise a
/// LOD upgrade that nudges the hit into the next chunk (or a grazing ray
/// flickering between hit and miss) replans and reloads every frame.
pub fn update_cursor_focus(hit: Res<CursorHit>, mut focus: ResMut<CursorFocusChunk>) {
    let Some(h) = hit.0.as_ref() else {
        return;
    };
    // Nudge inside the solid so a hit on a chunk boundary doesn't flicker.
    let new = world_pos_to_chunk_pos(&(h.pos - h.normal * 0.05));
    let inside_window = focus.0.is_some_and(|f| chebyshev(new, f) <= FOCUS_RADIUS);
    if !inside_window {
        focus.0 = Some(new);
    } // assign only on change, so is_changed stays meaningful
}
