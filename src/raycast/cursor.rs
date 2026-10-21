//! The terrain point under the mouse (`CursorHit`), the chunk focus it
//! drives, and the default brush (`CursorBrush`).
//!
//! The brush digs, builds, smooths, flattens, paints, fills, scrapes, hides
//! and reveals (with a sphere, or a cube for the last two), with the stroke
//! it's drawing in `BrushStroke`. F8 cycles the query backend.

use bevy::{
    ecs::{message::MessageWriter, system::SystemParam},
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};

use crate::{
    chunk::{CursorFocusChunk, FOCUS_RADIUS, chebyshev, world_pos_to_chunk_pos},
    field::{
        editor::{
            MaterialPaint, SdfCsgSphere, SdfSculptOp, SdfSculptSphere, VisibilityCsgBox,
            VisibilityCsgSphere,
        },
        ops::PlaneMode,
    },
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
    /// Like Flatten, but only adds material: fills dips up to the plane.
    Fill,
    /// Like Flatten, but only removes material: shaves bumps down to the
    /// plane.
    Scrape,
    /// Hides the sphere (exact CSG on the `VisibilityField`): the terrain
    /// inside it is cut away and the cut closed by a cap.
    Hide,
    /// Reveals the sphere again.
    Reveal,
}

impl BrushTool {
    pub const ALL: [Self; 9] = [
        Self::Dig,
        Self::Build,
        Self::Smooth,
        Self::Flatten,
        Self::Paint,
        Self::Fill,
        Self::Scrape,
        Self::Hide,
        Self::Reveal,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dig => "Dig",
            Self::Build => "Build",
            Self::Smooth => "Smooth",
            Self::Flatten => "Flatten",
            Self::Paint => "Paint",
            Self::Fill => "Fill",
            Self::Scrape => "Scrape",
            Self::Hide => "Hide",
            Self::Reveal => "Reveal",
        }
    }

    /// Dig and Build swap, and so do Fill and Scrape, and Hide and Reveal;
    /// the rest have no opposite.
    #[must_use]
    pub const fn inverted(self) -> Self {
        match self {
            Self::Dig => Self::Build,
            Self::Build => Self::Dig,
            Self::Fill => Self::Scrape,
            Self::Scrape => Self::Fill,
            Self::Hide => Self::Reveal,
            Self::Reveal => Self::Hide,
            other => other,
        }
    }

    /// The side of the stroke's plane this tool works on, if it works on
    /// one.
    #[must_use]
    pub const fn plane_mode(self) -> Option<PlaneMode> {
        match self {
            Self::Flatten => Some(PlaneMode::Flatten),
            Self::Fill => Some(PlaneMode::Fill),
            Self::Scrape => Some(PlaneMode::Scrape),
            _ => None,
        }
    }

    const fn color(self) -> Color {
        match self {
            Self::Dig => Color::srgb(1.0, 0.35, 0.3),
            Self::Build => Color::srgb(0.4, 1.0, 0.4),
            Self::Smooth => Color::srgb(0.4, 0.7, 1.0),
            Self::Flatten => Color::srgb(1.0, 0.9, 0.3),
            Self::Paint => Color::srgb(1.0, 0.4, 1.0),
            Self::Fill => Color::srgb(0.6, 1.0, 0.8),
            Self::Scrape => Color::srgb(1.0, 0.6, 0.2),
            Self::Hide => Color::srgb(0.5, 0.5, 0.6),
            Self::Reveal => Color::srgb(0.9, 0.9, 1.0),
        }
    }
}

/// The plane `BrushTool::Flatten`, `Fill` and `Scrape` pull toward,
/// through the point where the stroke started.
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

/// What `BrushTool::Hide` and `Reveal` edit: the brush sphere, or the
/// axis-aligned cube of the same half extent (fog-of-war cells, rooms).
/// The other tools are always spheres.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrushShape {
    #[default]
    Sphere,
    Cube,
}

impl BrushShape {
    #[must_use]
    pub const fn toggled(self) -> Self {
        match self {
            Self::Sphere => Self::Cube,
            Self::Cube => Self::Sphere,
        }
    }
}

/// The built-in mouse brush (`cursor_brush`).
///
/// Hold left mouse to apply `tool` at the `CursorHit`; hold Ctrl as well to
/// swap Dig and Build (or Fill and Scrape). Change the fields at runtime to
/// switch tools, or set `enabled: false` and edit from `CursorHit` yourself.
#[derive(Resource, Clone, Debug)]
pub struct CursorBrush {
    pub enabled: bool,
    pub tool: BrushTool,
    /// World units. Grown to 1.5 voxels on a coarse-LOD hit so the edit
    /// doesn't vanish between voxels.
    pub radius: f32,
    /// How fast the brush works, at its center (slower toward the rim).
    /// Dig and Build: world units per second the surface moves; they stop
    /// at the sphere either way. Smooth, Flatten, Fill and Scrape: the
    /// fraction of the way voxels move per second.
    pub strength: f32,
    /// Palette id that Paint (and Build, with `paint_on_build`) writes.
    /// Needs a `MaterialFieldPlugin`; without one painting does nothing.
    pub material: u8,
    pub paint_on_build: bool,
    /// The plane Flatten, Fill and Scrape pull toward.
    pub flatten: FlattenPlane,
    /// Sphere or cube for Hide and Reveal.
    pub shape: BrushShape,
    /// Draw the brush sphere (or cube) at the cursor.
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
            shape: BrushShape::default(),
            show_gizmo: true,
        }
    }
}

impl CursorBrush {
    /// The tool in effect this frame: `tool`, with Ctrl swapping it for its
    /// opposite (`BrushTool::inverted`).
    #[must_use]
    pub fn active_tool(&self, keys: &ButtonInput<KeyCode>) -> BrushTool {
        if keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]) {
            self.tool.inverted()
        } else {
            self.tool
        }
    }

    /// Whether `tool` edits the cube rather than the sphere.
    #[must_use]
    pub fn uses_cube(&self, tool: BrushTool) -> bool {
        self.shape == BrushShape::Cube && matches!(tool, BrushTool::Hide | BrushTool::Reveal)
    }

    /// The unit normal of the plane a stroke starting at `hit` would use.
    fn flatten_normal(&self, hit: &CursorSurfaceHit) -> Vec3 {
        match self.flatten {
            FlattenPlane::Level => Vec3::Y,
            FlattenPlane::Slope => hit.normal.normalize_or(Vec3::Y),
        }
    }

    /// The sphere `tool` edits for `hit`: (center, radius) in world space;
    /// for a cube (`uses_cube`) the radius is its half extent. Build sits
    /// half a radius out of the surface so it adds material.
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
    visibility: MessageWriter<'w, VisibilityCsgSphere>,
    visibility_box: MessageWriter<'w, VisibilityCsgBox>,
}

/// A stroke that jumps further than this many brush radii in one frame (the
/// cursor slid off a ledge onto distant terrain) starts a new segment
/// instead of sweeping straight through whatever lies between.
const MAX_SWEEP_RADII: f32 = 4.0;

/// The stroke `cursor_brush` is drawing: from pressing left mouse to
/// releasing it or switching tools. `None` between strokes.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct BrushStroke(pub Option<Stroke>);

/// What a stroke keeps from frame to frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    pub tool: BrushTool,
    /// The hit the stroke started on.
    pub start: Vec3,
    /// The plane Flatten, Fill and Scrape pull toward, through `start`:
    /// (point, unit normal pointing out of the terrain).
    pub plane: (Vec3, Vec3),
    /// The last edit's center, which the next edit sweeps from. `None`
    /// after a frame with nothing under the cursor, or a jump.
    pub last: Option<Vec3>,
}

impl Stroke {
    /// Where an edit at `center` sweeps from (`None`: a plain sphere), and
    /// makes `center` the last.
    fn sweep_to(&mut self, center: Vec3, radius: f32) -> Option<Vec3> {
        let from = self
            .last
            .filter(|last| last.distance(center) <= radius * MAX_SWEEP_RADII);
        self.last = Some(center);
        from
    }
}

/// Applies `CursorBrush` while left mouse is held.
///
/// Dig, Build, Smooth, Flatten, Fill and Scrape are gradual, scaled by the
/// frame time, and sweep from the last frame's center so a fast stroke
/// leaves no gaps. Dig, Build, Paint, Hide and Reveal stop at the brush
/// sphere, so holding them in place doesn't pile up. A cube (`BrushShape`)
/// is placed per frame and doesn't sweep.
pub fn cursor_brush(
    brush: Res<CursorBrush>,
    input: BrushInput,
    hit: Res<CursorHit>,
    terrain: TerrainRay,
    mut stroke: ResMut<BrushStroke>,
    mut out: BrushWriters,
) {
    let tool = brush.active_tool(&input.keys);
    let held = input.mouse.pressed(MouseButton::Left);
    if stroke.0.is_some_and(|s| !held || s.tool != tool) {
        stroke.0 = None;
    }
    if !held {
        return;
    }
    let Some(h) = hit.0 else {
        // Nothing under the cursor: the next hit starts a new segment.
        if let Some(s) = &mut stroke.0 {
            s.last = None;
        }
        return;
    };

    let (center, radius) = brush.sphere(tool, &h, &terrain);
    let s = stroke.0.get_or_insert_with(|| Stroke {
        tool,
        start: h.pos,
        plane: (h.pos, brush.flatten_normal(&h)),
        last: None,
    });
    let sweep_from = s.sweep_to(center, radius);
    let (point, normal) = s.plane;

    let dt = input.time.delta_secs();
    let amount = (brush.strength * dt).clamp(0.0, 1.0);
    match tool {
        BrushTool::Dig | BrushTool::Build => {
            let build = tool == BrushTool::Build;
            out.csg.write(SdfCsgSphere {
                center,
                radius,
                sweep_from,
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
                sweep_from,
                amount,
                op: SdfSculptOp::Smooth,
            });
        }
        BrushTool::Flatten | BrushTool::Fill | BrushTool::Scrape => {
            let mode = tool.plane_mode().unwrap_or_default();
            out.sculpt.write(SdfSculptSphere {
                center,
                radius,
                sweep_from,
                amount,
                op: SdfSculptOp::Flatten {
                    point,
                    normal,
                    mode,
                },
            });
        }
        BrushTool::Paint => {
            out.paint.write(MaterialPaint {
                center,
                radius,
                material: brush.material,
            });
        }
        BrushTool::Hide | BrushTool::Reveal => {
            let hide = tool == BrushTool::Hide;
            if brush.uses_cube(tool) {
                out.visibility_box.write(VisibilityCsgBox {
                    center,
                    half_size: Vec3::splat(radius),
                    hide,
                    max_step: None,
                });
            } else {
                out.visibility.write(VisibilityCsgSphere {
                    center,
                    radius,
                    sweep_from,
                    hide,
                    max_step: None,
                });
            }
        }
    }
}

/// Outlines the sphere (or cube) `cursor_brush` would edit, colored by
/// tool, and for Flatten, Fill and Scrape the plane under it. Does nothing
/// without `GizmoPlugin`.
pub fn draw_cursor_brush(
    brush: Res<CursorBrush>,
    keys: Res<ButtonInput<KeyCode>>,
    hit: Res<CursorHit>,
    stroke: Res<BrushStroke>,
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
    if brush.uses_cube(tool) {
        let cube = Transform::from_translation(center).with_scale(Vec3::splat(radius * 2.0));
        gizmos.cube(cube, color);
    } else {
        gizmos.sphere(Isometry3d::from_translation(center), radius, color);
    }
    if tool.plane_mode().is_some() {
        // The stroke's plane, or the one a stroke started here would use,
        // drawn where it cuts under the brush.
        let (point, normal) = stroke
            .0
            .map_or_else(|| (h.pos, brush.flatten_normal(&h)), |s| s.plane);
        let on_plane = center - normal * (center - point).dot(normal);
        let rotation = Quat::from_rotation_arc(Vec3::Z, normal);
        gizmos.circle(Isometry3d::new(on_plane, rotation), radius, color);
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

#[cfg(test)]
mod tests {
    use super::{BrushTool, Stroke, Vec3};

    #[test]
    fn stroke_sweeps_from_its_last_center_but_not_across_a_jump() {
        let mut stroke = Stroke {
            tool: BrushTool::Dig,
            start: Vec3::ZERO,
            plane: (Vec3::ZERO, Vec3::Y),
            last: None,
        };
        // The first edit is a plain sphere, the next sweeps from it.
        assert_eq!(stroke.sweep_to(Vec3::ZERO, 1.0), None);
        assert_eq!(stroke.sweep_to(Vec3::X * 3.0, 1.0), Some(Vec3::ZERO));
        // Further than four radii: a new segment.
        assert_eq!(stroke.sweep_to(Vec3::X * 8.0, 1.0), None);
        assert_eq!(stroke.last, Some(Vec3::X * 8.0));
    }
}
