//! A CPU mirror of `surface_nets_pass3.wesl`'s LOD-seam lattice, and the
//! classifier it makes possible: which vertices the seam stitch pins to a
//! chunk edge.
//!
//! # What pass3 does at a seam
//!
//! A cell next to a coarser chunk takes the vertex of a *shared* cell, so
//! every chunk meshing that spot emits the identical vertex and the meshes
//! meet. Where a below-neighbour can't see one lattice step past a face,
//! `seam_collapse` flattens the shared cell onto that face. Flattened on two
//! axes it is a segment on a chunk edge (on three, a corner); with no sign
//! change on it, `relaxed_vertex` keeps its flat axes where they are, so the
//! vertex sits on the edge **however far the surface is**.
//!
//! # The threshold is not a bound
//!
//! Such a vertex's distance off the surface is the terrain's height above
//! the edge it is pinned to, and it is continuous in that height: along one
//! logged edge it reads 1.28, 1.68, 2.07, 2.47 and 2.87 Medium voxels at
//! successive lattice steps. Nothing holds it near 2.0, so the bench's
//! `MESH_MAX_OFF_SURFACE = 2.0` sits on no structural bound and cannot
//! separate these vertices from a real defect. The three logged failures
//! (2.320 and 2.074 Medium voxels, and a 1.477 High voxel corner under a bad
//! normal) reproduce exactly on the shipped shader with every neighbour
//! resident: the bench's `seam_gpu` test. Neighbour residency is not the
//! trigger. [`pinned_cell`] identifies these vertices instead, so a check can
//! exempt them the way it exempts folds, and keep its teeth everywhere else.
//!
//! # Known issue: a chunk can hold a seam rank its neighbour disagrees with
//!
//! The ranks are a snapshot taken when the chunk was last extracted, and a
//! chunk can keep one that no longer matches a loaded neighbour's LOD:
//! measured on toth in 10 of 12 bench runs, 1 to 9 directions each, always
//! the same shape -- the chunk's *own* rank stored for a neighbour that is
//! loaded at a coarser LOD, which is `seam_lod_ranks`' substitution for an
//! unreadable neighbour, left behind.
//!
//! It matters because the mesher rescales an apron value, and places the
//! layer it sits in, by the rank stored for that region. A chunk whose
//! apron was filled from a Medium neighbour but whose rank says High reads
//! that whole region at half its true distance: measured at 4953 texels of
//! one chunk, up to 2.06 voxels out. Vertices there are unaffected (they
//! come from the shared cell), but `compute_normal` is a central difference
//! of that same padded SDF, so a vertex on the boundary gets a normal from
//! half-scale values. That is the bench's intermittent
//! `hold/check:mesh_on_surface` normal at `[4,-1,4]`: 34.97 degrees where
//! the same vertex, in the same place, reads 12.7 when the rank is current.
//!
//! **Tried and rejected** (wave 5): making `plan_jobs` re-pad instead of
//! restitching when a rank the apron depends on changes, on the theory that
//! a metadata-only restitch was splitting the data and the ranks across two
//! extractions. It is gated green but changed nothing -- 12 of 12 runs still
//! held stale ranks -- so the restitch path is not where they come from, and
//! it was reverted rather than kept for its cost in extra re-pads. What is
//! established is that the slot's ranks disagree with the main world's own
//! view at probe time, after loading has settled and while nothing is
//! re-meshing; where that split is introduced is not pinned down.
//!
//! The bench reports it: `STALE SEAM RANKS` in every mesh report, naming
//! each chunk and direction. The errors are small on screen, so this is
//! recorded rather than chased.
//!
//! # Keeping it in step
//!
//! This is a port: change pass3's `seam_*` functions and this must change
//! with them. `seam_repro`'s tests pin it to the logged sites, and the
//! bench's `seam_gpu` test holds the shipped shader to the same answers.

use bevy::math::Vec3;

use crate::num::{f32_from_u32, usize_from};

type V3 = [i32; 3];
type U3 = [u32; 3];

const AXES: [usize; 3] = [0, 1, 2];

/// How far a coordinate may be from a lattice face and still be on it, in
/// the chunk's voxels. Pinned vertices sit on it exactly (a copy of the
/// texel position); a solved vertex lands there only by coincidence.
const ON_FACE: f32 = 1e-3;

/// log2 of each chunk's voxels per edge around the meshing chunk, by
/// neighbour offset in -1..=1 per axis, as pass3's `seam_log2` reads them.
pub(crate) struct Seam {
    log2: [i32; 27],
}

/// pass3 `rank_log2_size`: 4, 16, 32, 64 voxels for ranks 0..=3.
const fn rank_log2_size(rank: u32) -> i32 {
    match rank & 3 {
        0 => 2,
        1 => 4,
        2 => 5,
        _ => 6,
    }
}

impl Seam {
    /// From a chunk's `ChunkMeta::seam_ranks` (see `seam_lod_ranks`).
    pub(crate) fn from_ranks(ranks: [u32; 2]) -> Self {
        let mut log2 = [0; 27];
        for (k, l) in (0u32..).zip(log2.iter_mut()) {
            let word = ranks.get(usize_from(k / 16)).copied().unwrap_or(0);
            *l = rank_log2_size(word.wrapping_shr((k % 16).wrapping_mul(2)));
        }
        Self { log2 }
    }

    /// Every neighbour at `own`, then `(offset, log2)` overrides.
    #[cfg(test)]
    pub(crate) fn new(own: i32, others: &[(V3, i32)]) -> Self {
        let mut log2 = [own; 27];
        for &(o, l) in others {
            if let Some(slot) = log2.get_mut(index(o)) {
                *slot = l;
            }
        }
        Self { log2 }
    }

    /// Back to `seam_ranks` words: the inverse of [`Self::from_ranks`].
    #[cfg(test)]
    pub(crate) fn ranks(&self) -> [u32; 2] {
        let mut ranks = [0u32; 2];
        for (k, &l) in (0u32..).zip(&self.log2) {
            let rank = match l {
                2 => 0,
                4 => 1,
                5 => 2,
                _ => 3,
            };
            if let Some(word) = ranks.get_mut(usize_from(k / 16)) {
                *word |= rank << (k % 16).wrapping_mul(2);
            }
        }
        ranks
    }

    pub(crate) fn at(&self, o: V3) -> i32 {
        self.log2.get(index(o)).copied().unwrap_or(0)
    }

    pub(crate) fn own(&self) -> i32 {
        self.at([0; 3])
    }
}

fn index(o: V3) -> usize {
    let k = o[0]
        .wrapping_add(1)
        .wrapping_add(o[1].wrapping_add(1).wrapping_mul(3))
        .wrapping_add(o[2].wrapping_add(1).wrapping_mul(9));
    usize::try_from(k).unwrap_or(13)
}

fn get<T: Copy + Default>(v: [T; 3], a: usize) -> T {
    v.get(a).copied().unwrap_or_default()
}

fn add(a: V3, b: V3) -> V3 {
    AXES.map(|i| get(a, i).wrapping_add(get(b, i)))
}

/// pass3 `below_offset`.
fn below_offset(i: u32) -> V3 {
    [i & 1, (i >> 1) & 1, (i >> 2) & 1].map(|b| b.cast_signed().wrapping_neg())
}

/// Whether below-offset `d` touches a cell starting at 0 on `zero`.
fn touches(d: V3, zero: [bool; 3]) -> bool {
    d.iter().zip(zero).all(|(&d, z)| d >= 0 || z)
}

/// pass3 `seam_shift`.
fn seam_shift(seam: &Seam, o: V3, o_log2: i32, zero: [bool; 3]) -> i32 {
    (1..8)
        .map(below_offset)
        .filter(|&d| touches(d, zero))
        .map(|d| o_log2.wrapping_sub(seam.at(add(o, d))))
        .fold(0, i32::max)
}

/// pass3 `seam_depth_shift`.
fn seam_depth_shift(seam: &Seam, o: V3, o_log2: i32, id_o: U3) -> i32 {
    let mut s = 0;
    for d in (1..8).map(below_offset) {
        let sd = o_log2.wrapping_sub(seam.at(add(o, d)));
        if sd <= 0 {
            continue;
        }
        let reach = 1u32.wrapping_shl(sd.cast_unsigned());
        if id_o.iter().zip(d).all(|(&c, d)| c < reach || d == 0) {
            s = s.max(sd);
        }
    }
    s
}

/// pass3 `seam_collapse`, with `skip` dropping below-neighbours (to ask
/// what a candidate fix would change; pass3 skips none).
fn seam_collapse(
    seam: &Seam,
    o: V3,
    o_log2: i32,
    zero: [bool; 3],
    s: i32,
    skip: &impl Fn(V3) -> bool,
) -> [bool; 3] {
    let mut collapse = [false; 3];
    for d in (1..8).map(below_offset) {
        if !touches(d, zero) || skip(add(o, d)) {
            continue;
        }
        let l = seam.at(add(o, d));
        for (b, c) in collapse.iter_mut().enumerate() {
            if get(d, b) == 0 {
                continue;
            }
            let mut across = d;
            if let Some(x) = across.get_mut(b) {
                *x = 0;
            }
            let seen = o_log2.wrapping_sub(l.min(seam.at(add(o, across))));
            if seen != s {
                *c = true;
            }
        }
    }
    collapse
}

/// The shared cell pass3's `seam_cell` builds for a cell: its first corner
/// `lo` and its extent `step` per axis, in the meshing chunk's texels. A
/// step of 0 is a flattened axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedCell {
    pub lo: [u32; 3],
    pub step: [u32; 3],
}

impl SharedCell {
    /// Axes the cell is flat on: a vertex from it cannot leave `lo` there.
    #[must_use]
    pub fn flat(&self) -> usize {
        self.step.iter().filter(|&&s| s == 0).count()
    }
}

/// The meshing chunk's edge in voxels, and the cells within `reach` of a
/// face that may be shared (pass3 `seam_reach`).
fn extent(seam: &Seam) -> (u32, u32) {
    let own = seam.own();
    let n = 1u32.wrapping_shl(own.cast_unsigned());
    let coarsest = seam.log2.iter().copied().fold(own, i32::min);
    (
        n,
        1u32.wrapping_shl(own.wrapping_sub(coarsest).cast_unsigned()),
    )
}

/// pass3 `seam_cell`'s lattice (everything but the vertex itself), with
/// `skip` as in [`seam_collapse`].
pub(crate) fn seam_cell(seam: &Seam, id: U3, skip: &impl Fn(V3) -> bool) -> Option<SharedCell> {
    let own_log2 = seam.own();
    let (n, reach) = extent(seam);
    if id.iter().all(|&c| c >= reach && c < n) {
        return None;
    }
    let o = id.map(|c| i32::from(c >= n));
    let o_log2 = seam.at(o);
    let up = own_log2.wrapping_sub(o_log2);
    let up_u = up.max(0).cast_unsigned();
    let down_u = up.wrapping_neg().max(0).cast_unsigned();
    let id_o: U3 = AXES.map(|a| {
        if get(o, a) == 1 {
            0
        } else {
            get(id, a).wrapping_shr(up_u).wrapping_shl(down_u)
        }
    });
    let snap = |s: i32| {
        id_o.map(|c| {
            c.wrapping_shr(s.cast_unsigned())
                .wrapping_shl(s.cast_unsigned())
        })
    };
    let mut s = seam_depth_shift(seam, o, o_log2, id_o);
    let mut lo = snap(s);
    for _ in 0..3 {
        let s2 = seam_shift(seam, o, o_log2, lo.map(|c| c == 0));
        if s2 == s {
            break;
        }
        s = s2;
        lo = snap(s);
    }
    let collapse = seam_collapse(seam, o, o_log2, lo.map(|c| c == 0), s, skip);
    if s == 0 && up == 0 && !collapse.contains(&true) {
        return None;
    }
    let face_shift =
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(|axis| own_log2.wrapping_sub(seam.at(axis)).max(0));
    let r = 1u32.wrapping_shl(s.cast_unsigned());
    let mut cell = SharedCell {
        lo: [0; 3],
        step: [0; 3],
    };
    for a in AXES {
        let (lo_t, step) = if get(o, a) == 1 {
            if get(face_shift, a).wrapping_sub(own_log2) != s.wrapping_sub(o_log2)
                && !get(collapse, a)
            {
                return None;
            }
            (n, 1)
        } else {
            (
                get(lo, a).wrapping_shl(up_u).wrapping_shr(down_u),
                r.wrapping_shl(up_u).wrapping_shr(down_u),
            )
        };
        if let (Some(l), Some(st)) = (cell.lo.get_mut(a), cell.step.get_mut(a)) {
            *l = lo_t;
            *st = if get(collapse, a) { 0 } else { step };
        }
    }
    Some(cell)
}

/// The shared cell pass3 builds for cell `id` of a chunk meshed with
/// `ranks` (its `ChunkMeta::seam_ranks`), or `None` where the cell keeps
/// its own vertex.
#[must_use]
pub fn shared_cell(ranks: [u32; 2], id: [u32; 3]) -> Option<SharedCell> {
    seam_cell(&Seam::from_ranks(ranks), id, &|_| false)
}

/// Whether the vertex at `local` (in its chunk's voxels) is one the seam
/// stitch pins to a chunk edge or corner, and if so the shared cell it
/// came from.
///
/// The test: some cell near the vertex gets, from the ranks the chunk was
/// meshed with, a shared cell flattened on two axes or more, and the vertex
/// lies on that degenerate cell: exactly on its flat coordinates (both on a
/// chunk face), and within its extent on the free one. A vertex from such a
/// cell can only move along that segment, whatever the field says, so its
/// distance off the surface is the seam geometry's, not the data's. A vertex
/// moved off the surface by anything else (a wrong SDF, a stale apron, a
/// solve gone wrong) is not on such a segment and is not classified: the
/// test is an identity of position against the mesher's own lattice, never
/// a magnitude.
///
/// Most vertices are rejected at once: a pinned one has two coordinates on
/// chunk faces (0 or the chunk's size), which a solved vertex almost never
/// has.
#[must_use]
pub fn pinned_cell(ranks: [u32; 2], local: Vec3) -> Option<SharedCell> {
    holding_cell(ranks, local, 2)
}

/// The flattest shared cell, flat on `min_flat` axes or more, that holds
/// the vertex at `local`.
///
/// [`pinned_cell`] is this with 2. `None` if no shared cell near the vertex
/// holds it: then it is its own cell's vertex, not the stitch's. A vertex
/// on fewer than `min_flat` chunk faces is rejected at once.
#[must_use]
pub fn holding_cell(ranks: [u32; 2], local: Vec3, min_flat: usize) -> Option<SharedCell> {
    let seam = Seam::from_ranks(ranks);
    let (n, reach) = extent(&seam);
    let nf = f32_from_u32(n);
    let p = local.to_array();
    let on_face = |x: f32| x.abs() <= ON_FACE || (x - nf).abs() <= ON_FACE;
    if p.iter().filter(|&&x| on_face(x)).count() < min_flat {
        return None;
    }
    // Where a texel sits, per axis (pass3 `texel_pos`): past the edge, one
    // apron step of the face neighbour's coarseness.
    let face_shift =
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(|axis| seam.own().wrapping_sub(seam.at(axis)).max(0));
    let pos = |a: usize, t: u32| -> f32 {
        if t > n {
            nf + f32_from_u32(1u32.wrapping_shl(get(face_shift, a).cast_unsigned()))
        } else {
            f32_from_u32(t)
        }
    };
    // Every cell whose shared cell could hold the vertex: up to a lattice
    // step (`reach`) below it on each axis, and the one it is in.
    let range = |x: f32| {
        // Apron vertices (past `n`) belong to the cells at `n`.
        let at = if x <= 0.0 {
            0
        } else {
            crate::num::u32_from_f32(x.floor()).min(n)
        };
        at.saturating_sub(reach)..=at.saturating_add(1).min(n)
    };
    let [px, py, pz] = p;
    let mut best = None;
    for z in range(pz) {
        for y in range(py) {
            for x in range(px) {
                let Some(cell) = seam_cell(&seam, [x, y, z], &|_| false) else {
                    continue;
                };
                if cell.flat() < min_flat
                    || best.is_some_and(|b: SharedCell| b.flat() >= cell.flat())
                {
                    continue;
                }
                let holds = AXES.into_iter().all(|a| {
                    let lo = get(cell.lo, a);
                    let v = get(p, a);
                    let from = pos(a, lo);
                    let to = pos(a, lo.saturating_add(get(cell.step, a)));
                    v >= from - ON_FACE && v <= to + ON_FACE
                });
                if holds {
                    best = Some(cell);
                }
            }
        }
    }
    best
}
