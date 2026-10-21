//! The terrain as a ray collider: `TerrainRay`, the `LayerCollider` of
//! `TerrainLayer`.
//!
//! Same frame, every LOD; used by the cursor and usable as terrain
//! collision. Sees what is drawn: the `SDF` clipped by the
//! `VisibilityField`, as the mesher and the GPU queries do (with the
//! `visibility-mask` feature the mesh isn't clipped, and neither is this).
//! `TerrainRay::unclipped` sees the terrain under its hidden regions.

use crate::{plugin::TerrainLayer, raycast::collider::LayerCollider};

/// System param: `sample` the world SDF or `cast` a ray at the terrain.
pub type TerrainRay<'w, 's> = LayerCollider<'w, 's, TerrainLayer>;

/// A hit on the terrain. The same type as any collider's hit.
pub type TerrainHit = crate::raycast::collider::RayHit;
