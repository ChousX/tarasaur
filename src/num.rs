//! Numeric conversions that `From`/`TryFrom` can't express.
//!
//! The crate denies `as` casts. Std has no non-`as` way to turn a float into
//! an integer, an integer into a float beyond 16 bits, or a `u32` into a
//! `usize`, so each of those conversions lives here once, with the reason it
//! is sound for how this crate uses it.
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

// `u32 -> usize` is lossless on every target this crate builds for.
const _: () = assert!(usize::BITS >= u32::BITS);

/// Lossless: `usize` is at least 32 bits (asserted above).
#[inline]
#[must_use]
pub const fn usize_from(v: u32) -> usize {
    v as usize
}

/// Exact below 2^24, which covers every grid coordinate, size and count the
/// crate converts (grids are at most a few hundred voxels across).
#[inline]
#[must_use]
pub const fn f32_from_u32(v: u32) -> f32 {
    v as f32
}

/// Exact for `|v| < 2^24`; see [`f32_from_u32`].
#[inline]
#[must_use]
pub const fn f32_from_i32(v: i32) -> f32 {
    v as f32
}

/// Exact below 2^24; see [`f32_from_u32`].
#[inline]
#[must_use]
pub const fn f32_from_usize(v: usize) -> f32 {
    v as f32
}

/// Saturating float -> int: out-of-range values clamp to the type's bounds
/// and NaN becomes 0. Round or floor first; this truncates toward zero.
#[inline]
#[must_use]
pub const fn i32_from_f32(v: f32) -> i32 {
    v as i32
}

/// Saturating like [`i32_from_f32`]; negative values become 0.
#[inline]
#[must_use]
pub const fn u32_from_f32(v: f32) -> u32 {
    v as u32
}

/// Saturating like [`i32_from_f32`].
#[inline]
#[must_use]
pub const fn i16_from_f32(v: f32) -> i16 {
    v as i16
}

/// Saturating like [`i32_from_f32`]; negative values become 0.
#[inline]
#[must_use]
pub const fn u8_from_f32(v: f32) -> u8 {
    v as u8
}

// `usize -> u64` is lossless on every target this crate builds for.
const _: () = assert!(usize::BITS <= u64::BITS);

/// Lossless: `usize` is at most 64 bits (asserted above). For byte sizes and
/// offsets handed to wgpu, which takes `u64`.
#[inline]
#[must_use]
pub const fn u64_from_usize(v: usize) -> u64 {
    v as u64
}
