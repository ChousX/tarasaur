//! Where chunk data comes from and goes: procedural generators, the
//! off-thread load pipeline that installs a chunk's fields at its
//! `TargetLOD`, and save files.

pub mod generator;
pub mod loading;
pub mod persistence;
