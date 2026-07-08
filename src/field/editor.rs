use std::marker::PhantomData;

// fields/editor.rs
use super::Field;
use bevy::math::primitives::Primitive3d;
use bevy::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum EditMode<V> {
    /// Instantly replaces the data with a static value
    Absolute,
    /// Adds/Subtracts a step size smoothly over time (e.g., for sculpting SDFs)
    Accumulate { delta: V },
    /// Smoothly moves current value towards target value with a falloff rate
    Blend { rate: f32 },
}

#[derive(Message)]
pub struct EditFieldMessage<F: Field<V>, S: Primitive3d, V: Copy + Default + Send + Sync + 'static>
{
    pub center: Vec3,
    pub shape: S,
    pub val: V,
    pub mode: EditMode<V>,
    phantom: PhantomData<F>,
}

impl<F: Field<V>, S: Primitive3d, V: Copy + Default + Send + Sync + 'static>
    EditFieldMessage<F, S, V>
{
    pub fn new(center: Vec3, shape: S, val: V, mode: EditMode<V>) -> Self {
        Self {
            center,
            shape,
            val,
            mode,
            phantom: PhantomData,
        }
    }
}
