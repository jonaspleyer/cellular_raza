use crate::concepts::errors::CalcError;

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use num::Zero;
use std::marker::{Send, Sync};

// TODO use trait alias when available
/* pub trait Position = Sized
+ Add<Self, Output = Self>
+ AddAssign
+ Sub<Output = Self>
+ SubAssign
+ Clone
+ Debug
+ Send
+ Sync
+ Mul<f64, Output = Self>;*/
/// Represents the current position of a cell-agent.
pub trait Position:
    Sized
    + Add<Self, Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Clone
    + Debug
    + Send
    + Sync
    + Mul<f64, Output = Self>
{
}
impl<T> Position for T where
    T: Sized
        + Add<Self, Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Clone
        + Debug
        + Send
        + Sync
        + Mul<f64, Output = Self>
{
}

// TODO use trait alias when available
/* pub trait Force = Sized
+ Add<Self, Output = Self>
+ AddAssign
+ Sub<Output = Self>
+ SubAssign
+ Clone
+ Debug
+ Zero
+ Send
+ Sync
+ Mul<f64, Output = Self>;*/
/// Represents a force which can act between two cells.
pub trait Force:
    Sized
    + Add<Self, Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Clone
    + Debug
    + Zero
    + Send
    + Sync
    + Mul<f64, Output = Self>
{
}
impl<T> Force for T where
    T: Sized
        + Add<Self, Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Clone
        + Debug
        + Zero
        + Send
        + Sync
        + Mul<f64, Output = Self>
{
}

// TODO use trait alias when available
/* pub trait Velocity = Sized
+ Add<Self, Output = Self>
+ AddAssign
+ Sub<Output = Self>
+ SubAssign
+ Clone
+ Debug
+ Zero
+ Send
+ Sync
+ Mul<f64, Output = Self>;*/
/// Represents the velocity of a cell.
pub trait Velocity:
    Sized
    + Add<Self, Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Clone
    + Debug
    + Zero
    + Send
    + Sync
    + Mul<f64, Output = Self>
{
}
impl<T> Velocity for T where
    T: Sized
        + Add<Self, Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Clone
        + Debug
        + Zero
        + Send
        + Sync
        + Mul<f64, Output = Self>
{
}

/// This trait should be merged with the interaction trait
// TODO merge this with the interaction trait!
pub trait Mechanics<Pos, For, Vel> {
    fn pos(&self) -> Pos;
    fn velocity(&self) -> Vel;
    fn set_pos(&mut self, pos: &Pos);
    fn set_velocity(&mut self, velocity: &Vel);
    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError>;
}
