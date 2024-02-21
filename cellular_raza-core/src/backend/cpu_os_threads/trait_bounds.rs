use cellular_raza_concepts::*;

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

/// Encapsulates all concepts that can be specified for a [Agent]
///
/// We hope to be deprecating this trait in the future and only rely on individual traits instead.
/// While this trait could be manually implemented, it is often not necessary (see [cellular_raza-building-blocks](https://docs.rs/cellular_raza-building-blocks))
pub trait Agent<Pos: Position, Vel: Velocity, For: Force, Inf, Float = f64>:
    Cycle<Self, Float>
    + Interaction<Pos, Vel, For, Inf>
    + Mechanics<Pos, Vel, For, Float>
    + Sized
    + Send
    + Sync
    + Clone
    + serde::Serialize
    + for<'a> serde::Deserialize<'a>
{
}
impl<Pos, Vel, For, Inf, Float, A> Agent<Pos, Vel, For, Inf, Float> for A
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: Cycle<Self, Float>
        + Interaction<Pos, Vel, For, Inf>
        + Mechanics<Pos, Vel, For, Float>
        + Sized
        + Send
        + Sync
        + Clone
        + serde::Serialize
        + for<'a> serde::Deserialize<'a>,
{
}