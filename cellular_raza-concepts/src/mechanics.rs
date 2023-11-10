use crate::errors::{CalcError, RngError};

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

// TODO merge this with the interaction trait!
/// Describes the position of a cell-agent and allows to calculate increments and set/get information of the agent.
pub trait Mechanics<Pos, Vel, For> {
    /// Gets the cells current position.
    fn pos(&self) -> Pos;
    /// Gets the cells current velocity.
    fn velocity(&self) -> Vel;
    /// Sets the cells current position.
    fn set_pos(&mut self, pos: &Pos);
    /// Sets the cells current velocity.
    fn set_velocity(&mut self, velocity: &Vel);

    /// Define a new random variable in case that the mechanics type contains a random aspect to its motion.
    /// By default this function does nothing.
    #[allow(unused)]
    fn set_random_variable(
        &mut self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: f64,
    ) -> Result<Option<f64>, RngError> {
        Ok(None)
    }

    /// Calculate the time-derivative of force and velocity given all the forces that act on the cell.
    /// Simple dampening effects should be included in this trait if not explicitly given by the [Voxel](super::domain::Voxel).
    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError>;
}
