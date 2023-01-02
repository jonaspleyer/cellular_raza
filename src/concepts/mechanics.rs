use crate::concepts::errors::CalcError;

use core::fmt::Debug;
use core::ops::{Add,AddAssign,Sub,SubAssign,Mul};
use num::Zero;
use std::marker::Send;


pub trait Position = Sized + Add<Self,Output=Self> + AddAssign + Sub<Output=Self> + SubAssign + Clone + Debug + Send + Mul<f64,Output=Self>;
pub trait Force = Sized + Add<Self,Output=Self> + AddAssign + Sub<Output=Self> + SubAssign + Debug + Zero + Send + Mul<f64,Output=Self>;
pub trait Velocity = Sized + Add<Self,Output=Self> + AddAssign + Sub<Output=Self> + SubAssign + Debug + Zero + Send + Mul<f64,Output=Self>;


pub trait Mechanics<Pos, For, Vel>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
{
    fn pos(&self) -> Pos;
    fn velocity(&self) -> Vel;
    fn set_pos(&mut self, pos: &Pos);
    fn set_velocity(&mut self, velocity: &Vel);
    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError>;
}
