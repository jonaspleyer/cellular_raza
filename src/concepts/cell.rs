use crate::concepts::cycle::*;
use crate::concepts::mechanics::{Position,Force,Velocity,Mechanics};
use crate::concepts::interaction::*;

use std::marker::Send;

use uuid::Uuid;


pub trait Cell<Pos: Position, For: Force, Vel: Velocity> = Cycle<Self> + Interaction<Pos, For> + Mechanics<Pos, For, Vel> + Sized + Id + Send;

pub trait Id {
    fn get_uuid(&self) -> Uuid;
}
