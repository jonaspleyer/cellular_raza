use crate::concepts::cycle::*;
use crate::concepts::mechanics::*;
use crate::concepts::interaction::*;

use uuid::Uuid;


pub trait Cell<Pos, Force, Velocity> = Cycle<Self> + Interaction<Pos, Force> + Mechanics<Pos, Velocity> + Sized + Id;

pub trait Id {
    fn get_uuid(&self) -> Uuid;
}
