use crate::concepts::cycle::*;
use crate::concepts::mechanics::{Position,Force,Velocity,Mechanics};
use crate::concepts::interaction::*;
use crate::concepts::errors::{CalcError,SimulationError};

use std::marker::{Send,Sync};

use uuid::Uuid;

use serde::{Serialize,Deserialize};


// pub trait CellAgent<Pos: Position, For: Force, Vel: Velocity> = Cycle<Self> + Interaction<Pos, For> + Mechanics<Pos, For, Vel> + Sized + Id + Send + Sync + Clone;
pub trait CellAgent<Pos: Position, For: Force, Inf, Vel: Velocity>: Cycle<Self> + Interaction<Pos, For, Inf> + Mechanics<Pos, For, Vel> + Sized + Send + Sync + Clone + Serialize + for<'a> serde::Deserialize<'a>{}
impl<Pos, For, Inf, Vel, A> CellAgent<Pos, For, Inf, Vel> for A
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: Cycle<Self> + Interaction<Pos, For, Inf> + Mechanics<Pos, For, Vel> + Sized + Send + Sync + Clone + Serialize + for<'a> serde::Deserialize<'a>,
{}


/// Obtains the unique identifier of a cell
pub trait Id {
    fn get_uuid(&self) -> Uuid;
}


/// A container struct containing meta-information of a given Cell
/// Some variables such as Uuid are not required and not desired to be
/// initialized by the user. This CellAgentBox acts as a container around the cell
/// to hold these variables.
#[derive(Serialize,Deserialize,Debug,Clone,PartialEq)]
pub struct CellAgentBox<A>
where
    A: Serialize + for<'a> Deserialize<'a>
{
    id: Uuid,
    #[serde(bound = "")]
    pub cell: A,
}


impl<A> Id for CellAgentBox<A>
where
    A: Serialize + for<'a> Deserialize<'a>
{
    fn get_uuid(&self) -> Uuid {
        self.id
    }
}


// Auto-implement traits for CellAgentBox which where also implemented for CellAgent
impl<A> Cycle<CellAgentBox<A>> for CellAgentBox<A>
where
    A: Cycle<A> + Serialize + for<'a> Deserialize<'a>
{
    fn update_cycle(dt: &f64, c: &mut CellAgentBox<A>) {
        A::update_cycle(dt, &mut c.cell);
    }
}


impl<Pos, For, Inf, A> Interaction<Pos, For, Inf> for CellAgentBox<A>
where
    Pos: Position,
    For: Force,
    A: Interaction<Pos, For, Inf> + Serialize + for<'a> Deserialize<'a>
{
    fn get_interaction_information(&self) -> Option<Inf> {
        self.cell.get_interaction_information()
    }

    fn calculate_force_on(&self, own_pos: &Pos, ext_pos: &Pos, ext_information: &Option<Inf>) -> Option<Result<For, CalcError>> {
        self.cell.calculate_force_on(own_pos, ext_pos, ext_information)
    }
}


impl<Pos, For, Vel, A> Mechanics<Pos, For, Vel> for CellAgentBox<A>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: Mechanics<Pos, For, Vel> + Serialize + for<'a>Deserialize<'a>,
{
    fn pos(&self) -> Pos {
        self.cell.pos()
    }

    fn velocity(&self) -> Vel {
        self.cell.velocity()
    }

    fn set_pos(&mut self, pos: &Pos) {
        self.cell.set_pos(pos);
    }

    fn set_velocity(&mut self, velocity: &Vel) {
        self.cell.set_velocity(velocity);
    }

    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError> {
        self.cell.calculate_increment(force)
    }
}


impl<C> From<(u32, u16, u16, u64, C)> for CellAgentBox<C>
where
    C: Serialize + for<'a> Deserialize<'a>
{
    fn from(comb: (u32, u16, u16, u64, C)) -> CellAgentBox<C> {
        CellAgentBox::<C> {
            id: Uuid::from_fields(comb.0, comb.1, comb.2, &comb.3.to_be_bytes()),
            cell: comb.4,
        }
    }
}
