use crate::concepts::cycle::*;
use crate::concepts::mechanics::{Position,Force,Velocity,Mechanics};
use crate::concepts::interaction::*;
use crate::concepts::errors::CalcError;

use std::marker::{Send,Sync, PhantomData};

use uuid::Uuid;

use serde::{Serialize,Deserialize};


// pub trait CellAgent<Pos: Position, For: Force, Vel: Velocity> = Cycle<Self> + Interaction<Pos, For> + Mechanics<Pos, For, Vel> + Sized + Id + Send + Sync + Clone;
pub trait CellAgent<Pos: Position, For: Force, Vel: Velocity>: Cycle<Self> + Interaction<Pos, For> + Mechanics<Pos, For, Vel> + Sized + Send + Sync + Clone + Serialize + for<'a> serde::Deserialize<'a>{}
impl<Pos, For, Vel, A> CellAgent<Pos, For, Vel> for A
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: Cycle<Self> + Interaction<Pos, For> + Mechanics<Pos, For, Vel> + Sized + Send + Sync + Clone + Serialize + for<'a> serde::Deserialize<'a>,
{}


/// Obtains the unique identifier of a cell
pub trait Id {
    fn get_uuid(&self) -> Uuid;
}


/// A container struct containing meta-information of a given Cell
/// Some variables such as Uuid are not required and not desired to be
/// initialized by the user. This CellAgentBox acts as a container around the cell
/// to hold these variables.
#[derive(Serialize,Deserialize,Debug,Clone)]
pub struct CellAgentBox<Pos, For, Vel, A>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: CellAgent<Pos, For, Vel>,
{
    id: Uuid,
    #[serde(bound = "")]
    pub cell: A,

    phantom_pos: PhantomData<Pos>,
    phantom_for: PhantomData<For>,
    phantom_vel: PhantomData<Vel>,
}


impl<Pos, For, Vel, A> Id for CellAgentBox<Pos, For, Vel, A>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: CellAgent<Pos, For, Vel>,
{
    fn get_uuid(&self) -> Uuid {
        self.id
    }
}


// Auto-implement traits for CellAgentBox which where also implemented for CellAgent
impl<Pos, For, Vel, A> Cycle<CellAgentBox<Pos, For, Vel, A>> for CellAgentBox<Pos, For, Vel, A>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: CellAgent<Pos, For, Vel>
{
    fn update_cycle(dt: &f64, c: &mut CellAgentBox<Pos, For, Vel, A>) {
        A::update_cycle(dt, &mut c.cell);
    }
}


impl<Pos, For, Vel, A> Interaction<Pos, For> for CellAgentBox<Pos, For, Vel, A>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: CellAgent<Pos, For, Vel>
{
    fn force(&self, own_pos: &Pos, ext_pos: &Pos) -> Option<Result<For, CalcError>> {
        self.cell.force(own_pos, ext_pos)
    }
}


impl<Pos, For, Vel, A> Mechanics<Pos, For, Vel> for CellAgentBox<Pos, For, Vel, A>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: CellAgent<Pos, For, Vel>
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


impl<Pos, For, Vel, C> From<(u128, C)> for CellAgentBox<Pos, For, Vel, C>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    C: CellAgent<Pos, For, Vel>,
{
    fn from(comb: (u128, C)) -> CellAgentBox<Pos, For, Vel, C> {
        CellAgentBox::<Pos, For, Vel, C> {
            id: Uuid::from_u128(comb.0),
            cell: comb.1,

            phantom_for: PhantomData,
            phantom_pos: PhantomData,
            phantom_vel: PhantomData,
        }
    }
}