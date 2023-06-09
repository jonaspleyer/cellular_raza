use crate::cycle::*;
use crate::errors::CalcError;
use crate::interaction::*;
use crate::mechanics::{Force, Mechanics, Position, Velocity};

use std::marker::{Send, Sync};

use serde::{Deserialize, Serialize};

// TODO this concept should probably be not necessary
// Prefer a combination of many traits
/// Encapsulates all concepts that can be specified for a [CellAgent]
///
/// We hope to be deprecating this trait in the future and only rely on individual traits instead.
/// While this trait could be manually implemented, it is often not necessary (see [cellular_raza-building-blocks](https://docs.rs/cellular_raza-building-blocks))
pub trait CellAgent<Pos: Position, Vel: Velocity, For: Force, Inf>:
    Cycle<Self>
    + Interaction<Pos, Vel, For, Inf>
    + Mechanics<Pos, Vel, For>
    + Sized
    + Send
    + Sync
    + Clone
    + Serialize
    + for<'a> serde::Deserialize<'a>
{
}
impl<Pos, Vel, For, Inf, A> CellAgent<Pos, Vel, For, Inf> for A
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: Cycle<Self>
        + Interaction<Pos, Vel, For, Inf>
        + Mechanics<Pos, Vel, For>
        + Sized
        + Send
        + Sync
        + Clone
        + Serialize
        + for<'a> serde::Deserialize<'a>,
{
}

/// This is a unique identifer which is deterministic even in multi-threading situations.
/// Its components are
/// 1. PlainIndex of Voxel where it was created
/// 2. Count the number of cells that have already been created in this voxel since simulation begin.
// TODO consider making this an associated type of the Id trait
pub type CellularIdentifier = (u64, u64);

/// Obtains the unique identifier of a cell
// TODO consider making this trait generic for cell and voxel with associated type varying
pub trait Id {
    /// Simple method to access the identifier of the cell
    fn get_id(&self) -> CellularIdentifier;
}

/// A container struct containing meta-information of a given Cell
/// Some variables such as id are not required and not desired to be
/// initialized by the user. This CellAgentBox acts as a container around the cell
/// to hold these variables.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CellAgentBox<Cel>
where
    Cel: Serialize + for<'a> Deserialize<'a>,
{
    id: CellularIdentifier,
    parent_id: Option<CellularIdentifier>,
    #[serde(bound = "")]
    /// The user-defined cell which is stored inside this container.
    pub cell: Cel,
}

impl<Cel> Id for CellAgentBox<Cel>
where
    Cel: Serialize + for<'a> Deserialize<'a>,
{
    fn get_id(&self) -> CellularIdentifier {
        self.id
    }
}

impl<Cel> CellAgentBox<Cel>
where
    Cel: Serialize + for<'a> Deserialize<'a>,
{
    /// Simple method to retrieve the [CellularIdentifier] of the parent cell if existing.
    pub fn get_parent_id(&self) -> Option<CellularIdentifier> {
        self.parent_id
    }
}

// Auto-implement traits for CellAgentBox which where also implemented for CellAgent
impl<Pos, Vel, For, Inf, A> Interaction<Pos, Vel, For, Inf> for CellAgentBox<A>
where
    A: Interaction<Pos, Vel, For, Inf> + Serialize + for<'a> Deserialize<'a>,
{
    fn get_interaction_information(&self) -> Option<Inf> {
        self.cell.get_interaction_information()
    }

    fn calculate_force_on(
        &self,
        own_pos: &Pos,
        own_vel: &Vel,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_information: &Option<Inf>,
    ) -> Option<Result<For, CalcError>> {
        self.cell
            .calculate_force_on(own_pos, own_vel, ext_pos, ext_vel, ext_information)
    }
}

impl<Pos, Vel, For, A> Mechanics<Pos, Vel, For> for CellAgentBox<A>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    A: Mechanics<Pos, Vel, For> + Serialize + for<'a> Deserialize<'a>,
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

impl<Cel> CellAgentBox<Cel>
where
    Cel: Serialize + for<'a> Deserialize<'a>,
{
    /// Create a new [CellAgentBox] at a specific voxel with a voxel-unique number
    /// of cells that has already been created at this position.
    // TODO make this generic with respect to voxel_index
    pub fn new(
        voxel_index: u64,
        n_cell: u64,
        cell: Cel,
        parent_id: Option<CellularIdentifier>,
    ) -> CellAgentBox<Cel> {
        CellAgentBox::<Cel> {
            id: (voxel_index, n_cell),
            parent_id,
            cell,
        }
    }
}
