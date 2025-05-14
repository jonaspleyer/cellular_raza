use crate::errors::{CalcError, RngError};
use crate::interaction::*;
use crate::mechanics::{Mechanics, Position, Velocity};

use serde::{Deserialize, Serialize};

// TODO move this module to cpu_os_threads backend except for traits

/// This is a unique identifier which is deterministic even in multi-threading situations.
/// Its components are
/// 1. PlainIndex of Voxel where it was created
/// 2. Count the number of cells that have already been created in this voxel since simulation begin.
// TODO consider making this an associated type of the Id trait
pub type CellularIdentifier = (u64, u64);

/// Specifies how to retrieve a unique identifier of an object.
pub trait Id {
    /// The identifier type is usually chosen to be completely unique and repeatable across
    /// different simulations.
    type Identifier;

    /// Retrieves the Identifier from the object.
    fn get_id(&self) -> Self::Identifier;
    /// Returns a reference to the id of the object.
    fn ref_id(&self) -> &Self::Identifier;
}

/// A container struct containing meta-information of a given Cell
/// Some variables such as id are not required and not desired to be
/// initialized by the user. This [CellAgentBox] acts as a container around the cell
/// to hold these variables.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CellAgentBox<Cel> {
    id: CellularIdentifier,
    parent_id: Option<CellularIdentifier>,
    /// The user-defined cell which is stored inside this container.
    pub cell: Cel,
}

impl<Cel> Id for CellAgentBox<Cel> {
    type Identifier = CellularIdentifier;

    fn get_id(&self) -> CellularIdentifier {
        self.id
    }

    fn ref_id(&self) -> &CellularIdentifier {
        &self.id
    }
}

impl<Cel> CellAgentBox<Cel> {
    /// Simple method to retrieve the [CellularIdentifier] of the parent cell if existing.
    pub fn get_parent_id(&self) -> Option<CellularIdentifier> {
        self.parent_id
    }
}

// Auto-implement traits for CellAgentBox which where also implemented for Agent
impl<Pos, Vel, For, Inf, A> Interaction<Pos, Vel, For, Inf> for CellAgentBox<A>
where
    A: Interaction<Pos, Vel, For, Inf> + Serialize + for<'a> Deserialize<'a>,
{
    fn get_interaction_information(&self) -> Inf {
        self.cell.get_interaction_information()
    }

    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        own_vel: &Vel,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_information: &Inf,
    ) -> Result<(For, For), CalcError> {
        self.cell
            .calculate_force_between(own_pos, own_vel, ext_pos, ext_vel, ext_information)
    }

    fn is_neighbor(&self, own_pos: &Pos, ext_pos: &Pos, ext_inf: &Inf) -> Result<bool, CalcError> {
        self.cell.is_neighbor(own_pos, ext_pos, ext_inf)
    }

    fn react_to_neighbors(&mut self, neighbors: usize) -> Result<(), CalcError> {
        self.cell.react_to_neighbors(neighbors)
    }
}

impl<A, Pos> Position<Pos> for CellAgentBox<A>
where
    A: Position<Pos>,
{
    fn pos(&self) -> Pos {
        self.cell.pos()
    }

    fn set_pos(&mut self, pos: &Pos) {
        self.cell.set_pos(pos)
    }
}

impl<A, Vel> Velocity<Vel> for CellAgentBox<A>
where
    A: Velocity<Vel>,
{
    fn velocity(&self) -> Vel {
        self.cell.velocity()
    }

    fn set_velocity(&mut self, velocity: &Vel) {
        self.cell.set_velocity(velocity)
    }
}

impl<Pos, Vel, For, Float, A> Mechanics<Pos, Vel, For, Float> for CellAgentBox<A>
where
    A: Mechanics<Pos, Vel, For, Float>,
{
    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: Float,
    ) -> Result<(Pos, Vel), RngError> {
        self.cell.get_random_contribution(rng, dt)
    }

    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError> {
        self.cell.calculate_increment(force)
    }
}

impl<Cel> CellAgentBox<Cel> {
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

#[doc(inline)]
pub use cellular_raza_concepts_derive::CellAgent;
