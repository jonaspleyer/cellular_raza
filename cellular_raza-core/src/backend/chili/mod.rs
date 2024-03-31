use serde::{Deserialize, Serialize};

/// Identifier for voxels used internally to get rid of user-defined ones.
#[derive(Clone, Copy, Debug, Deserialize, Hash, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct VoxelPlainIndex(usize);

/// Identifer or subdomains
#[derive(Clone, Copy, Debug, Deserialize, Hash, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct SubDomainPlainIndex(usize);

/// Unique identifier which is given to every cell in the simulation
///
/// The identifier is comprised of the [VoxelPlainIndex] in which the cell was first spawned.
/// This can be due to initial setup or due to other methods such as division in a cell cycle.
/// The second parameter is a counter which is unique for each voxel.
/// This ensures that each cell obtains a unique identifier over the course of the simulation.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct CellIdentifier(pub VoxelPlainIndex, pub u64);

/// Contains structs to store aspects of the simulation and macros to construct them.
mod aux_storage;
#[doc(hidden)]
pub mod compatibility_tests;
mod datastructures;
mod errors;
mod proc_macro;
mod setup;
mod simulation_flow;
mod solvers;
mod update_cycle;
mod update_mechanics;

pub use aux_storage::*;
pub use datastructures::*;
pub use errors::*;
pub use proc_macro::*;
pub use setup::*;
pub use simulation_flow::*;
pub use solvers::*;
pub use update_cycle::*;
pub use update_mechanics::*;
