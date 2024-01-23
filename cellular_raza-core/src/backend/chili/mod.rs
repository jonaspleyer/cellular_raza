use serde::{Deserialize, Serialize};

/// Identifier for voxels used internally to get rid of user-defined ones.
#[derive(Clone, Copy, Debug, Deserialize, Hash, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct VoxelPlainIndex(usize);

/// Identifer or subdomains
#[derive(Clone, Copy, Debug, Deserialize, Hash, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct SubDomainPlainIndex(usize);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct CellIdentifier(VoxelPlainIndex, u64);

/// Contains structs to store aspects of the simulation and macros to construct them.
mod aux_storage;
mod datastructures;
mod errors;
mod setup;
mod simulation_flow;
mod solvers;

pub use aux_storage::*;
pub use datastructures::*;
pub use errors::*;
pub use setup::*;
pub use simulation_flow::*;
pub use solvers::*;
