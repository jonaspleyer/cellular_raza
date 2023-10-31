use serde::{Deserialize, Serialize};

/// Identifier for voxels used internally to get rid of user-defined ones.
// #[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub type VoxelPlainIndex = u128;

/// Identifer or subdomains
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct SubDomainPlainIndex(usize);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct CellIdentifier(VoxelPlainIndex, u64);

/// Contains structs to store aspects of the simulation and macros to construct them.
pub mod aux_storage;

pub mod errors;

pub mod datastructures;

/// Specify a way to set-up and start the simulation.
pub mod setup;

/// Numerical integration and solving of the model.
pub mod simulation_flow;
