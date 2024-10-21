use serde::{Deserialize, Serialize};

/// Identifier for voxels used internally to get rid of user-defined ones.
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[derive(Clone, Copy, Debug, Deserialize, Hash, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct VoxelPlainIndex(pub usize);

/// This is mainly used by the simulation_flow::cocmmunicator for testing purposes
#[allow(unused)]
#[doc(hidden)]
impl VoxelPlainIndex {
    pub fn new(value: usize) -> Self {
        Self(value)
    }
}

/// Identifier or subdomains
#[derive(Clone, Copy, Debug, Deserialize, Hash, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct SubDomainPlainIndex(usize);

/// Unique identifier which is given to every cell in the simulation
///
/// The identifier is comprised of the [VoxelPlainIndex] in which the cell was first spawned.
/// This can be due to initial setup or due to other methods such as division in a cell cycle.
/// The second parameter is a counter which is unique for each voxel.
/// This ensures that each cell obtains a unique identifier over the course of the simulation.
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[derive(Clone, Copy, Debug, Deserialize, Hash, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub struct CellIdentifier(pub VoxelPlainIndex, pub u64);

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl CellIdentifier {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.eq(other)
    }

    fn __hash__(&self) -> u64 {
        use core::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __getstate__(&self) -> pyo3::PyResult<(usize, u64)> {
        Ok((self.0.0, self.1))
    }

    fn __setstate(&mut self, state: (usize, u64)) -> pyo3::PyResult<()> {
        self.0 = VoxelPlainIndex(state.0);
        self.1 = state.1;
        Ok(())
    }
}

/// Contains structs to store aspects of the simulation and macros to construct them.
mod aux_storage;
#[doc(hidden)]
pub mod compatibility_tests;
mod datastructures;
mod errors;
mod proc_macro;
mod result;
mod setup;
mod simulation_flow;
mod solvers;
mod update_cycle;
mod update_mechanics;
mod update_reactions;

pub use aux_storage::*;
pub use datastructures::*;
pub use errors::*;
pub use proc_macro::*;
pub use result::*;
pub use setup::*;
pub use simulation_flow::*;
pub use solvers::*;
pub use update_cycle::*;
pub use update_mechanics::*;
pub use update_reactions::*;
