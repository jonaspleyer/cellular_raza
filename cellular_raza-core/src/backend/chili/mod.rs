//! 🌶️ A modular, reusable, general-purpose backend
//!
//! # Internals
//! The [chili](self) backend uses procedural macros to generate code which
//! results in a fully working simulation.
//! The methods, functions and objects used in this way are formualted with generics.
//! This enables us to write general-purpose solvers for a wide range of problems.
//! The most important macro is the [run_simulation] macro which can be solely used to run
//! simulations.
//! This macro itself can be broken down into smaller pieces.
//! - [prepare_types] Defines types used in simulation
//!     - [build_aux_storage] AuxStorage struct used to store intermediate
//!       values for update steps of aspects
//!     - [build_communicator] Type which handles communication between
//!       threads
//! - [test_compatibility] Test compatibility of all types involved
//! - [run_main] Defines main loop and performs the simulation
//!
//! These macros take a subset of keyword arguments of the [run_simulation] macro.
//! The arguments are documented under the [run_simulation] macro.
//!
//! # Main Loop
//! The [run_main] macro constructs the main loop of the simulation.
//! It inserts functions depending on the given simulation aspects.
//! They can be grouped into 6 steps
//! Below, we show a list of all these functions and their corresponding aspects.
//!
//! #### Step 1 - Send Information
//!
//! In this step, each sub
//!
//! <style>
//!     div table {
//!         width: 100%;
//!     }
//!     table th:first-of-type {
//!         width: 20%;
//!     }
//!     table th:nth-of-type(2) {
//!         width: 35%;
//!     }
//!     table th:nth-of-type(3) {
//!         width: 45%;
//!     }
//! </style>
//!
//! | Aspects | Function | Purpose |
//! | --- | --- | --- |
#![doc = "\
    | `Mechanics && Interaction`\
    | [update_mechanics_interaction_step_1](SubDomainBox::update_mechanics_interaction_step_1)\
    | Send [PosInformation](PosInformation) between threads to get back \
      [ForceInformation](ForceInformation) |"]
#![doc = "\
    | `DomainForce`\
    | [calculate_custom_domain_force](SubDomainBox::calculate_custom_domain_force)\
    | Uses the [SubDomainForce](cellular_raza_concepts::SubDomainForce) trait to add \
      custom external force. |"]
#![doc = "\
    | `ReactionsContact`\
    | [update_contact_reactions_step_1]\
      (SubDomainBox::update_contact_reactions_step_1) \
    | Sends [ReactionsContactInformation](ReactionsContactInformation) between threads. |"]
#![doc = "\
    | `ReactionsExtra` \
    | [update_reactions_extra_step_1](SubDomainBox::update_reactions_extra_step_1) \
    | Sends [ReactionsExtraBorderInfo](ReactionsExtraBorderInfo) between threads. |"]
#![doc = "\
    | | [sync](SubDomainBox::sync) | Wait for threads to have finished until proceeding. |"]
//!
//! #### Step 2 - Calculate and Return
//! | Aspects | Function | Purpose |
//! | --- | --- | --- |
#![doc = "\
    | `Mechanics && Interaction` \
    | [update_mechanics_interaction_step_2](SubDomainBox::update_mechanics_interaction_step_2) \
    | Calculate forces and return [ForceInformation](ForceInformation) to the original \
      sender. |"]
#![doc = "\
    | `ReactionsContact` \
    | [update_contact_reactions_step_2](SubDomainBox::update_contact_reactions_step_2) \
    | Calculates the combined increment and returns \
      [ReactionsContactReturn](ReactionsContactReturn) |"]
#![doc = "\
    | `ReactionsExtra` \
    | [update_reactions_extra_step_2](SubDomainBox::update_reactions_extra_step_2) \
    | Returns [ReactionsExtraBorderReturn](ReactionsExtraBorderReturn) |"]
//!
//! #### Step 3 - Receive and Apply
//! | Aspects | Function | Purpose |
//! | --- | --- | --- |
#![doc = "\
    | `Mechanics && Interaction` \
    | [update_mechanics_interaction_step_3](SubDomainBox::update_mechanics_interaction_step_3) \
    | Receives the [ForceInformation](ForceInformation) and adds within the \
      `aux_storage`. |"]
#![doc = "\
    | `ReactionsContact` \
    | [update_contact_reactions_step_3](SubDomainBox::update_contact_reactions_step_3) \
    | Receives the [ReactionsContactReturn](ReactionsContactReturn) and adds within the `aux_storage`. |"]
#![doc = "\
    | `ReactionsExtra` \
    | [update_reactions_extra_step_3](SubDomainBox::update_reactions_extra_step_3) \
    | Receives the [ReactionsExtraBorderReturn](ReactionsExtraBorderReturn). |"]
//!
//! #### Pure Local Functions - Perform Update
//! | Aspects | Function | Purpose |
//! | --- | --- | --- |
#![doc = "\
    | `Mechanics` \
    | [local_mechanics_update](local_mechanics_update) \
    | Performs numerical integration of the position and velocity. |"]
#![doc = "\
    | `Interaction` \
    | [local_interaction_react_to_neighbors](local_interaction_react_to_neighbors) \
    | Performs changes due to neighbor counting. |"]
#![doc = "\
    | `Cycle` \
    | [local_cycle_update](local_cycle_update) \
    | Advances the cycle of the cell. This may introduce a\
      [CycleEvent](cellular_raza_concepts::CycleEvent) |"]
#![doc = "\
    | `Reactions` \
    | [local_reactions_intracellular](local_reactions_intracellular) \
    | Calculates increment from purely intracellular reactions. |"]
#![doc = "\
    | `ReactionsContact` \
    | [local_update_contact_reactions](local_update_contact_reactions) \
    | Performs the update of the contact reactions. |"]
#![doc = "\
    | `ReactionsExtra` \
    | [local_subdomain_update_reactions_extra](local_subdomain_update_reactions_extra) \
    | Performs the update of the extracellular reactions. |"]
#![doc = "\
    | `Reactions` &#124;&#124; `ReactionsContact` &#124;&#124; `ReactionsExtra` \
    | [local_reactions_use_increment](local_reactions_use_increment) \
    | Calculates increment from purely intracellular reactions. |"]
//!
//! #### Step 4 - Treat Cell Positions
//! | Aspects | Function | Purpose |
//! | --- | --- | --- |
#![doc = "\
    | `Mechanics` \
    | [apply_boundary](SubDomainBox::apply_boundary) \
    | Apply a boundary condition. |"]
#![doc = "\
    | `Cycle` \
    | [update_cell_cycle_4](SubDomainBox::update_cell_cycle_4) \
    | Performs cell-division and other cycle events. |"]
#![doc = "\
    | `Mechanics` \
    | [sort_cells_in_voxels_step_1](SubDomainBox::sort_cells_in_voxels_step_1) \
    | Checks if cells need to be sent to different subdomain and sends them. |"]
//!
//! #### Step 5 - Include new Cells
//! | Aspects | Function | Purpose |
//! | --- | --- | --- |
#![doc = "\
    | `Mechanics` \
    | [sort_cells_in_voxels_step_2](SubDomainBox::sort_cells_in_voxels_step_2) \
    | Include newly received cells into correct subdomains. |"]
//!
//! # Return Type
//! After the simulation is done, we return a [StorageAccess] struct to interoperate with stored
//! results.

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
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
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
    /// Constructs a new CellIdentifier
    #[new]
    pub fn new(voxel_plain_index: VoxelPlainIndex, counter: u64) -> Self {
        Self(voxel_plain_index, counter)
    }

    /// Returns an identical clone of the identifier
    pub fn __deepcopy__(&self, _memo: pyo3::Bound<pyo3::types::PyDict>) -> Self {
        self.clone()
    }

    /// Returns an identical clone of the identifier
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Returns an identical clone of the identifier
    pub fn __copy__(&self) -> Self {
        self.clone()
    }

    /// Formats the CellIdentifier
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    /// Performs the `==` operation.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.eq(other)
    }

    /// Calculates a hash value of type `u64`
    pub fn __hash__(&self) -> u64 {
        use core::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Used to pickle the object
    #[deprecated(note = "This method exists as part of experimentation and may change behaviour or\
        be removed in the future")]
    pub fn __reduce__<'a>(&'a self, py: pyo3::Python<'a>) -> pyo3::Bound<pyo3::types::PyTuple> {
        use pyo3::prelude::*;
        use pyo3::PyTypeInfo;
        pyo3::types::PyTuple::new_bound(
            py,
            [
                Self::type_object_bound(py).to_object(py),
                pyo3::types::PyTuple::new_bound(py, [self.0.0 as u64, self.1]).to_object(py),
            ],
        )
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
