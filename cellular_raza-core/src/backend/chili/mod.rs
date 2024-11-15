//! 🌶️ A modular, reusable, general-purpose backend
//!
//! # Usage
//!
//! In the following example, we provide an extremely short basic usage of the [run_simulation]
//! macro.
//! It performs the numerical integration of a well-defined problem.
//! We assume that the `MyAgent` struct and `MyDomain` have already been defined and implement the
//! [Mechanics](cellular_raza_concepts::Mechanics) concept.
//! More details about how to use pre-existing building blocks or derive their functionality is
//! provided by the
//! [cellular_raza_building_blocks](https://cellular-raza.com/docs/cellular_raza_building_blocks)
//! crate.
//! We will then solve our system for the
//! [`Mechanics`](https://cellular-raza.com/internals/concepts/cell/mechanics) aspect.
//!
//! ```
//! # use cellular_raza_core::backend::chili::*;
//! # use cellular_raza_core::{storage::*, time::*};
//! # use cellular_raza_concepts::*;
//! # use rand_chacha::ChaCha8Rng;
//! # use serde::{Deserialize, Serialize};
//! # use std::collections::{BTreeMap, BTreeSet};
//! # // Define agents and domain
//! # #[derive(Clone, Debug, Deserialize, Serialize)]
//! # struct MyAgent {
//! #     pos: f64,
//! #     vel: f64
//! # }
//! # impl Position<f64> for MyAgent {
//! #     fn pos(&self) -> f64 {
//! #         self.pos
//! #     }
//! #     fn set_pos(&mut self, pos: &f64) {
//! #         self.pos = *pos
//! #     }
//! # }
//! # impl Velocity<f64> for MyAgent {
//! #     fn velocity(&self) -> f64 {
//! #         self.vel
//! #     }
//! #     fn set_velocity(&mut self, vel: &f64) {
//! #         self.vel = *vel
//! #     }
//! # }
//! # impl Mechanics<f64, f64, f64> for MyAgent {
//! #     fn get_random_contribution(
//! #         &self,
//! #         _rng: &mut ChaCha8Rng,
//! #         _dt: f64,
//! #     ) -> Result<(f64, f64), RngError> {
//! #         Ok((0.0, 0.0))
//! #     }
//! #     fn calculate_increment(&self, force: f64) -> Result<(f64, f64), CalcError> {
//! #         Ok((self.vel, force))
//! #     }
//! # }
//! # #[derive(Clone, Debug, Deserialize, Serialize)]
//! # struct MyDomain {};
//! # impl<Ci> Domain<MyAgent, MyDomain, Ci> for MyDomain
//! # where
//! #     Ci: IntoIterator<Item = MyAgent>
//! # {
//! #     type VoxelIndex = usize;
//! #     type SubDomainIndex = usize;
//! #     fn decompose(
//! #         self,
//! #         _: core::num::NonZeroUsize,
//! #         cells: Ci,
//! #     ) -> Result<
//! #         DecomposedDomain<Self::SubDomainIndex, MyDomain, MyAgent>,
//! #         cellular_raza_concepts::DecomposeError,
//! #     > {
//! #         Ok(DecomposedDomain {
//! #             n_subdomains: 1.try_into().unwrap(),
//! #             index_subdomain_cells: vec![(1, MyDomain {}, cells.into_iter().collect())],
//! #             neighbor_map: BTreeMap::from([(1, BTreeSet::new())]),
//! #             rng_seed: 1,
//! #         })
//! #     }
//! # }
//! # impl SubDomain for MyDomain {
//! #     type VoxelIndex = usize;
//! #     fn get_neighbor_voxel_indices(&self, _: &Self::VoxelIndex) -> Vec<Self::VoxelIndex> {
//! #         Vec::new()
//! #     }
//! #     fn get_all_indices(&self) -> Vec<Self::VoxelIndex> {
//! #         vec![1]
//! #     }
//! # }
//! # impl SortCells<MyAgent> for MyDomain {
//! #     type VoxelIndex = usize;
//! #     fn get_voxel_index_of(
//! #         &self,
//! #         _: &MyAgent,
//! #     ) -> Result<Self::VoxelIndex, cellular_raza_concepts::BoundaryError> {
//! #         Ok(1)
//! #     }
//! # }
//! # impl SubDomainMechanics<f64, f64> for MyDomain {
//! #     fn apply_boundary(
//! #         &self,
//! #         _: &mut f64,
//! #         _: &mut f64,
//! #     ) -> Result<(), cellular_raza_concepts::BoundaryError> {
//! #         Ok(())
//! #     }
//! # }
//! # let t0 = 0.0;
//! # let dt = 0.1;
//! # let tmax = 20.0;
//! # let save_interval = 2.0;
//! # let initial_vel = 0.1;
//! # let n_threads = 1.try_into().unwrap();
//! // Initialize agents, domain, solving time and how to store results
//! let agents = (0..10).map(|n| MyAgent {
//!         /* Define the agent's properties */
//! #       pos: n as f64,
//! #       vel: initial_vel,
//!     });
//! let domain = MyDomain {/* Define the domain*/};
//! let time = FixedStepsize::from_partial_save_interval(t0, dt, tmax, save_interval)?;
//! let storage = StorageBuilder::new().priority([StorageOption::Memory]);
//!
//! // Group them together
//! let settings = Settings {
//!     n_threads,
//!     time,
//!     storage,
//!     show_progressbar: false,
//! };
//!
//! // This will perform the numerical simulation
//! let storage_access = run_simulation!(
//!     agents: agents,
//!     domain: domain,
//!     settings: settings,
//!     aspects: [Mechanics],
//!     core_path: cellular_raza_core,
//! )?;
//!
//! // Use calculated results
//! let history = storage_access.cells.load_all_elements()?;
//! for (iteration, cells) in history {
//!     // ...
//! #     assert!(iteration > 0);
//! #     assert_eq!(cells.len(), 10);
//! #     for (_, (cbox, _)) in cells {
//! #         let calculated = cbox.get_id().1 as f64 + iteration as f64 * 0.1 * initial_vel;
//! #         let cr = cbox.cell.0;
//! #         assert!((calculated - cr).abs() < 1e-5);
//! #     }
//! }
//! # Ok::<(), SimulationError>(())
//! ```
//!
//! This example cannot contain all the functionality which the [chili](self) backend provides.
//! We encourage the user to look at the
//! [cellular-raza.com/guides](https://cellular-raza.com/guides) and
//! [cellular-raza.com/showcase](https://cellular-raza.com/showcase) sections to get started.
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
