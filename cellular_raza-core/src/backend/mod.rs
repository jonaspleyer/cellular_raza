//! Numerically solve a given simulation setup.
//!
//! In the future, we plan on expanding the list of available backends.
//! We hope to provide specialized solvers for highly efficient GPU usage via the OpenCL standard.
//!
//! ## Supported Simulation Aspects
//! Not every backend does support all simulation aspects.
//! We aim to provide one general-purpose backend able to solve any given simulation that adheres
//! to the [cellular_raza_concepts] with the 🌶️ [chili] backend.
//!
//! | Aspect | 🐧 [cpu_os_threads] | 🌶️ [chili] |
//! | --- |:---:|:---:|
//! | [Cycle](cellular_raza_concepts::Cycle) | ✅¹ | ✅ |
//! | [Mechanics](cellular_raza_concepts::Mechanics) | ✅¹ | ✅ |
//! | [Interaction](cellular_raza_concepts::Interaction) | ✅ | ✅ |
//! | [Reactions](cellular_raza_concepts::CellularReactions) | ✅ | ❌ |
//! | [Volume](cellular_raza_concepts::Volume) | ✅¹ | ❌ |
//! | [Domain](cellular_raza_concepts::Domain) | ❌ | ✅ |
//! | [DomainOld](cellular_raza_concepts::domain_old::Domain) | ✅ | ❌ |
//! | [Controller](cellular_raza_concepts::domain_old::Controller) | ✅ | ❌ |
//! | [Plotting](cellular_raza_concepts::PlotSelf) | ✅ | ❌ |
//!
//! ¹Only supports `Float=f64`.

/// 🐧 Use multiple os-threads and cpu-only resources
///
/// Parallelization is achieved by splitting the simulation domain into as many chunks as
/// threads are desired. Communication between threads is handled by
/// [crossbeam_channel](https://docs.rs/crossbeam-channel/latest/crossbeam_channel/)
/// and synchronization by [hurdles::Barrier](https://docs.rs/hurdles/latest/hurdles/).
///
/// The user can manage the simulation flow by means of individual functions or by creating a
/// [SimulationSupervisor](cpu_os_threads::SimulationSupervisor).
// TODO deprecate this!
// #[deprecated]
// #[allow(deprecated)]
pub mod cpu_os_threads;

/// 🌶️ A modular, reusable, general purpose backend
///
/// # Overview
/// The [chili](crate::backend::chili) backend uses procedural macros to generate code which
/// results in a fully working simulation.
/// The methods, functions and objects used in this way are formualted with generics.
/// This enables us to write general-purpose solvers for a wide range of problems.
///
/// Since [cellular_raza](https://cellular-raza.com) is based on simulation
/// [aspects](https://cellular-raza.com/aspects/concepts)
///
/// # Simulation Flow
/// The [run_simulation](crate::backend::chili::run_simulation) macro (which calls the
/// [run_main](crate::backend::chili::run_main) macro) generates a set of functions
/// depending on the choice of simulation aspects.
/// Below, we show a list of all these functions and their corresponding aspects.
///
/// | Aspects | Function | Purpose |
/// | --- | --- | --- |
#[doc = "\
    | `Mechanics && Interaction`\
    | [update_mechanics_interaction_step_1](chili::SubDomainBox::update_mechanics_interaction_step_1)\
    | Send [PosInformation](chili::PosInformation) between threads to get back \
      [ForceInformation](chili::ForceInformation) |"]
#[doc = "\
    | `DomainForce`\
    | [calculate_custom_domain_force](chili::SubDomainBox::calculate_custom_domain_force)\
    | Uses the [SubDomainForce](cellular_raza_concepts::SubDomainForce) trait to add \
      custom external force. |"]
#[doc = "\
    | `ReactionsContact`\
    | [update_contact_reactions_step_1]\
      (chili::SubDomainBox::update_contact_reactions_step_1) \
    | Sends [ReactionsContactInformation](chili::ReactionsContactInformation) between threads. |"]
#[doc = "\
    | | [sync](chili::SubDomainBox::sync) | Wait for threads to have finished until proceeding. |"]
#[doc = "\
    | `ReactionsExtra` \
    | [update_reactions_extra_step_1](chili::SubDomainBox::update_reactions_extra_step_1) \
    | Sends [ReactionsExtraBorderInfo](chili::ReactionsExtraBorderInfo) between threads. |"]
#[doc = "\
    | `Mechanics && Interaction` \
    | [update_mechanics_interaction_step_2](chili::SubDomainBox::update_mechanics_interaction_step_2) \
    | Calculate forces and return [ForceInformation](chili::ForceInformation) to the original \
      sender. |"]
#[doc = "\
    | `ReactionsContact` \
    | [update_contact_reactions_step_2](chili::SubDomainBox::update_contact_reactions_step_2) \
    | Calculates the combined increment and returns \
      [ReactionsContactReturn](chili::ReactionsContactReturn) |"]
#[doc = "\
    | `ReactionsExtra` \
    | [update_reactions_extra_step_2](chili::SubDomainBox::update_reactions_extra_step_2) \
    | Returns [ReactionsExtraBorderReturn](chili::ReactionsExtraBorderReturn) |"]
#[doc = "\
    | \
    | [sync](chili::SubDomainBox::sync) \
    | Wait for threads to have finished until proceeding. |"]
#[doc = "\
    | `Mechanics && Interaction` \
    | [update_mechanics_interaction_step_3](chili::SubDomainBox::update_mechanics_interaction_step_3) \
    | Receives the [ForceInformation](chili::ForceInformation) and adds within the \
      `aux_storage`. |"]
#[doc = "\
    | `ReactionsContact` \
    | [update_contact_reactions_step_3](chili::SubDomainBox::update_contact_reactions_step_3) \
    | Receives the [ReactionsContactReturn](chili::ReactionsContactReturn) and adds within the `aux_storage`. |"]
#[doc = "\
    | `ReactionsExtra` \
    | [update_reactions_extra_step_3](chili::SubDomainBox::update_reactions_extra_step_3) \
    | Receives the [ReactionsExtraBorderReturn](chili::ReactionsExtraBorderReturn). |"]
#[doc = "\
    | \
    | [sync](chili::SubDomainBox::sync) \
    | Wait for threads to have finished until proceeding. |"]
#[doc = "\
    | `ReactionsExtra` \
    | [local_subdomain_update_reactions_extra](chili::SubDomainBox::local_subdomain_update_reactions_extra) \
    | Perform the update of the extracellular reactions. |"]
#[doc = "\
    | `Mechanics` \
    | [local_mechanics_update_step_3](chili::local_mechanics_update_step_3) \
    | Performs numerical integration of the position and velocity. |"]
#[doc = "\
    | `Interaction` \
    | [local_interaction_react_to_neighbors](chili::local_interaction_react_to_neighbors) \
    | Performs changes due to neighbor counting. |"]
#[doc = "\
    | `Cycle` \
    | [local_cycle_update](chili::local_cycle_update) \
    | Advance the cycle of the cell. This may introduce [CycleEvent](cellular_raza_concepts::CycleEvent) |"]
#[doc = "\
    | `Mechanics` \
    | [local_mechanics_set_random_variable](chili::local_mechanics_set_random_variable) \
    | Sets the random variable used for stochastic mechanical processes. |"]
#[doc = "\
    | `Reactions` \
    | [local_reactions_intracellular](chili::local_reactions_intracellular) \
    | Calculates increment from purely intracellular reactions. |"]
#[doc = "\
    | `ReactionsContact` \
    | [local_reactions_intracellular](chili::local_reactions_intracellular) \
    | Calculates increment from purely intracellular reactions. |"]
///
pub mod chili;

/// 🐯 GPU-centered backend using [OpenCL](https://www.khronos.org/opencl/)
pub mod cara {}
