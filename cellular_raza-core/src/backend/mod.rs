//! Numerically solve a given simulation setup.
//!
//! In the future, we plan on expanding the list of available backends.
//! We hope to provide specialized solvers for highly efficient GPU usage via the OpenCL standard.
//!
//! ## Supported Simulation Aspects
//! Not every backend does support all simulation aspects.
//! We aim to provide one general-purpose backend ablet to solve any given simulation that adheres
//! to the [cellular_raza_concepts] with the 🌶️ [chili](crate::backend::chili) backend.
//!
//! | Aspect | 🐧 [cpu_os_threads](crate::backend::cpu_os_threads) | 🌶️ [chili](crate::backend::chili) |
//! | --- |:---:|:---:|
//! | [Cycle](cellular_raza_concepts::Cycle) | ✅¹ | ✅ |
//! | [Mechanics](cellular_raza_concepts::Mechanics) | ✅¹ | ✅ |
//! | [Interaction](cellular_raza_concepts::Interaction) | ✅ | ✅ |
//! | [Reactions](cellular_raza_concepts::CellularReactions) | ✅ | ❌ |
//! | [Volume](cellular_raza_concepts::Volume) | ✅¹ | ❌ |
//! | [Domain](cellular_raza_concepts::domain::Domain) | ✅ | ❌ |
//! | [DomainNew](cellular_raza_concepts::domain_new::Domain) | ❌ | ✅ |
//! | [Controller](cellular_raza_concepts::Controller) | ✅ | ❌ |
//! | [Plotting](cellular_raza_concepts::plotting) | ✅ | ❌ |
//!
//! ¹Only supports `Float=f64`.

/// 🐧 Use multiple os-threads and cpu-only resources
///
/// Parallelization is achieved by splitting the simualtion domain into as many chunks as
/// threads are desired. Communication between threads is handled by
/// [crossbeam_channel](https://docs.rs/crossbeam-channel/latest/crossbeam_channel/)
/// and synchronization by [hurdles::Barrier](https://docs.rs/hurdles/latest/hurdles/).
///
/// The user can manage the simulation flow by means of individual funtions or by creating a
/// [SimulationSupervisor](cpu_os_threads::supervisor::SimulationSupervisor).
pub mod cpu_os_threads;

/// 🌶️ A modular, reusable, general purpose backend
///
/// This backend delivers a modular approach to building a fully working simulation.
/// Individual structs such as the [AuxStorage](aux_storage) to track properties of the cell
/// and help in updating its values can be construced with similarly named
/// [macros](crate::proc_macro).
///
/// In the future it will take over the role of the [cpu_os_threads](crate::backend::cpu_os_threads)
/// backend as the default backend.
pub mod chili;
