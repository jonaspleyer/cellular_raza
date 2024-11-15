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

pub mod chili;

/// 🐯 GPU-centered backend using [OpenCL](https://www.khronos.org/opencl/)
pub mod cara {}
