#![feature(drain_filter)]
#![feature(trait_alias)]

#![doc(html_logo_url = "../../../doc/cellular_raza_square.png")]

//! > “What I cannot create, I do not understand.”
//! >
//! >  --- Richard P. Feynman 
//!
//! [cellular_raza](crate) is an agent-based modeling tool to simulate individual biological cells with a mechanistically
//! driven mindset.
//! This means, properties of cells are individually driven by strictly local phenomena.
//!
//! # Simulating Cellular Biology
//! This crate aims to provide interfaces to simulate the following mechanisms found in biological cells.
//! - [x] Cellular mechanics via force interaction
//! - [ ] Intra- and Extracellular Reactions
//!     - [x] ODEs
//!     - [ ] Stochastic ODEs and PDEs
//! - [ ] Fluid Dynamics
//!     - [x] Diffusion
//!     - [ ] ...
//! - [x] Cell-Proliferation
//! - [x] Death
//! - [ ] Differentiation
//! - [ ] Contact functions
//! - [ ] Custom (individual-only) rules
//!
//! # Features and Techniques
// TODO extend list here: look at our own paper
//! - [ ] Adaptive and Multi-Scale solving
//! - [ ] Provide predefined Set of Cellular Agents and Simulation Domains
//! - [x] Separation of Concepts and Implementations
//! - [x] Parallelization
//!     - [ ] Scheduling for highly-parallelized NUMA instances
//! - [x] Deterministic (once fixing initial Seed, if random Processes take place)
//! ## Concepts
//! This crate provides well-defined traits to implement own cellular types with own rules.
//!
//! ## Simulation Snapshots
//! - [x] Take a simulation snapshot at any given time
//! - [ ] Start from any saved Snapshot
//!     - [ ] Modify parameter values before beginning simulation
//!
//! ## Deterministic Results
//! - This crates stochastic implementaions should always produce results which are deterministically reproducible when fixing the seed.
//!   That is if the provided functionalities are used correctly (see [concepts]).
//! - Changing the number of threads should not alter the sequence at which random numbers are being generated and thus provided to the agents and domains/voxels.
//!
//! These statements must be taken with a grain of salt.
//! While we strive for binary reproducability, one problem is non-associativity of floating point operations, meaning `a+(b+c)!=(a+b)+c` on the binary level of almost all of modern hardware.
//! This situation is not a data-race! For more information look at the implementation inside the [pipelines] module and read "[What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)".
//! This situation can only be reliably circumvented when running on a single execution thread.
//!
//! # Layers of Abstraction of this crate
//! ## Concepts
//! ## Cellular + Domain Implementations
//! ## Pipelines
// TODO where does plotting and storage fit in here?

// TODO move this to the corresponding pipeline
//! ## Implementation Details
//! - Intrinsic parallelization over physical simulation domain
//! - Strongly prefer static dispatch (frequent use of generics, avoid `Box<dyn Trait>`)
//! - Consider memory-locality, efficient cache-usage and thread-to-thread latency
//! - reduce heap allocations
//!
//! # What this crate does not aim to offer
//! - A graphical user interface
//!     - Models should be created with native Rust Code (or in the future by interacting with Python or Julia)
//!     - Data Analysis can be done with Python/Julia
//! - The option to freely change everything however you like during runtime
//! 
//! # Inherent Assumptions and Mindset
//! - Additions are commutative
//!     - This is inherently not true in almost all modern computers using floating point arithmetics
//!       It can lead to results which are not identical on a binary level, but should agree within their numerical uncertainty respective of the solver used.
//! - Multiplication with floats like f64 are commutative
//! - There should only be local rules
//!     - CellAgents are interacting with themselves, their neighbors and the extracellular properties around them.
//!       There is no exact definition of "local" but the first idea was to restrict interaction ranges.
//! - Any multithreading or implementaional details of how information is transported on an actual computer should be hidden from the end user.
//!     - Defining new cell-properties and domain/voxels should not involve thinking about parallelization etc.
//! - All traits should work in a continuous fashion
//!     - Use getters/setters to alter internal variables
//!     - Objects calculate increments which can be used in complex solvers
//!     - In the future we want to also calculate errors and have adaptive techniques for solving
//! - Goal: Any panic that can occur should be user-generated
//!     - we want to catch all errors
//!     - Goal: evaluate error type and reduce step-size for solving (or solver altogether) and try again from last breakpoint

//! # More Notes to include later
//! - Since this is a scientific library, casts between different numeric types are necessary
//!     - reduce them as much as possible
//!     - try to only need to cast from integer to floating point types and when doing the other way around be really careful
//! - Code Style/Guideline

#[cfg(all(feature = "db_sled", feature = "db_mongodb"))]
compile_error!("feature \"db_sled\" and feature \"db_mongodb\" cannot be enabled at the same time");

#[cfg(all(feature = "no_db", any(feature = "db_sled", feature = "db_mongodb")))]
compile_error!("feature \"no_db\" and database feature cannot be enabled at the same time");


/// Abstract concepts to describe cell properties, the domain and possible errors
///
/// These concepts should be implemented by the user and then be used by a simulation
/// [pipelines](crate::pipelines) which actually integrates the defined cellular properties.
pub mod concepts;
/// Database interface
pub mod storage;
/// Implementations of various concepts that can be readily used. Contains cellular properties and domains.
pub mod implementations;
/// The pipline controls the simulation flow. Multiple variants could be available.
pub mod pipelines;
/// Methods for plotting simulation results
pub mod plotting;
