#![feature(drain_filter)]
#![feature(trait_alias)]

//! This crate is an agent-based tool to simulate individual biological cells with a mechanistically driven mindset.
//! This means, properties of cells are thought to be individually driven by strictly local phenomena.
//! # Simulating Cellular Biology
//! This crate aims to provide interfaces to simulate the following mechanisms found in biological cells.
//! - [x] Cellular mechanics via force interaction
//! - [ ] Intra- and Extracellular Reactions by (Stochastic) ODEs and PDEs
//! - [ ] Fluid Dynamics (Diffusion, etc.)
//! - [ ] Cell-Proliferation
//! - [ ] Death
//! - [ ] Differentiation
//! - [ ] Contact functions
//! - [ ] Custom (individual-only) rules
//!
//! # Features and Techniques
// TODO extend list here: look at our own paper
//! - [ ] Adaptive and Multi-Scale solving
//! - [ ] Provide predefined Set of Cellular Agents and Simulation Domains
//! - [x] Separation of Concepts and Implementations
//! - [ ] Provide predefined Set of Cellular Agents and Simulation Domains
//! - [x] Parallelization
//!     - [ ] Scheduling for highly-parallelized NUMA instances
//! - [x] Deterministic (once fixing initial Seed, if random Processes take place)
//! ## Concepts
//! This crate provides well-defined traits to implement own cellular types with own rules.
//!
//! ## Simulation Snapshots
//! - Take a simulation snapshot at any given time
//! - Start from any saved Snapshot, even with possibly different parameter-values
//! ## Deterministic Results
//! This crate should always produce results which are deterministically reproducible.
//! That is if the provided functionalities are used correctly (see [concepts]).
//! Changing the number of threads should not alter the behaviour.
//! ## Implementation Details
//! - Intrinsic parallelization over physical simulation domain
//! - Strongly prefer static dispatch (frequent use of generics)
//! - Consider memory-locality and thread-to-thread latency
// TODO
//! - Efficient Cache-Usage
//!
//! # What this crate does not aim to offer
//! - A user graphical Interface
//!     - Models should be created with native Rust Code or by interacting with Python or Julia
//!     - Data Analysis can be done with Python/Julia as well
//! - The option to freely change everything however you like during runtime



#[cfg(all(feature = "no_db", any(feature = "db_sled", feature = "db_mongodb")))]
compile_error!("feature \"no_db\" and database feature cannot be enabled at the same time");


/// Semi-Abstract concepts for cells and simulation engine
pub mod concepts;
/// Database interface
pub mod storage;
/// Example implementations of cell models
pub mod impls_cell_models;
/// Implementations of individual cell properties
pub mod impls_cell_properties;
/// Implementations of Domains
pub mod impls_domain;
/// Objects and Methods for initializing and controlling the simulation flow
pub mod init;
/// Methods for plotting simulation results
pub mod plotting;
/// Re-exports of all objects and methods
pub mod prelude;
