#![feature(drain_filter)]
#![feature(trait_alias)]

//! This crate is an agent-based tool to simulate individual biological cells with a mechanistically driven mindset.
//! This means, properties of cells are thought to be individually driven by local phenomena.
//!
//! This crate aims to offer the following features in the future
//! - Intra- and Extracellular Reactions by (Stochastic) ODEs and PDEs
// TODO extend list here: look at our own paper
//! - Adaptive and Multi-Scale solving
//! - Intra- and Extracellular Reactions via Gillespie and PDE+ODE solvers
//! - Fluid Dynamics (Diffusion, etc.)
//! - Provide predefined Set of Cellular Agents and Simulation Domains
//! - Separation of Concepts and Implementations such that individual Implementations can be realized
//! - Efficiently parallelized
//! - Deterministic (once fixing initial Seed, if random Processes take place)
//! - Python and Julia bindings
//!
//! What this crate does not aim to offer
//! - A user graphical Interface
//!     - Models should be created with native Rust Code or by interacting with Python or Julia
//!     - Data Analysis can be done with Python/Julia as well
//! - The option to freely change everything however you like during runtime
//!
//! # Simulating Cell-Biology
//!
//! # Concepts
//!
//! # Features and Techniques
//! ## Simulation Snapshots
//! - Take a simulation snapshot at any given time
//! - Start from any saved Snapshot, even with possibly different parameter-values
//! ## Deterministic Results
//! - Results should be (if makeing corret use of the provided traits) deterministically reproducible, even when using parallelization
//! ## Efficient Hardware Usage
//! - Intrinsic parallelization over physical Simulation Domain
// TODO
//! - Efficient Cache-Usage

/// Semi-Abstract concepts for cells and simulation engine
pub mod concepts;
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
