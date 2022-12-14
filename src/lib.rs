#![feature(drain_filter)]
#![feature(trait_alias)]

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
