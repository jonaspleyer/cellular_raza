#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! > “What I cannot create, I do not understand.”
//! >
//! >  --- Richard P. Feynman
//!
//! [cellular_raza](crate) is an agent-based modeling tool to simulate individual biological cells with a mechanistically
//! driven mindset.
//! This means, properties of cells are individually driven by strictly local phenomena.

/// The backend controls the simulation flow. Multiple variants could be available in the future.
pub mod backend;

pub mod proc_macro;

/// Interface and methods to store and load simulation aspects.
pub mod storage;

// Some re-exports
pub use cellular_raza_concepts as concepts;
