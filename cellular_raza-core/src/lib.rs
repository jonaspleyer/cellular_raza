// TODO make this be deny instead of warning
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

/// The backend controls the simulation flow. Multiple variants could be available in the future.
pub mod backend;

pub mod proc_macro;

/// Interface and methods to store and load simulation aspects.
pub mod storage;

// Some re-exports
pub use cellular_raza_concepts as concepts;
