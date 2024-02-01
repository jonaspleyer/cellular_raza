#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
//! > “What I cannot create, I do not understand.”
//! >
//! >  --- Richard P. Feynman
//!
//! [cellular_raza](crate) is an agent-based modeling tool to simulate individual biological
//! cells with a mechanistically
//! driven mindset.
//! This means, properties of cells are individually driven by strictly local phenomena.

pub use cellular_raza_building_blocks as building_blocks;

pub use cellular_raza_concepts_derive as concepts_derive;

pub use cellular_raza_concepts as concepts;

pub use cellular_raza_core as core;

/// Re-exports the default simulation types and traits.
pub mod prelude;
