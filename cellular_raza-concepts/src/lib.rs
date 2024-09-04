#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
//! This crate encapsulates concepts which govern an agent-based model specified by
//! [cellular_raza](https://docs.rs/cellular_raza).
//! To learn more about the math and philosophy behind these concepts please refer to
//! [cellular-raza.com](https://cellular-raza.com).

mod cell;
mod cycle;
mod domain;
mod reactions;
/// Contains traits and types which specify cellular reactions specific to the [cpu_os_threads]
/// backend.
pub mod reactions_old;

/// Traits and types which will eventually replace the old [Domain] definition.
// TODO #[deprecated]
pub mod domain_old;
mod errors;
mod interaction;
mod mechanics;
mod plotting;

pub use cell::*;
pub use cycle::*;
pub use domain::*;
pub use errors::*;
pub use interaction::*;
pub use mechanics::*;
pub use plotting::*;
pub use reactions::*;
