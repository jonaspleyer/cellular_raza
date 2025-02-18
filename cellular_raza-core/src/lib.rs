#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![cfg_attr(docsrs, feature(doc_cfg))]
//! This crate collects objects and methods needed to run a numerical simulation of
//! given objects that satisfy the given [concepts](cellular_raza_concepts).
//!
//! ## Backends
//! This crate supports multiple types of backends.
//! Currently, the [backend::cpu_os_threads] backend is the general-purpose solver which
//! can deal with (almost) all simulation [concepts](cellular_raza_concepts).
//! In the future, the [backend::chili] backend will be replacing it, delivering
//! better performance, modularity while also updating [concepts](cellular_raza_concepts).
//!
//! ## Storage
//! We distinguish between a full (de-)serialization of the simulation
//! and exporting data from individual simulation steps.
//!
//! ### Full (de)serialization
//! The first approach allows for a full reload of the total simulation which in principle
//! enables methods such as starting/stopping the simulation and continuing from the last
//! known point.
//! This can also be used to avoid numerical solving problems by restarting from the
//! last known good save point.
//! However, the latter functionalities do not exist currently but are planned for future releases.
//!
//! ### Exporting
//! This approach allows to take cells or domain objects and extract information to then
//! save these in a given format.
//! The methods needed to do this have not yet been developed and are part of future releases.

pub mod backend;

pub mod storage;

pub mod time;

#[doc(hidden)]
pub use rayon;

#[cfg(feature = "tracing")]
#[doc(hidden)]
pub use tracing;
