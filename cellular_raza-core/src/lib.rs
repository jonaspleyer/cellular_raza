#![doc(html_logo_url = "../../../doc/cellular_raza_square.png")]

//! > “What I cannot create, I do not understand.”
//! >
//! >  --- Richard P. Feynman
//!
//! [cellular_raza](crate) is an agent-based modeling tool to simulate individual biological cells with a mechanistically
//! driven mindset.
//! This means, properties of cells are individually driven by strictly local phenomena.
//!
// TODO move this to the corresponding backend
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

/// The backend controls the simulation flow. Multiple variants could be available in the future.
pub mod backend;

/// Database interface
pub mod storage;
