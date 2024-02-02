//! # cellular_raza - Building Blocks
//!
//! Building blocks allow users to quickly construct complex cellular agents.
//! The simplest approach is to use fully-defined cell models which are
//! contained in [cell_models] with fitting [domains].
//! However, users can also build their own complex models by combining existing ones.
//!
//! To create your own agent with physical mechanics and interactions, we need to include
//! the building blocks for them as fields of our agent struct.
//! ```rust
//! # use cellular_raza_building_blocks::prelude::*;
//! struct MyAgent {
//!     mechanics: NewtonDamped2D,
//!     interaction: BoundLennardJones,
//! }
//! ```
//! Furthermore, we can derive desired concepts by using the [CellAgent](cellular_raza_concepts_derive::CellAgent)
//! derive macro.
//! ```rust
//! # use cellular_raza_concepts::*;
//! # use cellular_raza_building_blocks::prelude::*;
//! # use nalgebra::Vector2;
//! # use cellular_raza_concepts_derive::CellAgent;
//! #[derive(CellAgent)]
//! struct MyAgent {
//!     #[Mechanics(Vector2<f64>, Vector2<f64>, Vector2<f64>)]
//!     mechanics: NewtonDamped2D,
//!     #[Interaction(Vector2<f64>, Vector2<f64>, Vector2<f64>)]
//!     interaction: BoundLennardJones,
//! }
//! # let mut agent = MyAgent {
//! #     mechanics: NewtonDamped2D {
//! #         pos: Vector2::<f64>::from([0.0; 2]),
//! #         vel: Vector2::<f64>::from([0.0; 2]),
//! #         damping_constant: 0.1,
//! #         mass: 2.0,
//! #     },
//! #     interaction: BoundLennardJones {
//! #         epsilon: 1.0,
//! #         sigma: 2.2,
//! #         bound: 1.2,
//! #         cutoff: 6.0,
//! #     },
//! # };
//! #
//! # agent.set_pos(&[1.0, 2.0].into());
//! # assert_eq!(agent.pos(), Vector2::from([1.0, 2.0]));
//! ```
//! For technical reasons, we are required to also once more specify the types for position,
//! velocity and force when specifying which struct field to derive from.
//! The optional `Inf` generic parameter of the [Interaction](cellular_raza_concepts::Interaction) trait was left out and thus defaults to `()`.
//! It can and needs to also be specified when choosing interactions with non-trivial
//! interaction information.
//!
//! # Optional Features
//! Features guard implementations which introduce additional dependencies.
//! To simplify usability, we enable commonly used features by default.
//!
//! - [pyo3](https://docs.rs/pyo3/latest/pyo3/) Rust bindings to the Python interpreter

#![deny(missing_docs)]
#![cfg_attr(doc_cfg, feature(doc_cfg))]

/// Construct cells from individual components
pub mod cell_building_blocks;

/// Collection of complete cell models
pub mod cell_models;

/// Physical domains which contain cells
///
/// These domains can be used with blocks from [cell_building_blocks](crate::cell_building_blocks) and [cell_models](crate::cell_models).
pub mod domains;

/// Handy re-exports of every building block.
pub mod prelude;
