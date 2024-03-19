#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
//! This crate encapsulates concepts which govern an agent-based model specified by [cellular_raza](https://docs.rs/cellular_raza).
//! To learn how to design your own concepts please refer to the [cellular_raza book](https://jonaspleyer.github.io/cellular_raza/DevelopersDesigningNewConcepts.html).
//!
//! # [Mechanics]
//! Defines how to adjust position,force and velocity of the individual cells.
//!
//! This concept is extremely central to every cell-agent since it defines the spatial
//! representation of the cell which can be seen by other cells. This means that as long as all
//! cells which should be included in the simulation can be represented by this shared trait,
//! [cellular_raza](crate) should be able to simulate them.
//!
//! While a point of the simulation in general is not additive in the general theoretical formulation (due to boundaries for
//! example), it is necessary to have an additive type to effectively use adaptive solvers. We thus
//! need to check and apply boundary conditions independantly.
//!
//! # [Interaction]
//! Interactions can arise due to many different mechanisms. The following table shows a short
//! summary of the possible traits which can be used. Some of them should not make sense to use in
//! combination. An engine may chose to implement only certain traits while omitting others.
//!
//! | Interaction Trait | Description |
//! | --- | --- |
//! | [Interaction] | Cells are interacting by forces which can have a range larger than the cell itself. Foe example, users can choose to implement their own repulsive and attractive forces. |
//! | [CellularReactions] | Intracellular reactions may be coupled to an extracellular environment. We can model these reactions via [ODEs](https://en.wikipedia.org/wiki/Ordinary_differential_equation). |
//!
//! # [Cycle]
//! The Cycle trait is responsible for the implementation of cycles and updates them incrementally.
//! The main `update` function is responsible for returning an optional cycle event.
//! These events can have different effects. For example a cell-division event triggers the
//! `divide` method. A mapping of events and functions is depicted in the table below.
//!
//! | Event | Effect |
//! | ----- -------- | ------ |
//! | [`Division`](CycleEvent::Division) | The [`divide`](Cycle::divide) function returns (creates) a new cell and modifies the existing cell in-place. This means, that the user is responsible to make sure, that every field of the cell struct is modified correctly in order to simulate cell-division. |
//! | [`PhasedDeath`](CycleEvent::PhasedDeath) | The cell enters a dying process and is still continuously updated via [`update_conditional_phased_death`](Cycle::update_conditional_phased_death). Once the corresponding function returns `true` the process is considered complete and the cell is removed. |
//! | [`Remove`](CycleEvent::Remove) | This event removes the cell from the simulation without any further actions. |
//!
//! # Errors
//! For Backends it may be useful to define a singular error type (eg. `SimulationError`) which should be derivable from errors
//! arising during the simulation process.
//! It is required for custom error types `MyCustomError` of the engine to implement the `From<MyCustomError> for SimulationError`.
//! Errors should be seperated by their ability to be recovered, ignored or handled otherwise.
//! Since this crate aims to provide an adaptive solving aproach, it is desired to have a fallback
//! mechanism which can be called for errors which may arise due to precision problems.
//!
//! The following table shows a summary of the errors currently supported.
//! Backends need to be aware of them and implement custom handling schemes to overcome or work around them.
//! Aborting the simulation is an option but must be documented well without introducing undefined behaviour.
//!
//! | ErrorType | Possible Error Reasons |
//! | --- | --- |
//! | BoundaryError | Solver Accuracy, Domain Implementaion bug, Internal engine error |
//! | CalcError | Solver Accuracy, Bug by user implementation of corresponding function, Internal engine error |
// TODO implement the handling of these errors!
//! Captures traits and types related to interactions between cells.
//!
//! # Plotting
//! Visualize components of the simulation directly via the [plotters](https://docs.rs/plotters) library.

mod cell;
mod cycle;
mod domain;
/// Traits and types which will eventually replace the old [Domain] definition.
pub mod domain_new;
mod errors;
mod interaction;
mod mechanics;
mod plotting;
mod test_derive_cell_agent;

pub use cell::*;
pub use cellular_raza_concepts_derive::*;
pub use cycle::*;
pub use domain::*;
pub use errors::*;
pub use interaction::*;
pub use mechanics::*;
pub use plotting::*;
