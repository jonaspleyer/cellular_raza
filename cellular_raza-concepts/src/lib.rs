#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! This crate encapsulates concepts which govern an agent-based model specified by [cellular_raza](https://docs.rs/cellular_raza).
//! To learn how to design your own concepts please refer to the [cellular_raza book](https://jonaspleyer.github.io/cellular_raza/DevelopersDesigningNewConcepts.html).

/// Traits and types to define a complete cell-agent
pub mod cell;

/// Traits and types needed to define the cell cycle.
///
/// The Cycle trait is responsible for the implementation of cycles and updates them incrementally.
/// The main `update` function is responsible for returning an optional cycle event.
/// These events can have different effects. For example a cell-division event triggers the
/// `divide` method. A mapping of events and functions is depicted in the table below.
///
/// | Event | Effect |
/// | ----- -------- | ------ |
/// | [`Division`](cycle::CycleEvent::Division) | The [`divide`](cycle::Cycle::divide) function returns (creates) a new cell and modifies the existing cell in-place. This means, that the user is responsible to make sure, that every field of the cell struct is modified correctly in order to simulate cell-division. |
/// | [`PhasedDeath`](cycle::CycleEvent::PhasedDeath) | The cell enters a dying process and is still continuously updated via [`update_conditional_phased_death`](cycle::Cycle::update_conditional_phased_death). Once the corresponding function returns `true` the process is considered complete and the cell is removed. |
/// | [`Remove`](cycle::CycleEvent::Remove) | This event removes the cell from the simulation without any further actions. |
///
/// # Stochasticity
/// In order to make sure that results are reproducible, the provided rng parameter should be used.
/// Should a user fall back to the option to use the threaded rng, this simulation cannot guarantee
/// deterministic results anymore.
/// We plan to include the stochastic aspect into individual [`Event`](cycle::CycleEvent) variants such that the correct
/// handling of integrating the underlying stochastic process can be carried out by the [backend](https://docs.rs/cellular_raza-core/backend).
///
/// # Example implementation
/// This could be an example of a very simplified cell-agent.
/// The user is free to do anything with this function that is desired but is also responsible for
/// keeping track of all the variables. This means for example that intracellular values might need
/// to be adjusted (most often halfed) and new positions need to be assigned to the cells such that
/// the cells are not overlapping ...
///
/// ```
/// use rand::Rng;
/// use rand_chacha::ChaCha8Rng;
/// use cellular_raza_concepts::{
///     cycle::{Cycle,CycleEvent},
///     errors::DivisionError
/// };
///
/// // We define our cell struct with all parameters needed for this cell-agent.
/// #[derive(Clone)]
/// struct Cell {
///     // Size of the cell (spherical)
///     radius: f64,
///     // Track the age of the cell
///     current_age: f64,
///     // Used in cycle later. Divide cell if older than maximum age.
///     maximum_age: f64,
///     // The position of the cell. We cannot have two positions which are the same. Thus we need
///     // to update the position as well.
///     position: [f64; 2],
/// }
///
/// impl Cycle<Cell> for Cell {
///     fn update_cycle(rng: &mut ChaCha8Rng, dt: &f64, cell: &mut Cell) -> Option<CycleEvent> {
///         // Increase the current age of the cell
///         cell.current_age += dt;
///
///         // If the cell is older than the current age, return a division event
///         if cell.current_age > cell.maximum_age {
///             return Some(CycleEvent::Division)
///         }
///         None
///     }
///
///     fn divide(rng: &mut ChaCha8Rng, cell: &mut Cell) -> Result<Cell, DivisionError> {
///         // Prepare the original cell for division.
///         // Set the radius of both cells to half of the original radius.
///         cell.radius *= 0.5;
///
///         // Also set the current age of the cell to zero again
///         cell.current_age = 0.0;
///
///         // Clone the existing cell
///         let mut new_cell = (*cell).clone();
///
///         // Define a new position for both cells
///         // To do this: Pick a random number as an angle.
///         let angle = rng.gen_range(0.0..2.0*std::f64::consts::PI);
///
///         // Calculate the new position of the original and new cell with this angle
///         let pos = [
///             cell.radius * angle.cos(),
///             cell.radius * angle.sin()
///         ];
///         let new_pos = [
///             cell.radius * (angle+std::f64::consts::FRAC_PI_2).cos(),
///             cell.radius * (angle+std::f64::consts::FRAC_PI_2).sin()
///         ];
///
///         // Set new positions
///         cell.position = pos;
///         new_cell.position = new_pos;
///
///         // Finally return the new cell
///         return Ok(new_cell);
///     }
/// }
/// ```
pub mod cycle;

/// Traits and types used to define a domain where cells live in.
pub mod domain;

/// traits and types used to define a domain where cells live in (new Version, will replace [domain] eventually).
pub mod domain_new;

/// Error types which can arise during the simulation.
///
/// For Backends it may be useful to define a singular error type (eg. `SimulationError`) which should be derivable from errors
/// arising during the simulation process.
/// It is required for custom error types `MyCustomError` of the engine to implement the `From<MyCustomError> for SimulationError`.
/// Errors should be seperated by their ability to be recovered, ignored or handled otherwise.
/// Since this crate aims to provide an adaptive solving aproach, it is desired to have a fallback
/// mechanism which can be called for errors which may arise due to precision problems.
///
/// The following table shows a summary of the errors currently supported.
/// Backends need to be aware of them and implement custom handling schemes to overcome or work around them.
/// Aborting the simulation is an option but must be documented well without introducing undefined behaviour.
///
/// | ErrorType | Possible Error Reasons |
/// | --- | --- | --- |
/// | BoundaryError | Solver Accuracy, Domain Implementaion bug, Internal engine error |
/// | CalcError | Solver Accuracy, Bug by user implementation of corresponding function, Internal engine error |
// TODO implement the handling of these errors!
pub mod errors;

/// Captures traits and types related to interactions between cells.
///
/// Interactions can arise due to many different mechanisms. The following table shows a short
/// summary of the possible traits which can be used. Some of them should not make sense to use in
/// combination. An engine may chose to implement only certain traits while omitting others.
///
/// | Interaction Trait | Description |
/// | --- | --- |
/// | [Interaction](interaction::Interaction) | Cells are interacting by forces which can have a range larger than the cell itself. Foe example, users can choose to implement their own repulsive and attractive forces. |
/// | [CellularReactions](interaction::CellularReactions) | Intracellular reactions may be coupled to an extracellular environment. We can model these reactions via [ODEs](https://en.wikipedia.org/wiki/Ordinary_differential_equation). |
pub mod interaction;

/// Defines how to adjust position,force and velocity of the individual cells.
///
/// This concept is extremely central to every cell-agent since it defines the spatial
/// representation of the cell which can be seen by other cells. This means that as long as all
/// cells which should be included in the simulation can be represented by this shared trait,
/// [cellular_raza](crate) should be able to simulate them.
///
/// While a point of the simulation in general is not additive in the general theoretical formulation (due to boundaries for
/// example), it is necessary to have an additive type to effectively use adaptive solvers. We thus
/// need to check and apply boundary conditions independantly.
pub mod mechanics;

/// Visualize components of the simulation directly via the [plotters](https://docs.rs/plotters) library.
#[deprecated]
pub mod plotting;

/// Exports which can be used to simply access all concepts.
pub mod prelude;
