use crate::errors::{DeathError, DivisionError};

use serde::{Deserialize, Serialize};

/// Contains all events which can arise during the cell cycle and need to be communciated to
/// the simulation engine (see also [Cycle]).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CycleEvent {
    /// A cell-event which calls the [Cycle::divide] method which will
    /// spawn an additional cell and modify the existing one.
    Division,
    /// Immediately removes the cell from the simulation domain. No function will be called.
    Remove,
    /// The cell enters a dying mode.
    /// It is still continuously updating via the [Cycle::update_conditional_phased_death] its
    /// properties but now checking if the death phase is completed.
    /// [CycleEvent::Remove] will be carried out when the condition reaches true.
    PhasedDeath,
}

/// This trait represents all cycles of a cell and works in tandem with the [CycleEvent] enum.
///
/// The `update_cycle` function is designed to be called frequently and return only something if a
/// specific cycle event is supposed to be occuring. Backends should implement
/// the functionality to call the corresponding functions as needed.
///
/// ## Stochasticity
/// In order to make sure that results are reproducible, the provided rng parameter should be used.
/// Should a user fall back to the option to use the threaded rng, this simulation cannot guarantee
/// deterministic results anymore.
/// We plan to include the stochastic aspect into individual [`Event`](CycleEvent) variants such
/// that the correct handling of integrating the underlying stochastic process can be
/// carried out by the [backend](https://docs.rs/cellular_raza-core/backend).
///
/// ## Example implementation
/// This could be an example of a very simplified cell-agent.
/// The user is free to do anything with this function that is desired but is also responsible for
/// keeping track of all the variables. This means for example that intracellular values might need
/// to be adjusted (most often halfed) and new positions need to be assigned to the cells such that
/// the cells are not overlapping ...
///
/// ```
/// use rand::Rng;
/// use rand_chacha::ChaCha8Rng;
/// use cellular_raza_concepts::{Cycle, CycleEvent, DivisionError};
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
pub trait Cycle<Cell = Self, Float = f64> {
    /// Continuously updates cellular properties and may spawn a [CycleEvent] which
    /// then calls the corresponding functions (see also [CycleEvent]).
    #[must_use]
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &Float,
        cell: &mut Cell,
    ) -> Option<CycleEvent>;

    /// Performs division of the cell by modifying the existing one and spawning an additional cell.
    /// The user is responsible for correctly adjusting cell-specific values such as intracellular
    /// concentrations or position of the two resulting cells.
    /// Corresponds to [CycleEvent::Division].
    #[must_use]
    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Cell) -> Result<Cell, DivisionError>;

    /// Method corresponding to the [CycleEvent::PhasedDeath] event.
    /// Update the cell while returning a boolean which indicates if the updating procedure has
    /// finished. As soon as the return value is `true` the cell is removed.
    #[allow(unused)]
    #[must_use]
    fn update_conditional_phased_death(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &Float,
        cell: &mut Cell,
    ) -> Result<bool, DeathError> {
        Ok(true)
    }
}

#[allow(unused)]
#[doc(hidden)]
mod test_derive {
    /// ```
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError, CellAgent};
    /// struct MyCycle;
    ///
    /// impl Cycle<MyAgent> for MyCycle {
    ///     fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut MyAgent) -> Result<MyAgent, DivisionError> {
    ///         panic!("This should never be called")
    ///     }
    ///
    ///     fn update_cycle(
    ///             rng: &mut rand_chacha::ChaCha8Rng,
    ///             dt: &f64,
    ///             cell: &mut MyAgent,
    ///         ) -> Option<CycleEvent> {
    ///         panic!("This should never be called")
    ///     }
    /// }
    ///
    /// #[derive(CellAgent)]
    /// struct MyAgent {
    ///     #[Cycle]
    ///     cycle: MyCycle
    /// }
    /// ```
    fn derive_cycle_default() {}

    /// ```
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError, CellAgent};
    /// struct MyCycle;
    ///
    /// impl Cycle<MyAgent, f32> for MyCycle {
    ///     fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut MyAgent) -> Result<MyAgent, DivisionError> {
    ///         panic!("This should never be called")
    ///     }
    ///
    ///     fn update_cycle(
    ///             rng: &mut rand_chacha::ChaCha8Rng,
    ///             dt: &f32,
    ///             cell: &mut MyAgent,
    ///         ) -> Option<CycleEvent> {
    ///         panic!("This should never be called")
    ///     }
    /// }
    ///
    /// #[derive(CellAgent)]
    /// struct MyAgent {
    ///     #[Cycle]
    ///     cycle: MyCycle
    /// }
    /// ```
    fn derive_cycle_f32() {}

    /// ```
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError, CellAgent};
    /// struct MyCycle<F> {
    ///     some_property: F,
    /// }
    ///
    /// impl<F> Cycle<MyAgent<F>, F> for MyCycle<F> {
    ///     fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut MyAgent<F>) -> Result<MyAgent<F>, DivisionError> {
    ///         panic!("This should never be called")
    ///     }
    ///
    ///     fn update_cycle(
    ///             rng: &mut rand_chacha::ChaCha8Rng,
    ///             dt: &F,
    ///             cell: &mut MyAgent<F>,
    ///         ) -> Option<CycleEvent> {
    ///         panic!("This should never be called")
    ///     }
    /// }
    ///
    /// #[derive(CellAgent)]
    /// struct MyAgent<F> {
    ///     #[Cycle]
    ///     cycle: MyCycle<F>
    /// }
    /// ```
    fn derive_cycle_generic_float() {}

    /// ```
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError, CellAgent};
    /// struct MyCycle<G> {
    ///     some_property: G,
    /// }
    ///
    /// impl<G> Cycle<MyAgent<G>> for MyCycle<G>
    /// where
    ///     G: Clone
    /// {
    ///     fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut MyAgent<G>) -> Result<MyAgent<G>, DivisionError> {
    ///         panic!("This should never be called")
    ///     }
    ///
    ///     fn update_cycle(
    ///             rng: &mut rand_chacha::ChaCha8Rng,
    ///             dt: &f64,
    ///             cell: &mut MyAgent<G>,
    ///         ) -> Option<CycleEvent> {
    ///         panic!("This should never be called")
    ///     }
    /// }
    ///
    /// #[derive(CellAgent)]
    /// struct MyAgent<G>
    /// where
    ///     G: Clone
    /// {
    ///     #[Cycle]
    ///     cycle: MyCycle<G>
    /// }
    /// ```
    fn derive_cycle_generic_float_where_clause() {}

    /// ```
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError, CellAgent};
    /// struct MyCycle;
    ///
    /// impl Cycle<MyAgent> for MyCycle {
    ///     fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut MyAgent) -> Result<MyAgent, DivisionError> {
    ///         panic!("This should never be called")
    ///     }
    ///
    ///     fn update_cycle(
    ///             rng: &mut rand_chacha::ChaCha8Rng,
    ///             dt: &f64,
    ///             cell: &mut MyAgent,
    ///         ) -> Option<CycleEvent> {
    ///         panic!("This should never be called")
    ///     }
    /// }
    ///
    /// #[derive(CellAgent)]
    /// struct MyAgent(
    ///     #[Cycle]
    ///     MyCycle
    /// );
    /// ```
    fn derive_cycle_unnamed() {}

    /// ```
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError, CellAgent};
    /// struct MyCycle;
    ///
    /// impl Cycle<MyAgent, f32> for MyCycle {
    ///     fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut MyAgent) -> Result<MyAgent, DivisionError> {
    ///         panic!("This should never be called")
    ///     }
    ///
    ///     fn update_cycle(
    ///             rng: &mut rand_chacha::ChaCha8Rng,
    ///             dt: &f32,
    ///             cell: &mut MyAgent,
    ///         ) -> Option<CycleEvent> {
    ///         panic!("This should never be called")
    ///     }
    /// }
    ///
    /// #[derive(CellAgent)]
    /// struct MyAgent(
    ///     #[Cycle]
    ///     MyCycle
    /// );
    /// ```
    fn derive_cycle_f32_unnamed() {}

    /// ```
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError, CellAgent};
    /// struct MyCycle<F> {
    ///     some_property: F,
    /// }
    ///
    /// impl<F> Cycle<MyAgent<F>, F> for MyCycle<F> {
    ///     fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut MyAgent<F>) -> Result<MyAgent<F>, DivisionError> {
    ///         panic!("This should never be called")
    ///     }
    ///
    ///     fn update_cycle(
    ///             rng: &mut rand_chacha::ChaCha8Rng,
    ///             dt: &F,
    ///             cell: &mut MyAgent<F>,
    ///         ) -> Option<CycleEvent> {
    ///         panic!("This should never be called")
    ///     }
    /// }
    ///
    /// #[derive(CellAgent)]
    /// struct MyAgent<F>(
    ///     #[Cycle]
    ///     MyCycle<F>
    /// );
    /// ```
    fn derive_cycle_generic_float_unnamed() {}

    /// ```
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError, CellAgent};
    /// struct MyCycle<G> {
    ///     some_property: G,
    /// }
    ///
    /// impl<G> Cycle<MyAgent<G>> for MyCycle<G>
    /// where
    ///     G: Clone
    /// {
    ///     fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut MyAgent<G>) -> Result<MyAgent<G>, DivisionError> {
    ///         panic!("This should never be called")
    ///     }
    ///
    ///     fn update_cycle(
    ///             rng: &mut rand_chacha::ChaCha8Rng,
    ///             dt: &f64,
    ///             cell: &mut MyAgent<G>,
    ///         ) -> Option<CycleEvent> {
    ///         panic!("This should never be called")
    ///     }
    /// }
    ///
    /// #[derive(CellAgent)]
    /// struct MyAgent<G>
    /// (
    ///     #[Cycle]
    ///     MyCycle<G>
    /// )
    /// where
    ///     G: Clone;
    /// ```
    fn derive_cycle_generic_float_where_clause_unnamed() {}
}
