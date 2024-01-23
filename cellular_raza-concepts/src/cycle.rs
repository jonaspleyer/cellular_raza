use crate::errors::{DeathError, DivisionError};

use serde::{Deserialize, Serialize};

/// Contains all events which can arise during the cell cycle and need to be communciated to
/// the simulation engine (see also [Cycle]).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CycleEvent {
    /// A cell-event which calls the [Cycle::divide] method which will spawn an additional cell and modify the existing one.
    Division,
    /// Immediately removes the cell from the simulation domain. No function will be called.
    Remove,
    /// The cell enters a dying mode.
    /// It is still continuously updating via the [Cycle::update_conditional_phased_death] its properties but now checking if the death phase is completed.
    /// [CycleEvent::Remove] will be carried out when the condition reaches true.
    PhasedDeath,
}

/// This trait represents all cycles of a cell and works in tandem with the [CycleEvent] enum.
///
/// The `update_cycle` function is designed to be called frequently and return only something if a
/// specific cycle event is supposed to be occuring. Backends should implement
/// the functionality to call the corresponding functions as needed.
pub trait Cycle<Cell, Float = f64> {
    /// Continuously updates cellular properties and may spawn a [CycleEvent] which then calls the corresponding functions (see also [CycleEvent]).
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
    /// Update the cell while returning a boolean which indicates if the updating procedure has finished.
    /// As soon as the return value is `true` the cell is removed.
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
    /// use cellular_raza_concepts_derive::CellAgent;
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError};
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
    /// use cellular_raza_concepts_derive::CellAgent;
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError};
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
    ///     #[Cycle(f32)]
    ///     cycle: MyCycle
    /// }
    /// ```
    fn derive_cycle_f32() {}

    /// ```
    /// use cellular_raza_concepts_derive::CellAgent;
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError};
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
    ///     #[Cycle(F)]
    ///     cycle: MyCycle<F>
    /// }
    /// ```
    fn derive_cycle_generic_float() {}

    /// ```
    /// use cellular_raza_concepts_derive::CellAgent;
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError};
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
    /// use cellular_raza_concepts_derive::CellAgent;
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError};
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
    /// use cellular_raza_concepts_derive::CellAgent;
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError};
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
    ///     #[Cycle(f32)]
    ///     MyCycle
    /// );
    /// ```
    fn derive_cycle_f32_unnamed() {}

    /// ```
    /// use cellular_raza_concepts_derive::CellAgent;
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError};
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
    ///     #[Cycle(F)]
    ///     MyCycle<F>
    /// );
    /// ```
    fn derive_cycle_generic_float_unnamed() {}

    /// ```
    /// use cellular_raza_concepts_derive::CellAgent;
    /// use cellular_raza_concepts::{Cycle, CycleEvent, DeathError, DivisionError};
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
