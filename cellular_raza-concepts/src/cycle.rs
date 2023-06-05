use crate::errors::{DeathError, DivisionError};

use serde::{Deserialize, Serialize};

/// Contains all events which can arise during the cell cycle and need to be communciated to
/// the simulation engine (see also [Cycle]).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CycleEvent {
    Division,
    Death,
}

/// This trait represents all cycles of a cell and works in tandem with the [CycleEvent] enum.
///
/// The `update_cycle` function is designed to be called frequently and return only something if a
/// specific cycle event is supposed to be occuring. Backends should implement
/// the functionality to call the corresponding functions as needed.
pub trait Cycle<Cell> {
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Cell,
    ) -> Option<CycleEvent>;
    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Cell) -> Result<Cell, DivisionError>;
    #[allow(unused)]
    fn die(rng: &mut rand_chacha::ChaCha8Rng, cell: Cell) -> Result<(), DeathError> {
        Ok(())
    }
}
