use crate::concepts::errors::{DeathError,DivisionError};

use serde::{Serialize,Deserialize};


/// Contains all events which can arise during the cell cycle and need to be communciated to
/// the simulation engine (see also [Cycle]).
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
pub enum CycleEvent {
    Division,
    Death,
}


/// This trait represents all cycles of a cell and works in tandem with the [CycleEvent] enum.
pub trait Cycle<Cell> {
    fn update_cycle(rng: &mut rand_chacha::ChaCha8Rng, dt: &f64, cell: &mut Cell) -> Option<CycleEvent>;
    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Cell) -> Result<Option<Cell>, DivisionError>;
    fn die(_rng: &mut rand_chacha::ChaCha8Rng, _cell: Cell) -> Result<(), DeathError> {Ok(())}
}
