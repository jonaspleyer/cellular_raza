use crate::concepts::errors::{DeathError,DivisionError};

use serde::{Serialize,Deserialize};

#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
pub enum CycleEvent {
    Division,
    Death,
}

pub trait Cycle<Cell> {
    fn update_cycle(rng: &mut rand_chacha::ChaCha8Rng, dt: &f64, cell: &mut Cell) -> Option<CycleEvent>;
    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Cell) -> Result<Option<Cell>, DivisionError>;
    fn die(_rng: &mut rand_chacha::ChaCha8Rng, _cell: Cell) -> Result<(), DeathError> {Ok(())}
}
