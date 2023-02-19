use crate::concepts::errors::DivisionError;

pub enum CycleEvent {
    Division,
}

pub trait Cycle<Cell> {
    fn update_cycle(rng: &mut rand_chacha::ChaCha8Rng, dt: &f64, c: &mut Cell) -> Option<CycleEvent>;
    fn divide(rng: &mut rand_chacha::ChaCha8Rng, c: &mut Cell) -> Result<Option<Cell>, DivisionError>;
}
