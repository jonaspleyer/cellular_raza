use crate::concepts::cycle::*;
// use crate::impls_cell_properties::cell_model::CellModel;

use serde::{Serialize, Deserialize};


#[derive(Clone,Debug,Serialize, Deserialize)]
pub struct NoCycle {}

impl<Cel> Cycle<Cel> for NoCycle {
    fn update_cycle(_: &mut rand_chacha::ChaCha8Rng, _: &f64, _: &mut Cel) -> Option<CycleEvent> {None}
    fn divide(_: &mut rand_chacha::ChaCha8Rng, _: &mut Cel) -> Result<Option<Cel>, crate::concepts::errors::DivisionError> {
        Ok(None)
    }
}
