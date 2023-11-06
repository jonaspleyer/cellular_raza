use cellular_raza_concepts::cycle::*;
// use crate::impls_cell_properties::cell_model::CellModel;

use serde::{Deserialize, Serialize};

/// No cycle of the cell.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoCycle {}

impl<Cel> Cycle<Cel> for NoCycle {
    fn update_cycle(_: &mut rand_chacha::ChaCha8Rng, _: &f64, _: &mut Cel) -> Option<CycleEvent> {
        None
    }
    fn divide(
        _: &mut rand_chacha::ChaCha8Rng,
        _: &mut Cel,
    ) -> Result<Cel, cellular_raza_concepts::errors::DivisionError> {
        panic!("This function should never be called. This is a backend error. Please report!")
    }
}
