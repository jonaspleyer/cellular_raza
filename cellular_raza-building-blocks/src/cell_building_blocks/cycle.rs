use cellular_raza_concepts::cycle::*;
// use crate::impls_cell_properties::cell_model::CellModel;

use serde::{Deserialize, Serialize};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// No cycle of the cell.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct NoCycle;

impl<Cel, Float> Cycle<Cel, Float> for NoCycle {
    fn update_cycle(_: &mut rand_chacha::ChaCha8Rng, _: &Float, _: &mut Cel) -> Option<CycleEvent> {
        None
    }

    fn divide(
        _: &mut rand_chacha::ChaCha8Rng,
        _: &mut Cel,
    ) -> Result<Cel, cellular_raza_concepts::errors::DivisionError> {
        panic!("This is the divide() function of the NoCycle struct which should never be called. This is a backend error. Please report!")
    }
}
