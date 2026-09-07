use cellular_raza::building_blocks::MorsePotential;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use serde::{Deserialize, Serialize};

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Clone, Deserialize, Serialize)]
pub struct Fungus {
    pub mechanics: cellular_raza::building_blocks::RodMechanics<f64, 2>,
    pub interaction: cellular_raza::building_blocks::RodInteraction<MorsePotential>,
}
