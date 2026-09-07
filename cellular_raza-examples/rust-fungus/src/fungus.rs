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

#[gen_stub_pymethods]
#[pymethods]
impl Fungus {
    #[new]
    fn new(
        pos: Bound<numpy::PyArray2<f64>>,
        diffusion_constant: f64,
        spring_tension: f64,
        rigidity: f64,
        spring_length: f64,
        damping: f64,
        radius: f64,
        potential_stiffness: f64,
        cutoff: f64,
        strength: f64,
    ) -> Self {
        let pos = crate::py_array_to_matrix(pos).transpose();
        let vel = 0.0 * &pos;
        let mechanics = cellular_raza::building_blocks::RodMechanics {
            pos,
            vel,
            diffusion_constant,
            spring_tension,
            rigidity,
            spring_length,
            damping,
        };
        let interaction = cellular_raza::building_blocks::RodInteraction(MorsePotential {
            radius,
            potential_stiffness,
            cutoff,
            strength,
        });

        Fungus {
            mechanics,
            interaction,
        }
    }

    #[getter]
    pub fn get_position<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        crate::matrix_to_py_array(py, &self.mechanics.pos.transpose())
    }

    #[setter]
    pub fn set_position(&mut self, position: Bound<numpy::PyArray2<f64>>) {
        self.mechanics.pos = crate::py_array_to_matrix(position).transpose();
    }

    #[getter]
    fn get_radius(&self) -> f64 {
        self.interaction.0.radius
    }
}
