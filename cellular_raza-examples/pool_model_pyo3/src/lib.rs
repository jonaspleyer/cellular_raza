mod simulation;
mod bacteria_properties;

use simulation::*;
use bacteria_properties::*;

use pyo3::{prelude::*, exceptions::PyValueError};

/// Python version function of [run_simulation](simulation::run_simulation)
#[pyfunction]
fn run_simulation(simulation_settings: SimulationSettings) -> Result<std::path::PathBuf, PyErr> {
    match run_simulation_rs(simulation_settings) {
        Ok(b) => Ok(b),
        Err(e) => Err(PyValueError::new_err(format!("{:?}", e))),
    }
}

#[pymodule]
fn pool_model_pyo3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;

    m.add_class::<SimulationSettings>()?;
    m.add_class::<Bacteria>()?;
    m.add_class::<BacteriaInteraction>()?;
    m.add_class::<BacteriaMechanicsModel2D>()?;
    m.add_class::<BacteriaReactions>()?;

    Ok(())
}
