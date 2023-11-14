mod simulation;

use pyo3::{exceptions::PyValueError, prelude::*};
pub use simulation::*;

/// Python version function of [run_simulation](simulation::run_simulation)
#[pyfunction]
fn run_simulation(simulation_settings: SimulationSettings) -> Result<std::path::PathBuf, PyErr> {
    match run_simulation_rs(simulation_settings) {
        Ok(b) => Ok(b),
        Err(e) => Err(PyValueError::new_err(format!("{:?}", e))),
    }
}

#[pymodule]
fn cr_autophagy_pyo3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;

    m.add_class::<SimulationSettings>()?;
    m.add_class::<Species>()?;
    m.add_class::<CellSpecificInteraction>()?;
    m.add_class::<Brownian3D>()?;

    Ok(())
}
