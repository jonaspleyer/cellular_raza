mod simulation;
pub use simulation::*;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python version function of [run_simulation](simulation::run_simulation)
#[pyfunction]
// TODO
// #[pyo3(signature = (particles=Vec<Particle> simulation_settings=SimulationSettings))]
///
/// Runs a simulation containing particles with the settings given by simulation_settings.
fn run_simulation(
    // particles: Vec<Particle>,
    simulation_settings: SimulationSettings,
) -> Result<std::path::PathBuf, PyErr> {
    // println!("{:#?}", particles);
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
    m.add_class::<TypedInteraction>()?;
    m.add_class::<Brownian3D>()?;
    m.add_class::<Particle>()?;

    Ok(())
}
