mod simulation;
pub use simulation::*;

use pyo3::prelude::*;

#[pymodule]
fn cr_autophagy_pyo3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;

    m.add_class::<SimulationSettings>()?;
    m.add_class::<Species>()?;
    m.add_class::<TypedInteraction>()?;
    m.add_class::<Langevin3D>()?;
    m.add_class::<ParticleTemplate>()?;

    Ok(())
}
