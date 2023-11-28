mod bacteria_properties;
mod simulation;

use bacteria_properties::*;
use simulation::*;

use pyo3::prelude::*;

#[pymodule]
fn cr_pool_model_pyo3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;

    m.add_class::<SimulationSettings>()?;
    m.add_class::<Bacteria>()?;
    m.add_class::<BacteriaInteraction>()?;
    m.add_class::<cellular_raza::building_blocks::cell_building_blocks::mechanics::Langevin2D>()?;
    m.add_class::<BacteriaReactions>()?;

    Ok(())
}
