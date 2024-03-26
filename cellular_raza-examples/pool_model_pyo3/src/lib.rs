mod bacteria_properties;
mod simulation;

use bacteria_properties::*;
use simulation::*;

use pyo3::prelude::*;

#[pymodule]
fn cr_pool_model_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(generate_cells, m)?)?;

    m.add_class::<Bacteria>()?;
    m.add_class::<BacteriaTemplate>()?;
    m.add_class::<Species>()?;
    m.add_class::<cellular_raza::prelude::NewtonDamped2D>()?;
    m.add_class::<BacteriaCycle>()?;
    m.add_class::<BacteriaReactions>()?;

    m.add_class::<MetaParams>()?;
    m.add_class::<Domain>()?;

    Ok(())
}
