use cellular_raza::building_blocks::cartesian_cuboid_n_old::*;
use cellular_raza::building_blocks::*;
use cellular_raza::core::backend::cpu_os_threads::*;
use cellular_raza::core::storage::*;

use num::Zero;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use serde::{Deserialize, Serialize};

use pyo3::prelude::*;

use crate::bacteria_properties::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct Domain {
    pub diffusion_constants: [f64; NUMBER_OF_REACTION_COMPONENTS],
    pub initial_concentrations: [f64; NUMBER_OF_REACTION_COMPONENTS],

    pub size: f64,
    pub n_voxels: Option<usize>,
}

#[pymethods]
impl Domain {
    fn __repr__(&self) -> String {
        format!("{self:#?}")
    }

    #[new]
    #[pyo3(signature = (
        diffusion_constants=[10.0, 5.0],
        initial_concentrations=[2.0, 0.0],
        size=200.0,
        n_voxels=None
    ))]
    fn new(
        diffusion_constants: [f64; NUMBER_OF_REACTION_COMPONENTS],
        initial_concentrations: [f64; NUMBER_OF_REACTION_COMPONENTS],
        size: f64,
        n_voxels: Option<usize>,
    ) -> Self {
        Self {
            diffusion_constants,
            initial_concentrations,
            size,
            n_voxels,
        }
    }

    /* #[staticmethod]
    fn default() -> Self {
        Self {
            diffusion_constants: [2.0, 0.2],
            initial_concentrations: [1.0, 0.0],

            size: 250.0,
            n_voxels: None,
        }
    }*/
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct MetaParams {
    // General Settings
    pub random_seed: u64,

    // SIMULATION FLOW SETTINGS
    pub n_times: usize,
    pub dt: f64,
    pub t_start: f64,
    pub save_interval: usize,
    pub n_threads: usize,
    pub show_progressbar: bool,
    pub save_path: String,
    pub save_add_date: bool,
}

#[pymethods]
impl MetaParams {
    #[new]
    #[pyo3(signature = (
        random_seed=1,
        n_times=20_001,
        dt=0.5,
        t_start=0.0,
        save_interval=100,
        n_threads=1,
        show_progressbar=true,
        save_path="out/pool_model".into(),
        save_add_date=true,
    ))]
    fn new(
        random_seed: u64,
        n_times: usize,
        dt: f64,
        t_start: f64,
        save_interval: usize,
        n_threads: usize,
        show_progressbar: bool,
        save_path: String,
        save_add_date: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            random_seed,
            n_times,
            dt,
            t_start,
            save_interval,
            n_threads,
            show_progressbar,
            save_path,
            save_add_date,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

fn save_initial_state(
    path: &std::path::PathBuf,
    cells: &Vec<BacteriaTemplate>,
    domain: &Domain,
    meta_params: &MetaParams,
) -> PyResult<()> {
    // Also save the SimulationSettings into the same folder
    let mut save_path_cells = path.clone();
    let mut save_path_domain = path.clone();
    let mut save_path_meta_params = path.clone();
    save_path_cells.push("initial_cells.json");
    save_path_domain.push("domain.json");
    save_path_meta_params.push("meta_params.json");

    let f_cells = std::fs::File::create(save_path_cells)?;
    let f_domain = std::fs::File::create(save_path_domain)?;
    let f_meta_params = std::fs::File::create(save_path_meta_params)?;

    let writer_cells = std::io::BufWriter::new(f_cells);
    let writer_domain = std::io::BufWriter::new(f_domain);
    let writer_meta_params = std::io::BufWriter::new(f_meta_params);

    serde_json::to_writer_pretty(writer_cells, &cells).or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("serde_json error in writing simulation settings to file: {e}"),
        ))
    })?;
    serde_json::to_writer_pretty(writer_domain, &domain).or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("serde_json error in writing simulation settings to file: {e}"),
        ))
    })?;
    serde_json::to_writer_pretty(writer_meta_params, &meta_params).or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("serde_json error in writing simulation settings to file: {e}"),
        ))
    })?;

    Ok(())
}

#[pyfunction]
pub fn generate_cells(
    py: Python,
    n_bacteria_1: usize,
    n_bacteria_2: usize,
    domain: Domain,
    meta_params: MetaParams,
    bacteria_template: Option<BacteriaTemplate>,
) -> PyResult<Vec<BacteriaTemplate>> {
    // ###################################### DEFINE CELLS IN SIMULATION ######################################

    let mut rng = ChaCha8Rng::seed_from_u64(meta_params.random_seed);
    let bacteria = match bacteria_template {
        Some(template) => template.clone(),
        None => BacteriaTemplate::default(py)?,
    };

    let cells = (0..n_bacteria_1 + n_bacteria_2)
        .map(|n_cell| {
            let mut new_bacteria = bacteria.clone();
            let x = rng.gen_range(0.0..domain.size);
            let y = rng.gen_range(0.0..domain.size);

            if n_cell >= n_bacteria_1 {
                let mut reactions = new_bacteria
                    .cellular_reactions
                    .extract::<BacteriaReactions>(py)?;
                reactions.species = Species::S2;
                new_bacteria.cellular_reactions = Py::new(py, reactions)?;
            }
            let mut mechanics = new_bacteria.mechanics.extract::<NewtonDamped2D>(py)?;
            mechanics.set_pos([x, y]);
            new_bacteria.mechanics = Py::new(py, mechanics)?;
            Ok(new_bacteria)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(cells)
}

#[pyfunction]
pub fn run_simulation(
    py: Python,
    cells: Vec<BacteriaTemplate>,
    domain: Domain,
    meta_params: MetaParams,
) -> PyResult<std::path::PathBuf> {
    // ###################################### CREATE SUPERVISOR AND RUN SIMULATION ######################################
    let simulation_domain = match domain.n_voxels {
        Some(n_voxels) => CartesianCuboid2::from_boundaries_and_n_voxels(
            [0.0; 2],
            [domain.size; 2],
            [n_voxels; 2],
        ),
        None => {
            let mut max_cell_radius = cells
                .iter()
                .map(|cell| {
                    cell.cellular_reactions
                        .extract::<BacteriaReactions>(py)
                        .unwrap()
                        .cell_radius()
                })
                .fold(f64::MIN, f64::max);
            if max_cell_radius <= 0.0 {
                max_cell_radius = domain.size / 3.0;
            }
            CartesianCuboid2::from_boundaries_and_interaction_ranges(
                [0.0; 2],
                [domain.size; 2],
                [2.0 * max_cell_radius; 2],
            )
        }
    }
    .or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Rust error in construction of simulation domain: {e}"),
        ))
    })?;

    // TODO try to avoid this cloning right here!
    let simulation_cells = cells
        .clone()
        .into_iter()
        .map(|cell_template| Bacteria::from(py, cell_template))
        .collect::<PyResult<Vec<_>>>()?;

    let time = TimeSetup {
        t_start: 0.0,
        t_eval: (0..meta_params.n_times)
            .map(|n| {
                (
                    n as f64 * meta_params.dt,
                    n % meta_params.save_interval == 0,
                )
            })
            .collect(),
    };

    let simulation_meta_params = SimulationMetaParams {
        n_threads: meta_params.n_threads,
        rng_seed: meta_params.random_seed,
    };

    let storage = StorageBuilder::new()
        .location(&meta_params.save_path)
        .add_date(meta_params.save_add_date)
        .init();
    // storage.export_formats = vec![ExportOptions::Vtk];

    let simulation_setup = create_simulation_setup!(
        Domain: simulation_domain,
        Cells: simulation_cells,
        Time: time,
        MetaParams: simulation_meta_params,
        Storage: storage
    );

    let voxel_definition_strategy = |voxel: &mut CartesianCuboidVoxel2<2>| {
        voxel.diffusion_constant = ReactionVector::from([domain.diffusion_constants]);
        voxel.extracellular_concentrations = ReactionVector::from([domain.initial_concentrations]);
        voxel.degradation_rate = ReactionVector::zero();
        voxel.production_rate = ReactionVector::zero();
    };

    let strategies = Strategies {
        voxel_definition_strategies: &voxel_definition_strategy,
    };

    // let simulation_result = run_full_simulation!(setup, settings, [Mechanics, Interaction, Cycle])?;
    let mut supervisor =
        SimulationSupervisor::initialize_with_strategies(simulation_setup, strategies);
    supervisor.config.show_progressbar = meta_params.show_progressbar;

    save_initial_state(
        &supervisor.storage.get_full_path(),
        &cells,
        &domain,
        &meta_params,
    )?;

    let simulation_result = supervisor.run_full_sim().unwrap();

    Ok(simulation_result.storage.get_full_path())
}
