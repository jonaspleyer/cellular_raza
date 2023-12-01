use cellular_raza::prelude::*;

use nalgebra::Vector2;

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
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct SimulationSettings {
    pub random_seed: u64,

    // DOMAIN SETTINGS
    pub domain: Py<Domain>,

    // BACTERIA SETTINGS
    pub n_bacteria_initial_1: usize,
    pub n_bacteria_initial_2: usize,
    pub bacteria_mechanics: Py<NewtonDamped2D>,
    pub bacteria_interaction: Py<BacteriaInteraction>,
    pub bacteria_cycle: Py<BacteriaCycle>,
    pub bacteria_reactions: Py<BacteriaReactions>,

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
impl SimulationSettings {
    #[new]
    fn new(py: Python) -> PyResult<Self> {
        let domain_size = 250.0;
        let bacteria_radius: f64 = 1.5;
        let bacteria_volume = std::f64::consts::PI * bacteria_radius.powf(2.0);

        Ok(Self {
            random_seed: 1,

            // DOMAIN SETTINGS
            domain: Py::new(
                py,
                Domain {
                    diffusion_constants: [2.0, 0.2],
                    initial_concentrations: [1.0, 0.0],

                    size: domain_size,
                    n_voxels: None,
                },
            )?,

            // BACTERIA SETTINGS
            n_bacteria_initial_1: 5,
            n_bacteria_initial_2: 5,

            bacteria_mechanics: Py::new(
                py,
                // For the mass conversion also see the dedicated [volume_to_mass] function!
                NewtonDamped2D::new(
                    [0.0; 2],               // pos
                    [0.0; 2],               // vel
                    0.5,                    // damping
                    1.09 * bacteria_volume, // mass
                ),
            )?,

            bacteria_interaction: Py::new(
                py,
                BacteriaInteraction {
                    potential_strength: 0.5,
                    cell_radius: bacteria_radius,
                },
            )?,

            bacteria_cycle: Py::new(
                py,
                BacteriaCycle {
                    food_to_volume_conversion: 1e-5,
                    volume_division_threshold: 2.0 * bacteria_volume,
                    lag_phase_transition_rate_1: 0.005,
                    lag_phase_transition_rate_2: 0.008,
                },
            )?,

            bacteria_reactions: Py::new(
                py,
                BacteriaReactions {
                    lag_phase_active: true,
                    // By default we make the cells species 1
                    species: Species::S1,
                    intracellular_concentrations: [0.0; NUMBER_OF_REACTION_COMPONENTS].into(),
                    uptake_rate: 0.01,
                    inhibition_production_rate: 0.1,
                    inhibition_coefficient: 0.1,
                },
            )?,

            // SIMULATION FLOW SETTINGS
            n_times: 50_001,
            dt: 0.5,
            t_start: 0.0,
            save_interval: 1_000,
            n_threads: 1,
            show_progressbar: true,
            save_path: "out/pool_model".into(),
            save_add_date: true,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

fn save_simulation_settings(
    path: &std::path::PathBuf,
    simulation_settings: &SimulationSettings,
) -> PyResult<()> {
    // Also save the SimulationSettings into the same folder
    let mut save_path = path.clone();
    save_path.push("simulation_settings.json");
    let f = std::fs::File::create(save_path)?;
    let writer = std::io::BufWriter::new(f);
    serde_json::to_writer_pretty(writer, &simulation_settings).or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("serde_json error in writing simulation settings to file: {e}"),
        ))
    })?;
    Ok(())
}

#[pyfunction]
pub fn run_simulation(
    simulation_settings: SimulationSettings,
    py: Python,
) -> PyResult<std::path::PathBuf> {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(simulation_settings.random_seed);

    // ###################################### DEFINE CELLS IN SIMULATION ######################################
    let mechanics: NewtonDamped2D = simulation_settings.bacteria_mechanics.extract(py)?;
    let cycle: BacteriaCycle = simulation_settings.bacteria_cycle.extract(py)?;
    let interaction: BacteriaInteraction = simulation_settings.bacteria_interaction.extract(py)?;
    let cellular_reactions: BacteriaReactions =
        simulation_settings.bacteria_reactions.extract(py)?;
    let domain_setup: Domain = simulation_settings.domain.extract(py)?;

    let cells = (0..simulation_settings.n_bacteria_initial_1+simulation_settings.n_bacteria_initial_2)
        .map(|n_cell| {
            let x = rng.gen_range(0.0..domain_setup.size);
            let y = rng.gen_range(0.0..domain_setup.size);

            // Set new position of bacteria
            let pos = Vector2::from([x, y]);
            let mut mechanics_new = mechanics.clone();
            let mut cellular_reactions_new = cellular_reactions.clone();
            if n_cell >= simulation_settings.n_bacteria_initial_1 {
                cellular_reactions_new.species = Species::S2;
            }
            mechanics_new.set_pos(&pos);
            Ok(Bacteria {
                mechanics: mechanics_new,
                cycle: cycle.clone(),
                interaction: interaction.clone(),
                cellular_reactions: cellular_reactions_new,
                interactionextracellulargradient: GradientSensing,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    // ###################################### CREATE SUPERVISOR AND RUN SIMULATION ######################################
    let domain = match domain_setup.n_voxels {
        Some(n_voxels) => CartesianCuboid2::from_boundaries_and_n_voxels(
            [0.0; 2],
            [domain_setup.size; 2],
            [n_voxels; 2],
        ),
        None => CartesianCuboid2::from_boundaries_and_interaction_ranges(
            [0.0; 2],
            [domain_setup.size; 2],
            [2.0 * interaction.cell_radius; 2],
        ),
    }
    .or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Rust error in construction of simulation domain: {e}"),
        ))
    })?;

    let time = TimeSetup {
        t_start: 0.0,
        t_eval: (0..simulation_settings.n_times)
            .map(|n| {
                (
                    n as f64 * simulation_settings.dt,
                    n % simulation_settings.save_interval == 0,
                )
            })
            .collect(),
    };

    let meta_params = SimulationMetaParams {
        n_threads: simulation_settings.n_threads,
        rng_seed: simulation_settings.random_seed,
    };

    let storage = StorageConfig::from_path(std::path::Path::new(&simulation_settings.save_path))
        .add_date(simulation_settings.save_add_date);
    // storage.export_formats = vec![ExportOptions::Vtk];

    let simulation_setup = create_simulation_setup!(
        Domain: domain,
        Cells: cells,
        Time: time,
        MetaParams: meta_params,
        Storage: storage
    );

    let voxel_definition_strategy =
        |voxel: &mut CartesianCuboidVoxel2<NUMBER_OF_REACTION_COMPONENTS>| {
            voxel.diffusion_constant = ReactionVector::from([domain_setup.diffusion_constants]);
            voxel.extracellular_concentrations =
                ReactionVector::from([domain_setup.initial_concentrations]);
            voxel.degradation_rate = ReactionVector::zero();
            voxel.production_rate = ReactionVector::zero();
        };

    let strategies = Strategies {
        voxel_definition_strategies: &voxel_definition_strategy,
    };

    // let simulation_result = run_full_simulation!(setup, settings, [Mechanics, Interaction, Cycle])?;
    let mut supervisor =
        SimulationSupervisor::initialize_with_strategies(simulation_setup, strategies);
    supervisor.config.show_progressbar = simulation_settings.show_progressbar;

    save_simulation_settings(&supervisor.storage.get_location(), &simulation_settings)?;

    let simulation_result = supervisor.run_full_sim().unwrap();

    Ok(simulation_result.storage.get_location())
}
