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
    pub n_bacteria_initial: usize,
    pub bacteria_mechanics: Py<Langevin2D>,
    pub bacteria_interaction: Py<BacteriaInteraction>,
    pub bacteria_cycle: Py<BacteriaCycle>,
    pub bacteria_reactions: Py<BacteriaReactions>,

    // SIMULATION FLOW SETTINGS
    pub n_times: usize,
    pub dt: f64,
    pub t_start: f64,
    pub save_interval: usize,
    pub n_threads: usize,
}

#[pymethods]
impl SimulationSettings {
    #[new]
    fn new(py: Python) -> PyResult<Self> {
        let domain_size = 2_000.0;
        let bacteria_radius: f64 = 6.0;
        let bacteria_volume = std::f64::consts::PI * bacteria_radius.powf(2.0);
        let dt = 0.01;
        Ok(Self {
            random_seed: 1,

            // DOMAIN SETTINGS
            domain: Py::new(
                py,
                Domain {
                    diffusion_constants: [20.0, 15.0],
                    initial_concentrations: [12.0, 0.0],

                    size: domain_size,
                    n_voxels: None,
                },
            )?,

            // BACTERIA SETTINGS
            n_bacteria_initial: 400,

            bacteria_mechanics: Py::new(
                py,
                Langevin2D::new(
                    [0.0; 2], // pos
                    [0.0; 2], // vel
                    // For this field also see the dedicated [volume_to_mass] function!
                    0.1 * bacteria_volume, // mass
                    0.5,                   // damping
                    0.01,                  // kb_temperature
                    5,                     // update_interval
                ),
            )?,

            bacteria_interaction: Py::new(
                py,
                BacteriaInteraction {
                    potential_strength: 0.02,
                    cell_radius: bacteria_radius,
                },
            )?,

            bacteria_cycle: Py::new(
                py,
                BacteriaCycle {
                    food_to_volume_conversion: 0.001,
                    volume_division_threshold: 2.0 * bacteria_volume,
                    lag_phase_transition_rate: 0.0005,
                },
            )?,

            bacteria_reactions: Py::new(
                py,
                BacteriaReactions {
                    lag_phase_active: true,
                    intracellular_concentrations: [1.0; NUMBER_OF_REACTION_COMPONENTS].into(),
                    production_rates: [0.0, 0.01].into(),
                    uptake_rates: [0.001, 0.002].into(),
                    inhibitions: [0.0, 0.1].into(),
                },
            )?,

            /* bacteria_cycle_division_age_max: 70.0,
            bacteria_cycle_growth_rate: 0.1,
            bacteria_cycle_food_threshold: 2.0,
            bacteria_cycle_food_growth_rate_multiplier: 10.0,
            bacteria_cycle_food_division_threshold: bacteria_food_initial_concentration * 0.8,

            bacteria_food_initial_concentration,
            bacteria_food_turnover_rate: 0.025,
            bacteria_food_uptake_rate: 0.05,

            intracellular_concentrations: [1.0; NUMBER_OF_REACTION_COMPONENTS],
            turnover_rate: [0.025; NUMBER_OF_REACTION_COMPONENTS],
            production_term: [0.0; NUMBER_OF_REACTION_COMPONENTS],
            degradation_rate: [0.0; NUMBER_OF_REACTION_COMPONENTS],
            secretion_rate: [0.0; NUMBER_OF_REACTION_COMPONENTS],
            uptake_rate: [0.05; NUMBER_OF_REACTION_COMPONENTS],
            */
            // SIMULATION FLOW SETTINGS
            n_times: 20_001,
            dt,
            t_start: 0.0,
            save_interval: 250,
            n_threads: 1,
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
    let mechanics: Langevin2D = simulation_settings.bacteria_mechanics.extract(py)?;
    let cycle: BacteriaCycle = simulation_settings.bacteria_cycle.extract(py)?;
    let interaction: BacteriaInteraction = simulation_settings.bacteria_interaction.extract(py)?;
    let cellular_reactions: BacteriaReactions =
        simulation_settings.bacteria_reactions.extract(py)?;
    let domain_setup: Domain = simulation_settings.domain.extract(py)?;

    let cells = (0..simulation_settings.n_bacteria_initial)
        .map(|_| {
            let x = rng.gen_range(0.0..domain_setup.size);
            let y = rng.gen_range(0.0..domain_setup.size);

            // Set new position of bacteria
            let pos = Vector2::from([x, y]);
            let mut mechanics_new = mechanics.clone();
            mechanics_new.set_pos(&pos);
            Ok(Bacteria {
                mechanics: mechanics_new,
                cycle: cycle.clone(),
                interaction: interaction.clone(),
                cellular_reactions: cellular_reactions.clone(),
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

    let storage = StorageConfig::from_path(std::path::Path::new("out/pool_model"));
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

    save_simulation_settings(&supervisor.storage.get_location(), &simulation_settings)?;

    let simulation_result = supervisor.run_full_sim().unwrap();

    Ok(simulation_result.storage.get_location())
}
