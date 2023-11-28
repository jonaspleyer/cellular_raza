use cellular_raza::prelude::*;

use nalgebra::Vector2;

use num::Zero;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use serde::{Deserialize, Serialize};

use pyo3::prelude::*;

use crate::bacteria_properties::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass]
pub struct SimulationSettings {
    // DOMAIN SETTINGS
    #[pyo3(get, set)]
    pub voxel_food_diffusion_constant: f64,
    #[pyo3(get, set)]
    pub voxel_food_initial_concentration: f64,

    #[pyo3(get, set)]
    pub domain_size: f64,

    #[pyo3(get, set)]
    pub starting_domain_x_low: f64,
    #[pyo3(get, set)]
    pub starting_domain_x_high: f64,
    #[pyo3(get, set)]
    pub starting_domain_y_low: f64,
    #[pyo3(get, set)]
    pub starting_domain_y_high: f64,

    // BACTERIA SETTINGS
    #[pyo3(get, set)]
    pub n_bacteria_initial: usize,

    #[pyo3(get, set)]
    pub bacteria_mechanics_velocity_reduction: f64,
    #[pyo3(get, set)]
    pub bacteria_mechanics_radius: f64,

    #[pyo3(get, set)]
    pub bacteria_interaction_potential_strength: f64,
    #[pyo3(get, set)]
    pub bacteria_interaction_relative_range: f64,

    #[pyo3(get, set)]
    pub bacteria_cycle_division_age_max: f64,
    #[pyo3(get, set)]
    pub bacteria_cycle_growth_rate: f64,
    #[pyo3(get, set)]
    pub bacteria_cycle_food_threshold: f64,
    #[pyo3(get, set)]
    pub bacteria_cycle_food_growth_rate_multiplier: f64,
    #[pyo3(get, set)]
    pub bacteria_cycle_food_division_threshold: f64,

    #[pyo3(get, set)]
    pub bacteria_food_initial_concentration: f64,
    #[pyo3(get, set)]
    pub bacteria_food_turnover_rate: f64,
    #[pyo3(get, set)]
    pub bacteria_food_uptake_rate: f64,

    #[pyo3(get, set)]
    pub intracellular_concentrations: [f64; NUMBER_OF_REACTION_COMPONENTS],
    #[pyo3(get, set)]
    pub turnover_rate: [f64; NUMBER_OF_REACTION_COMPONENTS],
    #[pyo3(get, set)]
    pub production_term: [f64; NUMBER_OF_REACTION_COMPONENTS],
    #[pyo3(get, set)]
    pub degradation_rate: [f64; NUMBER_OF_REACTION_COMPONENTS],
    #[pyo3(get, set)]
    pub secretion_rate: [f64; NUMBER_OF_REACTION_COMPONENTS],
    #[pyo3(get, set)]
    pub uptake_rate: [f64; NUMBER_OF_REACTION_COMPONENTS],

    // SIMULATION FLOW SETTINGS
    #[pyo3(get, set)]
    pub n_times: usize,
    #[pyo3(get, set)]
    pub dt: f64,
    #[pyo3(get, set)]
    pub t_start: f64,
    #[pyo3(get, set)]
    pub save_interval: usize,
    #[pyo3(get, set)]
    pub n_threads: usize,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        let domain_size = 3_000.0;
        let bacteria_food_initial_concentration = 1.0;
        Self {
            // DOMAIN SETTINGS
            voxel_food_diffusion_constant: 25.0,
            voxel_food_initial_concentration: 12.0,

            domain_size,

            starting_domain_x_low: domain_size / 2.0 - 150.0,
            starting_domain_x_high: domain_size / 2.0 + 150.0,
            starting_domain_y_low: domain_size / 2.0 - 150.0,
            starting_domain_y_high: domain_size / 2.0 + 150.0,

            // BACTERIA SETTINGS
            n_bacteria_initial: 400,

            bacteria_mechanics_velocity_reduction: 2.0,
            bacteria_mechanics_radius: 6.0,

            bacteria_interaction_potential_strength: 0.3,
            bacteria_interaction_relative_range: 1.5,

            bacteria_cycle_division_age_max: 70.0,
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

            // SIMULATION FLOW SETTINGS
            n_times: 20_001,
            dt: 0.01,
            t_start: 0.0,
            save_interval: 250,
            n_threads: 1,
        }
    }
}

#[pymethods]
impl SimulationSettings {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

pub fn run_simulation_rs(
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

    simulation_settings: SimulationSettings,
) -> Result<std::path::PathBuf, SimulationError> {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // ###################################### DEFINE CELLS IN SIMULATION ######################################
    let cells = (0..simulation_settings.n_bacteria_initial)
        .map(|_| {
            let x = rng.gen_range(
                simulation_settings.starting_domain_x_low
                    ..simulation_settings.starting_domain_x_high,
            );
            let y = rng.gen_range(
                simulation_settings.starting_domain_y_low
                    ..simulation_settings.starting_domain_y_high,
            );

            let pos = Vector2::from([x, y]);
            Bacteria {
                mechanics: BacteriaMechanicsModel2D {
                    pos,
                    vel: Vector2::zero(),
                    dampening_constant: simulation_settings.bacteria_mechanics_velocity_reduction,
                    mass: 1.0,
                },
                cycle: BacteriaCycle::new(
                    rng.gen_range(0.0..simulation_settings.bacteria_cycle_division_age_max),
                    simulation_settings.bacteria_mechanics_radius,
                    simulation_settings.bacteria_cycle_growth_rate,
                    simulation_settings.bacteria_cycle_food_threshold,
                    simulation_settings.bacteria_cycle_food_growth_rate_multiplier,
                    simulation_settings.bacteria_cycle_food_division_threshold,
                ),
                interaction: BacteriaInteraction {
                    potential_strength: simulation_settings.bacteria_interaction_potential_strength,
                    relative_interaction_range: simulation_settings
                        .bacteria_interaction_relative_range,
                    cell_radius: simulation_settings.bacteria_mechanics_radius,
                },
                cellular_reactions: BacteriaReactions {
                    intracellular_concentrations: simulation_settings
                        .intracellular_concentrations
                        .into(),
                    turnover_rate: simulation_settings.turnover_rate.into(),
                    production_term: simulation_settings.production_term.into(),
                    degradation_rate: simulation_settings.degradation_rate.into(),
                    secretion_rate: simulation_settings.secretion_rate.into(),
                    uptake_rate: simulation_settings.uptake_rate.into(),
                },
                interactionextracellulargradient: NoExtracellularGradientSensing,
            }
        })
        .collect::<Vec<_>>();

    // ###################################### CREATE SUPERVISOR AND RUN SIMULATION ######################################
    let domain = CartesianCuboid2::from_boundaries_and_n_voxels(
        [0.0; 2],
        [simulation_settings.domain_size; 2],
        [3; 2],
    )?;

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
            voxel.diffusion_constant =
                ReactionVector::from([simulation_settings.voxel_food_diffusion_constant]);
            voxel.extracellular_concentrations =
                ReactionVector::from([simulation_settings.voxel_food_initial_concentration]);
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
