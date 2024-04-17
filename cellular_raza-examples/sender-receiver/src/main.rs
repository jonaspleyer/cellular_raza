use cellular_raza::prelude::*;

use nalgebra::Vector2;

use num::Zero;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ###########
// ## UNITS ##
// ###########

pub const MICRO_METRE: f64 = 1.0;
pub const MINUTE: f64 = 60.0;
pub const SECOND: f64 = 1.0;

// Number of cells to put into simulation in the Beginning
pub const N_CELLS_INITIAL_SENDER: u32 = 50;
pub const N_CELLS_INITIAL_RECEIVER: u32 = 50;

// Mechanical parameters
pub const CELL_MECHANICS_RADIUS: f64 = 12.0 * MICRO_METRE;
pub const CELL_MECHANICS_RELATIVE_INTERACTION_RANGE: f64 = 1.6;
// TODO this value probably needs to be adjusted :shrug:
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 =
    2.0 * MICRO_METRE * MICRO_METRE / (MINUTE * MINUTE);
pub const CELL_MECHANICS_DAMPING: f64 = 2.0 / MINUTE;

// Reaction parameters of the cell
pub const CELL_LIGAND_TURNOVER_RATE: f64 = 0.025 / MINUTE;
pub const CELL_LIGAND_UPTAKE_RATE: f64 = 0.05 / MINUTE;

// Parameters for domain
pub const DOMAIN_SIZE: f64 = 300.0;

// Parameters for Voxel Reaction+Diffusion
pub const VOXEL_LIGAND_DIFFUSION_CONSTANT: f64 = 4.0 * MICRO_METRE * MICRO_METRE / SECOND;

// Controller parameters
pub const TARGET_AVERAGE_CONC: f64 = 2.0;

// Time parameters
pub const DT: f64 = 0.5 * SECOND;
pub const T_START: f64 = 0.0 * SECOND;
pub const T_END: f64 = 50.0 * MINUTE;
pub const SAVE_INTERVAL: usize = 250;

// Meta Parameters to control solving
pub const N_THREADS: usize = 1;
pub const N_PLOTTING_THREADS: usize = 14;

mod bacteria_properties;
mod plotting;

use bacteria_properties::*;
use plotting::*;

fn voxel_definition_strategy(voxel: &mut CartesianCuboidVoxel2<NUMBER_OF_REACTION_COMPONENTS>) {
    voxel.diffusion_constant = [VOXEL_LIGAND_DIFFUSION_CONSTANT].into();
    voxel.extracellular_concentrations = [0.0].into();
    voxel.degradation_rate = ReactionVector::zero();
    voxel.production_rate = ReactionVector::zero();
}

fn create_domain() -> Result<CartesianCuboid2, CalcError> {
    CartesianCuboid2::from_boundaries_and_interaction_ranges(
        [0.0; 2],
        [DOMAIN_SIZE, DOMAIN_SIZE],
        [CELL_MECHANICS_RADIUS * CELL_MECHANICS_RELATIVE_INTERACTION_RANGE; 2],
    )
}

fn main() -> Result<(), SimulationError> {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // ###########################
    // ## DEFINE DOMAIN & CELLS ##
    // ###########################
    let domain = create_domain().unwrap();
    let cells = (0..N_CELLS_INITIAL_SENDER + N_CELLS_INITIAL_RECEIVER)
        .map(|n_cell| {
            let y = DOMAIN_SIZE * rng.gen_range(0.3..0.7);
            let (species, x) = if n_cell < N_CELLS_INITIAL_SENDER {
                let x = DOMAIN_SIZE * rng.gen_range(0.0..0.2);
                (Species::Sender, x)
            } else {
                let x = DOMAIN_SIZE * rng.gen_range(0.8..1.0);
                (Species::Receiver, x)
            };

            let pos = Vector2::from([x, y]);
            Ok(ModularCell {
                mechanics: NewtonDamped2D {
                    pos,
                    vel: Vector2::zero(),
                    damping_constant: CELL_MECHANICS_DAMPING,
                    mass: 1.0,
                },
                interaction: MiePotential::<3, 1>::new(
                    CELL_MECHANICS_RADIUS,
                    CELL_MECHANICS_POTENTIAL_STRENGTH,
                    1.5 * CELL_MECHANICS_POTENTIAL_STRENGTH,
                    CELL_MECHANICS_RELATIVE_INTERACTION_RANGE * CELL_MECHANICS_RADIUS,
                )?,
                interaction_extracellular: NoExtracellularGradientSensing,
                cycle: NoCycle,
                cellular_reactions: OwnReactions {
                    species,
                    intracellular_concentrations: [0.0].into(),
                    turnover_rate: [CELL_LIGAND_TURNOVER_RATE].into(),
                    production_term: ReactionVector::zero(),
                    secretion_rate: ReactionVector::zero(),
                    uptake_rate: [CELL_LIGAND_UPTAKE_RATE].into(),
                },
                volume: 2.0 * std::f64::consts::PI * CELL_MECHANICS_RADIUS.powf(2.0),
            })
        })
        .collect::<Result<Vec<_>, CalcError>>()?;

    // ##########################################
    // ## CREATE SUPERVISOR AND RUN SIMULATION ##
    // ##########################################
    let n_times = ((T_END - T_START) / DT).ceil() as usize + 1;
    let setup = SimulationSetup::new(
        domain,
        cells,
        TimeSetup {
            t_start: 0.0,
            t_eval: (0..n_times)
                .map(|i| (T_START + DT * i as f64, i % SAVE_INTERVAL == 0))
                .collect::<Vec<(f64, bool)>>(),
        },
        SimulationMetaParams {
            n_threads: N_THREADS,
            ..Default::default()
        },
        StorageBuilder::new().location("out/bacteria_population"),
        ConcentrationController {
            target_average_conc: TARGET_AVERAGE_CONC,
            k_p: 0.01,
            t_d: 1.0 * MINUTE,
            t_i: 20.0 * MINUTE,
            previous_values: vec![],
        },
    );

    let strategies = Strategies {
        voxel_definition_strategies: &voxel_definition_strategy,
    };

    let mut supervisor = SimulationSupervisor::initialize_with_strategies(setup, strategies);

    let mut simulation_result = supervisor.run_full_sim()?;

    // ######################
    // ## PLOT THE RESULTS ##
    // ######################
    simulation_result.plotting_config = PlottingConfig {
        n_threads: Some(N_PLOTTING_THREADS),
        image_size: 1500,
        image_type: ImageType::BitMap,
        ..Default::default()
    };

    simulation_result
        .plot_spatial_all_iterations_custom_cell_voxel_functions(plot_modular_cell, plot_voxel)?;
    Ok(())
}
