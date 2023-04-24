use cellular_raza::implementations::cell_models::modular_cell::ModularCell;
use cellular_raza::pipelines::cpu_os_threads::prelude::*;

use nalgebra::Vector2;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// Number of cells to put into simulation in the Beginning
pub const N_CELLS_ORGANOID: u32 = 800;
pub const N_CELLS_SENDER: u32 = 000;

// Mechanical parameters
pub const CELL_1_MECHANICS_RADIUS: f64 = 6.0;
pub const CELL_1_MECHANICS_RELATIVE_INTERACTION_RANGE: f64 = 1.6;
pub const CELL_1_MECHANICS_POTENTIAL_STRENGTH: f64 = 2.0;
pub const CELL_1_MECHANICS_VELOCITY_REDUCTION: f64 = 2.0;

// Reaction parameters of the cell
pub const CELL_1_BMP4_INITIAL_CONCENTRATION: f64 = 0.0;
pub const CELL_1_BMP4_PRODUCTION_RATE: f64 = 0.05;
pub const CELL_1_BMP4_DEGRADATION_RATE: f64 = 0.05;
pub const CELL_1_BMP4_SECRETION_RATE: f64 = 0.1;
pub const CELL_1_BMP4_UPTAKE_RATE: f64 = 0.1;
pub const CELL_1_BMP4_MODIFIER: f64 = 1.0;
pub const CELL_1_BMP4_HILL_COEFFICIENT: f64 = 0.05;

pub const CELL_1_FOOD_INITIAL_CONCENTRATION: f64 = 1.0;
pub const CELL_1_FOOD_SECRETION_RATE: f64 = 0.0;
pub const CELL_1_FOOD_TURNOVER_RATE: f64 = 0.025;
pub const CELL_1_FOOD_UPTAKE_RATE: f64 = 0.05;

// Parameters for cell cycle
pub const CELL_1_CYCLE_DIVISION_AGE_MIN: f64 = 60.0;
pub const CELL_1_CYCLE_DIVISION_AGE_MAX: f64 = 70.0;
pub const CELL_1_CYCLE_GROWTH_RATE: f64 = 0.1;
pub const CELL_1_CYCLE_FOOD_THRESHOLD: f64 = 2.0;
pub const CELL_1_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER: f64 = 10.0;
pub const CELL_1_CYCLE_FOOD_DIVISION_THRESHOLD: f64 = CELL_1_FOOD_INITIAL_CONCENTRATION * 0.8;

// Parameters for second cell type
// Parameters used to contol BMP4
pub const CELL_2_BMP4_SECRETION_RATE: f64 = 0.3;
pub const CELL_2_BMP4_PRODUCTION_RATE: f64 = 0.05;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 10_000.0;
pub const DOMAIN_SIZE_Y: f64 = 10_000.0;

// Where will the cells be placed initially
pub const STARTING_DOMAIN_X_LOW: f64 = DOMAIN_SIZE_X / 2.0 - 250.0;
pub const STARTING_DOMAIN_X_HIGH: f64 = DOMAIN_SIZE_X / 2.0 + 250.0;
pub const STARTING_DOMAIN_Y_LOW: f64 = DOMAIN_SIZE_Y / 2.0 - 250.0;
pub const STARTING_DOMAIN_Y_HIGH: f64 = DOMAIN_SIZE_Y / 2.0 + 250.0;

// Place the sender cells at the bottom of the simulation
pub const STARTING_DOMAIN_X_SENDER_LOW: f64 = DOMAIN_SIZE_X / 2.0 - 500.0;
pub const STARTING_DOMAIN_X_SENDER_HIGH: f64 = DOMAIN_SIZE_X / 2.0 + 500.0;
pub const STARTING_DOMAIN_Y_SENDER_LOW: f64 = DOMAIN_SIZE_Y - 100.0;
pub const STARTING_DOMAIN_Y_SENDER_HIGH: f64 = DOMAIN_SIZE_Y;

// Parameters for Voxel Reaction+Diffusion
pub const VOXEL_BMP4_DEGRADATION_RATE: f64 = 0.1;
pub const VOXEL_BMP4_DIFFUSION_CONSTANT: f64 = 0.0;
pub const VOXEL_BMP4_INITIAL_CONCNENTRATION: f64 = 0.0;

pub const VOXEL_FOOD_PRODUCTION_RATE: f64 = 0.0;
pub const VOXEL_FOOD_DEGRADATION_RATE: f64 = 0.0;
pub const VOXEL_FOOD_DIFFUSION_CONSTANT: f64 = 25.0;
pub const VOXEL_FOOD_INITIAL_CONCENTRATION: f64 = 12.0;
pub const VOXEL_FOOD_INITIAL_DIFFERENCE: f64 = 0.0;

// Time parameters
pub const N_TIMES: usize = 100_001;
pub const DT: f64 = 0.25;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: usize = 1000;
pub const FULL_SAVE_INTERVAL: usize = 1000;

// Meta Parameters to control solving
pub const N_THREADS: usize = 14;
pub const N_PLOTTING_THREADS: usize = 40;

mod cell_properties;
mod plotting;

use cell_properties::*;
use plotting::*;

fn voxel_definition_strategy(voxel: &mut CartesianCuboidVoxel2Reactions2) {
    voxel.diffusion_constant =
        ReactionVector::from([VOXEL_BMP4_DIFFUSION_CONSTANT, VOXEL_FOOD_DIFFUSION_CONSTANT]);
    voxel.extracellular_concentrations = ReactionVector::from([
        VOXEL_BMP4_INITIAL_CONCNENTRATION,
        VOXEL_FOOD_INITIAL_CONCENTRATION,
    ]);
    voxel.degradation_rate =
        ReactionVector::from([VOXEL_BMP4_DEGRADATION_RATE, VOXEL_FOOD_DEGRADATION_RATE]);
    voxel.production_rate = ReactionVector::from([0.0, VOXEL_FOOD_PRODUCTION_RATE]);

    // Increase voxel food from bottom to top
    let middle = voxel.get_middle();
    let t = (middle[0].powf(2.0)+middle[1].powf(2.0)).sqrt()/(DOMAIN_SIZE_X.powf(2.0) + DOMAIN_SIZE_Y.powf(2.0)).sqrt();
    let q = VOXEL_FOOD_INITIAL_DIFFERENCE;
    voxel.extracellular_concentrations[1] *= 1.0 - q + q*t;
}

fn create_domain() -> Result<CartesianCuboid2, CalcError> {
    CartesianCuboid2::from_boundaries_and_interaction_ranges(
        [0.0; 2],
        [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
        [CELL_1_MECHANICS_RADIUS * CELL_1_MECHANICS_RELATIVE_INTERACTION_RANGE; 2],
    )
}

fn main() {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // ###################################### DEFINE SIMULATION DOMAIN ######################################
    let domain = create_domain().unwrap();

    // ###################################### DEFINE CELLS IN SIMULATION ######################################
    let mut cells = (0..N_CELLS_ORGANOID)
        .map(|cell_id| {
            let x = rng.gen_range(STARTING_DOMAIN_X_LOW..STARTING_DOMAIN_X_HIGH);
            let y = rng.gen_range(STARTING_DOMAIN_Y_LOW..STARTING_DOMAIN_Y_HIGH);

            let pos = Vector2::from([x, y]);
            ModularCell {
                mechanics: MechanicsModel2D {
                    pos,
                    vel: Vector2::from([0.0, 0.0]),
                    dampening_constant: CELL_1_MECHANICS_VELOCITY_REDUCTION,
                },
                interaction: CellSpecificInteraction {
                    potential_strength: CELL_1_MECHANICS_POTENTIAL_STRENGTH,
                    relative_interaction_range: CELL_1_MECHANICS_RELATIVE_INTERACTION_RANGE,
                    cell_radius: CELL_1_MECHANICS_RADIUS,
                },
                interaction_extracellular: GradientSensing {},
                cycle: OwnCycle::new(
                    rng.gen_range(0.0..CELL_1_CYCLE_DIVISION_AGE_MAX),
                    CELL_1_MECHANICS_RADIUS,
                    CELL_1_CYCLE_GROWTH_RATE,
                    CELL_1_CYCLE_FOOD_THRESHOLD,
                    CELL_1_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER,
                    CELL_1_CYCLE_FOOD_DIVISION_THRESHOLD,
                    true,
                    cell_id,
                ),
                cellular_reactions: OwnReactions {
                    intracellular_concentrations: ReactionVector::from([
                        0.0,
                        CELL_1_FOOD_INITIAL_CONCENTRATION,
                    ]),
                    turnover_rate: ReactionVector::from([
                        CELL_1_BMP4_DEGRADATION_RATE,
                        CELL_1_FOOD_TURNOVER_RATE,
                    ]),
                    production_term: ReactionVector::from([0.0, 0.0]),
                    degradation_rate: ReactionVector::from([CELL_1_BMP4_DEGRADATION_RATE, 0.0]),
                    secretion_rate: ReactionVector::from([0.0, CELL_1_FOOD_SECRETION_RATE]),
                    uptake_rate: ReactionVector::from([
                        CELL_1_BMP4_UPTAKE_RATE,
                        CELL_1_FOOD_UPTAKE_RATE,
                    ]),

                    bmp4_mod: CELL_1_BMP4_MODIFIER,
                    bmp4_hill: CELL_1_BMP4_HILL_COEFFICIENT,
                },
            }
        })
        .collect::<Vec<_>>();

    let cells_sender = (N_CELLS_ORGANOID..N_CELLS_ORGANOID + N_CELLS_SENDER)
        .map(|cell_id| {
            let x = rng.gen_range(STARTING_DOMAIN_X_SENDER_LOW..STARTING_DOMAIN_X_SENDER_HIGH);
            let y = rng.gen_range(STARTING_DOMAIN_Y_SENDER_LOW..STARTING_DOMAIN_Y_SENDER_HIGH);

            let pos = Vector2::from([x, y]);
            ModularCell {
                mechanics: MechanicsModel2D {
                    pos,
                    vel: Vector2::from([0.0, 0.0]),
                    dampening_constant: CELL_1_MECHANICS_VELOCITY_REDUCTION,
                },
                interaction: CellSpecificInteraction {
                    potential_strength: CELL_1_MECHANICS_POTENTIAL_STRENGTH,
                    relative_interaction_range: CELL_1_MECHANICS_RELATIVE_INTERACTION_RANGE,
                    cell_radius: CELL_1_MECHANICS_RADIUS,
                },
                interaction_extracellular: GradientSensing {},
                cycle: OwnCycle::new(
                    rng.gen_range(0.0..CELL_1_CYCLE_DIVISION_AGE_MAX),
                    CELL_1_MECHANICS_RADIUS,
                    0.0,
                    0.0,
                    0.0,
                    10_000.0,
                    true,
                    cell_id,
                ),
                cellular_reactions: OwnReactions {
                    intracellular_concentrations: ReactionVector::from([
                        0.0,
                        CELL_1_FOOD_INITIAL_CONCENTRATION,
                    ]),
                    turnover_rate: ReactionVector::from([
                        CELL_1_BMP4_DEGRADATION_RATE,
                        CELL_1_FOOD_TURNOVER_RATE,
                    ]),
                    production_term: ReactionVector::from([CELL_2_BMP4_PRODUCTION_RATE, 0.0]),
                    degradation_rate: ReactionVector::from([CELL_1_BMP4_DEGRADATION_RATE, 0.0]),
                    secretion_rate: ReactionVector::from([
                        CELL_2_BMP4_SECRETION_RATE,
                        CELL_1_FOOD_SECRETION_RATE,
                    ]),
                    uptake_rate: ReactionVector::from([
                        CELL_1_BMP4_UPTAKE_RATE,
                        CELL_1_FOOD_UPTAKE_RATE,
                    ]),

                    bmp4_mod: 0.0,
                    bmp4_hill: 1.0,
                },
            }
        })
        .collect::<Vec<_>>();
    cells.extend(cells_sender);

    // ###################################### CREATE SUPERVISOR AND RUN SIMULATION ######################################
    let setup = SimulationSetup {
        domain,
        cells,
        time: TimeSetup {
            t_start: 0.0,
            t_eval: (0..N_TIMES)
                .map(|i| {
                    (
                        T_START + DT * i as f64,
                        i % SAVE_INTERVAL == 0,
                        i % FULL_SAVE_INTERVAL == 0,
                    )
                })
                .collect::<Vec<(f64, bool, bool)>>(),
        },
        meta_params: SimulationMetaParams {
            n_threads: N_THREADS,
        },
        storage: StorageConfig {
            location: "out/organoid_simple".to_owned().into(),
        },
    };

    let strategies = Strategies {
        voxel_definition_strategies: Some(voxel_definition_strategy),
    };

    let mut supervisor = SimulationSupervisor::initialize_with_strategies(setup, strategies);

    let mut simulation_result = supervisor.run_full_sim().unwrap();

    // ###################################### PLOT THE RESULTS ######################################
    simulation_result.plotting_config = PlottingConfig {
        n_threads: Some(N_PLOTTING_THREADS),
        image_size: 1500,
        image_type: ImageType::BitMap,
    };

    simulation_result.plot_spatial_all_iterations_custom_cell_voxel_functions(
        plot_modular_cell,
        plot_voxel
    ).unwrap();
}
