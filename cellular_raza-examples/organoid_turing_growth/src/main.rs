use cellular_raza::prelude::*;

use nalgebra::Vector2;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use plotting::*;

// Number of cells to put into simulation in the Beginning
pub const N_CELLS_ORGANOID: u32 = 1000;
pub const N_CELLS_SENDER: u32 = 0;

// Mechanical parameters
pub const CELL_MECHANICS_RADIUS: f64 = 6.0;
pub const CELL_MECHANICS_RELATIVE_INTERACTION_RANGE: f64 = 1.5;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 2.0;
pub const CELL_MECHANICS_VELOCITY_REDUCTION: f64 = 2.0;

// Reaction parameters of the cell
pub const CELL_BMP4_INITIAL_CONCENTRATION: f64 = 0.0;
pub const CELL_BMP4_PRODUCTION_RATE: f64 = 0.05;
pub const CELL_BMP4_DEGRADATION_RATE: f64 = 0.005;
pub const CELL_BMP4_SATURATION: f64 = CELL_BMP4_PRODUCTION_RATE / CELL_BMP4_DEGRADATION_RATE;
pub const CELL_BMP4_SECRETION_RATE: f64 = 0.1;
pub const CELL_BMP4_UPTAKE_RATE: f64 = 0.1;

pub const CELL_FOOD_INITIAL_CONCENTRATION: f64 = 10.0;
pub const CELL_FOOD_CONSUMPTION_RATE: f64 = 0.1;
pub const CELL_FOOD_SECRETION_RATE: f64 = 0.0001;
pub const CELL_FOOD_SATURATION: f64 = 10.0;
pub const CELL_FOOD_UPTAKE_RATE: f64 = 0.02;

// Parameters for cell turing pattern
pub const CELL_TURING_PATTERN_K1: f64 = 10.0;
pub const CELL_TURING_PATTERN_K2: f64 = 0.1;
pub const CELL_TURING_PATTERN_K3: f64 = 2.938271604938272e-07;
pub const CELL_TURING_PATTERN_K4: f64 = 80.0;

pub const VOXEL_TURING_PATTERN_DIFFUSION_CONSTANT_1: f64 = 50.0;
pub const VOXEL_TURING_PATTERN_DIFFUSION_CONSTANT_2: f64 = 2500.0;

// Parameters for cell cycle
pub const CELL_CYCLE_DIVISION_AGE_MIN: f64 = 60.0;
pub const CELL_CYCLE_DIVISION_AGE_MAX: f64 = 70.0;
pub const CELL_CYCLE_GROWTH_RATE: f64 = 0.2;
pub const CELL_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER: f64 = 0.5;
pub const CELL_CYCLE_FOOD_DEATH_THRESHOLD: f64 = CELL_FOOD_SATURATION * 0.4;
pub const CELL_CYCLE_FOOD_DIVISION_THRESHOLD: f64 = CELL_FOOD_SATURATION * 0.8;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 2500.0;
pub const DOMAIN_SIZE_Y: f64 = 2500.0;

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
pub const VOXEL_BMP4_DEGRADATION_RATE: f64 = 0.2;
pub const VOXEL_BMP4_DIFFUSION_CONSTANT: f64 = 20.0;
pub const VOXEL_BMP4_INITIAL_CONCENTRATION: f64 = 0.0;

pub const VOXEL_FOOD_PRODUCTION_RATE: f64 = 2.5;
pub const VOXEL_FOOD_DEGRADATION_RATE: f64 = 0.05;
pub const VOXEL_FOOD_DIFFUSION_CONSTANT: f64 = 1.0;
pub const VOXEL_FOOD_INITIAL_CONCENTRATION: f64 = 50.0;

// Time parameters
pub const N_TIMES: usize = 100_000;
pub const DT: f64 = 0.01;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: usize = 500;

// Meta Parameters to control solving
pub const N_THREADS: usize = 2;

mod cell_properties;
mod plotting;

use cell_properties::*;

fn voxel_definition_strategy(voxel: &mut CartesianCuboidVoxel2<NUMBER_OF_REACTION_COMPONENTS>) {
    voxel.diffusion_constant = ReactionVector::from([
        VOXEL_BMP4_DIFFUSION_CONSTANT,
        VOXEL_FOOD_DIFFUSION_CONSTANT,
        VOXEL_TURING_PATTERN_DIFFUSION_CONSTANT_1,
        VOXEL_TURING_PATTERN_DIFFUSION_CONSTANT_2,
    ]);
    voxel.extracellular_concentrations = ReactionVector::from([
        VOXEL_BMP4_INITIAL_CONCENTRATION,
        VOXEL_FOOD_INITIAL_CONCENTRATION,
        0.0,
        0.0,
    ]);
    voxel.degradation_rate = ReactionVector::from([
        VOXEL_BMP4_DEGRADATION_RATE,
        VOXEL_FOOD_DEGRADATION_RATE,
        0.0,
        0.0,
    ]);
    voxel.production_rate = ReactionVector::from([0.0, VOXEL_FOOD_PRODUCTION_RATE, 0.0, 0.0]);
}

fn create_domain() -> Result<CartesianCuboid2, CalcError> {
    CartesianCuboid2::from_boundaries_and_interaction_ranges(
        [0.0; 2],
        [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
        [CELL_MECHANICS_RADIUS * CELL_MECHANICS_RELATIVE_INTERACTION_RANGE * 2.0; 2],
    )
}

fn main() {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // ###################################### DEFINE SIMULATION DOMAIN ######################################
    let domain = create_domain().unwrap();

    // ###################################### DEFINE CELLS IN SIMULATION ######################################
    let mut cells = (0..N_CELLS_ORGANOID as i32)
        .map(|_| {
            let x = rng.gen_range(STARTING_DOMAIN_X_LOW..STARTING_DOMAIN_X_HIGH);
            let y = rng.gen_range(STARTING_DOMAIN_Y_LOW..STARTING_DOMAIN_Y_HIGH);

            let pos = Vector2::from([x, y]);
            ModularCell {
                mechanics: NewtonDamped2D {
                    pos,
                    vel: Vector2::from([0.0, 0.0]),
                    damping_constant: CELL_MECHANICS_VELOCITY_REDUCTION,
                    mass: 1.0,
                },
                interaction: CellSpecificInteraction {
                    potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
                    relative_interaction_range: CELL_MECHANICS_RELATIVE_INTERACTION_RANGE,
                    cell_radius: CELL_MECHANICS_RADIUS,
                },
                interaction_extracellular: GradientSensing {},
                cycle: OwnCycle::new(
                    rng.gen_range(0.0..CELL_CYCLE_DIVISION_AGE_MAX),
                    CELL_MECHANICS_RADIUS,
                    CELL_CYCLE_GROWTH_RATE,
                    CELL_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER,
                    CELL_CYCLE_FOOD_DEATH_THRESHOLD,
                    CELL_CYCLE_FOOD_DIVISION_THRESHOLD,
                    true,
                ),
                cellular_reactions: OwnReactions {
                    intracellular_concentrations: ReactionVector::from([
                        0.0,
                        CELL_FOOD_INITIAL_CONCENTRATION,
                        rng.gen_range(200.0..500.0),
                        rng.gen_range(200.0..500.0),
                    ]),
                    intracellular_concentrations_saturation_level: ReactionVector::from([
                        CELL_BMP4_SATURATION,
                        CELL_FOOD_SATURATION,
                        0.0,
                        0.0,
                    ]),
                    production_term: ReactionVector::from([
                        0.0,
                        -CELL_FOOD_CONSUMPTION_RATE,
                        0.0,
                        0.0,
                    ]),
                    degradation_rate: ReactionVector::from([
                        CELL_BMP4_DEGRADATION_RATE,
                        0.0,
                        0.01,
                        0.0,
                    ]),
                    secretion_rate: ReactionVector::from([
                        0.0,
                        CELL_FOOD_SECRETION_RATE,
                        100.0,
                        100.0,
                    ]),
                    uptake_rate: ReactionVector::from([
                        CELL_BMP4_UPTAKE_RATE,
                        CELL_FOOD_UPTAKE_RATE,
                        100.0,
                        100.0,
                    ]),

                    p1: CELL_TURING_PATTERN_K1,
                    p2: CELL_TURING_PATTERN_K2,
                    p3: CELL_TURING_PATTERN_K3,
                    p4: CELL_TURING_PATTERN_K4,

                    bmp4_hill: 0.00,
                },
                volume: 4.0 / 3.0 * std::f64::consts::PI * CELL_MECHANICS_RADIUS.powf(3.0),
            }
        })
        .collect::<Vec<_>>();

    let mut cells2 = (0..N_CELLS_SENDER)
        .map(|_| {
            let x = rng.gen_range(STARTING_DOMAIN_X_SENDER_LOW..STARTING_DOMAIN_X_SENDER_HIGH);
            let y = rng.gen_range(STARTING_DOMAIN_Y_SENDER_LOW..STARTING_DOMAIN_Y_SENDER_HIGH);

            let pos = Vector2::from([x, y]);
            ModularCell {
                mechanics: NewtonDamped2D {
                    pos,
                    vel: Vector2::from([0.0, 0.0]),
                    damping_constant: CELL_MECHANICS_VELOCITY_REDUCTION,
                    mass: 1.0,
                },
                interaction: CellSpecificInteraction {
                    potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
                    relative_interaction_range: CELL_MECHANICS_RELATIVE_INTERACTION_RANGE,
                    cell_radius: CELL_MECHANICS_RADIUS,
                },
                interaction_extracellular: GradientSensing {},
                cycle: OwnCycle::new(
                    rng.gen_range(0.0..CELL_CYCLE_DIVISION_AGE_MAX),
                    CELL_MECHANICS_RADIUS,
                    0.0,
                    CELL_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER,
                    CELL_CYCLE_FOOD_DEATH_THRESHOLD,
                    1000.0,
                    true,
                ),
                cellular_reactions: OwnReactions {
                    intracellular_concentrations: ReactionVector::from([
                        0.0,
                        CELL_FOOD_INITIAL_CONCENTRATION,
                        rng.gen_range(200.0..500.0),
                        rng.gen_range(200.0..500.0),
                    ]),
                    intracellular_concentrations_saturation_level: ReactionVector::from([
                        CELL_BMP4_SATURATION,
                        CELL_FOOD_SATURATION,
                        0.0,
                        0.0,
                    ]),
                    production_term: ReactionVector::from([
                        CELL_BMP4_PRODUCTION_RATE,
                        -CELL_FOOD_CONSUMPTION_RATE,
                        0.0,
                        0.0,
                    ]),
                    degradation_rate: ReactionVector::from([0.0, 0.0, 0.01, 0.0]),
                    secretion_rate: ReactionVector::from([
                        CELL_BMP4_SECRETION_RATE,
                        CELL_FOOD_SECRETION_RATE,
                        0.0,
                        0.0,
                    ]),
                    uptake_rate: ReactionVector::from([
                        CELL_BMP4_UPTAKE_RATE,
                        CELL_FOOD_UPTAKE_RATE,
                        0.0,
                        0.0,
                    ]),

                    p1: 0.0,
                    p2: 0.0,
                    p3: 0.0,
                    p4: 0.0,

                    bmp4_hill: 0.02,
                },
                volume: 4.0 / 3.0 * std::f64::consts::PI * CELL_MECHANICS_RADIUS.powf(3.0),
            }
        })
        .collect::<Vec<_>>();
    cells.append(&mut cells2);

    // ###################################### CREATE SUPERVISOR AND RUN SIMULATION ######################################
    // Solve the initial steps more precisely when the Turint pattern is still forming
    let t_eval = (0..N_TIMES)
        .map(|n| (T_START + n as f64 * DT, n % SAVE_INTERVAL == 0))
        .collect::<Vec<(f64, bool)>>();

    let setup = SimulationSetup::new(
        domain,
        cells,
        TimeSetup {
            t_start: 0.0,
            t_eval,
        },
        SimulationMetaParams {
            n_threads: N_THREADS,
            ..Default::default()
        },
        StorageBuilder::new().location("out/organoid_turing_growth"),
        (),
    );

    let strategies = Strategies {
        voxel_definition_strategies: &voxel_definition_strategy,
    };

    let mut supervisor = SimulationSupervisor::initialize_with_strategies(setup, strategies);

    let mut simulation_result = supervisor.run_full_sim().unwrap();

    // ###################################### PLOT THE RESULTS ######################################
    simulation_result.plotting_config = PlottingConfig {
        n_threads: Some(N_THREADS),
        image_size: 1500,
        image_type: ImageType::BitMap,
        ..Default::default()
    };

    simulation_result
        .plot_spatial_all_iterations_custom_cell_voxel_functions(&plot_modular_cell, &plot_voxel)
        .unwrap();
}
