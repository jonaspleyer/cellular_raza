use cellular_raza::pipelines::cpu_os_threads::prelude::*;
use cellular_raza::implementations::cell_models::modular_cell::ModularCell;

use nalgebra::{Unit,Vector2};

use rand_chacha::ChaCha8Rng;
use rand::{SeedableRng,Rng};


// Number of cells to put into simulation in the Beginning
pub const N_CELLS: u32 = 200;

// Mechanical parameters
pub const CELL_MECHANICS_RADIUS: f64 = 6.0;
pub const CELL_MECHANICS_RELATIVE_INTERACTION_RANGE: f64 = 1.15;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 0.5;
pub const CELL_MECHANICS_VELOCITY_REDUCTION: f64 = 2.0;
pub const CELL_MECHANICS_ATTRACTION_MULTIPLIER: f64 = 0.5;

// Reaction parameters of the cell
pub const CELL_SPATIAL_SIGNALLING_MOLECULE_INITIAL_CONCENTRATION: f64 = 10.0;
pub const CELL_SPATIAL_SIGNALLING_MOLECULE_SATURATION: f64 = 10.0;
pub const CELL_SPATIAL_SIGNALLING_MOLECULE_PRODUCTION_RATE: f64 = 0.05;
pub const CELL_SPATIAL_SIGNALLING_MOLECULE_DEGRADATION_RATE: f64 = 0.004;
pub const CELL_SPATIAL_SIGNALLING_MOLECULE_SECRETION_RATE: f64 = 0.1;
pub const CELL_SPATIAL_SIGNALLING_MOLECULE_UPTAKE_RATE: f64 = 0.0;

pub const CELL_FOOD_INITIAL_CONCENTRATION: f64 = 20.0;
pub const CELL_FOOD_SATURATION: f64 = 20.0;
pub const CELL_FOOD_CONSUMPTION_RATE: f64 = 0.02;
pub const CELL_FOOD_SECRETION_RATE: f64 = 0.0001;
pub const CELL_FOOD_UPTAKE_RATE: f64 = 0.01;

// Parameters for cell turing pattern
pub const CELL_TURING_PATTERN_K1: f64 = 10.0;
pub const CELL_TURING_PATTERN_K2: f64 =  0.1;
pub const CELL_TURING_PATTERN_K3: f64 = 2.938271604938272e-07;
pub const CELL_TURING_PATTERN_K4: f64 = 80.0;

pub const VOXEL_TURING_PATTERN_DIFFUSION_CONSTANT_1: f64 =   50.0;
pub const VOXEL_TURING_PATTERN_DIFFUSION_CONSTANT_2: f64 = 2500.0;

// Parameters for cell cycle
pub const CELL_CYCLE_DIVISION_AGE_MIN: f64 = 60.0;
pub const CELL_CYCLE_DIVISION_AGE_MAX: f64 = 70.0;
pub const CELL_CYCLE_GROWTH_RATE: f64 = 0.3;
pub const CELL_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER: f64 = 2.0;
pub const CELL_CYCLE_FOOD_DEATH_THRESHOLD: f64 = CELL_FOOD_SATURATION * 0.4;
pub const CELL_CYCLE_FOOD_DIVISION_THRESHOLD: f64 = CELL_FOOD_SATURATION * 0.6;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 1200.0;
pub const DOMAIN_SIZE_Y: f64 = 1200.0;

// Where will the cells be placed initially
// Define a polygon by points
pub const STARTING_DOMAIN_X_LOW:  f64 = DOMAIN_SIZE_X/2.0 - 80.0;
pub const STARTING_DOMAIN_X_HIGH: f64 = DOMAIN_SIZE_X/2.0 + 80.0;
pub const STARTING_DOMAIN_Y_LOW:  f64 = DOMAIN_SIZE_Y/2.0 - 80.0;
pub const STARTING_DOMAIN_Y_HIGH: f64 = DOMAIN_SIZE_Y/2.0 + 80.0;

// Parameters for Voxel Reaction+Diffusion
pub const VOXEL_SPATIAL_SIGNALLING_MOLECULE_DEGRADATION_RATE: f64 = 0.003;
pub const VOXEL_SPATIAL_SIGNALLING_MOLECULE_DIFFUSION_CONSTANT: f64 = 75.0;
pub const VOXEL_SPATIAL_SIGNALLING_MOLECULE_INITIAL_CONCNENTRATION: f64 = 0.0;

pub const VOXEL_FOOD_PRODUCTION_RATE: f64 = 0.3;
pub const VOXEL_FOOD_DEGRADATION_RATE: f64 = 0.003;
pub const VOXEL_FOOD_DIFFUSION_CONSTANT: f64 = 0.0;
pub const VOXEL_FOOD_INITIAL_CONCENTRATION: f64 = 60.0;

// Time parameters
pub const N_TIMES: usize = 40_001;
pub const DT: f64 = 0.02;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: usize = 40;
pub const FULL_SAVE_INTERVAL: usize = 40;

// Meta Parameters to control solving
pub const N_THREADS: usize = 8;


mod plotting;
mod cell_properties;

use plotting::*;
use cell_properties::*;


fn voxel_definition_strategy(voxel: &mut CartesianCuboidVoxel2Reactions4) {
    voxel.diffusion_constant = ReactionVector::from([
        VOXEL_SPATIAL_SIGNALLING_MOLECULE_DIFFUSION_CONSTANT,
        VOXEL_FOOD_DIFFUSION_CONSTANT,
        VOXEL_TURING_PATTERN_DIFFUSION_CONSTANT_1,
        VOXEL_TURING_PATTERN_DIFFUSION_CONSTANT_2
    ]);
    voxel.extracellular_concentrations = ReactionVector::from([
        VOXEL_SPATIAL_SIGNALLING_MOLECULE_INITIAL_CONCNENTRATION,
        VOXEL_FOOD_INITIAL_CONCENTRATION,
        0.0,
        0.0
    ]);
    voxel.degradation_rate = ReactionVector::from([
        VOXEL_SPATIAL_SIGNALLING_MOLECULE_DEGRADATION_RATE,
        VOXEL_FOOD_DEGRADATION_RATE,
        0.0,
        0.0,
    ]);
    voxel.production_rate = ReactionVector::from([
        0.0,
        VOXEL_FOOD_PRODUCTION_RATE,
        0.0,
        0.0,
    ]);
}


fn main() {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // ###################################### DEFINE SIMULATION DOMAIN ######################################
    // Define the simulation domain
    let domain = CartesianCuboid2::from_boundaries_and_interaction_ranges(
        [0.0; 2],
        [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
        [CELL_MECHANICS_RELATIVE_INTERACTION_RANGE * CELL_MECHANICS_RADIUS * 2.0; 2],
    ).unwrap();

    // ###################################### DEFINE CELLS IN SIMULATION ######################################
    // Cells of Type 1
    let cells = (0..N_CELLS as i32).map(|n_cell| {
        let pos = Vector2::<f64>::from([
            rng.gen_range(STARTING_DOMAIN_X_LOW..STARTING_DOMAIN_X_HIGH),
            rng.gen_range(STARTING_DOMAIN_Y_LOW..STARTING_DOMAIN_Y_HIGH)]);
        ModularCell {
        mechanics: cellular_raza::implementations::cell_models::modular_cell::MechanicsOptions::Mechanics(MechanicsModel2D {
            pos,
            vel: Vector2::from([0.0, 0.0]),
            dampening_constant: CELL_MECHANICS_VELOCITY_REDUCTION,
        }),
        interaction: CellSpecificInteraction {
            potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
            attraction_multiplier: CELL_MECHANICS_ATTRACTION_MULTIPLIER,
            relative_interaction_range: CELL_MECHANICS_RELATIVE_INTERACTION_RANGE,
            cell_radius: CELL_MECHANICS_RADIUS,
            orientation: Unit::<Vector2<f64>>::new_normalize(Vector2::<f64>::from([0.0, 1.0])),
            polarity: n_cell,
        },
        interaction_extracellular: GradientSensing {},
        cycle: OwnCycle::new(
            rng.gen_range(CELL_CYCLE_DIVISION_AGE_MIN..CELL_CYCLE_DIVISION_AGE_MAX),
            CELL_MECHANICS_RADIUS,
            CELL_CYCLE_GROWTH_RATE,
            CELL_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER,
            CELL_CYCLE_FOOD_DEATH_THRESHOLD,
            CELL_CYCLE_FOOD_DIVISION_THRESHOLD
        ),
        cellular_reactions: OwnReactions {
            intracellular_concentrations: ReactionVector::from([
                CELL_SPATIAL_SIGNALLING_MOLECULE_INITIAL_CONCENTRATION,
                CELL_FOOD_INITIAL_CONCENTRATION,
                rng.gen_range(200.0..500.0),
                rng.gen_range(200.0..500.0)
            ]),
            intracellular_concentrations_saturation_level: ReactionVector::from([
                CELL_SPATIAL_SIGNALLING_MOLECULE_SATURATION,
                CELL_FOOD_SATURATION,
                0.0,
                0.0
            ]),
            production_term: ReactionVector::from([
                CELL_SPATIAL_SIGNALLING_MOLECULE_PRODUCTION_RATE,
                -CELL_FOOD_CONSUMPTION_RATE,
                0.0,
                0.0,
            ]),
            degradation_rate: ReactionVector::from([
                CELL_SPATIAL_SIGNALLING_MOLECULE_DEGRADATION_RATE,
                0.0,
                0.0,
                0.0,
            ]),
            secretion_rate: ReactionVector::from([
                CELL_SPATIAL_SIGNALLING_MOLECULE_SECRETION_RATE,
                CELL_FOOD_SECRETION_RATE,
                0.0,
                0.0
            ]),
            uptake_rate: ReactionVector::from([
                CELL_SPATIAL_SIGNALLING_MOLECULE_UPTAKE_RATE,
                CELL_FOOD_UPTAKE_RATE,
                0.0,
                0.0
            ]),
            p1: CELL_TURING_PATTERN_K1,
            p2: CELL_TURING_PATTERN_K2,
            p3: CELL_TURING_PATTERN_K3,
            p4: CELL_TURING_PATTERN_K4,
        },
    }}).collect::<Vec<_>>();

    // ###################################### CREATE SUPERVISOR AND RUN SIMULATION ######################################
    let setup = SimulationSetup {
        domain,
        cells,
        time: TimeSetup {
            t_start: 0.0,
            t_eval: (0..N_TIMES).map(|i| (T_START + DT * i as f64, i % SAVE_INTERVAL == 0, i % FULL_SAVE_INTERVAL == 0)).collect::<Vec<(f64, bool, bool)>>(),
        },
        meta_params: SimulationMetaParams {
            n_threads: N_THREADS
        },
        database: SledDataBaseConfig {
            name: "out/simulation_custom_cells".to_owned().into(),
        }
    };

    let strategies = Strategies {
        voxel_definition_strategies: Some(voxel_definition_strategy),
    };

    let mut supervisor = SimulationSupervisor::initialize_with_strategies(setup, strategies);

    supervisor.run_full_sim().unwrap();

    supervisor.end_simulation().unwrap();

    supervisor.plotting_config = PlottingConfig {
        n_threads: Some(20),
        image_size: 1500,
    };

    // ###################################### PLOT THE RESULTS ######################################
    // supervisor.plot_cells_at_every_iter_bitmap_with_cell_plotting_func(&plot_modular_cell).unwrap();
    supervisor.plot_cells_at_every_iter_bitmap_with_cell_plotting_func_and_voxel_plotting_func(&plot_modular_cell, &plot_voxel).unwrap();
}
