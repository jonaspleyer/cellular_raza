use cellular_raza::prelude::*;
use cellular_raza::impls_cell_models::modular_cell::ModularCell;

use nalgebra::{Unit,Vector2};

use rand_chacha::ChaCha8Rng;
use rand::{SeedableRng,Rng};


// Number of cells to put into simulation in the Beginning
pub const N_CELLS: u32 = 10;

// Mechanical parameters
pub const CELL_RADIUS: f64 = 20.0;
pub const CELL_RELATIVE_INTERACTION_RANGE: f64 = 1.75;
pub const CELL_POTENTIAL_STRENGTH: f64 = 0.5;
pub const CELL_VELOCITY_REDUCTION: f64 = 2.0;
pub const CELL_VELOCITY_REDUCTION_MAX: f64 = 20.0;
pub const CELL_VELOCITY_REDUCTION_RATE: f64 = 5e-4;
pub const ATTRACTION_MULTIPLIER: f64 = 2.0;

// Reactin parameters of the cell
pub const CELL_STUFF_INITIAL_CONCENTRATION: f64 = 0.0;
pub const CELL_STUFF_PRODUCTION_RATE: f64 = 0.005;
pub const CELL_STUFF_SECRETION_RATE: f64 = 0.001;
pub const CELL_STUFF_UPTAKE_RATE: f64 = 0.0;

// Parameters for cell cycle
pub const DIVISION_AGE_MIN: f64 = 45.0;
pub const DIVISION_AGE_MAX: f64 = 55.0;
pub const CELL_GROWTH_RATE: f64 = 0.3;
pub const CELL_GENERATION_BRANCHING_EVENT: i32 = 50;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 5000.0;
pub const DOMAIN_SIZE_Y: f64 = 5000.0;

// Where will the cells be placed initially
pub const STARTING_DOMAIN_X_LOW:  f64 = DOMAIN_SIZE_X/2.0 - 60.0;
pub const STARTING_DOMAIN_X_HIGH: f64 = DOMAIN_SIZE_X/2.0 + 60.0;
pub const STARTING_DOMAIN_Y_LOW:  f64 = 0.0;
pub const STARTING_DOMAIN_Y_HIGH: f64 = 60.0;

// Parameters for Voxel Reaction+Diffusion
pub const VOXEL_DEGRADATION_RATE: f64 = 0.003;
pub const VOXEL_DIFFUSION_CONSTANT: f64 = 100.0;
pub const VOXEL_INITIAL_CONCNENTRATION: f64 = 0.0;

// Time parameters
pub const N_TIMES: usize = 20_001;
pub const DT: f64 = 0.5;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: usize = 40;
pub const FULL_SAVE_INTERVAL: usize = 40;

// Meta Parameters to control solving
pub const N_THREADS: usize = 14;


mod plotting;
mod cell_properties;

use plotting::*;
use cell_properties::*;


fn voxel_definition_strategy(voxel: &mut CartesianCuboidVoxel2Reactions1) {
    voxel.diffusion_constant = nalgebra::SVector::<f64, 1>::from_element(VOXEL_DIFFUSION_CONSTANT);
    voxel.extracellular_concentrations = nalgebra::SVector::<f64, 1>::from_element(VOXEL_INITIAL_CONCNENTRATION);
    voxel.degradation_rate = nalgebra::SVector::<f64, 1>::from_element(VOXEL_DEGRADATION_RATE);
}


fn main() {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // ###################################### DEFINE SIMULATION DOMAIN ######################################
    // Define the simulation domain
    let domain = CartesianCuboid2::from_boundaries_and_interaction_ranges(
        [0.0; 2],
        [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
        [CELL_RELATIVE_INTERACTION_RANGE * CELL_RADIUS; 2],
    ).unwrap();

    // ###################################### DEFINE CELLS IN SIMULATION ######################################
    // Cells of Type 1
    let cells = (0..N_CELLS as i32).map(|n_cell| {
        let pos = Vector2::<f64>::from([
            rng.gen_range(STARTING_DOMAIN_X_LOW..STARTING_DOMAIN_X_HIGH),
            rng.gen_range(STARTING_DOMAIN_Y_LOW..STARTING_DOMAIN_Y_HIGH)]);
        ModularCell {
        mechanics: cellular_raza::impls_cell_models::modular_cell::MechanicsOptions::Mechanics(MechanicsModel2D {
            pos,
            vel: Vector2::from([0.0, 0.0]),
            dampening_constant: CELL_VELOCITY_REDUCTION,
        }),
        interaction: CellSpecificInteraction {
            potential_strength: CELL_POTENTIAL_STRENGTH,
            attraction_multiplier: ATTRACTION_MULTIPLIER,
            relative_interaction_range: CELL_RELATIVE_INTERACTION_RANGE,
            cell_radius: CELL_RADIUS,
            orientation: Unit::<Vector2<f64>>::new_normalize(Vector2::<f64>::from([1.0, 0.0])),
            polarity: n_cell,
        },
        cycle: OwnCycle::new(
            rng.gen_range(DIVISION_AGE_MIN..DIVISION_AGE_MAX),
            CELL_RADIUS,
            CELL_GROWTH_RATE,
            CELL_GENERATION_BRANCHING_EVENT
        ),
        cellular_reactions: OwnReactions {
            intracellular_concentrations: ReactionVector::from_element(CELL_STUFF_INITIAL_CONCENTRATION),
            production_term: ReactionVector::from_element(CELL_STUFF_PRODUCTION_RATE),
            secretion_rate: ReactionVector::from_element(CELL_STUFF_SECRETION_RATE),
            uptake_rate: ReactionVector::from_element(CELL_STUFF_UPTAKE_RATE),
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

    let mut supervisor = SimulationSupervisor::new_with_strategies(setup, strategies);

    supervisor.run_full_sim().unwrap();

    supervisor.end_simulation().unwrap();

    supervisor.plotting_config = PlottingConfig {
        n_threads: Some(20),
        image_size: 2000,
    };

    // ###################################### PLOT THE RESULTS ######################################
    // supervisor.plot_cells_at_every_iter_bitmap_with_cell_plotting_func(&plot_modular_cell).unwrap();
    supervisor.plot_cells_at_every_iter_bitmap_with_cell_plotting_func_and_voxel_plotting_func(&plot_modular_cell, &plot_voxel).unwrap();
}
