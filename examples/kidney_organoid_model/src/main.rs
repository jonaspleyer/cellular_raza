use cellular_raza::pipelines::cpu_os_threads::prelude::*;
use cellular_raza::implementations::cell_models::modular_cell::ModularCell;

use nalgebra::Vector2;

use rand_chacha::ChaCha8Rng;
use rand::{SeedableRng,Rng};


// Number of cells to put into simulation in the Beginning
pub const N_CELLS: u32 = 200;

// Mechanical parameters
pub const CELL_MECHANICS_AREA: f64 = 300.0;
pub const CELL_MECHANICS_MAXIMUM_AREA: f64 = 400.0;
pub const CELL_MECHANICS_INTERACTION_RANGE: f64 = 4.0;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 2.5;
pub const CELL_MECHANICS_DAMPENING_CONSTANT: f64 = 1.0;

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
pub const CELL_CYCLE_DIVISION_AGE_MIN: f64 = 1000.0;
pub const CELL_CYCLE_DIVISION_AGE_MAX: f64 = 1010.0;
pub const CELL_CYCLE_GROWTH_RATE: f64 = 0.3;
pub const CELL_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER: f64 = 2.0;
pub const CELL_CYCLE_FOOD_DEATH_THRESHOLD: f64 = CELL_FOOD_SATURATION * 0.4;
pub const CELL_CYCLE_FOOD_DIVISION_THRESHOLD: f64 = CELL_FOOD_SATURATION * 0.6;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 700.0;
pub const DOMAIN_SIZE_Y: f64 = 700.0;

// Where will the cells be placed initially
// Define a polygon by points
pub const STARTING_DOMAIN_X_LOW:  f64 = 150.0;
pub const STARTING_DOMAIN_X_HIGH: f64 = 550.0;
pub const STARTING_DOMAIN_Y_LOW:  f64 = 150.0;
pub const STARTING_DOMAIN_Y_HIGH: f64 = 550.0;

// Parameters for Voxel Reaction+Diffusion
pub const VOXEL_SPATIAL_SIGNALLING_MOLECULE_DEGRADATION_RATE: f64 = 0.003;
pub const VOXEL_SPATIAL_SIGNALLING_MOLECULE_DIFFUSION_CONSTANT: f64 = 75.0;
pub const VOXEL_SPATIAL_SIGNALLING_MOLECULE_INITIAL_CONCNENTRATION: f64 = 0.0;

pub const VOXEL_FOOD_PRODUCTION_RATE: f64 = 0.3;
pub const VOXEL_FOOD_DEGRADATION_RATE: f64 = 0.003;
pub const VOXEL_FOOD_DIFFUSION_CONSTANT: f64 = 0.0;
pub const VOXEL_FOOD_INITIAL_CONCENTRATION: f64 = 60.0;

// Time parameters
pub const N_TIMES: usize = 900_001;
pub const DT: f64 = 0.001;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: usize = 500;
pub const FULL_SAVE_INTERVAL: usize = 500;

// Meta Parameters to control solving
pub const N_THREADS: usize = 14;


mod plotting;
mod cell_properties;

use plotting::*;
use cell_properties::*;


fn voxel_definition_strategy(voxel: &mut CartesianCuboidVoxel2VertexReactions4<NUMBER_OF_VERTICES>) {
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
    let domain = CartesianCuboid2Vertex::from_boundaries_and_interaction_ranges(
        [0.0; 2],
        [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
        [2.0*CELL_MECHANICS_INTERACTION_RANGE.max((CELL_MECHANICS_AREA/std::f64::consts::PI).sqrt()); 2],
    ).unwrap();

    // ###################################### DEFINE CELLS IN SIMULATION ######################################
    // Cells of Type 1
    let estimated_cell_diameter = 2.3*(CELL_MECHANICS_AREA/std::f64::consts::PI).sqrt();
    let n_cells_per_row = ((STARTING_DOMAIN_X_HIGH-STARTING_DOMAIN_X_LOW)/estimated_cell_diameter).floor() as i32;

    let cells = (0..N_CELLS as i32).map(|n_cell| {
        let pos = Vector2::<f64>::from([
            STARTING_DOMAIN_X_LOW + (n_cell % n_cells_per_row) as f64 * estimated_cell_diameter,
            STARTING_DOMAIN_Y_LOW + (n_cell as f64/ n_cells_per_row as f64).floor() * estimated_cell_diameter
        ]);
        ModularCell {
        mechanics: MechanicsOptions::<VertexMechanics2D<NUMBER_OF_VERTICES>, VertexVector2<NUMBER_OF_VERTICES>>::Mechanics(VertexMechanics2D::new(
            pos,
            CELL_MECHANICS_AREA,
            2.0*std::f64::consts::PI/NUMBER_OF_VERTICES as f64 * n_cell as f64,
            4.0,
            3.0,
            CELL_MECHANICS_DAMPENING_CONSTANT,
            Some((0.03, rand_chacha::ChaCha8Rng::seed_from_u64(n_cell as u64)))
        )),
        interaction: VertexDerivedInteraction::from_two_forces(
            OutsideInteraction {
                potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
                interaction_range: CELL_MECHANICS_INTERACTION_RANGE,
            },
            InsideInteraction {
                potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
            }
        ),
        interaction_extracellular: GradientSensing {},
        cycle: OwnCycle::new(
            rng.gen_range(CELL_CYCLE_DIVISION_AGE_MIN..CELL_CYCLE_DIVISION_AGE_MAX),
            CELL_MECHANICS_AREA,
            CELL_MECHANICS_MAXIMUM_AREA,
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
        storage: StorageConfig {
            location: "out/kidney_organoid_model".to_owned().into(),
        }
    };

    let strategies = Strategies {
        voxel_definition_strategies: Some(voxel_definition_strategy),
    };

    let mut supervisor = SimulationSupervisor::initialize_with_strategies(setup, strategies);

    let mut simulation_result = supervisor.run_full_sim().unwrap();

    // ###################################### PLOT THE RESULTS ######################################
    simulation_result.plotting_config = PlottingConfig {
        n_threads: Some(20),
        image_size: 1500,
        image_type: ImageType::BitMap,
    };

    simulation_result.plot_spatial_all_iterations_custom_cell_voxel_functions(&plot_modular_cell, &plot_voxel).unwrap();
}
