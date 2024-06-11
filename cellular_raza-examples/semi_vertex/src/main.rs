use backend::chili;
use cellular_raza::prelude::*;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

// Number of cells
pub const N_CELLS: usize = 2_025;

// Mechanical parameters
pub const CELL_MECHANICS_AREA: f64 = 500.0;
pub const CELL_MECHANICS_SPRING_TENSION: f64 = 2.0;
pub const CELL_MECHANICS_CENTRAL_PRESSURE: f64 = 0.5;
pub const CELL_MECHANICS_MAXIMUM_AREA: f64 = 350.0;
pub const CELL_MECHANICS_INTERACTION_RANGE: f64 = 5.0;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 3.0;
pub const CELL_MECHANICS_DAMPING_CONSTANT: f64 = 1.0;
pub const CELL_MECHANICS_DIFFUSION_CONSTANT: f64 = 0.004;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 1_200.0;
pub const DOMAIN_SIZE_Y: f64 = 1_200.0;

// Time parameters
pub const N_TIMES: usize = 100_001;
pub const DT: f64 = 0.05;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: usize = 50;

// Meta Parameters to control solving
pub const N_THREADS: usize = 32;

mod cell_properties;
mod plotting;

use cell_properties::*;
use plotting::*;

fn voxel_definition_strategy(
    voxel: &mut CartesianCuboidVoxel2Vertex<NUMBER_OF_VERTICES, NUMBER_OF_REACTION_COMPONENTS>,
) {
    voxel.diffusion_constant = ReactionVector::from([
        VOXEL_SPATIAL_SIGNALLING_MOLECULE_DIFFUSION_CONSTANT,
        VOXEL_FOOD_DIFFUSION_CONSTANT,
    ]);
    voxel.extracellular_concentrations = ReactionVector::from([
        VOXEL_SPATIAL_SIGNALLING_MOLECULE_INITIAL_CONCNENTRATION,
        VOXEL_FOOD_INITIAL_CONCENTRATION,
    ]);
    voxel.degradation_rate = ReactionVector::from([
        VOXEL_SPATIAL_SIGNALLING_MOLECULE_DEGRADATION_RATE,
        VOXEL_FOOD_DEGRADATION_RATE,
    ]);
    voxel.production_rate = ReactionVector::from([0.0, VOXEL_FOOD_PRODUCTION_RATE]);
}

fn main() {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // ###################################### DEFINE SIMULATION DOMAIN ######################################
    // Define the simulation domain
    let domain = CartesianCuboid2Vertex::from_boundaries_and_interaction_ranges(
        [0.0; 2],
        [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
        [2.0 * CELL_MECHANICS_INTERACTION_RANGE
            .max((CELL_MECHANICS_AREA / std::f64::consts::PI).sqrt()); 2],
    )
    .unwrap();

    // ###################################### DEFINE CELLS IN SIMULATION ######################################

    let dx = 0.95 * CELL_MECHANICS_AREA.sqrt();
    let n_x_max = (0.8 * DOMAIN_SIZE_X / dx).floor();
    let n_y_max = (0.8 * DOMAIN_SIZE_Y / dx).floor();
    let cells = (0..N_CELLS)
        .map(|n_cell| {
            let n_x = n_cell as f64 % n_x_max;
            let n_y = (n_cell as f64 / n_y_max).floor();
            MyCell {
                mechanics: VertexMechanics2D::<6>::new(
                    [
                        0.1 * DOMAIN_SIZE_X + n_x * dx + 0.5 * (n_y % 2.0) * dx,
                        0.1 * DOMAIN_SIZE_Y + n_y * dx,
                        // rng.gen_range(0.2 * DOMAIN_SIZE_X..0.8 * DOMAIN_SIZE_X),
                        // rng.gen_range(0.2 * DOMAIN_SIZE_Y..0.8 * DOMAIN_SIZE_Y),
                    ]
                    .into(),
                    CELL_MECHANICS_AREA,
                    rng.gen_range(0.0..2.0 * std::f64::consts::PI),
                    CELL_MECHANICS_SPRING_TENSION,
                    CELL_MECHANICS_CENTRAL_PRESSURE,
                    CELL_MECHANICS_DAMPING_CONSTANT,
                    CELL_MECHANICS_DIFFUSION_CONSTANT,
                    None,
                ),
                interaction: VertexDerivedInteraction::from_two_forces(
                    OutsideInteraction {
                        potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
                        interaction_range: CELL_MECHANICS_INTERACTION_RANGE,
                    },
                    InsideInteraction {
                        potential_strength: 1.5 * CELL_MECHANICS_POTENTIAL_STRENGTH,
                        average_radius: CELL_MECHANICS_AREA.sqrt(),
                    },
                ),
            }
        })
        .collect::<Vec<_>>();

    // ###################################### CREATE SUPERVISOR AND RUN SIMULATION ######################################
    let setup = create_simulation_setup!(
        Domain: domain,
        Cells: cells,
        Time: TimeSetup {
            t_start: 0.0,
            t_eval: (0..N_TIMES)
                .map(|i| (T_START + DT * i as f64, i % SAVE_INTERVAL == 0))
                .collect::<Vec<(f64, bool)>>(),
        },
        MetaParams: SimulationMetaParams {
            n_threads: N_THREADS,
            ..Default::default()
        },
        Storage: StorageBuilder::new()
            .location("out/semi_vertex")
            .init(),
    );

    let strategies = Strategies {
        voxel_definition_strategies: &voxel_definition_strategy,
    };

    let mut supervisor = SimulationSupervisor::initialize_with_strategies(setup, strategies);

    let mut simulation_result = supervisor.run_full_sim().unwrap();

    // ###################################### PLOT THE RESULTS ######################################
    simulation_result.plotting_config = PlottingConfig {
        n_threads: Some(20),
        image_size: 1500,
        image_type: ImageType::BitMap,
        ..Default::default()
    };

    simulation_result
        .plot_spatial_all_iterations_custom_cell_voxel_functions(&plot_modular_cell, &plot_voxel)
        .unwrap();
}
