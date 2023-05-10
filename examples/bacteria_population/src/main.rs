use cellular_raza::backend::cpu_os_threads::prelude::*;
use cellular_raza::implementations::cell_models::modular_cell::ModularCell;

use nalgebra::Vector2;

use num::Zero;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use serde::{Deserialize, Serialize};

// Number of cells to put into simulation in the Beginning
pub const N_BACTERIA_INITIAL: u32 = 400;

// Mechanical parameters
pub const BACTERIA_MECHANICS_RADIUS: f64 = 6.0;
pub const BACTERIA_MECHANICS_RELATIVE_INTERACTION_RANGE: f64 = 1.6;
pub const BACTERIA_MECHANICS_POTENTIAL_STRENGTH: f64 = 2.0;
pub const BACTERIA_MECHANICS_VELOCITY_REDUCTION: f64 = 2.0;

// Reaction parameters of the cell
pub const BACTERIA_FOOD_INITIAL_CONCENTRATION: f64 = 1.0;
pub const BACTERIA_FOOD_TURNOVER_RATE: f64 = 0.025;
pub const BACTERIA_FOOD_UPTAKE_RATE: f64 = 0.05;

// Parameters for cell cycle
pub const BACTERIA_CYCLE_DIVISION_AGE_MIN: f64 = 60.0;
pub const BACTERIA_CYCLE_DIVISION_AGE_MAX: f64 = 70.0;
pub const BACTERIA_CYCLE_GROWTH_RATE: f64 = 0.1;
pub const BACTERIA_CYCLE_FOOD_THRESHOLD: f64 = 2.0;
pub const BACTERIA_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER: f64 = 10.0;
pub const BACTERIA_CYCLE_FOOD_DIVISION_THRESHOLD: f64 = BACTERIA_FOOD_INITIAL_CONCENTRATION * 0.8;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 2_000.0;
pub const DOMAIN_SIZE_Y: f64 = 2_000.0;

// Where will the cells be placed initially
pub const STARTING_DOMAIN_X_LOW: f64 = DOMAIN_SIZE_X / 2.0 - 150.0;
pub const STARTING_DOMAIN_X_HIGH: f64 = DOMAIN_SIZE_X / 2.0 + 150.0;
pub const STARTING_DOMAIN_Y_LOW: f64 = DOMAIN_SIZE_Y / 2.0 - 150.0;
pub const STARTING_DOMAIN_Y_HIGH: f64 = DOMAIN_SIZE_Y / 2.0 + 150.0;

// Parameters for Voxel Reaction+Diffusion
pub const VOXEL_FOOD_DIFFUSION_CONSTANT: f64 = 25.0;
pub const VOXEL_FOOD_INITIAL_CONCENTRATION: f64 = 12.0;

// Time parameters
pub const N_TIMES: usize = 1_001;
pub const DT: f64 = 0.25;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: usize = 100;

// Meta Parameters to control solving
pub const N_THREADS: usize = 4;
pub const N_PLOTTING_THREADS: usize = 40;

mod bacteria_properties;
mod plotting;

use bacteria_properties::*;
use plotting::*;

fn voxel_definition_strategy(voxel: &mut CartesianCuboidVoxel2<NUMBER_OF_REACTION_COMPONENTS>) {
    voxel.diffusion_constant = ReactionVector::from([VOXEL_FOOD_DIFFUSION_CONSTANT]);
    voxel.extracellular_concentrations = ReactionVector::from([VOXEL_FOOD_INITIAL_CONCENTRATION]);
    voxel.degradation_rate = ReactionVector::zero();
    voxel.production_rate = ReactionVector::zero();
}

fn create_domain() -> Result<CartesianCuboid2, CalcError> {
    CartesianCuboid2::from_boundaries_and_interaction_ranges(
        [0.0; 2],
        [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
        [BACTERIA_MECHANICS_RADIUS * BACTERIA_MECHANICS_RELATIVE_INTERACTION_RANGE; 2],
    )
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CellNumberController {
    pub target_number: i128,
    pub K_p: f64,
    pub K_d: f64,
}

type Observable = i128;

impl Controller<MyCellType, Observable> for CellNumberController {
    fn measure<'a, I>(&self, cells: I) -> Result<Observable, SimulationError>
    where
        I: IntoIterator<Item=&'a MyCellType>
    {
        let mut length = 0;
        for _ in cells {
            length+=1;
        }
        Ok(length)
    }

    fn adjust<'a, 'b, I, J>(&self, measurements: I, cells: J) -> Result<(), SimulationError>
    where
        Observable: 'a,
        MyCellType: 'b,
        I: Iterator<Item=&'a Observable>,
        J: Iterator<Item=&'b mut MyCellType>,
    {
        // Calculate difference between measured cells and target
        let delta = self.target_number - measurements.sum::<i128>() as i128;

        if delta < 0 {
            cells.into_iter().take((-delta) as usize).for_each(|c| {
                if c.mechanics.pos().y < DOMAIN_SIZE_Y/2.0 {
                    c.cycle.division_age += 15.0;
                }
            });
        }
        Ok(())
    }
}

fn main() {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // ###################################### DEFINE SIMULATION DOMAIN ######################################
    let domain = create_domain().unwrap();

    // ###################################### DEFINE CELLS IN SIMULATION ######################################
    let cells = (0..N_BACTERIA_INITIAL)
        .map(|_| {
            let x = rng.gen_range(STARTING_DOMAIN_X_LOW..STARTING_DOMAIN_X_HIGH);
            let y = rng.gen_range(STARTING_DOMAIN_Y_LOW..STARTING_DOMAIN_Y_HIGH);

            let pos = Vector2::from([x, y]);
            ModularCell {
                mechanics: MechanicsModel2D {
                    pos,
                    vel: Vector2::zero(),
                    dampening_constant: BACTERIA_MECHANICS_VELOCITY_REDUCTION,
                },
                interaction: CellSpecificInteraction {
                    potential_strength: BACTERIA_MECHANICS_POTENTIAL_STRENGTH,
                    relative_interaction_range: BACTERIA_MECHANICS_RELATIVE_INTERACTION_RANGE,
                    cell_radius: BACTERIA_MECHANICS_RADIUS,
                },
                interaction_extracellular: GradientSensing {},
                cycle: OwnCycle::new(
                    rng.gen_range(0.0..BACTERIA_CYCLE_DIVISION_AGE_MAX),
                    BACTERIA_MECHANICS_RADIUS,
                    BACTERIA_CYCLE_GROWTH_RATE,
                    BACTERIA_CYCLE_FOOD_THRESHOLD,
                    BACTERIA_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER,
                    BACTERIA_CYCLE_FOOD_DIVISION_THRESHOLD,
                ),
                cellular_reactions: OwnReactions {
                    intracellular_concentrations: ReactionVector::from([
                        BACTERIA_FOOD_INITIAL_CONCENTRATION,
                    ]),
                    turnover_rate: ReactionVector::from([BACTERIA_FOOD_TURNOVER_RATE]),
                    production_term: ReactionVector::zero(),
                    degradation_rate: ReactionVector::zero(),
                    secretion_rate: ReactionVector::zero(),
                    uptake_rate: ReactionVector::from([BACTERIA_FOOD_UPTAKE_RATE]),
                },
            }
        })
        .collect::<Vec<_>>();

    // ###################################### CREATE SUPERVISOR AND RUN SIMULATION ######################################
    let setup = SimulationSetup::new(
        domain,
        cells,
        TimeSetup {
            t_start: 0.0,
            t_eval: (0..N_TIMES)
                .map(|i| (T_START + DT * i as f64, i % SAVE_INTERVAL == 0))
                .collect::<Vec<(f64, bool)>>(),
        },
        SimulationMetaParams {
            n_threads: N_THREADS,
        },
        StorageConfig {
            location: "out/bacteria_population".to_owned().into(),
        },
        CellNumberController {
            target_number: 5000,
            K_p: 0.0,
            K_d: 0.0,
        },
    );

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
        ..Default::default()
    };

    simulation_result
        .plot_spatial_all_iterations_custom_cell_voxel_functions(plot_modular_cell, plot_voxel)
        .unwrap();
}
