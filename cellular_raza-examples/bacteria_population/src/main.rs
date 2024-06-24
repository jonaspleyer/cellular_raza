use cellular_raza::prelude::*;
use cellular_raza::building_blocks::cartesian_cuboid_n_old::*;
use cellular_raza::concepts::domain_old::Controller;

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
pub const DOMAIN_SIZE: f64 = 3_000.0;
pub const DOMAIN_MIDDLE: Vector2<f64> = nalgebra::vector![DOMAIN_SIZE / 2.0, DOMAIN_SIZE / 2.0];

// Where will the cells be placed initially
pub const STARTING_DOMAIN_X_LOW: f64 = DOMAIN_SIZE / 2.0 - 150.0;
pub const STARTING_DOMAIN_X_HIGH: f64 = DOMAIN_SIZE / 2.0 + 150.0;
pub const STARTING_DOMAIN_Y_LOW: f64 = DOMAIN_SIZE / 2.0 - 150.0;
pub const STARTING_DOMAIN_Y_HIGH: f64 = DOMAIN_SIZE / 2.0 + 150.0;

// Parameters for Voxel Reaction+Diffusion
pub const VOXEL_FOOD_DIFFUSION_CONSTANT: f64 = 25.0;
pub const VOXEL_FOOD_INITIAL_CONCENTRATION: f64 = 12.0;

// Time parameters
pub const N_TIMES: usize = 20_001;
pub const DT: f64 = 0.25;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: usize = 250;

// Meta Parameters to control solving
pub const N_THREADS: usize = 1;
pub const N_PLOTTING_THREADS: usize = 14;

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
        [DOMAIN_SIZE, DOMAIN_SIZE],
        [BACTERIA_MECHANICS_RADIUS * BACTERIA_MECHANICS_RELATIVE_INTERACTION_RANGE; 2],
    )
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CellNumberController {
    target_cell_number: i64,
    stored_ids: std::collections::HashSet<(u64, u64)>,
    full: bool,
}

type Observable = Option<(i64, Vec<(u64, u64)>)>;

impl Controller<MyCellType, Observable> for CellNumberController {
    fn measure<'a, I>(&self, cells: I) -> Result<Observable, CalcError>
    where
        I: IntoIterator<Item = &'a CellAgentBox<MyCellType>> + Clone,
    {
        if !self.full {
            let mut n_cells = 0_i64;
            let positions = cells
                .into_iter()
                .map(|c| {
                    n_cells += 1;
                    c.get_id()
                })
                .collect();
            Ok(Some((n_cells, positions)))
        } else {
            Ok(None)
        }
    }

    fn adjust<'a, 'b, I, J>(&mut self, measurements: I, cells: J) -> Result<(), ControllerError>
    where
        Observable: 'a,
        MyCellType: 'b,
        I: Iterator<Item = &'a Observable>,
        J: Iterator<Item = (&'b mut CellAgentBox<MyCellType>, &'b mut Vec<CycleEvent>)>,
    {
        // If we are not full, we
        if !self.full {
            let mut total_cell_number: i64 = 0;
            let all_ids: std::collections::HashSet<_> = measurements
                .into_iter()
                .filter_map(|meas| {
                    if let Some((n_cells, ids)) = meas {
                        total_cell_number += n_cells;
                        Some(ids.into_iter())
                    } else {
                        None
                    }
                })
                .flatten()
                .map(|&id| id)
                .collect();

            if total_cell_number > self.target_cell_number {
                self.stored_ids = all_ids;
                self.full = true;
            }
        }
        if self.full {
            // Kill all cells which do not match ids
            for (cell, cell_events) in cells.into_iter() {
                if !self.stored_ids.contains(&cell.get_id()) {
                    cell_events.push(CycleEvent::Remove);
                }
            }
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
                mechanics: NewtonDamped2D {
                    pos,
                    vel: Vector2::zero(),
                    damping_constant: BACTERIA_MECHANICS_VELOCITY_REDUCTION,
                    mass: 1.0,
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
                volume: (DOMAIN_SIZE / 151.0).powf(2.0), //2.0*std::f64::consts::PI*BACTERIA_MECHANICS_RADIUS.powf(2.0),
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
            ..Default::default()
        },
        StorageBuilder::new()
            .location("out/bacteria_population")
            .init(),
        CellNumberController {
            target_cell_number: 15_000,
            stored_ids: std::collections::HashSet::new(),
            full: false,
        },
    );

    let strategies = Strategies {
        voxel_definition_strategies: &voxel_definition_strategy,
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
