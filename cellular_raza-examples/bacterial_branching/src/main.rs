use backend::chili::Settings;
use cellular_raza::core::backend::chili;
use cellular_raza::prelude::*;

use nalgebra::Vector2;

use num::Zero;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use serde::{Deserialize, Serialize};

// Number of cells to put into simulation in the Beginning
pub const N_BACTERIA_INITIAL: u32 = 400;

// Mechanical parameters
pub const BACTERIA_MECHANICS_RADIUS: f32 = 6.0;
pub const BACTERIA_MECHANICS_RELATIVE_INTERACTION_RANGE: f32 = 1.6;
pub const BACTERIA_MECHANICS_POTENTIAL_STRENGTH: f32 = 1e-2;
pub const BACTERIA_MECHANICS_VELOCITY_REDUCTION: f32 = 2e-1;

// Reaction parameters of the cell
pub const BACTERIA_FOOD_INITIAL_CONCENTRATION: f32 = 1.0;
pub const BACTERIA_FOOD_TURNOVER_RATE: f32 = 0.025;
pub const BACTERIA_FOOD_UPTAKE_RATE: f32 = 0.05;

// Parameters for cell cycle
pub const BACTERIA_CYCLE_DIVISION_AGE_MIN: f32 = 60.0;
pub const BACTERIA_CYCLE_DIVISION_AGE_MAX: f32 = 70.0;
pub const BACTERIA_CYCLE_GROWTH_RATE: f32 = 0.1;
pub const BACTERIA_CYCLE_FOOD_THRESHOLD: f32 = 2.0;
pub const BACTERIA_CYCLE_FOOD_GROWTH_RATE_MULTIPLIER: f32 = 10.0;
pub const BACTERIA_CYCLE_FOOD_DIVISION_THRESHOLD: f32 = BACTERIA_FOOD_INITIAL_CONCENTRATION * 0.8;

// Parameters for domain
pub const DOMAIN_SIZE: f32 = 3_000.0;
pub const DOMAIN_MIDDLE: Vector2<f32> = nalgebra::vector![DOMAIN_SIZE / 2.0, DOMAIN_SIZE / 2.0];

// Where will the cells be placed initially
pub const STARTING_DOMAIN_X_LOW: f32 = DOMAIN_SIZE / 2.0 - 150.0;
pub const STARTING_DOMAIN_X_HIGH: f32 = DOMAIN_SIZE / 2.0 + 150.0;
pub const STARTING_DOMAIN_Y_LOW: f32 = DOMAIN_SIZE / 2.0 - 150.0;
pub const STARTING_DOMAIN_Y_HIGH: f32 = DOMAIN_SIZE / 2.0 + 150.0;

// Parameters for Voxel Reaction+Diffusion
pub const VOXEL_FOOD_DIFFUSION_CONSTANT: f32 = 25.0;
pub const VOXEL_FOOD_INITIAL_CONCENTRATION: f32 = 12.0;

// Time parameters
pub const DT: f32 = 0.1;
pub const T_START: f32 = 0.0;
pub const T_MAX: f32 = 100.0;
pub const SAVE_INTERVAL: usize = 10;

// Meta Parameters to control solving
pub const N_THREADS: usize = 1;

mod bacteria_properties;

use bacteria_properties::*;
use time::FixedStepsize;

fn main() -> Result<(), chili::SimulationError> {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    let cells = (0..N_BACTERIA_INITIAL)
        .map(|_| {
            let x = rng.gen_range(STARTING_DOMAIN_X_LOW..STARTING_DOMAIN_X_HIGH);
            let y = rng.gen_range(STARTING_DOMAIN_Y_LOW..STARTING_DOMAIN_Y_HIGH);

            let pos = Vector2::from([x, y]);
            MyAgent {
                mechanics: NewtonDamped2DF32 {
                    pos,
                    vel: Vector2::zero(),
                    damping_constant: BACTERIA_MECHANICS_VELOCITY_REDUCTION,
                    mass: 1.0,
                },
                interaction: MorsePotentialF32 {
                    length_repelling: BACTERIA_MECHANICS_RADIUS,
                    length_attracting: 1.0,
                    cutoff: BACTERIA_MECHANICS_RADIUS,
                    strength_repelling: BACTERIA_MECHANICS_POTENTIAL_STRENGTH,
                    strength_attracting: 0.0,
                },
                reactions: SimpleReactions {
                    intracellular: ReactionVector::from(
                        [BACTERIA_FOOD_INITIAL_CONCENTRATION; N_REACTIONS],
                    ),
                    turnover_rate: ReactionVector::from([BACTERIA_FOOD_TURNOVER_RATE; N_REACTIONS]),
                    production_term: ReactionVector::zero(),
                    secretion_rate: ReactionVector::zero(),
                    uptake_rate: ReactionVector::from([BACTERIA_FOOD_UPTAKE_RATE; N_REACTIONS]),
                },
            }
        })
        .collect::<Vec<_>>();

    let domain = MyDomain {
        domain: CartesianCuboid::from_boundaries_and_interaction_range(
            [0.0; 2],
            [DOMAIN_SIZE, DOMAIN_SIZE],
            BACTERIA_MECHANICS_RADIUS * BACTERIA_MECHANICS_RELATIVE_INTERACTION_RANGE,
        )?,
        reactions_dx: 50.0,
        diffusion_constant: 1.0,
        initial_value: ReactionVector::from([1.0]),
    };

    let storage = StorageBuilder::new().priority([StorageOption::Ron]);
    let time = FixedStepsize::from_partial_save_freq(0.0, DT, T_MAX, SAVE_INTERVAL)?;
    let settings = Settings {
        n_threads: N_THREADS.try_into().unwrap(),
        time,
        storage,
        show_progressbar: true,
    };

    let mut storager = chili::run_simulation!(
        agents: cells,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction, Reactions, ReactionsExtra],
        parallelizer: Rayon,
    )?;
    Ok(())
}