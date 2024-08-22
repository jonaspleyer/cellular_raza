use backend::chili::Settings;
use cellular_raza::core::backend::chili;
use cellular_raza::prelude::*;

use nalgebra::Vector2;

use num::Zero;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use serde::{Deserialize, Serialize};

// Number of cells to put into simulation in the Beginning
pub const N_BACTERIA_INITIAL: u32 = 1;

// Mechanical parameters
pub const BACTERIA_MECHANICS_RADIUS: f32 = 6.0;
pub const BACTERIA_MECHANICS_EXPONENT: f32 = 0.2;
pub const BACTERIA_MECHANICS_POTENTIAL_STRENGTH: f32 = 4.0;
pub const BACTERIA_MECHANICS_VELOCITY_REDUCTION: f32 = 1.0;

// Reaction parameters of the cell
pub const BACTERIA_FOOD_INITIAL_CONCENTRATION_LOW: f32 = 0.0;
pub const BACTERIA_FOOD_INITIAL_CONCENTRATION_HIGH: f32 = 0.0;
pub const BACTERIA_FOOD_TURNOVER_RATE: f32 = 0.0;
pub const BACTERIA_FOOD_UPTAKE_RATE: f32 = 0.5;

// Parameters for cell cycle
pub const BACTERIA_CYCLE_GROWTH_RATE: f32 = 1.2;

// Parameters for domain
pub const DOMAIN_SIZE: f32 = 500.0;
pub const DOMAIN_MIDDLE: Vector2<f32> = nalgebra::vector![DOMAIN_SIZE / 2.0, DOMAIN_SIZE / 2.0];

// Where will the cells be placed initially
pub const STARTING_DOMAIN_X_LOW: f32 = DOMAIN_SIZE / 2.0 - 50.0;
pub const STARTING_DOMAIN_X_HIGH: f32 = DOMAIN_SIZE / 2.0 + 50.0;
pub const STARTING_DOMAIN_Y_LOW: f32 = DOMAIN_SIZE / 2.0 - 50.0;
pub const STARTING_DOMAIN_Y_HIGH: f32 = DOMAIN_SIZE / 2.0 + 50.0;

// Parameters for Voxel Reaction+Diffusion
pub const FOOD_DIFFUSION_CONSTANT: f32 = 125.0;
pub const FOOD_INITIAL_CONCENTRATION: f32 = 10.0;

// Time parameters
pub const DT: f32 = 0.02;
pub const T_START: f32 = 0.0;
pub const T_MAX: f32 = 1_000.0;
pub const SAVE_INTERVAL: usize = 50;

// Meta Parameters to control solving
pub const N_THREADS: usize = 8;

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
                interaction: MyInteraction {
                    cell_radius: BACTERIA_MECHANICS_RADIUS,
                    exponent: BACTERIA_MECHANICS_EXPONENT,
                    potential_strength: BACTERIA_MECHANICS_POTENTIAL_STRENGTH,
                },
                uptake_rate: BACTERIA_FOOD_UPTAKE_RATE,
                division_radius: BACTERIA_MECHANICS_RADIUS * 2.0,
                growth_rate: BACTERIA_CYCLE_GROWTH_RATE,
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
        diffusion_constant: FOOD_DIFFUSION_CONSTANT,
        initial_value: ReactionVector::from([FOOD_INITIAL_CONCENTRATION]),
    };

    let storage = StorageBuilder::new().priority([StorageOption::SerdeJson]);
    let time = FixedStepsize::from_partial_save_freq(0.0, DT, T_MAX, SAVE_INTERVAL)?;
    let settings = Settings {
        n_threads: N_THREADS.try_into().unwrap(),
        time,
        storage,
        show_progressbar: true,
    };

    let _storager = chili::run_simulation!(
        agents: cells,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction, ReactionsExtra, Cycle],
        parallelizer: Rayon,
    )?;
    Ok(())
}
