use backend::chili;
use cellular_raza::prelude::*;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

// Mechanical parameters
pub const CELL_MECHANICS_AREA: f64 = 500.0;
pub const CELL_MECHANICS_SPRING_TENSION: f64 = 2.0;
pub const CELL_MECHANICS_CENTRAL_PRESSURE: f64 = 0.5;
pub const CELL_MECHANICS_INTERACTION_RANGE: f64 = 5.0;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 10.0;
pub const CELL_MECHANICS_DAMPING_CONSTANT: f64 = 0.2;
pub const CELL_MECHANICS_DIFFUSION_CONSTANT: f64 = 0.0;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 800.0;
pub const DOMAIN_SIZE_Y: f64 = 800.0;

// Time parameters
pub const N_TIMES: u64 = 20_001;
pub const DT: f64 = 0.005;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: u64 = 50;

// Meta Parameters to control solving
pub const N_THREADS: usize = 10;

mod cell_properties;
mod custom_domain;

use cell_properties::*;
use custom_domain::*;
use time::FixedStepsize;

fn main() -> Result<(), chili::SimulationError> {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // Define the simulation domain
    let domain = MyDomain {
        cuboid: CartesianCuboid::from_boundaries_and_interaction_range(
            [0.0; 2],
            [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
            2.0 * VertexMechanics2D::<6>::inner_radius_from_cell_area(CELL_MECHANICS_AREA),
        )?,
    };

    // Define cell agents
    let models = VertexMechanics2D::fill_rectangle_flat_top(
        CELL_MECHANICS_AREA,
        CELL_MECHANICS_SPRING_TENSION,
        CELL_MECHANICS_CENTRAL_PRESSURE,
        CELL_MECHANICS_DAMPING_CONSTANT,
        CELL_MECHANICS_DIFFUSION_CONSTANT,
        [
            [0.1 * DOMAIN_SIZE_X, 0.1 * DOMAIN_SIZE_Y].into(),
            [0.9 * DOMAIN_SIZE_X, 0.9 * DOMAIN_SIZE_Y].into(),
        ],
    );
    println!("Generated {} cells", models.len());

    let growth_rate = 5.0;
    let cells = models
        .into_iter()
        .map(|model| MyCell {
            mechanics: model,
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
            growth_rate: rng.random_range(0.0..1.0) * growth_rate,
        })
        .collect::<Vec<_>>();

    // Define settings for storage and time solving
    let settings = chili::Settings {
        time: FixedStepsize::from_partial_save_steps(0.0, DT, N_TIMES, SAVE_INTERVAL)?,
        n_threads: N_THREADS.try_into().unwrap(),
        progressbar: Some("".into()),
        storage: StorageBuilder::new()
            .location("out/semi_vertex")
            .priority([StorageOption::SerdeJson]),
    };

    // Run the simulation
    let _storager = chili::run_simulation!(
        agents: cells,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction, Cycle],
    )?;
    Ok(())
}
