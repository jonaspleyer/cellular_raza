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
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 6.0;
pub const CELL_MECHANICS_DAMPING_CONSTANT: f64 = 1.0;
pub const CELL_MECHANICS_DIFFUSION_CONSTANT: f64 = 0.0;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 800.0;
pub const DOMAIN_SIZE_Y: f64 = 800.0;

// Time parameters
pub const N_TIMES: u64 = 10_001;
pub const DT: f64 = 0.05;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: u64 = 10;

// Meta Parameters to control solving
pub const N_THREADS: usize = 4;

mod cell_properties;
mod custom_domain;
mod plotting;

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
            2.0 * CELL_MECHANICS_INTERACTION_RANGE
                .max((CELL_MECHANICS_AREA / std::f64::consts::PI).sqrt()),
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
    let cells = models
        .into_iter()
        .map(|model| {
            MyCell {
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
                intracellular: nalgebra::Vector2::from([
                    rng.gen_range(5.0..6.0),
                    rng.gen_range(0.0..1.0),
                ]),
                k1: 0.01,
                k2: 0.02,
                k3: 0.1,
                exchange: nalgebra::Vector2::from([0.001, 0.1]),
            }
        })
        .collect::<Vec<_>>();
    for cell in cells.iter() {
        assert!(cell.k2 > 0.0);
        assert!(cell.k2.powi(3) < cell.k1 * cell.k3.powi(2));
        assert!(cell.exchange.y / cell.exchange.x * cell.k2.powi(3) > cell.k1 * cell.k3.powi(2));
    }

    // Define settings for storage and time solving
    let settings = chili::Settings {
        time: FixedStepsize::from_partial_save_steps(0.0, DT, N_TIMES, SAVE_INTERVAL)?,
        n_threads: N_THREADS.try_into().unwrap(),
        show_progressbar: true,
        storage: StorageBuilder::new().location("out/semi_vertex"),
    };

    // Run the simulation
    let _storager = chili::run_simulation!(
        agents: cells,
        domain: domain,
        settings: settings,
        aspects: [Reactions, ReactionsContact],
    )?;
    Ok(())
}
