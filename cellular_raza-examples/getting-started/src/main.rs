use cellular_raza::prelude::*;

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(CellAgent, Clone, Deserialize, Serialize)]
struct Agent {
    #[Mechanics]
    mechanics: NewtonDamped2D,
    #[Interaction]
    interaction: MorsePotential,
}

struct Parameters {
    domain_size: f64,
    cell_number: usize,
    cell_mechanics: NewtonDamped2D,
    cell_interaction: MorsePotential,
    time_dt: f64,
    time_save_interval: f64,
    time_start: f64,
    time_end: f64,
    n_threads: std::num::NonZeroUsize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            domain_size: 100.0,
            cell_number: 20,
            cell_mechanics: NewtonDamped2D {
                pos: [0.0; 2].into(),
                vel: [0.0; 2].into(),
                damping_constant: 0.1,
                mass: 1.0,
            },
            cell_interaction: MorsePotential {
                length_repelling: 5.0,
                length_attracting: 10.0,
                strength_repelling: 3.0,
                strength_attracting: 0.5,
                cutoff: 15.0,
            },
            time_dt: 0.1,
            time_save_interval: 1.0,
            time_start: 0.0,
            time_end: 10_000.0,
            n_threads: 1.try_into().unwrap(),
        }
    }
}

fn main() -> Result<(), SimulationError> {
    let parameters = Parameters::default();

    // Define the seed
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);

    let cells = (0..parameters.cell_number)
        .map(|_| {
            let pos = [
                rng.gen_range(0.0..parameters.domain_size),
                rng.gen_range(0.0..parameters.domain_size),
            ];
            Agent {
                mechanics: NewtonDamped2D {
                    pos: pos.into(),
                    ..parameters.cell_mechanics.clone()
                },
                interaction: parameters.cell_interaction.clone(),
            }
        });

    let domain = CartesianCuboid2New::from_boundaries_and_interaction_ranges(
        [0.0; 2],
        [parameters.domain_size; 2],
        [parameters.cell_interaction.cutoff * 2.0; 2],
    )?;

    let time = FixedStepsize::from_partial_save_interval(
        parameters.time_start,
        parameters.time_dt,
        parameters.time_end,
        parameters.time_save_interval,
    )?;
    let storage_builder = StorageBuilder::new().location("out/cell_sorting");

    let settings = Settings {
        n_threads: parameters.n_threads,
        time,
        storage: storage_builder,
        show_progressbar: true,
    };

    run_simulation!(
        domain: domain,
        agents: cells,
        settings: settings,
        aspects: [Mechanics, Interaction]
    )?;
    Ok(())
}
