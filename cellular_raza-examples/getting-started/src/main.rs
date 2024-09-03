use cellular_raza::prelude::*;

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(CellAgent, Clone, Deserialize, Serialize)]
struct Agent {
    #[Mechanics]
    mechanics: Langevin2D,
    #[Interaction]
    interaction: MorsePotential,
}

struct Parameters {
    domain_size: f64,
    cell_number: usize,
    cell_mechanics: Langevin2D,
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
            domain_size: 100.0, // µm
            cell_number: 60,
            cell_mechanics: Langevin2D {
                pos: [0.0; 2].into(), // µm
                vel: [0.0; 2].into(), // µm
                damping: 2.0,         // 1/min
                mass: 1.0,            // picogram
                kb_temperature: 0.3,  // picogram µm^2 / min^2
            },
            cell_interaction: MorsePotential {
                radius: 2.0,          // µm
                potential_width: 1.0, // 1/µm
                cutoff: 6.0,          // µm
                strength: 0.01,       // picogram * µm / min^2
            },
            time_dt: 0.01,           // min
            time_save_interval: 1.0, // min
            time_start: 0.0,         // min
            time_end: 1_000.0,       // min
            n_threads: 1.try_into().unwrap(),
        }
    }
}

fn main() -> Result<(), SimulationError> {
    let parameters = Parameters::default();

    // Define the seed
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);

    let cells = (0..parameters.cell_number).map(|_| {
        let pos = [
            rng.gen_range(0.0..parameters.domain_size),
            rng.gen_range(0.0..parameters.domain_size),
        ];
        Agent {
            mechanics: Langevin2D {
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
    let storage_builder = StorageBuilder::new().location("out");

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
