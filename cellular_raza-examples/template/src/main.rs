use cellular_raza::building_blocks::{
    BoundLennardJonesF32, CartesianCuboid2NewF32, NewtonDamped2DF32,
};
use cellular_raza::concepts::{CalcError, CellAgent, Interaction, Mechanics, RngError, Volume};

use cellular_raza::core::backend::chili;

use nalgebra::Vector2;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

pub struct SimulationSettings {
    n_agents: usize,
    domain_size: f32,
    n_voxels: usize,
    n_threads: usize,
    dt: f32,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            n_agents: 200,
            domain_size: 30.0,
            n_voxels: 3,
            n_threads: 4,
            dt: 0.002,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct Vol(f64);

impl Volume for Vol {
    fn get_volume(&self) -> f64 {
        self.0
    }
}

#[derive(CellAgent, Clone, Deserialize, Serialize)]
struct Agent {
    #[Mechanics(Vector2<f32>, Vector2<f32>, Vector2<f32>, f32)]
    pub mechanics: NewtonDamped2DF32,
    #[Interaction(Vector2<f32>, Vector2<f32>, Vector2<f32>)]
    pub interaction: BoundLennardJonesF32,
}

fn main() -> Result<(), chili::SimulationError> {
    use rand::Rng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let simulation_settings = SimulationSettings::default();

    // Agents setup
    let agent = Agent {
        mechanics: NewtonDamped2DF32 {
            pos: Vector2::from([0.0, 0.0]),
            vel: Vector2::from([0.0, 0.0]),
            damping_constant: 1.0,
            mass: 1.0,
        },
        interaction: BoundLennardJonesF32 {
            epsilon: 0.01,
            sigma: 1.0,
            bound: 0.1,
            cutoff: 1.0,
        },
    };

    let domain_size = simulation_settings.domain_size;
    let agents = (0..simulation_settings.n_agents).map(|_| {
        let mut new_agent = agent.clone();
        new_agent.set_pos(&Vector2::from([
            rng.gen_range(0.0..domain_size),
            rng.gen_range(0.0..domain_size),
        ]));
        new_agent
    });

    // Domain Setup
    let domain = CartesianCuboid2NewF32::from_boundaries_and_n_voxels(
        [0.0; 2],
        [simulation_settings.domain_size; 2],
        [simulation_settings.n_voxels; 2],
    )?;

    // Storage Setup
    let mut storage_priority = cellular_raza::prelude::UniqueVec::new();
    storage_priority.push(cellular_raza::prelude::StorageOption::SerdeJson);
    let storage_builder = cellular_raza::prelude::StorageBuilder::new()
        .priority(storage_priority)
        .location("./out");

    // Time Setup
    let t0: f32 = 0.0;
    let dt = simulation_settings.dt;
    let save_points = vec![5.0, 10.0, 15.0, 20.0];
    let time_stepper = cellular_raza::prelude::time::FixedStepsize::from_partial_save_points(
        t0,
        dt,
        save_points.clone(),
    )?;

    let settings = chili::Settings {
        n_threads: simulation_settings.n_threads.try_into().unwrap(),
        time: time_stepper,
        storage: storage_builder,
        show_progressbar: true,
    };

    chili::run_simulation!(
        domain: domain,
        agents: agents,
        settings: settings,
        aspects: [Mechanics, Interaction],
    )?;
    Ok(())
}
