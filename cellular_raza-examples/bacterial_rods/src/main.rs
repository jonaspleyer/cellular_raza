use cellular_raza::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

pub const METRE: f64 = 1.0;
pub const MILI_METRE: f64 = 1e-3;
pub const MICRO_METRE: f64 = 1e-6;

pub const MINUTE: f64 = 1.0;
pub const HOUR: f64 = 60.0 * MINUTE;

pub const GROWTH_BASE_RATE: f64 = 0.1 * MICRO_METRE / MINUTE;

#[derive(CellAgent, Clone, Deserialize, Serialize)]
pub struct Agent {
    #[Mechanics]
    mechanics: RodMechanics<f64, 3>,

    // Interaction
    interaction: RodInteraction<MorsePotential>,

    // Cycle
    growth_rate: f64,
    spring_length_threshold: f64,
}

impl
    cellular_raza::concepts::Interaction<
        nalgebra::MatrixXx3<f64>,
        nalgebra::MatrixXx3<f64>,
        nalgebra::MatrixXx3<f64>,
        f64,
    > for Agent
{
    fn calculate_force_between(
        &self,
        own_pos: &nalgebra::MatrixXx3<f64>,
        own_vel: &nalgebra::MatrixXx3<f64>,
        ext_pos: &nalgebra::MatrixXx3<f64>,
        ext_vel: &nalgebra::MatrixXx3<f64>,
        ext_info: &f64,
    ) -> Result<(nalgebra::MatrixXx3<f64>, nalgebra::MatrixXx3<f64>), CalcError> {
        self.interaction
            .calculate_force_between(own_pos, own_vel, ext_pos, ext_vel, ext_info)
    }

    fn get_interaction_information(&self) -> f64 {
        self.interaction.0.radius
    }

    fn is_neighbor(
        &self,
        own_pos: &nalgebra::MatrixXx3<f64>,
        ext_pos: &nalgebra::MatrixXx3<f64>,
        _ext_inf: &f64,
    ) -> Result<bool, CalcError> {
        for own_point in own_pos.row_iter() {
            for ext_point in ext_pos.row_iter() {
                if (own_point - ext_point).norm() < 2.0 * self.interaction.0.radius {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn react_to_neighbors(&mut self, neighbors: usize) -> Result<(), CalcError> {
        if neighbors > 0 {
            self.growth_rate = (GROWTH_BASE_RATE * (8.0 - neighbors as f64) / 8.0).max(0.0);
        } else {
            self.growth_rate = GROWTH_BASE_RATE;
        }
        Ok(())
    }
}

impl Cycle<Agent> for Agent {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Agent,
    ) -> Option<cellular_raza::prelude::CycleEvent> {
        cell.mechanics.spring_length += cell.growth_rate * dt;
        if cell.mechanics.spring_length > cell.spring_length_threshold {
            Some(CycleEvent::Division)
        } else {
            None
        }
    }

    fn divide(
        _rng: &mut rand_chacha::ChaCha8Rng,
        cell: &mut Agent,
    ) -> Result<Agent, cellular_raza::prelude::DivisionError> {
        let c2_mechanics = cell.mechanics.divide(cell.interaction.0.radius)?;
        let mut c2 = cell.clone();
        c2.mechanics = c2_mechanics;
        Ok(c2)
    }
}

fn main() -> Result<(), SimulationError> {
    // Define the dimensionality of the problem
    let nrows: usize = 5;

    // Define initial random seed
    use rand::Rng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);

    // Give agent default values
    let agent = Agent {
        mechanics: RodMechanics {
            pos: nalgebra::MatrixXx3::zeros(nrows),
            vel: nalgebra::MatrixXx3::zeros(nrows),
            diffusion_constant: 0.0 * MICRO_METRE.powf(2.0) / MINUTE,
            spring_tension: 10.0 / MINUTE.powf(2.0),
            rigidity: 2.0 * MICRO_METRE / MINUTE.powf(2.0),
            damping: 1.5 / MINUTE,
            spring_length: 3.0 * MICRO_METRE,
        },
        interaction: RodInteraction(MorsePotential {
            radius: 3.0 * MICRO_METRE,
            potential_stiffness: 0.5 / MICRO_METRE,
            strength: 0.1 * MICRO_METRE.powf(2.0) / MINUTE.powf(2.0),
            cutoff: 5.0 * MICRO_METRE,
        }),
        spring_length_threshold: 6.0 * MICRO_METRE,
        growth_rate: GROWTH_BASE_RATE,
    };

    // Place agents in simulation domain
    let domain_size = 50.0 * MICRO_METRE;
    let delta_x = agent.mechanics.spring_length * nrows as f64;
    let agents = (0..5).map(|_| {
        let mut new_agent = agent.clone();
        new_agent.mechanics.spring_length = rng.gen_range(1.5..2.5) * MICRO_METRE;
        let mut pos = nalgebra::MatrixXx3::zeros(nrows);
        pos[(0, 0)] = rng.gen_range(delta_x..2.0 * delta_x);
        pos[(0, 1)] = rng.gen_range(delta_x / 3.0..delta_x * 2.0 / 3.0);
        pos[(0, 2)] = rng.gen_range(delta_x..2.0 * delta_x);
        let theta = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        for i in 1..pos.nrows() {
            let phi =
                theta + rng.gen_range(-std::f64::consts::FRAC_PI_8..std::f64::consts::FRAC_PI_8);
            let mut direction = nalgebra::Vector3::zeros();
            direction[0] = phi.cos();
            direction[1] = phi.sin();
            let new_pos = pos.row(i - 1) + agent.mechanics.spring_length * (direction).transpose();
            use core::ops::AddAssign;
            pos.row_mut(i).add_assign(new_pos);
        }
        new_agent.mechanics.set_pos(&pos);
        new_agent
    });

    // Domain Setup
    let domain_sizes = [4.0 * domain_size, delta_x, 3.0 * delta_x];
    let domain_segments = [8, 1, 2];
    let domain = CartesianCuboidRods {
        domain: CartesianCuboid::from_boundaries_and_n_voxels(
            [0.0; 3],
            domain_sizes,
            domain_segments,
        )?,
    };

    // Storage Setup
    let storage_builder = cellular_raza::prelude::StorageBuilder::new().location("./out");

    // Time Setup
    let t0 = 0.0 * MINUTE;
    let dt = 0.1 * MINUTE;
    let save_interval = 2.5 * MINUTE;
    let t_max = 30.0 * HOUR;
    let time_stepper = cellular_raza::prelude::time::FixedStepsize::from_partial_save_interval(
        t0,
        dt,
        t_max,
        save_interval,
    )?;

    let settings = Settings {
        n_threads: 8.try_into().unwrap(),
        time: time_stepper,
        storage: storage_builder,
        show_progressbar: true,
    };

    println!("Running Simulation");
    run_simulation!(
        domain: domain,
        agents: agents,
        settings: settings,
        aspects: [Mechanics, Interaction, Cycle],
        zero_force_default: |c: &Agent| {
            nalgebra::MatrixXx3::zeros(c.mechanics.pos.nrows())
        },
    )?;
    Ok(())
}
