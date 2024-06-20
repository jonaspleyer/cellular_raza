use std::ops::Div;

use cellular_raza::building_blocks::{nearest_point_from_point_to_line, CartesianCuboid};
use cellular_raza::concepts::{CalcError, CellAgent, Cycle, CycleEvent, Mechanics};
use cellular_raza::core::backend::chili;

use rand::SeedableRng;
use serde::{Deserialize, Serialize};

mod custom_domain;
use custom_domain::*;

pub const METRE: f64 = 1.0;
pub const MILI_METRE: f64 = 1e-3;
pub const MICRO_METRE: f64 = 1e-6;

pub const SECOND: f64 = 1.0;
pub const MINUTE: f64 = 60.0 * SECOND;
pub const HOUR: f64 = 24.0 * MINUTE;

/// # Mechanics
/// The cell consists of two 2D points which are coupled via a spring.
/// We represent this by a matrix
/// ```
/// # use nalgebra::Matrix5x2;
/// let cell_pos = Matrix5x2::new(
///     1.0, -2.0,
///     -2.0, 1.0
/// );
/// let cell_pos_point_0 = cell_pos.row(0);
/// let cell_pos_point_1 = cell_pos.row(1);
/// ```
#[derive(CellAgent, Clone, Deserialize, Serialize)]
pub struct Agent<const D1: usize, const D2: usize> {
    // Mechanics
    pos: nalgebra::SMatrix<f64, D1, D2>,
    vel: nalgebra::SMatrix<f64, D1, D2>,
    random_velocity: nalgebra::SMatrix<f64, D1, D2>,
    diffusion_constant: f64,
    spring_tension: f64,
    angle_stiffness: f64,
    spring_length: f64,

    // Interaction
    interaction_potential: f64,
    radius: f64,
    interaction_range: f64,

    // Cycle
    growth_rate: f64,
    max_spring_length: f64,
}

impl<const D1: usize, const D2: usize>
    cellular_raza::concepts::Mechanics<
        nalgebra::SMatrix<f64, D1, D2>,
        nalgebra::SMatrix<f64, D1, D2>,
        nalgebra::SMatrix<f64, D1, D2>,
    > for Agent<D1, D2>
{
    fn pos(&self) -> nalgebra::SMatrix<f64, D1, D2> {
        self.pos.clone()
    }

    fn velocity(&self) -> nalgebra::SMatrix<f64, D1, D2> {
        self.vel.clone()
    }

    fn set_pos(&mut self, pos: &nalgebra::SMatrix<f64, D1, D2>) {
        self.pos = pos.clone();
    }

    fn set_velocity(&mut self, velocity: &nalgebra::SMatrix<f64, D1, D2>) {
        self.vel = velocity.clone();
    }

    fn calculate_increment(
        &self,
        force: nalgebra::SMatrix<f64, D1, D2>,
    ) -> Result<
        (
            nalgebra::SMatrix<f64, D1, D2>,
            nalgebra::SMatrix<f64, D1, D2>,
        ),
        CalcError,
    > {
        use core::ops::AddAssign;
        // Calculate internal force between the two points of the Agent
        let mut total_force = force;

        // Calculate force exerted by spring action between individual vertices
        let dist_internal =
            self.pos.rows(0, self.pos.nrows() - 1) - self.pos.rows(1, self.pos.nrows() - 1);
        dist_internal.row_iter().enumerate().for_each(|(i, dist)| {
            let dir = dist.normalize();
            let force_internal = -self.spring_tension * (dist.norm() - self.spring_length) * dir;
            total_force.row_mut(i).add_assign(0.5 * force_internal);
            total_force.row_mut(i + 1).add_assign(-0.5 * force_internal);
        });

        // Calculate force exerted by angle-contributions
        use itertools::Itertools;
        dist_internal
            .row_iter()
            .tuple_windows::<(_, _)>()
            .enumerate()
            .for_each(|(i, (conn1, conn2))| {
                let angle = conn1.angle(&-conn2);
                let force_direction = (conn1.normalize() - conn2.normalize()).normalize();
                let force = self.angle_stiffness * (std::f64::consts::PI - angle) * force_direction;
                total_force.row_mut(i).add_assign(-0.5 * force);
                total_force.row_mut(i + 1).add_assign(force);
                total_force.row_mut(i + 2).add_assign(-0.5 * force);
            });
        Ok((self.vel.clone() + self.random_velocity, total_force))
    }

    fn set_random_variable(
        &mut self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: f64,
    ) -> Result<(), cellular_raza::prelude::RngError> {
        let distr = match rand_distr::Normal::new(0.0, dt.sqrt()) {
            Ok(e) => Ok(e),
            Err(e) => Err(cellular_raza::concepts::RngError(format!("{e}"))),
        }?;
        self.random_velocity = std::f64::consts::SQRT_2
            * self.diffusion_constant
            * nalgebra::SMatrix::<f64, D1, D2>::from_distribution(&distr, rng)
            / dt;

        Ok(())
    }
}

impl<const D1: usize, const D2: usize>
    cellular_raza::concepts::Interaction<
        nalgebra::SMatrix<f64, D1, D2>,
        nalgebra::SMatrix<f64, D1, D2>,
        nalgebra::SMatrix<f64, D1, D2>,
        f64,
    > for Agent<D1, D2>
{
    fn calculate_force_between(
        &self,
        own_pos: &nalgebra::SMatrix<f64, D1, D2>,
        _own_vel: &nalgebra::SMatrix<f64, D1, D2>,
        ext_pos: &nalgebra::SMatrix<f64, D1, D2>,
        _ext_vel: &nalgebra::SMatrix<f64, D1, D2>,
        ext_radius: &f64,
    ) -> Result<
        (
            nalgebra::SMatrix<f64, D1, D2>,
            nalgebra::SMatrix<f64, D1, D2>,
        ),
        CalcError,
    > {
        use core::ops::AddAssign;
        let mut force_own = nalgebra::SMatrix::<f64, D1, D2>::zeros();
        let mut force_ext = nalgebra::SMatrix::<f64, D1, D2>::zeros();
        use itertools::Itertools;
        for (i, p1) in own_pos.row_iter().enumerate() {
            for (j, (p2_n0, p2_n1)) in ext_pos.row_iter().tuple_windows::<(_, _)>().enumerate() {
                // Calculate the closest point of the external position
                let (_, nearest_point, rel_length) = nearest_point_from_point_to_line(
                    &p1.transpose(),
                    (p2_n0.transpose(), p2_n1.transpose()),
                );
                let dist = p1 - nearest_point.transpose();
                let r = dist.norm();
                if r < ext_radius + self.radius + self.interaction_range {
                    let sigma = r / (self.radius + ext_radius);
                    let strength = (1.0 / sigma.powf(4.0) - 1.0 / sigma.powf(2.0)).min(0.2);
                    let force_strength = self.interaction_potential * strength * dist.normalize();
                    force_own.row_mut(i).add_assign(-force_strength);
                    force_ext
                        .row_mut(j)
                        .add_assign(-(1.0 - rel_length) / 2.0 * force_strength);
                    force_ext
                        .row_mut((j + 1) % D1)
                        .add_assign(-rel_length / 2.0 * force_strength);
                }
            }
        }
        Ok((-force_own, force_ext))
    }

    fn get_interaction_information(&self) -> f64 {
        self.radius
    }

    fn is_neighbour(
        &self,
        own_pos: &nalgebra::SMatrix<f64, D1, D2>,
        ext_pos: &nalgebra::SMatrix<f64, D1, D2>,
        _ext_inf: &f64,
    ) -> Result<bool, CalcError> {
        for own_point in own_pos.row_iter() {
            for ext_point in ext_pos.row_iter() {
                if (own_point - ext_point).norm() < 2.0 * self.radius {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn react_to_neighbours(&mut self, neighbours: usize) -> Result<(), CalcError> {
        let base_rate = 3.0 * MICRO_METRE / MINUTE;
        if neighbours > 0 {
            self.growth_rate = (base_rate * (10.0 - neighbours as f64) / 10.0).max(0.0);
        } else {
            self.growth_rate = base_rate;
        }
        Ok(())
    }
}

impl<const D1: usize, const D2: usize> Cycle<Agent<D1, D2>> for Agent<D1, D2> {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Agent<D1, D2>,
    ) -> Option<cellular_raza::prelude::CycleEvent> {
        if cell.spring_length < cell.max_spring_length {
            cell.spring_length += dt * cell.growth_rate;
            None
        } else {
            Some(CycleEvent::Division)
        }
    }

    fn divide(
        rng: &mut rand_chacha::ChaCha8Rng,
        cell: &mut Agent<D1, D2>,
    ) -> Result<Agent<D1, D2>, cellular_raza::prelude::DivisionError> {
        use rand::Rng;
        let c1 = cell;
        let mut c2 = c1.clone();

        let base_rate = 1.5 * MICRO_METRE / MINUTE;
        c1.growth_rate = rng.gen_range(0.8 * base_rate..1.2 * base_rate);
        c2.growth_rate = rng.gen_range(0.8 * base_rate..1.2 * base_rate);

        let n_rows = c1.pos.nrows();
        // Calculate the fraction of how much we need to scale down the individual spring lengths
        // in order for the distances to still work.
        let div_factor = 0.5 - c1.radius / (n_rows as f64 * c1.spring_length);

        // Shrink spring length
        c1.spring_length *= div_factor;
        c2.spring_length *= div_factor;

        // Define new positions
        let middle = if D1 % 2 == 0 {
            1.0 * c1.pos.row(D1.div(2))
        } else {
            0.5 * (c1.pos.row(D1.div(2)) + c1.pos.row(D1.div(2) + 1))
        };
        let p1 = c1.pos.row(0).to_owned();
        let p2 = c1.pos.row(n_rows - 1).to_owned();
        let d1 = (p1 - middle) * (1.0 - div_factor);
        let d2 = (p2 - middle) * (1.0 - div_factor);
        c1.pos.row_iter_mut().for_each(|mut r| {
            r -= middle;
            r *= div_factor;
            r += middle;
            r += d1
        });
        c2.pos.row_iter_mut().for_each(|mut r| {
            r -= middle;
            r *= div_factor;
            r += middle;
            r += d2;
        });
        Ok(c2)
    }
}

fn main() -> Result<(), chili::SimulationError> {
    // Define the dimensionality of the problem
    const D1: usize = 5;
    const D2: usize = 3;

    // Define initial random seed
    use rand::Rng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);

    // Give agent default values
    let agent = Agent {
        pos: nalgebra::SMatrix::<f64, D1, D2>::zeros(),
        vel: nalgebra::SMatrix::<f64, D1, D2>::zeros(),
        random_velocity: nalgebra::SMatrix::<f64, D1, D2>::zeros(),
        diffusion_constant: 0.1 * MICRO_METRE.powf(2.0) / SECOND,
        spring_tension: 20.0 / SECOND.powf(2.0),
        angle_stiffness: 2.0 * MICRO_METRE / SECOND.powf(2.0),
        interaction_potential: 3e6 * MICRO_METRE.powf(2.0) / SECOND.powf(2.0),
        spring_length: 3.0 * MICRO_METRE,
        max_spring_length: 6.0 * MICRO_METRE,
        radius: 3.0 * MICRO_METRE,
        interaction_range: 0.25 * MICRO_METRE,
        growth_rate: 3.0 * MICRO_METRE / MINUTE,
    };

    // Place agents in simulation domain
    let domain_size = 50.0 * MICRO_METRE;
    let delta_x = agent.spring_length * D1 as f64;
    let agents = (0..5).map(|_| {
        let mut new_agent = agent.clone();
        new_agent.spring_length = rng.gen_range(1.5..2.5) * MICRO_METRE;
        let mut pos = nalgebra::SMatrix::<f64, D1, D2>::zeros();
        pos[(0, 0)] = rng.gen_range(delta_x..2.0 * delta_x);
        pos[(0, 1)] = rng.gen_range(delta_x / 3.0..delta_x * 2.0 / 3.0);
        pos[(0, 2)] = rng.gen_range(delta_x..2.0 * delta_x);
        let theta = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        for i in 1..pos.nrows() {
            let phi =
                theta + rng.gen_range(-std::f64::consts::FRAC_PI_8..std::f64::consts::FRAC_PI_8);
            let mut direction = nalgebra::SVector::<f64, D2>::zeros();
            if D2 > 0 {
                direction[0] = phi.cos();
            }
            if D2 > 1 {
                direction[1] = phi.sin();
            }
            let new_pos = pos.row(i - 1) + agent.spring_length * (direction).transpose();
            use core::ops::AddAssign;
            pos.row_mut(i).add_assign(new_pos);
        }
        new_agent.set_pos(&pos);
        new_agent
    });

    // Domain Setup
    let domain_sizes = [4.0 * domain_size, delta_x, 3.0 * delta_x];
    let domain_segments = [8, 1, 2];
    let domain = MyDomain {
        cuboid: CartesianCuboid::from_boundaries_and_n_voxels(
            [0.0; D2],
            domain_sizes,
            domain_segments,
        )?,
        // TODO enable this
        gravity: 0.0 * 2e-7 * 9.81 * METRE / SECOND.powf(2.0),
        damping: 1.5 / SECOND,
    };

    // Storage Setup
    let storage_builder = cellular_raza::prelude::StorageBuilder::new().location("./out");

    // Time Setup
    let t0 = 0.0 * MINUTE;
    let dt = 0.05 * SECOND;
    let save_interval = 2.0 * SECOND;
    let t_max = 30.0 * MINUTE;
    let time_stepper = cellular_raza::prelude::time::FixedStepsize::from_partial_save_interval(
        t0,
        dt,
        t_max,
        save_interval,
    )?;

    let settings = chili::Settings {
        n_threads: 8.try_into().unwrap(),
        time: time_stepper,
        storage: storage_builder,
        show_progressbar: true,
    };

    println!("Running Simulation");
    let _storage = chili::run_simulation!(
        domain: domain,
        agents: agents,
        settings: settings,
        aspects: [Mechanics, Interaction, DomainForce, Cycle],
    )?;

    Ok(())
}
