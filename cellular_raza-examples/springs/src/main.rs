use cellular_raza::building_blocks::CartesianCuboid;
use cellular_raza::concepts::{CalcError, CellAgent, Mechanics};
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
    pos: nalgebra::SMatrix<f64, D1, D2>,
    vel: nalgebra::SMatrix<f64, D1, D2>,
    spring_tension: f64,
    angle_stiffness: f64,
    interaction_potential: f64,
    spring_length: f64,
    radius: f64,
    interaction_range: f64,
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
        // Calculate internal force between the two points of the Agent<D1, D2>
        let mut total_force = force;

        // Calculate force exerted by spring action between individual vertices
        let dist_internal = self.pos.rows(0, 4) - self.pos.rows(1, 4);
        dist_internal.row_iter().enumerate().for_each(|(i, dist)| {
            let dir = dist.normalize();
            let force_internal = -self.spring_tension * (dist.norm() - self.spring_length) * dir;
            use core::ops::AddAssign;
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
                use core::ops::AddAssign;
                total_force.row_mut(i).add_assign(-0.5 * force);
                total_force.row_mut(i + 1).add_assign(force);
                total_force.row_mut(i + 2).add_assign(-0.5 * force);
            });
        Ok((self.vel.clone(), total_force))
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
    ) -> Result<nalgebra::SMatrix<f64, D1, D2>, CalcError> {
        use core::ops::AddAssign;
        let mut force = nalgebra::SMatrix::<f64, D1, D2>::zeros();
        for (i, p1) in own_pos.row_iter().enumerate() {
            for (j, p2) in ext_pos.row_iter().enumerate() {
                let dist = p1 - p2;
                let r = dist.norm();
                if r < ext_radius + self.radius + self.interaction_range {
                    let sigma = r / (self.radius + ext_radius);
                    let strength = (1.0 / sigma.powf(4.0) - 1.0 / sigma.powf(2.0)).min(0.2);
                    force
                        .row_mut(i)
                        .add_assign(-self.interaction_potential * strength * dist.normalize());
                    force
                        .row_mut(j)
                        .add_assign(-self.interaction_potential * strength * dist.normalize());
                }
            }
        }
        Ok(force)
    }

    fn get_interaction_information(&self) -> f64 {
        self.radius
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
        spring_tension: 2.0 / SECOND.powf(2.0),
        angle_stiffness: 20.0 * MICRO_METRE / SECOND.powf(2.0),
        interaction_potential: 2.0 * MICRO_METRE.powf(2.0) / SECOND.powf(2.0),
        spring_length: 3.0 * MICRO_METRE,
        radius: 3.0 * MICRO_METRE,
        interaction_range: 1.5 * MICRO_METRE,
    };

    // Place agents in simulation domain
    let domain_size = 100.0 * MICRO_METRE;
    let agents = (0..40).map(|_| {
        let mut new_agent = agent.clone();
        let mut pos = nalgebra::SMatrix::<f64, D1, D2>::zeros();

        let delta_x = agent.spring_length * pos.nrows() as f64;
        let lower = delta_x;
        let upper = domain_size - delta_x;
        pos[(0, 0)] = rng.gen_range(lower..upper);
        pos[(0, 1)] = rng.gen_range(lower..upper);
        if D2 > 2 {
            pos[(0, 2)] = domain_size - delta_x / 80.0 * rng.gen_range(0.0..1.0);
        }
        let theta = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        for i in 1..pos.nrows() {
            let phi =
                theta + rng.gen_range(-std::f64::consts::FRAC_PI_4..std::f64::consts::FRAC_PI_4);
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
    let domain_sizes = [domain_size; D2];
    let domain = MyDomain {
        cuboid: CartesianCuboid::from_boundaries_and_n_voxels([0.0; D2], domain_sizes, [4; D2])?,
        gravity: 2e-7 * 9.81 * METRE / SECOND.powf(2.0),
        damping: 1.5 / SECOND,
    };

    // Storage Setup
    let storage_builder = cellular_raza::prelude::StorageBuilder::new().location("./out");

    // Time Setup
    let t0 = 0.0 * MINUTE;
    let dt = 0.025 * SECOND;
    let save_interval = 0.2 * SECOND;
    let t_max = 1.0 * MINUTE;
    let time_stepper = cellular_raza::prelude::time::FixedStepsize::from_partial_save_interval(
        t0,
        dt,
        t_max,
        save_interval,
    )?;

    let settings = chili::Settings {
        n_threads: 1.try_into().unwrap(),
        time: time_stepper,
        storage: storage_builder,
        show_progressbar: true,
    };

    let _storage = chili::run_simulation!(
        domain: domain,
        agents: agents,
        settings: settings,
        aspects: [Mechanics, Interaction, DomainForce],
    )?;

    Ok(())
}
