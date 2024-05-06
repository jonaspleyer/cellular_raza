use cellular_raza::building_blocks::{CartesianCuboid, CartesianSubDomain};
use cellular_raza::concepts::domain_new::*;
use cellular_raza::concepts::{
    BoundaryError, CalcError, CellAgent, DecomposeError, IndexError, Mechanics,
};
use cellular_raza::core::backend::chili;

use rand::SeedableRng;
use serde::{Deserialize, Serialize};

///
///
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
struct Agent<const D1: usize, const D2: usize> {
    pos: nalgebra::SMatrix<f64, D1, D2>,
    vel: nalgebra::SMatrix<f64, D1, D2>,
    spring_tension: f64,
    angle_stiffness: f64,
    interaction_potential: f64,
    spring_length: f64,
    damping: f64,
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
        Ok((self.vel.clone(), total_force - self.damping * self.vel))
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

#[derive(Clone, Domain)]
struct MyDomain<const D2: usize> {
    #[DomainRngSeed]
    cuboid: CartesianCuboid<f64, D2>,
}

impl<const D2: usize> cellular_raza::concepts::domain_new::DomainCreateSubDomains<MySubDomain<D2>>
    for MyDomain<D2>
{
    type SubDomainIndex = usize;
    type VoxelIndex = [usize; D2];

    fn create_subdomains(
        &self,
        n_subdomains: std::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<Item = (Self::SubDomainIndex, MySubDomain<D2>, Vec<Self::VoxelIndex>)>,
        DecomposeError,
    > {
        let subdomains = self.cuboid.create_subdomains(n_subdomains)?;
        Ok(subdomains
            .into_iter()
            .map(|(subdomain_index, subdomain, voxels)| {
                (subdomain_index, MySubDomain { subdomain }, voxels)
            }))
    }
}

impl<const D1: usize, const D2: usize> cellular_raza::concepts::domain_new::SortCells<Agent<D1, D2>>
    for MyDomain<D2>
{
    type VoxelIndex = [usize; D2];

    fn get_voxel_index_of(&self, cell: &Agent<D1, D2>) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos();
        let index = (pos.row_mean().transpose() - self.cuboid.get_min())
            .component_div(&self.cuboid.get_dx());
        let res: [usize; D2] = index.try_cast::<usize>().unwrap().into();
        Ok(res)
    }
}

#[derive(Clone, SubDomain)]
struct MySubDomain<const D2: usize> {
    #[Base]
    subdomain: CartesianSubDomain<f64, D2>,
}

impl<const D1: usize, const D2: usize>
    cellular_raza::concepts::domain_new::SubDomainMechanics<
        nalgebra::SMatrix<f64, D1, D2>,
        nalgebra::SMatrix<f64, D1, D2>,
    > for MySubDomain<D2>
{
    fn apply_boundary(
        &self,
        pos: &mut nalgebra::SMatrix<f64, D1, D2>,
        vel: &mut nalgebra::SMatrix<f64, D1, D2>,
    ) -> Result<(), BoundaryError> {
        // TODO refactor this with matrix multiplication!!!
        // This will probably be much more efficient and less error-prone!

        // For each position in the springs Agent<D1, D2>
        pos.row_iter_mut()
            .zip(vel.row_iter_mut())
            .for_each(|(mut p, mut v)| {
                // For each dimension in the space
                for i in 0..p.ncols() {
                    // Check if the particle is below lower edge
                    if p[i] < self.subdomain.get_domain_min()[i] {
                        p[i] = 2.0 * self.subdomain.get_domain_min()[i] - p[i];
                        v[i] = v[i].abs();
                    }

                    // Check if the particle is over the edge
                    if p[i] > self.subdomain.get_domain_max()[i] {
                        p[i] = 2.0 * self.subdomain.get_domain_max()[i] - p[i];
                        v[i] = -v[i].abs();
                    }
                }
            });

        // If new pos is still out of boundary return error
        for j in 0..pos.nrows() {
            let p = pos.row(j);
            for i in 0..pos.ncols() {
                if p[i] < self.subdomain.get_domain_min()[i]
                    || p[i] > self.subdomain.get_domain_max()[i]
                {
                    return Err(BoundaryError(format!(
                        "Particle is out of domain at pos {:?}",
                        pos
                    )));
                }
            }
        }
        Ok(())
    }
}

impl<const D1: usize, const D2: usize> cellular_raza::concepts::domain_new::SortCells<Agent<D1, D2>>
    for MySubDomain<D2>
{
    type VoxelIndex = [usize; D2];

    fn get_voxel_index_of(&self, cell: &Agent<D1, D2>) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos();
        let mut out = [0; D2];

        for i in 0..pos.ncols() {
            out[i] = ((pos[i] - self.subdomain.get_domain_min()[0]) / self.subdomain.get_dx()[i])
                as usize;
            out[i] = out[i]
                .min(self.subdomain.get_domain_n_voxels()[i] - 1)
                .max(0);
        }
        Ok(out.into())
    }
}

fn main() -> Result<(), chili::SimulationError> {
    // Define the dimensionality of the problem
    const D1: usize = 5;
    const D2: usize = 2;

    // Define initial random seed
    use rand::Rng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);

    // Give agent default values
    let agent = Agent {
        pos: nalgebra::SMatrix::<f64, D1, D2>::zeros(),
        vel: nalgebra::SMatrix::<f64, D1, D2>::zeros(),
        spring_tension: 2.0,
        angle_stiffness: 20.0,
        interaction_potential: 3.0,
        spring_length: 3.0,
        damping: 0.75,
        radius: 3.0,
        interaction_range: 1.5,
    };

    // Place agents in simulation domain
    let domain_size = 100.0;
    let agents = (0..40).map(|_| {
        let mut new_agent = agent.clone();
        let mut pos = nalgebra::SMatrix::<f64, D1, D2>::zeros();

        let delta_x = agent.spring_length * pos.nrows() as f64;
        // let lower = domain_size / 2.0 - delta_x;
        // let upper = domain_size / 2.0 + delta_x;
        let lower = delta_x;
        let upper = domain_size - delta_x;
        pos[(0, 0)] = rng.gen_range(lower..upper);
        pos[(0, 1)] = rng.gen_range(lower..upper);
        if D2 > 2 {
            pos[(0, 2)] = domain_size / 20.0 + delta_x / 80.0 * rng.gen_range(-1.0..1.0);
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
    let mut domain_sizes = [domain_size; D2];
    if D2 > 2 {
        domain_sizes[2] /= 10.0;
    }
    let domain = MyDomain {
        cuboid: CartesianCuboid::from_boundaries_and_n_voxels([0.0; D2], domain_sizes, [4; D2])?,
    };

    // Storage Setup
    let storage_builder = cellular_raza::prelude::StorageBuilder::new().location("./out");

    // Time Setup
    let t0 = 0.0;
    let dt = 0.0025;
    let save_interval = 0.2;
    let t_max = 50.0;
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
        aspects: [Mechanics, Interaction],
    )?;

    Ok(())
}
