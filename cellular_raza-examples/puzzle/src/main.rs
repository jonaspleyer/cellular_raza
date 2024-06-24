use std::marker::PhantomData;

use cellular_raza::building_blocks::{CartesianCuboid, CartesianSubDomain};
use cellular_raza::concepts::*;
use cellular_raza::core::backend::chili;
use cellular_raza::core::time::FixedStepsize;
use cellular_raza::prelude::StorageBuilder;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

mod puzzle_mechanics;

use puzzle_mechanics::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct InsideInteraction<F> {
    pub strength: F,
    pub radius: F,
}

impl<F> Interaction<Vector2<F>, Vector2<F>, Vector2<F>> for InsideInteraction<F>
where
    F: Copy + nalgebra::RealField,
{
    fn calculate_force_between(
        &self,
        own_pos: &Vector2<F>,
        _own_vel: &Vector2<F>,
        ext_pos: &Vector2<F>,
        _ext_vel: &Vector2<F>,
        _ext_info: &(),
    ) -> Result<(Vector2<F>, Vector2<F>), CalcError> {
        let dir = own_pos - ext_pos;
        let r2 = dir.norm_squared() / self.radius.powf(F::one() + F::one());
        let force: Vector2<F> = if !r2.is_zero() {
            dir * self.strength / r2
        } else {
            Vector2::zeros()
        };
        Ok((force, -force))
    }

    fn get_interaction_information(&self) -> () {}
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutsideInteraction<F> {
    pub attraction: F,
}

impl<F> Interaction<Vector2<F>, Vector2<F>, Vector2<F>> for OutsideInteraction<F>
where
    F: Copy + nalgebra::RealField,
    Vector2<F>: core::ops::Mul<F, Output = Vector2<F>>,
{
    fn calculate_force_between(
        &self,
        own_pos: &Vector2<F>,
        _own_vel: &Vector2<F>,
        ext_pos: &Vector2<F>,
        _ext_vel: &Vector2<F>,
        _ext_info: &(),
    ) -> Result<(Vector2<F>, Vector2<F>), CalcError> {
        let dir = own_pos - ext_pos;
        let r = dir.norm();
        let force = if !r.is_zero() {
            dir * self.attraction / r
        } else {
            Vector2::zeros()
        };
        Ok((-force, force))
    }

    fn get_interaction_information(&self) -> () {}
}

#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
pub struct Agent {
    #[Mechanics]
    #[Interaction]
    mechanics: PuzzleInteraction<f64, InsideInteraction<f64>, OutsideInteraction<f64>, (), ()>,
}

#[derive(Clone, Domain)]
pub struct MyDomain {
    #[DomainRngSeed]
    pub cuboid: CartesianCuboid<f64, 2>,
}

impl SortCells<Agent> for MyDomain {
    type VoxelIndex = [usize; 2];

    fn get_voxel_index_of(&self, cell: &Agent) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.mechanics.pos();
        let mean = pos.mean();
        self.cuboid.get_voxel_index_of_raw(&mean)
    }
}

impl DomainCreateSubDomains<MySubDomain> for MyDomain {
    type SubDomainIndex = usize;
    type VoxelIndex = [usize; 2];

    fn create_subdomains(
        &self,
        n_subdomains: core::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<Item = (Self::SubDomainIndex, MySubDomain, Vec<Self::VoxelIndex>)>,
        DecomposeError,
    > {
        Ok(self
            .cuboid
            .create_subdomains(n_subdomains)?
            .into_iter()
            .map(|(index, subdomain, voxels)| (index, MySubDomain { subdomain }, voxels)))
    }
}

#[derive(Clone, SubDomain)]
pub struct MySubDomain {
    #[Base]
    pub subdomain: CartesianSubDomain<f64, 2>,
}

impl SortCells<Agent> for MySubDomain {
    type VoxelIndex = [usize; 2];

    fn get_voxel_index_of(&self, cell: &Agent) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos().mean();
        let n_vox = (pos - self.subdomain.get_domain_min()).component_div(&self.subdomain.get_dx());
        let mut res = [0usize; 2];
        for i in 0..2 {
            res[i] = n_vox[i] as usize;
        }
        Ok(res.into())
    }
}

impl SubDomainMechanics<Vertices<f64>, Vertices<f64>> for MySubDomain {
    fn apply_boundary(
        &self,
        pos: &mut Vertices<f64>,
        vel: &mut Vertices<f64>,
    ) -> Result<(), BoundaryError> {
        pos.0
            .iter_mut()
            .zip(vel.0.iter_mut())
            .map(|(mut pos, mut vel)| {
                <CartesianSubDomain<f64, 2> as SubDomainMechanics<
                    nalgebra::SVector<f64, 2>,
                    nalgebra::SVector<f64, 2>,
                >>::apply_boundary(&self.subdomain, &mut pos, &mut vel)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }
}

fn generate_initial_points(domain_size: f64) -> Vec<nalgebra::Vector2<f64>> {
    use nalgebra::Vector2;
    let n_grid = 3;
    let dx = domain_size / n_grid as f64;
    let mut positions = vec![];
    for i in 0..(n_grid - 1) {
        for j in 0..(n_grid - 1) {
            let pos = Vector2::from([(1 + i) as f64 * dx, (1 + j) as f64 * dx]);
            positions.push(pos);
        }
    }
    positions
}

fn main() -> Result<(), chili::SimulationError> {
    let domain_size = 40.0;
    let n_vertices = 20;
    let angle_stiffness = 0.03;
    let surface_tension = 0.01;
    let boundary_length = 60.0;
    let cell_area = 150.0;
    let internal_pressure = 2.5e-3;
    let diffusion_constant = 0.0;
    let damping = 0.1;
    let agents = generate_initial_points(domain_size)
        .into_iter()
        .enumerate()
        .map(|(n_agent, middle)| Agent {
            mechanics: PuzzleInteraction {
                puzzle: Puzzle::new_equilibrium(
                    middle,
                    n_vertices,
                    angle_stiffness,
                    surface_tension,
                    boundary_length,
                    cell_area,
                    internal_pressure,
                    diffusion_constant,
                    damping,
                    Some((0.1, 10 * n_agent as u64)),
                ),
                bounding_min: [std::f64::NEG_INFINITY; 2].into(),
                bounding_max: [std::f64::INFINITY; 2].into(),
                inside_force: InsideInteraction {
                    strength: 1e-6,
                    radius: cell_area.sqrt(),
                },
                outside_force: OutsideInteraction { attraction: 0.0 },
                phantom_inf_outside: PhantomData,
                phantom_inf_inside: PhantomData,
            },
        });
    let domain = MyDomain {
        cuboid: CartesianCuboid::from_boundaries_and_n_voxels([0.0; 2], [domain_size; 2], [3; 2])?,
    };
    let settings = chili::Settings {
        n_threads: 1.try_into().unwrap(),
        time: FixedStepsize::from_partial_save_interval(0.0, 5e-1, 1e4, 2e1)?,
        storage: StorageBuilder::new().location("out/puzzles"),
        show_progressbar: true,
    };
    chili::run_simulation!(
        agents: agents,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction],
    )?;
    Ok(())
}
