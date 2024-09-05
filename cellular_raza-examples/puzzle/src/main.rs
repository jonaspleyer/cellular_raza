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
        let r = dir.norm() / self.radius;
        let force: Vector2<F> = if !r.is_zero() {
            dir / r * self.strength * (F::one() + r.powf(F::one() + F::one()))
        } else {
            Vector2::zeros()
        };
        Ok((-force, force))
    }

    fn get_interaction_information(&self) -> () {}
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutsideInteraction<F> {
    pub attraction: F,
    pub radius: F,
    pub cutoff: F,
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
        let r = dir.norm() / self.radius;
        let force = if !r.is_zero() && r < self.cutoff {
            dir * self.attraction / (F::one() + r)
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
    growth_factor: f64,
}

impl Cycle for Agent {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Self,
    ) -> Option<CycleEvent> {
        cell.mechanics.puzzle.boundary_length += cell.growth_factor * dt;
        None
    }

    fn divide(_: &mut rand_chacha::ChaCha8Rng, _: &mut Self) -> Result<Self, DivisionError> {
        todo!()
    }
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

#[derive(Clone, SubDomain, Serialize)]
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
        pos.iter_mut()
            .zip(vel.iter_mut())
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

fn generate_initial_points(n_cells: usize, domain_size: f64) -> Vec<nalgebra::Vector2<f64>> {
    let n_grid: usize = (n_cells as f64).sqrt().ceil() as usize;
    let dx = domain_size / (n_grid + 1) as f64;
    let mut positions = vec![];
    for i in 0..n_grid {
        for j in 0..n_grid {
            if i * n_grid + j < n_cells {
                let pos = Vector2::from([(1 + i) as f64 * dx, (j + 1) as f64 * dx]);
                positions.push(pos);
            }
        }
    }
    positions
}

fn main() -> Result<(), chili::SimulationError> {
    let radius = 5.0;
    let domain_size = 35.0;
    let n_vertices = 100;
    let angle_stiffness = 0.03;
    let surface_tension = 0.1;
    let boundary_length = 1.2 * 2.0 * std::f64::consts::PI * radius;
    let cell_area = std::f64::consts::PI * radius.powf(2.0);
    let internal_pressure = 2.5e-2;
    let diffusion_constant = 0.0;
    let damping = 0.1;
    let agents = generate_initial_points(9, domain_size)
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
                    strength: 5e-3,
                    radius,
                },
                outside_force: OutsideInteraction {
                    attraction: 3e-4,
                    radius,
                    cutoff: 0.3 * radius,
                },
                phantom_inf_outside: PhantomData,
                phantom_inf_inside: PhantomData,
            },
            growth_factor: 0.001,
        });
    let domain = MyDomain {
        cuboid: CartesianCuboid::from_boundaries_and_n_voxels([0.0; 2], [domain_size; 2], [3; 2])?,
    };
    let settings = chili::Settings {
        n_threads: 4.try_into().unwrap(),
        time: FixedStepsize::from_partial_save_interval(0.0, 5e-1, 4e4, 4e1)?,
        storage: StorageBuilder::new().location("out/puzzles"),
        show_progressbar: true,
    };
    chili::run_simulation!(
        agents: agents,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction, Cycle],
    )?;
    Ok(())
}
