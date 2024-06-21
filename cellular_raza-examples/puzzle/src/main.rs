use cellular_raza::building_blocks::{CartesianCuboid, CartesianSubDomain};
use cellular_raza::concepts::domain_new::*;
use cellular_raza::concepts::{
    BoundaryError, CalcError, CellAgent, DecomposeError, IndexError, Mechanics, RngError,
};
use cellular_raza::core::backend::chili;
use cellular_raza::core::time::FixedStepsize;
use cellular_raza::prelude::StorageBuilder;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

mod puzzle_mechanics;
// mod two_three_tree;

use puzzle_mechanics::*;

#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
pub struct Agent {
    #[Mechanics]
    mechanics: Puzzle<f64>,
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
    let dx = domain_size / 3.0;
    vec![
        Vector2::from([1.0 * dx, 1.0 * dx]),
        Vector2::from([2.0 * dx, 1.0 * dx]),
        Vector2::from([1.0 * dx, 2.0 * dx]),
        Vector2::from([2.0 * dx, 2.0 * dx]),
    ]
}

fn main() -> Result<(), chili::SimulationError> {
    let domain_size = 100.0;
    let n_vertices = 40;
    let angle_stiffness = 1.0;
    let surface_tension = 0.05;
    let boundary_length = 50.0;
    let cell_area = 300.0;
    let internal_pressure = 0.00025;
    let diffusion_constant = 0.0;
    let damping = 0.5;
    let agents = generate_initial_points(domain_size)
        .into_iter()
        .enumerate()
        .map(|(n_agent, middle)| Agent {
            mechanics: Puzzle::new_equilibrium(
                middle,
                n_vertices,
                angle_stiffness,
                surface_tension,
                boundary_length,
                cell_area,
                internal_pressure,
                diffusion_constant,
                damping,
                Some((0.5, 10 * n_agent as u64)),
            ),
        });
    let domain = MyDomain {
        cuboid: CartesianCuboid::from_boundaries_and_n_voxels([0.0; 2], [domain_size; 2], [1; 2])?,
    };
    let settings = chili::Settings {
        n_threads: 1.try_into().unwrap(),
        time: FixedStepsize::from_partial_save_interval(0.0, 1e-1, 1e2, 5e-1)?,
        storage: StorageBuilder::new().location("out/puzzles"),
        show_progressbar: true,
    };
    chili::run_simulation!(
        agents: agents,
        domain: domain,
        settings: settings,
        aspects: [Mechanics],
    )?;
    Ok(())
}
