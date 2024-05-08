use cellular_raza::building_blocks::{CartesianCuboid, CartesianSubDomain};
use cellular_raza::concepts::domain_new::*;
use cellular_raza::concepts::{BoundaryError, CalcError, DecomposeError, IndexError, Mechanics};

use crate::Agent;

#[derive(Clone, Domain)]
pub struct MyDomain<const D2: usize> {
    #[DomainRngSeed]
    pub cuboid: CartesianCuboid<f64, D2>,
    pub gravity: f64,
    pub damping: f64,
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
        let gravity = self.gravity;
        let damping = self.damping;
        let subdomains = self.cuboid.create_subdomains(n_subdomains)?;
        Ok(subdomains
            .into_iter()
            .map(move |(subdomain_index, subdomain, voxels)| {
                (
                    subdomain_index,
                    MySubDomain {
                        subdomain,
                        force: MySubDomainForce { gravity, damping },
                    },
                    voxels,
                )
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
pub struct MySubDomain<const D2: usize> {
    #[Base]
    pub subdomain: CartesianSubDomain<f64, D2>,
    #[Force]
    pub force: MySubDomainForce,
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

#[derive(Clone)]
pub struct MySubDomainForce {
    pub damping: f64, // 0.75 / SECOND,
    pub gravity: f64,
}

impl<const D1: usize, const D2: usize>
    SubDomainForce<
        nalgebra::SMatrix<f64, D1, D2>,
        nalgebra::SMatrix<f64, D1, D2>,
        nalgebra::SMatrix<f64, D1, D2>,
    > for MySubDomainForce
{
    fn calculate_custom_force(
        &self,
        _pos: &nalgebra::SMatrix<f64, D1, D2>,
        vel: &nalgebra::SMatrix<f64, D1, D2>,
    ) -> Result<nalgebra::SMatrix<f64, D1, D2>, cellular_raza::concepts::CalcError> {
        let mut force = nalgebra::SMatrix::<f64, D1, D2>::zeros();
        // Gravity
        force.column_mut(D2 - 1).add_scalar_mut(-self.gravity);

        // Damping force
        force -= self.damping * vel;
        Ok(force)
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
