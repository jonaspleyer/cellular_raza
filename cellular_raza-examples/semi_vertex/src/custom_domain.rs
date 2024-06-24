use cellular_raza::building_blocks::{CartesianCuboid, CartesianSubDomain};
use cellular_raza::concepts::*;

use crate::MyCell;

#[derive(Clone, Domain)]
pub struct MyDomain {
    #[DomainRngSeed]
    pub cuboid: CartesianCuboid<f64, 2>,
}

impl DomainCreateSubDomains<MySubDomain> for MyDomain {
    type SubDomainIndex = usize;
    type VoxelIndex = [usize; 2];

    fn create_subdomains(
        &self,
        n_subdomains: std::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<Item = (Self::SubDomainIndex, MySubDomain, Vec<Self::VoxelIndex>)>,
        DecomposeError,
    > {
        let subdomains = self.cuboid.create_subdomains(n_subdomains)?;
        Ok(subdomains
            .into_iter()
            .map(move |(subdomain_index, subdomain, voxels)| {
                (subdomain_index, MySubDomain { subdomain }, voxels)
            }))
    }
}

impl<const D: usize> SortCells<MyCell<D>> for MyDomain {
    type VoxelIndex = [usize; 2];

    fn get_voxel_index_of(&self, cell: &MyCell<D>) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos().row_mean().transpose();
        self.cuboid.get_voxel_index_of_raw(&pos)
    }
}

#[derive(Clone, SubDomain)]
pub struct MySubDomain {
    #[Base]
    pub subdomain: CartesianSubDomain<f64, 2>,
}

impl<const D: usize> SortCells<MyCell<D>> for MySubDomain {
    type VoxelIndex = [usize; 2];

    fn get_voxel_index_of(&self, cell: &MyCell<D>) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos().row_mean().transpose();
        self.subdomain.get_index_of(pos)
    }
}

impl<const D: usize>
    SubDomainMechanics<
        nalgebra::SMatrix<f64, D, 2>,
        nalgebra::SMatrix<f64, D, 2>,
    > for MySubDomain
{
    fn apply_boundary(
        &self,
        pos: &mut nalgebra::SMatrix<f64, D, 2>,
        vel: &mut nalgebra::SMatrix<f64, D, 2>,
    ) -> Result<(), BoundaryError> {
        // TODO refactor this with matrix multiplication!!!
        // This will probably be much more efficient and less error-prone!

        // For each position in the springs MyCell<D>
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
