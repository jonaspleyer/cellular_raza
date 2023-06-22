use std::collections::HashMap;

use cellular_raza_concepts::errors::{BoundaryError, CalcError};

use super::{
    aux_storage::{SubDomainBox, Voxel},
    simulation_flow::SyncSubDomains,
};

/// Provides an abstraction of the physical total simulation domain.
///
/// [cellular_raza](https://github.com/jonaspleyer/cellular_raza) uses domain-decomposition
/// algorithms to split up the computational workload over multiple physical regions.
/// That's why the domain itself is mostly responsible for being deconstructed into smaller [SubDmains](SubDomain)
/// which can then be used to numerically solve our system.
pub trait Domain<C, S> {
    /// Subdomains can be identified by their unique [SubDomainIndex](Domain::SubDomainIndex).
    /// The backend uses this property to construct a mapping (graph) between subdomains.
    type SubDomainIndex;
    /// Similarly to the [SubDomainIndex](Domain::SubDomainIndex), voxels can be accessed by their unique
    /// index. The backend will use this information to construct a mapping (graph) between voxels
    /// inside their respective subdomains.
    type VoxelIndex;

    /// Get indices of other subdomains which are neighbors of specified index.
    ///
    /// This function is mainly used to build a map (graph) between subdomains.
    fn get_neighbor_subdomains(&self, index: &Self::SubDomainIndex) -> Vec<Self::SubDomainIndex>;

    /// Retrieves all indices of subdomains.
    fn get_all_voxel_indices(&self) -> Vec<Self::VoxelIndex>;

    /// Deconstructs the [Domain] into its respective subdomains.
    ///
    /// In addition, we provide the initial cells for the simulation
    fn decompose(
        self,
        n_subdomains: usize,
        cells: Vec<C>,
    ) -> Result<DecomposedDomain<Self::SubDomainIndex, S, C>, CalcError>;
}

pub struct DecomposedDomain<I, S, C> {
    pub n_subdomains: usize,
    pub index_subdomain_cells: Vec<(I, S, Vec<C>)>,
    pub neighbor_map: HashMap<I, Vec<I>>,
    pub rng_seed: u64,
}

impl<I, S, C> DecomposedDomain<I, S, C>
where
    S: SubDomain<C>,
    S::VoxelIndex: Eq + Ord,
    I: Eq + PartialEq + core::hash::Hash + Clone,
{
    // TODO this is not a BoundaryError
    pub(crate) fn into_subdomain_boxes<A, Sy>(
        self,
    ) -> Result<Vec<SubDomainBox<S, C, A, Sy>>, BoundaryError>
    where
        A: Default,
        Sy: SyncSubDomains,
    {
        use rand::SeedableRng;
        let mut syncers = Sy::from_map(self.neighbor_map);

        self.index_subdomain_cells
            .into_iter()
            .map(|(index, subdomain, cells)| {
                let mut cells = cells.into_iter().map(|c| (c, None)).collect();
                let voxels = subdomain.get_all_indices().into_iter().map(|voxel_index| {
                    (
                        voxel_index,
                        Voxel {
                            cells: Vec::new(),
                            new_cells: Vec::new(),
                            id_counter: 0,
                            rng: rand_chacha::ChaCha8Rng::seed_from_u64(self.rng_seed),
                        },
                    )
                });
                let syncer = syncers.remove(&index).ok_or(BoundaryError {
                    message: "Index was not present in subdomain map".into(),
                })?;
                let mut subdomain_box = SubDomainBox {
                    subdomain,
                    voxels: voxels.collect(),
                    syncer,
                };
                subdomain_box.insert_cells(&mut cells)?;
                Ok(subdomain_box)
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

pub trait SubDomain<C> {
    type VoxelIndex;

    fn get_voxel_index_of(&self, cell: &C) -> Result<Self::VoxelIndex, BoundaryError>;
    fn apply_boundary(&self, cell: &mut C) -> Result<(), BoundaryError>;
    fn get_all_indices(&self) -> Vec<Self::VoxelIndex>;
}

pub trait Id {
    type Identifier;

    fn get_id(&self) -> Self::Identifier;
}
