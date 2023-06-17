use cellular_raza_concepts::errors::{BoundaryError, CalcError};

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
    fn into_subdomains(
        self,
        n: usize,
        cells: Vec<C>,
    ) -> Result<Vec<(Self::SubDomainIndex, S, Vec<C>)>, CalcError>;
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
