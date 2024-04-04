use crate::storage::StorageManager;
use super::{CellIdentifier, VoxelPlainIndex};

/// Gathers the [StorageManager] for cells and voxels of the previously run simulation
pub struct StorageAccess<C, V> {
    /// Access cells at their saved iteration steps
    pub cells: StorageManager<CellIdentifier, C>,
    /// Access voxels at their saved iteration steps
    pub voxels: StorageManager<VoxelPlainIndex, V>,
}
