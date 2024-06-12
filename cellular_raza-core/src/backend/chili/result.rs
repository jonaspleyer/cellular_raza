use super::{CellIdentifier, SimulationError, VoxelPlainIndex};
use crate::storage::StorageManager;

/// Gathers the [StorageManager] for cells and voxels of the previously run simulation
pub struct StorageAccess<C, V> {
    /// Access cells at their saved iteration steps
    pub cells: StorageManager<CellIdentifier, C>,
    /// Access voxels at their saved iteration steps
    pub voxels: StorageManager<VoxelPlainIndex, V>,
}

impl<C, V> StorageAccess<C, V> {
    /// Obtain the save path for cells and voxels of this simulation
    pub fn get_path(&self) -> Result<std::path::PathBuf, SimulationError> {
        match self.cells.extract_builder().get_full_path().parent() {
            Some(p) => Ok(p.to_path_buf()),
            None => Err(SimulationError::IoError(std::io::Error::from(
                std::io::ErrorKind::NotFound,
            ))),
        }
    }
}
