use crate::concepts::cell::{Id,CellularIdentifier};
use crate::concepts::errors::{SimulationError,IndexError};

use crate::storage::sled_database::io::{
    store_elements_in_tree,
    store_single_element_in_tree,
    deserialize_all_elements_from_tree,
    deserialize_all_elements_from_tree_group,
    deserialize_single_element_from_tree
};

use super::config::SimulationSetup;
use super::domain_decomposition::{PlainIndex,GetPlainIndex};

use std::collections::HashMap;

use serde::{Serialize,Deserialize};


// ############################################################
// Datastructures for storing cells, voxels, setup, etc.
// ############################################################

/// Datastructure for storing [Cells](crate::concepts::cell::CellAgent) at iteration of simulation
pub type CellStorage<Cbox> = HashMap<u64, Vec<Cbox>>;
/// Datastructure for storing [CellHistories](crate::concepts::cell::CellAgent)
pub type CellHistories<Cbox> = HashMap<CellularIdentifier, Vec<(u64, Cbox)>>;

/// Datastructure for storing [Voxels](crate::concepts::domain::Voxel)
pub type VoxelStorage<VBox> = HashMap<u64, Vec<VBox>>;
/// Datastructure for storing [VoxelHistories](crate::concepts::domain::Voxel)
pub type VoxelHistories<VBox> = HashMap<PlainIndex, Vec<(u64, VBox)>>;

/// Datastructure for storing [SimulationSetups](SimulationSetup)
pub type SetupStorage<SimSetup> = Vec<(u64, SimSetup)>;


// ############################################################
// Public - Functions for converting between different cell storage types
// ############################################################

pub fn convert_cell_storage_to_cell_histories<Cbox>
(
    cell_storage: CellStorage<Cbox>,
    sorted: bool
) -> CellHistories<Cbox>
where
    Cbox: Id,
{
    let mut histories = cell_storage.into_iter()
        .fold(CellHistories::<Cbox>::new(), |mut acc, (iteration, cells)| {
            cells.into_iter()
                .for_each(|cell| match acc.get_mut(&cell.get_id()) {
                    Some(cells) => cells.push((iteration, cell)),
                    None => {acc.insert(cell.get_id(), vec![(iteration, cell)]);},
                });
            acc
        });
    if sorted {
        histories.iter_mut()
            .for_each(|(_, cell_history)| cell_history.sort_by(|(iteration1, _), (iteration2, _)| iteration1.cmp(iteration2)));
    }
    histories
}


pub fn convert_cell_histories_to_cell_storage<Cbox>
(
    cell_histories: CellHistories<Cbox>
) -> CellStorage<Cbox>
where
    Cbox: Id,
{
    let storage = cell_histories.into_iter()
        .fold(CellStorage::<Cbox>::new(), |mut acc, (_ , cell_history)| {
            cell_history.into_iter()
                .for_each(|(iteration, cell)| match acc.get_mut(&iteration) {
                    Some(cells) => cells.push(cell),
                    None => {acc.insert(iteration, vec![cell]);},
                });
            acc
        });
    storage
}


// ############################################################
// Private - Conversion of database entry (String) and cell-properties
// ############################################################

fn iteration_id_to_db_key(iteration: u64, id: &CellularIdentifier) -> String
{
    format!("{:020.0}_{:020.0}_{:020.0}", iteration, id.0, id.1)
}


fn iteration_cell_to_db_key<Cbox>(iteration: u64, cell: &Cbox) -> String
where
    Cbox: Id,
{
    iteration_id_to_db_key(iteration, &cell.get_id())
}


fn db_key_to_iteration_id(db_key: &String) -> Result<(u64, CellularIdentifier), SimulationError>
{
    let mut res = db_key.split("_");
    let iter: u64 = res.next().unwrap().parse()?;
    let id0: u64 = res.next().unwrap().parse()?;
    let id1: u64 = res.next().unwrap().parse()?;
    Ok((iter, (id0, id1)))
}


// ############################################################
// Public - Storing Cells
// ############################################################

pub fn store_cell_in_tree<Cbox>(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    cell: Cbox
) -> Result<(), SimulationError>
where
    Cbox: Serialize + Id + Send + Sync + 'static,
{
    store_single_element_in_tree(tree.clone(), iteration, cell, iteration_cell_to_db_key)
}


pub fn store_cells_in_tree<Cbox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    cells: Vec<Cbox>
) -> Result<(), SimulationError>
where
    Cbox: Serialize + Id + Send + Sync + 'static,
{
    store_elements_in_tree(tree.clone(), iteration, cells, iteration_cell_to_db_key)
}


// ############################################################
// Private - buffer management
// ############################################################

fn get_cell_storage_with_buffer<Cbox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    buffer: Option<&mut CellStorage<Cbox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<CellStorage<Cbox>, SimulationError>
where
    Cbox: for<'a>Deserialize<'a> + Clone,
{
    // Get values from buffer if existant
    let all_cells = match &buffer {
        Some(buff) => Ok((*buff).clone()),
        None => deserialize_all_elements_from_tree_group(tree, db_key_to_iteration_id, progress_style),
    }?;

    // Update buffer
    buffer.and_then(|buff| {
        *buff = all_cells.clone();
        Some(buff)
    });
    
    Ok(all_cells)
}


fn get_cell_histories_with_buffer<Cbox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    buffer: Option<&mut CellHistories<Cbox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<CellHistories<Cbox>, SimulationError>
where
    Cbox: for<'a>Deserialize<'a> + Clone + Id,
{
    // Get values from buffer if existant
    let all_cell_histories = match &buffer {
        Some(buff) => (*buff).clone(),
        None => {
            let cell_storage = deserialize_all_elements_from_tree_group(tree, db_key_to_iteration_id, progress_style)?;
            convert_cell_storage_to_cell_histories(cell_storage, true)
        },
    };

    // Update buffer
    buffer.and_then(|buff| {
        *buff = all_cell_histories.clone();
        Some(buff)
    });

    Ok(all_cell_histories)
}


// ############################################################
// Public - Retrieving Cells
// ############################################################

pub fn get_cell<Cbox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    id: &CellularIdentifier
) -> Result<Option<Cbox>, SimulationError>
where
    Cbox: for<'a>Deserialize<'a> + 'static,
{
    deserialize_single_element_from_tree(tree, iteration, id, iteration_id_to_db_key)
}


pub fn get_all_cells<Cbox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    buffer: Option<&mut CellStorage<Cbox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<CellStorage<Cbox>, SimulationError>
where
    Cbox: for<'a>Deserialize<'a> + Clone,
{
    get_cell_storage_with_buffer(tree, buffer, progress_style)
}


pub fn get_cells_at_iter<Cbox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    buffer: Option<&mut CellStorage<Cbox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<Vec<Cbox>, SimulationError>
where
    Cbox: for<'a>Deserialize<'a> + Clone,
{
    let all_cells = get_cell_storage_with_buffer(tree, buffer, progress_style)?;

    // Get cells at certain iteration
    all_cells.get(&iteration)
        .map(|cells| (*cells).clone())
        .ok_or(IndexError{
            message: format!("could not obtain cells at iteration {} from database: no entry for this iteration found", iteration)
        }.into()
    )
}


pub fn get_cell_history<Cbox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    id: &CellularIdentifier,
    buffer: Option<&mut CellHistories<Cbox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<Option<Vec<(u64, Cbox)>>, SimulationError>
where
    Cbox: for<'a>Deserialize<'a> +Id + Clone
{
    let all_cell_histories = get_cell_histories_with_buffer(tree, buffer, progress_style)?;

    // Return cell history
    Ok(all_cell_histories.get(id).and_then(|x| Some(x.clone())))
}


pub fn get_all_cell_histories<Cbox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    buffer: Option<&mut CellHistories<Cbox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<CellHistories<Cbox>, SimulationError>
where
    Cbox: for<'a>Deserialize<'a> + Id + Clone
{
    get_cell_histories_with_buffer(tree, buffer, progress_style)
}


// ############################################################
// Functions for converting between different voxel storage types
// ############################################################

pub fn convert_voxel_storage_to_cell_histories<VBox>
(
    voxel_storage: VoxelStorage<VBox>,
    sorted: bool
) -> VoxelHistories<VBox>
where
    VBox: GetPlainIndex,
{
    let mut histories = voxel_storage.into_iter()
        .fold(VoxelHistories::<VBox>::new(), |mut acc, (iteration, voxels)| {
            voxels.into_iter()
                .for_each(|voxel| match acc.get_mut(&voxel.get_plain_index()) {
                    Some(voxels) => voxels.push((iteration, voxel)),
                    None => {acc.insert(voxel.get_plain_index(), vec![(iteration, voxel)]);},
                });
            acc
        });
    if sorted {
        histories.iter_mut()
            .for_each(|(_, voxel_history)| voxel_history.sort_by(|(iteration1, _), (iteration2, _)| iteration1.cmp(iteration2)));
    }
    histories
}


pub fn convert_voxel_histories_to_cell_storage<VBox>
(
    voxel_histories: VoxelHistories<VBox>
) -> VoxelStorage<VBox>
where
    VBox: GetPlainIndex,
{
    let storage = voxel_histories.into_iter()
        .fold(VoxelStorage::<VBox>::new(), |mut acc, (_ , voxel_history)| {
            voxel_history.into_iter()
                .for_each(|(iteration, voxel)| match acc.get_mut(&iteration) {
                    Some(voxels) => voxels.push(voxel),
                    None => {acc.insert(iteration, vec![voxel]);},
                });
            acc
        });
    storage
}



// ############################################################
// Private - Conversion of database entry (String) and voxel-properties
// ############################################################

fn iteration_voxel_to_db_key<VBox>(iteration: u64, voxel: &VBox) -> String
where
    VBox: Serialize + GetPlainIndex + Send + Sync,
{
    iteration_plain_index_to_db_key(iteration, &voxel.get_plain_index())
}


fn iteration_plain_index_to_db_key(iteration: u64, plain_index: &PlainIndex) -> String
{
    format!("{:010.0}_{:010.0}", iteration, plain_index)
}


fn db_key_to_iteration_plain_index(db_key: &String) -> Result<(u64, PlainIndex), SimulationError>
{
    let mut res = db_key.split("_");
    let iter: u64 = res.next().unwrap().parse()?;
    let plain_index: PlainIndex = res.next().unwrap().parse()?;
    Ok((iter, plain_index))
}


// ############################################################
// Public - Storing Voxels
// ############################################################

pub fn store_voxel_in_tree<VBox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    voxel: VBox
) -> Result<(), SimulationError>
where
    VBox: Serialize + GetPlainIndex + Send + Sync + 'static,
{
    store_single_element_in_tree(tree.clone(), iteration, voxel, iteration_voxel_to_db_key)
}

pub fn store_voxels_in_tree<VBox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    voxels: Vec<VBox>,
) -> Result<(), SimulationError>
where
    VBox: Serialize + GetPlainIndex + Send + Sync + 'static,
{
    store_elements_in_tree(tree.clone(), iteration, voxels, iteration_voxel_to_db_key)
}


// ############################################################
// Private - Buffer management
// ############################################################

fn get_voxel_storage_with_buffer<VBox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    buffer: Option<&mut VoxelStorage<VBox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<VoxelStorage<VBox>, SimulationError>
where
    VBox: for<'a>Deserialize<'a> + GetPlainIndex + Clone,
{
    // Get values from buffer if existant
    let all_cells = match &buffer {
        Some(buff) => Ok((*buff).clone()),
        None => deserialize_all_elements_from_tree_group(tree, db_key_to_iteration_plain_index, progress_style),
    }?;

    // Update buffer
    buffer.and_then(|buff| {
        *buff = all_cells.clone();
        Some(buff)
    });
    
    Ok(all_cells)
}


fn get_voxel_histories_with_buffer<VBox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    buffer: Option<&mut VoxelHistories<VBox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<VoxelHistories<VBox>, SimulationError>
where
    VBox: for<'a>Deserialize<'a> + GetPlainIndex + Clone,
{
    // Get values from buffer if existant
    let all_cells = match &buffer {
        Some(buff) => (*buff).clone(),
        None => {
            let all_voxels = deserialize_all_elements_from_tree_group(tree, db_key_to_iteration_plain_index, progress_style)?;
            convert_voxel_storage_to_cell_histories(all_voxels, true)
        },
    };

    // Update buffer
    buffer.and_then(|buff| {
        *buff = all_cells.clone();
        Some(buff)
    });
    
    Ok(all_cells)
}


// ############################################################
// Public - Retrieving Voxels
// ############################################################

pub fn get_voxel_at_iteration<VBox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    plain_index: &PlainIndex,
    iteration: u64,
) -> Result<VBox, SimulationError>
where
    VBox: for<'a>Deserialize<'a> + 'static,
{
    deserialize_single_element_from_tree(tree, iteration, plain_index, iteration_plain_index_to_db_key)
}


pub fn get_all_voxels<VBox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    buffer: Option<&mut VoxelStorage<VBox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<VoxelStorage<VBox>, SimulationError>
where
    VBox: for<'a>Deserialize<'a> + GetPlainIndex + Clone,
{
    get_voxel_storage_with_buffer(tree, buffer, progress_style)
}


pub fn get_voxel_history<VBox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    plain_index: &PlainIndex,
    buffer: Option<&mut VoxelStorage<VBox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<Vec<(u64, VBox)>, SimulationError>
where
    VBox: for<'a>Deserialize<'a> + GetPlainIndex + Clone,
{
    let res = get_voxel_storage_with_buffer(tree, buffer, progress_style)?
        .into_iter()
        .filter_map(|(iteration, mut voxels)| voxels
            .drain(..)
            .filter(|vox| vox.get_plain_index()==*plain_index).next()
            .and_then(|vox| Some((iteration, vox)))
        ).collect();
    Ok(res)
}


pub fn get_all_voxel_hitories<VBox>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    buffer: Option<&mut VoxelHistories<VBox>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<VoxelHistories<VBox>, SimulationError>
where
    VBox: for<'a>Deserialize<'a> + GetPlainIndex + Clone,
{
    get_voxel_histories_with_buffer(tree, buffer, progress_style)
}


// ############################################################
// Private - Conversions for setup
// ############################################################

fn db_key_to_iteration(db_key: &String) -> Result<u64, SimulationError>
{
    let mut res = db_key.split("_");
    let iter: u64 = res.nth(1).unwrap().parse()?;
    Ok(iter)
}

// ############################################################
// Private - Buffer management
// ############################################################

fn get_setup_storage_with_buffer<Dom, C>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    buffer: Option<&mut SetupStorage<SimulationSetup<Dom, C>>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<SetupStorage<SimulationSetup<Dom, C>>, SimulationError>
where
    SimulationSetup<Dom, C>: for<'a>Deserialize<'a> + Clone,
{
    // Get values from buffer if existant
    let all_setups = match &buffer {
        Some(buff) => Ok((*buff).clone()),
        None => deserialize_all_elements_from_tree(tree, db_key_to_iteration, progress_style),
    }?;

    // Update buffer
    buffer.and_then(|buff| {
        *buff = all_setups.clone();
        Some(buff)
    });
    
    Ok(all_setups)
}


// ############################################################
// Public - Get setups
// ############################################################

pub fn get_all_setups<Dom, C>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    buffer: Option<&mut SetupStorage<SimulationSetup<Dom, C>>>,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<SetupStorage<SimulationSetup<Dom, C>>, SimulationError>
where
    SimulationSetup<Dom, C>: for<'a>Deserialize<'a> + Clone,
{
    get_setup_storage_with_buffer(tree, buffer, progress_style)
}


// ############################################################
// Private - Testing
// ############################################################

#[cfg(test)]
mod test {
    use super::*;
    use crate::implementations::cell_models::standard_cell_2d::StandardCell2D;
    use crate::concepts::cell::{Id,CellAgentBox};
    use crate::storage::sled_database::io::store_single_element_in_tree;

    use nalgebra::Vector2;
    use rand::Rng;

    #[test]
    fn test_sled_read_write() {
        let db = sled::Config::new().temporary(true).open().unwrap();
        let tree = typed_sled::Tree::<String, Vec<u8>>::open(&db, "test");

        let mut cellboxes = Vec::new();

        for i in 0..1000 {
            let cell = StandardCell2D {
                pos: Vector2::<f64>::from([
                    rand::thread_rng().gen_range(-100.0..100.0),
                    rand::thread_rng().gen_range(-100.0..100.0)]),
                velocity: Vector2::<f64>::from([0.0, 0.0]),
                cell_radius: 4.0,
                potential_strength: 3.0,
                velocity_reduction: 0.0,
                maximum_age: 1000.0,
                remove: false,
                current_age: 0.0,
            };

            let cellbox = CellAgentBox::new(0, i, cell, None);

            store_single_element_in_tree(tree.clone(), 0, cellbox.clone(), iteration_cell_to_db_key).unwrap();

            cellboxes.push(cellbox);
        }

        let res: CellAgentBox<StandardCell2D> = deserialize_single_element_from_tree(&tree, 0, &cellboxes[0].get_id(), iteration_id_to_db_key).unwrap();

        assert_eq!(cellboxes[0].cell, res.cell);
    }


    #[cfg(not(feature = "test_exhaustive"))]
    const N_IDS: u64 = 1_000;

    #[cfg(not(feature = "test_exhaustive"))]
    const N_ITERS: u64 = 1_000;

    #[cfg(feature = "test_exhaustive")]
    const N_IDS: u64 = 10_000;

    #[cfg(feature = "test_exhaustive")]
    const N_ITERS: u64 = 10_000;

    #[test]
    fn test_iter_id_conversion() {
        #[cfg(feature = "test_exhaustive")]
        use rayon::prelude::*;

        let ids = (0..N_IDS).map(|i| (i, i)).collect::<Vec<_>>();

        #[cfg(feature = "test_exhaustive")]
        ids.into_par_iter().for_each(|id| {
            (0..N_ITERS).for_each(|iter| {
                let entry = cell_iteration_id_to_entry(&iter, &id);
                let res = cell_entry_to_iteration_id(&entry);
                let (iter_new, id_new) = res.unwrap();

                assert_eq!(id_new, id);
                assert_eq!(iter, iter_new);
            })
        });

        #[cfg(not(feature = "test_exhaustive"))]
        ids.into_iter().for_each(|id| {
            (0..N_ITERS).for_each(|iter| {
                let entry = iteration_id_to_db_key(iter, &id);
                let res = db_key_to_iteration_id(&entry);
                let (iter_new, id_new) = res.unwrap();

                assert_eq!(id_new, id);
                assert_eq!(iter, iter_new);
            })
        });
    }
}
