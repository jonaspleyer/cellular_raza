use crate::concepts::errors::SimulationError;
use crate::concepts::cell::{CellAgent,Id,CellAgentBox};
use crate::concepts::domain::Index;
use crate::concepts::mechanics::{Position,Force,Velocity};
use crate::prelude::VoxelBox;

use serde::{Serialize,Deserialize};

use uuid::Uuid;


fn cell_iteration_uuid_to_entry(iteration: &u32, uuid: &Uuid) -> String
{
    format!("{:010.0}_{}", iteration, uuid)
}


fn cell_entry_to_iteration_uuid(entry: &String) -> Result<(u32, Uuid), SimulationError>
{
    let mut res = entry.split("_");
    let iter: u32 = res.next().unwrap().parse()?;
    let uuid: Uuid = res.next().unwrap().parse()?;
    Ok((iter, uuid))
}


pub fn store_cell_in_database<C>(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iteration: &u32,
    cell: &CellAgentBox<C>
) -> Result<(), SimulationError>
where
    C: Serialize + for<'a>Deserialize<'a>,
{
    let serialized = bincode::serialize(cell)?;
    tree.insert(&cell_iteration_uuid_to_entry(iteration, &cell.get_uuid()), &serialized)?;
    Ok(())
}


pub fn store_cells_in_database<C>
(
    tree: typed_sled::Tree<String, Vec<u8>>,
    iteration: u32,
    cells: Vec<CellAgentBox<C>>
) -> Result<(), SimulationError>
where
    C: Serialize + for<'a>Deserialize<'a>,
{
    let cells_encoded: Vec<_> = cells
        .iter()
        .map(|cell| (cell.get_uuid(), bincode::serialize(&cell).unwrap()))
        .collect();

    async_std::task::spawn(async move {
        // TODO use batches to write multiple entries at once
        let mut batch = typed_sled::Batch::<String, Vec<u8>>::default();

        cells_encoded.into_iter().for_each(|(uuid, cell_enc)| {
            // TODO Find another format to save cells in sled database
            // TODO create features to use other databases
            let name = cell_iteration_uuid_to_entry(&iteration, &uuid);
            batch.insert(&name, &cell_enc);
        });
        match tree.apply_batch(batch) {
            Ok(()) => (),
            Err(error) => {
                print!("Storing cells: Database error: {}", error);
                println!("Continuing simulation anyway!");
            },
        }
    });
    
    Ok(())
}


pub fn get_cell_from_database<Pos, For, Inf, Vel, C: CellAgent<Pos, For, Inf, Vel>>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iteration: &u32,
    uuid: &Uuid
) -> Result<Option<CellAgentBox<C>>, SimulationError>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    C: for<'a> Deserialize<'a>,
{
    let retr = tree.get(&cell_iteration_uuid_to_entry(iteration, uuid))?;
    match retr {
        Some(retrieved) => {
            let cell = bincode::deserialize(&retrieved)?;
            Ok(Some(cell))
        },
        None => Ok(None),
    }
}


pub fn get_cell_history_from_database<C>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    uuid: &Uuid
) -> Result<Vec<(u32, CellAgentBox<C>)>, SimulationError>
where
    C: Serialize + for<'a> Deserialize<'a>,
{
    let mut res: Vec<_> = tree
        .iter()
        .map(|ret| {
            match ret {
                Ok((key, value)) => {
                    let val_des = bincode::deserialize::<CellAgentBox<C>>(&value);
                    match (cell_entry_to_iteration_uuid(&key), val_des) {
                        (Ok((iter, _)), Ok(val)) => Some((iter, val)),
                        _ => None,
                    }
                },
                Err(_) => None,
            }
        })
        .filter_map(|opt| opt)
        .filter(|(_, cell)| &cell.get_uuid()==uuid)
        .collect::<Vec<_>>();
    
    res.sort_by(|(k1, _), (k2, _)| k1.cmp(&k2));
    Ok(res)
}


pub fn get_all_cell_histories<C>
(
    tree: &typed_sled::Tree<String, Vec<u8>>
) -> Result<std::collections::HashMap<Uuid, Vec<(u32, CellAgentBox<C>)>>, SimulationError>
where
    C: Serialize + for<'a> Deserialize<'a> + Clone,
{
    // Obtain all histories
    let mut histories = tree
        .iter()
        .filter_map(|ret| ret.ok())
        .fold(std::collections::HashMap::<Uuid, Vec<(u32, CellAgentBox<C>)>>::new(), |mut acc, (key, val)| {
            let cb_res: Result<CellAgentBox<C>, _> = bincode::deserialize(&val);
            let entry_res = cell_entry_to_iteration_uuid(&key);
            match (cb_res, entry_res) {
                (Ok(cb), Ok((iter, uuid))) => {
                    let entries = acc.entry(uuid).or_insert_with(|| vec![(iter, cb.clone())]);
                    entries.push((iter, cb));
                },
                _ => (),
            }
            acc
        });

    // Sort them
    histories.iter_mut().for_each(|(_, entries)| entries.sort_by(|(it1, _), (it2, _)| it1.cmp(&it2)));

    Ok(histories)
}


pub fn get_cell_uuids_at_iter<C>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iter: &u32,
) -> Result<Vec<Uuid>, SimulationError>
where
    C: Serialize + for<'a> Deserialize<'a>,
{
    let res = tree
        .iter()
        .map(|ret| {
            match ret {
                Ok((key, _)) => Some(cell_entry_to_iteration_uuid(&key).ok()),
                _ => None,
            }
            
        })
        .filter_map(|opt| opt)
        .filter_map(|opt| opt)
        .filter_map(|(it, uuid)| match &it==iter {
            true => Some(uuid),
            false => None,
        })
        .collect();
    
    Ok(res)
}


pub fn deserialize_tree<C>(tree: &typed_sled::Tree<String, Vec<u8>>, progress_style: Option<indicatif::ProgressStyle>) -> Result<std::collections::HashMap<u32, Vec<CellAgentBox<C>>>, SimulationError>
where
    C: Serialize + for<'a>Deserialize<'a>,
{
    let bar = indicatif::ProgressBar::new(tree.len() as u64);
    match progress_style {
        Some(s) => bar.set_style(s),
        None => (),
    }
    println!("Reading from Database");
    let res = tree
        .iter()
        .filter_map(|opt| opt.ok())
        .filter_map(|(key, value)| {
            let cb: Option<CellAgentBox<C>> = bincode::deserialize(&value).ok();
            let res = cell_entry_to_iteration_uuid(&key).ok();
            match (cb, res) {
                (Some(cab), Some((it, _))) => Some((it, cab)),
                _ => None,
            }
        },)
        .fold(std::collections::HashMap::<u32, Vec<CellAgentBox<C>>>::new(), |mut acc, (it, cab)| -> std::collections::HashMap<u32, Vec<CellAgentBox<C>>> {
            match acc.get_mut(&it) {
                Some(cells) => cells.push(cab),
                None => {acc.insert(it, vec![cab]);},
            };
            bar.inc(1);
            acc
        });
    bar.finish();
    Ok(res)
}



// TODO This is deserializing the complete tree before filtering at every time step.
// This can be made much more efficient!
// Possibly introduce buffer to store deserialized tree in memory
pub fn get_cells_at_iter<C>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iter: &u32,
) -> Result<Vec<CellAgentBox<C>>, SimulationError>
where
    C: Serialize + for<'a> Deserialize<'a>,
{
    let res = tree
        .iter()
        .filter_map(|opt| opt.ok())
        .map(|(key, value)| {
            let cb: Option<CellAgentBox<C>> = bincode::deserialize(&value).ok();
            let res = cell_entry_to_iteration_uuid(&key).ok();
            match (cb, res) {
                (Some(cab), Some((it, _))) => Some((it, cab)),
                _ => None,
            }
        },)
        .filter_map(|opt| match opt {
            Some((it, cab)) => match &it==iter {
                true => Some(cab),
                false => None,
            },
            _ => None,
        })
        .collect();
    
    Ok(res)
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::impls_cell_models::standard_cell_2d::StandardCell2D;
    use crate::concepts::cell::{Id,CellAgentBox};

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

            let cellbox = CellAgentBox::new(0, 0, i, cell);

            store_cell_in_database(&tree, &0, &cellbox).unwrap();

            cellboxes.push(cellbox);
        }

        let res: CellAgentBox<StandardCell2D> = get_cell_from_database(&tree, &0, &cellboxes[0].get_uuid()).unwrap().unwrap();

        assert_eq!(cellboxes[0].cell, res.cell);
    }


    #[cfg(not(feature = "test_exhaustive"))]
    const N_UUIDS: u128 = 1_000;

    #[cfg(not(feature = "test_exhaustive"))]
    const N_ITERS: u32 = 1_000;

    #[cfg(feature = "test_exhaustive")]
    const N_UUIDS: u128 = 10_000;

    #[cfg(feature = "test_exhaustive")]
    const N_ITERS: u32 = 10_000;

    #[test]
    fn test_iter_uuid_conversion() {
        #[cfg(feature = "test_exhaustive")]
        use rayon::prelude::*;

        let uuids = (0..N_UUIDS).map(|i| uuid::Uuid::from_u128(i)).collect::<Vec<_>>();

        #[cfg(feature = "test_exhaustive")]
        uuids.into_par_iter().for_each(|uuid| {
            (0..N_ITERS).for_each(|iter| {
                let entry = cell_iteration_uuid_to_entry(&iter, &uuid);
                let res = cell_entry_to_iteration_uuid(&entry);
                let (iter_new, uuid_new) = res.unwrap();

                assert_eq!(uuid_new, uuid);
                assert_eq!(iter, iter_new);
            })
        });

        #[cfg(not(feature = "test_exhaustive"))]
        uuids.into_iter().for_each(|uuid| {
            (0..N_ITERS).for_each(|iter| {
                let entry = cell_iteration_uuid_to_entry(&iter, &uuid);
                let res = cell_entry_to_iteration_uuid(&entry);
                let (iter_new, uuid_new) = res.unwrap();

                assert_eq!(uuid_new, uuid);
                assert_eq!(iter, iter_new);
            })
        });
    }
}


use crate::concepts::domain::PlainIndex;


fn voxel_iteration_plain_index_to_entry(iteration: &u32, plain_index: &PlainIndex) -> String {
    format!("{:010.0}_{:010.0}", iteration, plain_index)
}


fn voxel_entry_to_iteration_plain_index(entry: &String) -> Result<(u32, PlainIndex), SimulationError>
{
    let mut res = entry.split("_");
    let iter: u32 = res.next().unwrap().parse()?;
    let plain_index: PlainIndex = res.next().unwrap().parse()?;
    Ok((iter, plain_index))
}


pub fn store_voxels_in_database<I, V, C, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>
(
    tree: typed_sled::Tree<String, Vec<u8>>,
    iteration: u32,
    voxels: Vec<VoxelBox<I, V, C, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>>,
) -> Result<(), SimulationError>
where
    I: Serialize + for<'a>Deserialize<'a>,
    V: Serialize + for<'a>Deserialize<'a>,
    C: Serialize + for<'a>Deserialize<'a>,
    Pos: Serialize + for<'a>Deserialize<'a>,
    For: Serialize + for<'a>Deserialize<'a>,
    Vel: Serialize + for<'a>Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a>Deserialize<'a>,
    ConcBoundaryExtracellular: Serialize + for<'a>Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a>Deserialize<'a>,
{
    let voxels_encoded: Vec<_> = voxels
        .iter()
        .map(|voxel| (voxel.plain_index, bincode::serialize(&voxel).unwrap()))
        .collect();

    async_std::task::spawn(async move {
        let mut batch = typed_sled::Batch::<String, Vec<u8>>::default();

        voxels_encoded.into_iter().for_each(|(uuid, cell_enc)| {
            // TODO Find another format to save voxels in sled database
            // TODO create features to use other databases
            let name = voxel_iteration_plain_index_to_entry(&iteration, &uuid);
            batch.insert(&name, &cell_enc);
        });
        match tree.apply_batch(batch) {
            Ok(()) => (),
            Err(error) => {
                print!("Storing cells: Database error: {}", error);
                println!("Continuing simulation anyway!");
            },
        }
    });
    
    Ok(())
}


pub fn voxels_deserialize_tree<I, V, C, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    progress_style: Option<indicatif::ProgressStyle>
) -> Result<std::collections::HashMap<u32, Vec<VoxelBox<I, V, C, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>>>, SimulationError>
where
    I: Serialize + for<'a>Deserialize<'a>,
    V: Serialize + for<'a>Deserialize<'a>,
    C: Serialize + for<'a>Deserialize<'a>,
    Pos: Serialize + for<'a>Deserialize<'a>,
    For: Serialize + for<'a>Deserialize<'a>,
    Vel: Serialize + for<'a>Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a>Deserialize<'a>,
    ConcBoundaryExtracellular: Serialize + for<'a>Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a>Deserialize<'a>,
{
    let bar = indicatif::ProgressBar::new(tree.len() as u64);
    match progress_style {
        Some(s) => bar.set_style(s),
        None => (),
    }
    println!("Reading from Database");
    let res = tree
        .iter()
        .filter_map(|opt| opt.ok())
        .filter_map(|(key, value)| {
            let cb: Option<VoxelBox<I, V, C, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>> = bincode::deserialize(&value).ok();
            let res = voxel_entry_to_iteration_plain_index(&key).ok();
            match (cb, res) {
                (Some(voxelbox), Some((it, _))) => Some((it, voxelbox)),
                _ => None,
            }
        },)
        .fold(std::collections::HashMap::<u32, Vec<VoxelBox<I, V, C, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>>>::new(), |mut acc, (it, voxelbox)| -> std::collections::HashMap<u32, Vec<VoxelBox<I, V, C, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>>> {
            match acc.get_mut(&it) {
                Some(voxels) => voxels.push(voxelbox),
                None => {acc.insert(it, vec![voxelbox]);},
            };
            bar.inc(1);
            acc
        });
    bar.finish();
    Ok(res)
}
