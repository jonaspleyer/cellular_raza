use crate::concepts::errors::SimulationError;
use crate::concepts::cell::{CellAgent,Id};
use crate::concepts::mechanics::{Position,Force,Velocity};
use crate::prelude::CellAgentBox;

use serde::{Deserialize};

use uuid::Uuid;


pub enum StorageIdent {
    Cell,
    Voxel,
    MultiVoxel,
}


impl StorageIdent {
    pub const fn value(self) -> u16 {
        match self {
            StorageIdent::Cell => 1,
            StorageIdent::Voxel => 2,
            StorageIdent::MultiVoxel => 3,
        }
    }
}


fn iteration_uuid_to_entry(iteration: &u32, uuid: &Uuid) -> String
{
    format!("{:010.0}_{}", iteration, uuid)
}


fn entry_to_iteration_uuid(entry: &String) -> Result<(u32, Uuid), SimulationError>
{
    let mut res = entry.split("_");
    let iter: u32 = res.next().unwrap().parse()?;
    let uuid: Uuid = res.next().unwrap().parse()?;
    Ok((iter, uuid))
}


pub fn store_cell_in_database<Pos, For, Vel, C: CellAgent<Pos, For, Vel>>
(
    tree: &sled::Db,
    iteration: &u32,
    cell: &CellAgentBox<Pos, For, Vel, C>
) -> Result<(), SimulationError>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
{
    let serialized = bincode::serialize(cell)?;
    tree.insert(iteration_uuid_to_entry(iteration, &cell.get_uuid()), serialized)?;
    Ok(())
}


pub fn store_cells_in_database<Pos, For, Vel, C: CellAgent<Pos, For, Vel>>
(
    tree: typed_sled::Tree<String, Vec<u8>>,
    iteration: u32,
    cells: Vec<CellAgentBox<Pos, For, Vel, C>>
) -> Result<(), SimulationError>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
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
            let name = iteration_uuid_to_entry(&iteration, &uuid);
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


pub fn get_cell_from_database<Pos, For, Vel, C: CellAgent<Pos, For, Vel>>
(
    tree: &sled::Db,
    iteration: &u32,
    uuid: &Uuid
) -> Result<Option<CellAgentBox<Pos, For, Vel, C>>, SimulationError>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    C: for<'a> Deserialize<'a>,
{
    let retr = tree.get(iteration_uuid_to_entry(iteration, uuid))?;
    match retr {
        Some(retrieved) => {
            let cell = bincode::deserialize(&retrieved)?;
            Ok(Some(cell))
        },
        None => Ok(None),
    }
}


pub fn get_cell_history_from_database<Pos, For, Vel, C: CellAgent<Pos, For, Vel>>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    uuid: &Uuid
) -> Result<Vec<(u32, CellAgentBox<Pos, For, Vel, C>)>, SimulationError>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    C: for<'a> Deserialize<'a>,
{
    let mut res: Vec<_> = tree
        .iter()
        .map(|ret| {
            match ret {
                Ok((key, value)) => {
                    let val_des = bincode::deserialize::<CellAgentBox<Pos, For, Vel, C>>(&value);
                    match (entry_to_iteration_uuid(&key), val_des) {
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


pub fn get_cell_uuids_at_iter<Pos, For, Vel, C: CellAgent<Pos, For, Vel>>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iter: &u32,
) -> Result<Vec<Uuid>, SimulationError>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    C: for<'a> Deserialize<'a>,
{
    let res = tree
        .iter()
        .map(|ret| {
            match ret {
                Ok((key, _)) => Some(entry_to_iteration_uuid(&key).ok()),
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


pub fn get_cells_at_iter<Pos, For, Vel, C: CellAgent<Pos, For, Vel>>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iter: &u32,
) -> Result<Vec<CellAgentBox<Pos, For, Vel, C>>, SimulationError>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    C: for<'a> Deserialize<'a>,
{
    let res = tree
        .iter()
        .filter_map(|opt| opt.ok())
        .map(|(key, value)| {
            let cb: Option<CellAgentBox<Pos, For, Vel, C>> = bincode::deserialize(&value).ok();
            let res = entry_to_iteration_uuid(&key).ok();
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
        let tree = sled::open("test_sled_db").unwrap();

        let mut cellboxes = Vec::new();

        for i in 0..1000 {
            let cell = StandardCell2D {
                pos: Vector2::<f64>::from([
                    rand::thread_rng().gen_range(-100.0..100.0),
                    rand::thread_rng().gen_range(-100.0..100.0)]),
                velocity: Vector2::<f64>::from([0.0, 0.0]),
                cell_radius: 4.0,
                potential_strength: 3.0,
                maximum_age: 1000.0,
                remove: false,
                current_age: 0.0,
            };

            let cellbox = CellAgentBox::from((
                0,
                StorageIdent::Cell.value(),
                0,
                i,
                cell
            ));

            store_cell_in_database(&tree, &0, &cellbox).unwrap();

            cellboxes.push(cellbox);
        }

        let res: CellAgentBox<Vector2<f64>, Vector2<f64>, Vector2<f64>, StandardCell2D> = get_cell_from_database(&tree, &0, &cellboxes[0].get_uuid()).unwrap().unwrap();

        assert_eq!(cellboxes[0].cell, res.cell);

        tree.drop_tree("test_sled_db").unwrap();
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
                let entry = iteration_uuid_to_entry(&iter, &uuid);
                let res = entry_to_iteration_uuid(&entry);
                let (iter_new, uuid_new) = res.unwrap();

                assert_eq!(uuid_new, uuid);
                assert_eq!(iter, iter_new);
            })
        });

        #[cfg(not(feature = "test_exhaustive"))]
        uuids.into_iter().for_each(|uuid| {
            (0..N_ITERS).for_each(|iter| {
                let entry = iteration_uuid_to_entry(&iter, &uuid);
                let res = entry_to_iteration_uuid(&entry);
                let (iter_new, uuid_new) = res.unwrap();

                assert_eq!(uuid_new, uuid);
                assert_eq!(iter, iter_new);
            })
        });
    }
}
