use crate::concepts::errors::*;
use crate::concepts::cell::*;
use crate::concepts::cycle::*;
use crate::concepts::interaction::*;
use crate::concepts::mechanics::*;
use crate::concepts::mechanics::{Position,Force,Velocity};

#[cfg(feature = "db_sled")]
use crate::storage::sled_database::io::store_cells_in_database;

use std::collections::{HashMap,BTreeMap};
use std::marker::{Send,Sync};

use core::hash::Hash;
use core::cmp::Eq;

use crossbeam_channel::{Sender,Receiver,SendError};
use hurdles::Barrier;

use num::Zero;
use serde::{Serialize,Deserialize};


pub trait Domain<C, I, V>: Send + Sync + Serialize + for<'a> Deserialize<'a>
{
    fn apply_boundary(&self, cell: &mut C) -> Result<(), BoundaryError>;
    fn get_neighbor_voxel_indices(&self, index: &I) -> Vec<I>;
    fn get_voxel_index(&self, cell: &C) -> I;
    fn get_all_indices(&self) -> Vec<I>;
    fn generate_contiguous_multi_voxel_regions(&self, n_regions: usize) -> Result<(usize, Vec<Vec<(I, V)>>), CalcError>;
}


#[derive(Clone,Serialize,Deserialize)]
pub struct DomainBox<D>
where
    D: Serialize + for<'a>Deserialize<'a>,
{
    #[serde(bound = "")]
    pub domain_raw: D,
}


impl<D> From<D> for DomainBox<D>
where
    D: Serialize + for<'a>Deserialize<'a>,
{
    fn from(domain: D) -> DomainBox<D> {
        DomainBox {
            domain_raw: domain,
        }
    }
}


impl<C, I, V, D> Domain<CellAgentBox<C>, I, V> for DomainBox<D>
where
    I: Index,
    D: Domain<C, I, V>,
    V: Send + Sync,
    C: Serialize + for<'a> Deserialize<'a> + Send + Sync
{
    fn apply_boundary(&self, cbox: &mut CellAgentBox<C>) -> Result<(), BoundaryError> {
        self.domain_raw.apply_boundary(&mut cbox.cell)
    }

    fn get_neighbor_voxel_indices(&self, index: &I) -> Vec<I> {
        self.domain_raw.get_neighbor_voxel_indices(index)
    }

    fn get_voxel_index(&self, cbox: &CellAgentBox<C>) -> I {
        self.domain_raw.get_voxel_index(&cbox.cell)
    }

    fn get_all_indices(&self) -> Vec<I> {
        self.domain_raw.get_all_indices()
    }

    fn generate_contiguous_multi_voxel_regions(&self, n_regions: usize) -> Result<(usize, Vec<Vec<(I, V)>>), CalcError> {
        self.domain_raw.generate_contiguous_multi_voxel_regions(n_regions)
    }
}


pub trait Index = Ord + Hash + Eq + Clone + Send + Sync + Serialize + std::fmt::Debug;

pub(crate) type PlainIndex = u32;

pub trait Voxel<I, Pos, Force>: Send + Sync + Clone + Serialize + for<'a> Deserialize<'a>
where
    I: Index,
{
    fn custom_force_on_cell(&self, _cell: &Pos) -> Option<Result<Force, CalcError>> {
        None
    }

    fn get_index(&self) -> I;
}


pub(crate) struct PosInformation<Pos, Inf> {
    pub pos: Pos,
    pub info: Option<Inf>,
    pub count: usize,
    pub index_sender: PlainIndex,
    pub index_receiver: PlainIndex,
}


pub(crate) struct ForceInformation<Force> {
    pub force: Force,
    pub count: usize,
    pub index_sender: PlainIndex,
}


#[derive(Serialize,Deserialize,Clone)]
pub(crate) struct VoxelBox<I, V,C,For> {
    pub plain_index: PlainIndex,
    pub index: I,
    pub voxel: V,
    pub neighbors: Vec<PlainIndex>,
    pub cells: Vec<(C, AuxiliaryCellPropertyStorage<For>)>,
    pub uuid_counter: u64,
}


#[derive(Serialize,Deserialize,Clone)]
pub(crate) struct AuxiliaryCellPropertyStorage<For> {
    force: For,
}


impl<For> Default for AuxiliaryCellPropertyStorage<For>
where
    For: Zero,
{
    fn default() -> AuxiliaryCellPropertyStorage<For> {
        AuxiliaryCellPropertyStorage { force: For::zero() }
    }
}


impl<I,V,C,For> VoxelBox<I,V,C,For> {
    fn calculate_custom_force_on_cells<Pos,Vel>(&mut self) -> Result<(), CalcError>
    where
        V: Voxel<I,Pos,For>,
        I: Index,
        Pos: Position,
        For: Force,
        Vel: Velocity,
        C: Mechanics<Pos,For,Vel>,
    {
        for (cell, aux_storage) in self.cells.iter_mut() {
            match self.voxel.custom_force_on_cell(&cell.pos()) {
                Some(Ok(force)) => Ok(aux_storage.force += force),
                Some(Err(e))    => Err(e),
                None            => Ok(()),
            }?;
        }
        Ok(())
    }

    fn calculate_force_between_cells_internally<Pos,Inf,Vel>(&mut self) -> Result<(), CalcError>
    where
        V: Voxel<I,Pos,For>,
        I: Index,
        Pos: Position,
        For: Force,
        Vel: Velocity,
        C: Interaction<Pos,For,Inf> + Mechanics<Pos,For,Vel> + Clone,
    {
        for n in 0..self.cells.len() {
            for m in 0..self.cells.len() {
                if n != m {
                    // Calculate the force which is exerted on
                    let pos_other = self.cells[m].0.pos();
                    let inf_other = self.cells[m].0.get_interaction_information();
                    let (cell, _) = self.cells.get_mut(n).unwrap();
                    match cell.calculate_force_on(&cell.pos(), &pos_other, &inf_other) {
                        Some(Ok(force)) => {
                            let (_, aux_storage) = self.cells.get_mut(m).unwrap();
                            Ok(aux_storage.force += force)
                        },
                        Some(Err(e))    => Err(e),
                        None            => Ok(()),
                    }?;
                }
            }
        }
        Ok(())
    }

    fn calculate_force_from_cells_on_other_cell<Pos,Inf,Vel>(&self, ext_pos: &Pos, ext_inf: &Option<Inf>) -> Result<For, CalcError>
    where
        V: Voxel<I,Pos,For>,
        I: Index,
        Pos: Position,
        For: Force,
        Vel: Velocity,
        C: Interaction<Pos,For,Inf> + Mechanics<Pos,For,Vel>,
    {
        let mut force = For::zero();
        for (cell, _) in self.cells.iter() {
            match cell.calculate_force_on(&cell.pos(), &ext_pos, &ext_inf) {
                Some(Ok(f))     => Ok(force+=f),
                Some(Err(e))    => Err(e),
                None            => Ok(()),
            }?;
        }
        Ok(force)
    }
}


/* impl<I,V,C,Pos,For> Voxel<PlainIndex,Pos,For> for VoxelBox<I, V,C,For>
where
    C: Clone + Serialize + for<'a> Deserialize<'a> + Send + Sync,
    Pos: Serialize + for<'a> Deserialize<'a> + Send + Sync,
    For: Clone + Serialize + for<'a> Deserialize<'a> + Send + Sync,
    I: Serialize + for<'a> Deserialize<'a> + Index,
    V: Serialize + for<'a> Deserialize<'a> + Voxel<I,Pos,For>,
{
    fn custom_force_on_cell(&self, cell: &Pos) -> Option<Result<For, CalcError>> {
        self.voxel.custom_force_on_cell(cell)
    }

    fn get_index(&self) -> PlainIndex {
        self.plain_index
    }
}*/


// This object has multiple voxels and runs on a single thread.
// It can communicate with other containers via channels.
pub(crate) struct MultiVoxelContainer<I, Pos, For, Inf, V, D, C>
where
    I: Index,
    Pos: Position,
    For: Force,
    V: Voxel<I, Pos, For>,
    C: Serialize + for<'a> Deserialize<'a> + Send + Sync,
    D: Domain<C, I, V>,
{
    pub voxels: BTreeMap<PlainIndex, VoxelBox<I,V,CellAgentBox<C>,For>>,

    // TODO
    // Maybe we need to implement this somewhere else since
    // it is currently not simple to change this variable on the fly.
    // However, maybe we should be thinking about specifying an interface to use this function
    // Something like:
    // fn update_domain(&mut self, domain: Domain) -> Result<(), BoundaryError>
    // And then automatically have the ability to change cell positions if the domain shrinks/grows for example
    // but then we might also want to change the number of voxels and redistribute cells accordingly
    // This needs much more though!
    pub domain: DomainBox<D>,
    pub index_to_plain_index: BTreeMap<I,PlainIndex>,
    pub plain_index_to_thread: BTreeMap<PlainIndex, usize>,
    pub index_to_thread: BTreeMap<I, usize>,

    // Where do we want to send cells, positions and forces
    // TODO use Vector of pointers in each voxel to get all neighbors.
    // Also store cells in this way.
    pub senders_cell: HashMap<usize, Sender<CellAgentBox<C>>>,
    pub senders_pos: HashMap<usize, Sender<PosInformation<Pos, Inf>>>,
    pub senders_force: HashMap<usize, Sender<ForceInformation<For>>>,

    // Same for receiving
    pub receiver_cell: Receiver<CellAgentBox<C>>,
    pub receiver_pos: Receiver<PosInformation<Pos, Inf>>,
    pub receiver_force: Receiver<ForceInformation<For>>,

    // TODO store datastructures for forces and neighboring voxels such that
    // memory allocation is minimized

    // Global barrier to synchronize threads and make sure every information is sent before further processing
    pub barrier: Barrier,

    #[cfg(not(feature = "no_db"))]
    pub database_cells: typed_sled::Tree<String, Vec<u8>>,

    pub mvc_id: u16,
}


impl<I, Pos, For, Inf, V, D, C> MultiVoxelContainer<I, Pos, For, Inf, V, D, C>
where
    // TODO abstract away these trait bounds to more abstract traits
    // these traits should be defined when specifying the individual cell components
    // (eg. mechanics, interaction, etc...)
    I: Index,
    V: Voxel<I, Pos, For>,
    D: Domain<C, I, V>,
    For: Force,
    Pos: Position,
    // C: CellAgent<Pos, For, Vel>,
    C: Serialize + for<'a>Deserialize<'a> + Send + Sync,
{
    fn update_cell_cycle(&mut self, dt: &f64)
    where
        C: Cycle<C>,
    {
        self.voxel_cells
            .iter_mut()
            .for_each(|(_, cs)| cs
                .iter_mut()
                .for_each(|c| CellAgentBox::<C>::update_cycle(dt, c))
            );
    }

    fn apply_boundaries(&mut self) -> Result<(), BoundaryError> {
        for cell in self.voxel_cells.iter_mut().map(|(_, cells)| cells.iter_mut()).flatten() {
            self.domain.apply_boundary(cell)?;
        }
        Ok(())
    }

    pub fn insert_cells(&mut self, index: &I, new_cells: &mut Vec<CellAgentBox<C>>) -> Result<(), CalcError>
    {
        self.voxel_cells.get_mut(index)
            .ok_or(CalcError{ message: "New cell has incorrect index".to_owned()})?
            .append(new_cells);
        Ok(())
    }

    // TODO add functionality
    pub fn sort_cell_in_voxel(&mut self, cell: CellAgentBox<C>) -> Result<(), SimulationError>
    {
        let index = self.domain.get_voxel_index(&cell);

        match self.voxel_cells.get_mut(&index) {
            Some(cells) => cells.push(cell),
            None => {
                match self.senders_cell.get(&index) {
                    Some(sender) => sender.send(cell),
                    None => Err(SendError(cell)),
                }?;
            },
        }
        Ok(())
    }

    fn calculate_forces_for_external_cells<Vel>(&self, pos_info: PosInformation<I, Pos, Inf>) -> Result<(), SimulationError>
    where
        Vel: Velocity,
        C: Interaction<Pos, For, Inf> + Mechanics<Pos, For, Vel>,
    {
        // Calculate force from cells in voxel
        let mut forces = Vec::new();
        for cell in self.voxel_cells[&pos_info.index_receiver].iter() {
            // TODO in which order do we need to insert these positions?
            match cell.calculate_force_on(&cell.pos(), &pos_info.pos, &pos_info.info) {
                Some(force) => forces.push(force),
                None => (),
            }
        }

        // Send back force information
        // println!("Thread: {} Senders indices: {:?} Sender Index: {:?} Receiver Index {:?}", std::thread::current().name().unwrap(), self.senders_force.iter().map(|(i, _)| i.clone()).collect::<Vec<I>>(), pos_info.index_sender, pos_info.index_receiver);
        self.senders_force[&pos_info.index_sender].send(
            ForceInformation{
                forces,
                uuid: pos_info.uuid
            }
        )?;
        Ok(())
    }

    pub fn update_mechanics<Vel>(&mut self, dt: &f64) -> Result<(), SimulationError>
    where
        Vel: Velocity,
        Inf: Clone,
        C: Interaction<Pos, For, Inf> + Mechanics<Pos, For, Vel> + Clone,
    {
        // General Idea of this function
        // for each cell
        //      for each neighbor_voxel in neighbors of voxel containing cell
        //              if neighbor_voxel is in current MultivoxelContainer
        //                      calculate forces of current cells on cell and store
        //                      calculate force from voxel on cell and store
        //              else
        //                      send PosInformation to other MultivoxelContainer
        // 
        // for each PosInformation received from other MultivoxelContainers
        //      calculate forces of current_cells on cell and send back
        //
        // for each ForceInformation received from other MultivoxelContainers
        //      store received force
        //
        // for each cell in this MultiVoxelContainer
        //      update pos and velocity with all forces obtained
        //      Simultanously

        // Define to calculate forces on a cell from external cells in a voxel with a certain index
        let mut calculate_force_from_cells_on_cell_and_store_or_send = |voxel_index: &I, cell: &CellAgentBox<C>| -> Result<(), SimulationError> {
            // Iterate over all cells which are in the voxel of interest
            match self.voxel_cells.get(voxel_index) {

                // If cells are present (which means the voxel is in this multivoxelcontainer), we can calculate the result immediately
                Some(ext_cells) => {
                    for ext_cell in ext_cells.iter().filter(|c| c.get_uuid()!=cell.get_uuid()) {
                        // Calculate the force and store the raw Result
                        match ext_cell.calculate_force_on(&ext_cell.pos(), &cell.pos(), &cell.get_interaction_information()) {
                            Some(force) => {
                                match self.cell_forces.get_mut(&cell.get_uuid()) {
                                    // We need to check if there is already an entry for the cell if this is the case, we can simply append there
                                    Some(forces) => {
                                        forces.push(force);
                                    },
                                    // If not then we create a new entry
                                    None => {
                                        self.cell_forces.insert(cell.get_uuid(), Vec::from([force]));
                                    },
                                };
                            },
                            None => (),
                        }
                    }
                    Ok(())
                },

                // If the voxel is not in this multicontainervoxel, we will send the required positional information to the corresponding container
                // such that the force will be calculated there. We can then later receive the information and include it in our calculation.
                None => self.senders_pos[voxel_index].send(
                    PosInformation {
                        index_sender: self.domain.get_voxel_index(cell),
                        index_receiver: voxel_index.clone(),
                        pos: cell.pos(),
                        info: cell.get_interaction_information(),
                        uuid: cell.get_uuid(),
                    }),
            }?;
            Ok(())
        };

        // Calculate forces for all cells from neighbors
        for (voxel_index, cells) in self.voxel_cells.iter() {
            
            for cell in cells.iter() {
                // Calculate force from own voxel
                calculate_force_from_cells_on_cell_and_store_or_send(&voxel_index, cell)?;
                
                
                // Calculate force from neighbors
                for neighbor_index in self.neighbor_indices[voxel_index].iter() {
                    calculate_force_from_cells_on_cell_and_store_or_send(&neighbor_index, cell)?;
                }
            }
        }

        // Calculate custom force of voxel on cell
        for (voxel_index, cells) in self.voxel_cells.iter() {
            for cell in cells.iter() {
                match self.voxels[voxel_index].custom_force_on_cell(&cell.pos()) {
                    Some(force) => {
                        match self.cell_forces.get_mut(&cell.get_uuid()) {
                            Some(forces) => forces.push(force),
                            None => {self.cell_forces.insert(cell.get_uuid(), Vec::from([force]));},
                        }
                    },
                    None => (),
                }
            }
        }

        // Wait for all threads to send PositionInformation
        self.barrier.wait();

        // Receive PositionInformation and send back ForceInformation
        for obt_pos in self.receiver_pos.try_iter() {
            self.calculate_forces_for_external_cells(obt_pos)?;
        }

        // Synchronize again such that every message reaches its receiver
        self.barrier.wait();
        
        // Update position and velocity of all cells with new information
        for mut obt_forces in self.receiver_force.try_iter() {
            match self.cell_forces.get_mut(&obt_forces.uuid) {
                Some(saved_forces) => {
                    saved_forces.append(&mut obt_forces.forces);
                },
                None => (),
            };
        }

        // Update position and velocity of cells
        for (_, cells) in self.voxel_cells.iter_mut() {
            for cell in cells.iter_mut() {
                let mut force = For::zero();
                match self.cell_forces.get_mut(&cell.get_uuid()) {
                    Some(new_forces) => {
                        for new_force in new_forces.drain(..) {
                            match new_force {
                                Ok(n_force) => {
                                    force += n_force;
                                    Ok(())
                                },
                                Err(error) => Err(error),
                            }?;
                        }
                    }
                    None => (),
                }

                // Update cell position and velocity
                let (dx, dv) = cell.calculate_increment(force)?;
                cell.set_pos(&(cell.pos() + dx * *dt));
                cell.set_velocity(&(cell.velocity() + dv * *dt));
            }
        }
        Ok(())
    }

    pub fn sort_cells_in_voxels(&mut self) -> Result<(), SimulationError> {
        // Store all cells which need to find a new home in this variable
        let mut find_new_home_cells = Vec::<_>::new();
        
        for (voxel_index, cells) in self.voxel_cells.iter_mut() {
            // Drain every cell which is currently not in the correct voxel
            let new_voxel_cells = cells.drain_filter(|c| &self.domain.get_voxel_index(&c)!=voxel_index);
            // Check if the cell needs to be sent to another multivoxelcontainer
            find_new_home_cells.append(&mut new_voxel_cells.collect::<Vec<_>>());
        }

        // Send cells to other multivoxelcontainer or keep them here
        for cell in find_new_home_cells {
            let cell_index = self.domain.get_voxel_index(&cell);
            match self.voxel_cells.get_mut(&cell_index) {
                Some(cells) => {
                    cells.push(cell);
                    Ok(())
                },
                None => {
                    match self.senders_cell.get(&cell_index) {
                        Some(sender) => {
                            sender.send(cell)?;
                            Ok(())
                        }
                        None => Err(IndexError {message: format!("Could not correctly send cell with uuid {}", cell.get_uuid())}),
                    }
                }
            }?;
        }

        // Wait until every cell has been sent
        self.barrier.wait();

        // Now receive new cells and insert them
        let mut new_cells = self.receiver_cell.try_iter().collect::<Vec<_>>();
        for cell in new_cells.drain(..) {
            self.sort_cell_in_voxel(cell)?;
        }
        Ok(())
    }


    #[cfg(not(feature = "no_db"))]
    pub fn save_cells_to_database(&self, iteration: &u32) -> Result<(), SimulationError>
    where
        CellAgentBox<C>: Clone,
    {
        let cells = self.voxel_cells.clone()
            .into_iter()
            .map(|(_, cells)| cells)
            .flatten()
            .collect::<Vec<_>>();

        #[cfg(feature = "db_sled")]
        store_cells_in_database(self.database_cells.clone(), *iteration, cells)?;

        Ok(())
    }


    pub fn insert_cell(&mut self, iteration: &u32, cell: C) -> Option<C> {
        match self.voxel_cells.get_mut(&self.domain.domain_raw.get_voxel_index(&cell)) {
            Some(cells) => {
                let cellagentbox = CellAgentBox::from((
                    *iteration,
                    StorageIdent::Cell.value(),
                    self.mvc_id,
                    self.uuid_counter,
                    cell
                ));
                cells.push(cellagentbox);
                self.uuid_counter += 1;
                None
            },
            None => Some(cell),
        }
    }


    pub fn run_full_update<Vel>(&mut self, _t: &f64, dt: &f64) -> Result<(), SimulationError>
    where
        Vel: Velocity,
        C: Cycle<C> + Mechanics<Pos, For, Vel> + Interaction<Pos, For, Inf> + Clone,
    {
        self.update_cell_cycle(dt);

        self.update_mechanics(dt)?;

        self.apply_boundaries()?;

        self.sort_cells_in_voxels()?;
        Ok(())
    }
}
