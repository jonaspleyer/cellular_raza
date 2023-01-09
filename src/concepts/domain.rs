use crate::concepts::errors::*;
use crate::concepts::cell::*;
use crate::concepts::cycle::*;
use crate::concepts::interaction::*;
use crate::concepts::mechanics::*;
use crate::concepts::mechanics::{Position,Force,Velocity};

use crate::database::io::{StorageIdent,store_cells_in_database};

use std::collections::{HashMap};
use std::marker::{Send,Sync};

use core::marker::PhantomData;
use core::hash::Hash;
use core::cmp::Eq;

use crossbeam_channel::{Sender,Receiver,SendError};
use hurdles::Barrier;

use uuid::Uuid;


pub trait Domain<C, I, V>: Send + Sync
{
    fn apply_boundary(&self, cell: &mut C) -> Result<(), BoundaryError>;
    fn get_neighbor_voxel_indices(&self, index: &I) -> Vec<I>;
    fn get_voxel_index(&self, cell: &C) -> I;
    fn generate_contiguous_multi_voxel_regions(&self, n_regions: usize) -> Result<(usize, Vec<Vec<(I, V)>>), CalcError>;
}


#[derive(Clone)]
pub struct DomainBox<C, I, V, D>
where
    I: Index,
    D: Domain<C, I, V>,
{
    domain_raw: D,

    phantom_cel: PhantomData<C>,
    phantom_ind: PhantomData<I>,
    phantom_vox: PhantomData<V>,
}


impl<C, I, V, D> From<D> for DomainBox<C, I, V, D>
where
    I: Index,
    D: Domain<C, I, V>,
{
    fn from(domain: D) -> DomainBox<C, I, V, D> {
        DomainBox {
            domain_raw: domain,

            phantom_cel: PhantomData,
            phantom_ind: PhantomData,
            phantom_vox: PhantomData,
        }
    }
}


impl<Pos, For, Vel, C, I, V, D> Domain<CellAgentBox<Pos, For, Vel, C>, I, V> for DomainBox<C, I, V, D>
where
    Pos: Position,
    For: Force,
    Vel: Velocity,
    I: Index,
    V: Voxel<I, Pos, For>,
    C: CellAgent<Pos, For, Vel>,
    D: Domain<C, I, V>
{
    fn apply_boundary(&self, cbox: &mut CellAgentBox<Pos, For, Vel, C>) -> Result<(), BoundaryError> {
        self.domain_raw.apply_boundary(&mut cbox.cell)
    }

    fn get_neighbor_voxel_indices(&self, index: &I) -> Vec<I> {
        self.domain_raw.get_neighbor_voxel_indices(index)
    }

    fn get_voxel_index(&self, cbox: &CellAgentBox<Pos, For, Vel, C>) -> I {
        self.domain_raw.get_voxel_index(&cbox.cell)
    }

    fn generate_contiguous_multi_voxel_regions(&self, n_regions: usize) -> Result<(usize, Vec<Vec<(I, V)>>), CalcError> {
        self.domain_raw.generate_contiguous_multi_voxel_regions(n_regions)
    }
}


pub trait Index = Hash + Eq + Clone + Send + Sync + std::fmt::Debug;


pub trait Voxel<I, Pos, Force>: Send + Sync + Clone
where
    I: Index,
{
    fn custom_force_on_cell(&self, _cell: &Pos) -> Option<Result<Force, CalcError>> {
        None
    }

    fn get_index(&self) -> I;
}


pub struct PosInformation<I, Pos> {
    pub pos: Pos,
    pub uuid: Uuid,
    pub index_sender: I,
    pub index_receiver: I,
}


pub struct ForceInformation<Force> {
    pub forces: Vec<Result<Force, CalcError>>,
    pub uuid: Uuid,
}


// This object has multiple voxels and runs on a single thread.
// It can communicate with other containers via channels.
pub struct MultiVoxelContainer<I, Pos, For, Vel, V, D, C>
where
    I: Index,
    Pos: Position,
    For: Force,
    Vel: Velocity,
    V: Voxel<I, Pos, For>,
    C: CellAgent<Pos, For, Vel>,
    D: Domain<C, I, V>,
{
    pub voxels: HashMap<I, V>,
    pub voxel_cells: HashMap<I, Vec<CellAgentBox<Pos, For, Vel, C>>>,

    // TODO
    // Maybe we need to implement this somewhere else since
    // it is currently not simple to change this variable on the fly.
    // However, maybe we should be thinking about specifying an interface to use this function
    // Something like:
    // fn update_domain(&mut self, domain: Domain) -> Result<(), BoundaryError>
    // And then automatically have the ability to change cell positions if the domain shrinks/grows for example
    // but then we might also want to change the number of voxels and redistribute cells accordingly
    // This needs much more though!
    pub domain: DomainBox<C, I, V, D>,

    // Where do we want to send cells, positions and forces
    pub senders_cell: HashMap<I, Sender<CellAgentBox<Pos, For, Vel, C>>>,
    pub senders_pos: HashMap<I, Sender<PosInformation<I,Pos>>>,
    pub senders_force: HashMap<I, Sender<ForceInformation<For>>>,

    // Same for receiving
    pub receiver_cell: Receiver<CellAgentBox<Pos, For, Vel, C>>,
    pub receiver_pos: Receiver<PosInformation<I,Pos>>,
    pub receiver_force: Receiver<ForceInformation<For>>,

    // TODO store datastructures for forces and neighboring voxels such that
    // memory allocation is minimized
    pub cell_forces: HashMap<Uuid, Vec<Result<For, CalcError>>>,
    pub neighbor_indices: HashMap<I, Vec<I>>,

    // Global barrier to synchronize threads and make sure every information is sent before further processing
    pub barrier: Barrier,

    pub database: typed_sled::Tree<String, Vec<u8>>,

    pub uuid_counter: u64,
    pub mvc_id: u16,
}


impl<I, Pos, For, Vel, V, D, C> MultiVoxelContainer<I, Pos, For, Vel, V, D, C>
where
    // TODO abstract away these trait bounds to more abstract traits
    // these traits should be defined when specifying the individual cell components
    // (eg. mechanics, interaction, etc...)
    I: Index,
    V: Voxel<I, Pos, For>,
    D: Domain<C, I, V>,
    Vel: Velocity,
    For: Force,
    Pos: Position,
    C: CellAgent<Pos, For, Vel>,
{
    fn update_cell_cycle(&mut self, dt: &f64) {
        self.voxel_cells
            .iter_mut()
            .for_each(|(_, cs)| cs
                .iter_mut()
                .for_each(|c| CellAgentBox::<Pos, For, Vel, C>::update_cycle(dt, c))
            );
    }

    fn apply_boundaries(&mut self) -> Result<(), BoundaryError> {
        self.voxel_cells
            .iter_mut()
            .map(|(_, cells)| cells.iter_mut())
            .flatten()
            .map(|cell| self.domain.apply_boundary(cell))// TODO catch this error
            .collect()
    }

    pub fn insert_cells(&mut self, index: &I, new_cells: &mut Vec<CellAgentBox<Pos, For, Vel, C>>) -> Result<(), CalcError>
    {
        self.voxel_cells.get_mut(index)
            .ok_or(CalcError{ message: "New cell has incorrect index".to_owned()})?
            .append(new_cells);
        Ok(())
    }

    // TODO add functionality
    pub fn sort_cell_in_voxel(&mut self, cell: CellAgentBox<Pos, For, Vel, C>) -> Result<(), SimulationError>
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

    fn calculate_forces_for_external_cells(&self, pos_info: PosInformation<I, Pos>) -> Result<(), SimulationError> {
        // Calculate force from cells in voxel
        let mut forces = Vec::new();
        for cell in self.voxel_cells[&pos_info.index_receiver].iter() {
            match cell.force(&cell.pos(), &pos_info.pos) {
                Some(force) => forces.push(force),
                None => (),
            }
        }

        // Send back force information
        // println!("Thread: {} Senders indices: {:?} Sender Index: {:?} Receiver Index {:?}", std::thread::current().name().unwrap(), self.senders_force.iter().map(|(i, _)| i.clone()).collect::<Vec<I>>(), pos_info.index_sender, pos_info.index_receiver);
        self.senders_force[&pos_info.index_sender].send(
            ForceInformation{
                forces: forces,
                uuid: pos_info.uuid
            }
        )?;
        Ok(())
    }

    pub fn update_mechanics(&mut self, dt: &f64) -> Result<(), SimulationError> {
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
        let mut calculate_force_from_cells_on_cell_and_store_or_send = |voxel_index: &I, cell: &CellAgentBox<Pos, For, Vel, C>| -> Result<(), SimulationError> {
            // Iterate over all cells which are in the voxel of interest
            match self.voxel_cells.get(voxel_index) {

                // If cells are present (which means the voxel is in this multivoxelcontainer), we can calculate the result immediately
                Some(ext_cells) => {
                    for ext_cell in ext_cells.iter().filter(|c| c.get_uuid()!=cell.get_uuid()) {
                        // Calculate the force and store the raw Result
                        match ext_cell.force(&ext_cell.pos(), &cell.pos()) {
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


    pub fn save_cells_to_database(&self, iteration: &u32) -> Result<(), SimulationError> {
        let cells = self.voxel_cells.clone()
            .into_iter()
            .map(|(_, cells)| cells)
            .flatten()
            .collect::<Vec<_>>();

        store_cells_in_database(self.database.clone(), *iteration, cells)?;

        Ok(())
    }

    fn insert_cell(&mut self, iteration: &u32, cell: C) -> Option<C> {
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


    pub fn run_full_update(&mut self, _t: &f64, dt: &f64) -> Result<(), SimulationError> {
        self.update_cell_cycle(dt);

        self.update_mechanics(dt)?;

        self.apply_boundaries()?;

        self.sort_cells_in_voxels()?;
        Ok(())
    }
}
