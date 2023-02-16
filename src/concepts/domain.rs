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
    Pos: Position,
    For: Force,
    Inf: Clone,
    // C: CellAgent<Pos, For, Vel>,
    C: Serialize + for<'a>Deserialize<'a> + Send + Sync,
{
    fn update_cell_cycle(&mut self, dt: &f64)
    where
        C: Cycle<C>,
    {
        self.voxels
            .iter_mut()
            .for_each(|(_, vox)| vox.cells
                .iter_mut()
                .for_each(|(c, _)| CellAgentBox::<C>::update_cycle(dt, c))
            );
    }

    fn apply_boundaries(&mut self) -> Result<(), BoundaryError> {
        self.voxels
            .iter_mut()
            .map(|(_, vox)| vox.cells.iter_mut())
            .flatten()
            .map(|(c, _)| self.domain.apply_boundary(c))
            .collect::<Result<(), BoundaryError>>()
    }

    pub fn insert_cells(&mut self, index: &I, new_cells: Vec<CellAgentBox<C>>) -> Result<(), CalcError>
    {
        let vox = self.voxels.get_mut(&self.index_to_plain_index[&index]).ok_or(CalcError{ message: "New cell has incorrect index".to_owned()})?;
        let mut cells_with_storage = new_cells.into_iter().map(|c| (c, AuxiliaryCellPropertyStorage{ force: For::zero() })).collect::<Vec<_>>();
        vox.cells.append(&mut cells_with_storage);
        Ok(())
    }

    // TODO add functionality
    pub fn sort_cell_in_voxel(&mut self, cell: CellAgentBox<C>) -> Result<(), SimulationError>
    {
        let index = self.index_to_plain_index[&self.domain.get_voxel_index(&cell)];
        let aux_storage = AuxiliaryCellPropertyStorage { force: For::zero() };

        match self.voxels.get_mut(&index) {
            Some(vox) => vox.cells.push((cell, aux_storage)),
            None => {
                let thread_index = self.plain_index_to_thread[&index];
                match self.senders_cell.get(&thread_index) {
                    Some(sender) => sender.send(cell),
                    None => Err(SendError(cell)),
                }?;
            },
        }
        Ok(())
    }

    fn calculate_forces_for_external_cells<Vel>(&self, pos_info: PosInformation<Pos, Inf>) -> Result<(), SimulationError>
    where
        Vel: Velocity,
        C: Interaction<Pos, For, Inf> + Mechanics<Pos, For, Vel>,
    {
        let vox = self.voxels.get(&pos_info.index_receiver).ok_or(IndexError {message: format!("EngineError: Voxel with index {:?} of PosInformation can not be found in this thread.", pos_info.index_receiver)})?;
        // Calculate force from cells in voxel
        let force = vox.calculate_force_from_cells_on_other_cell(&pos_info.pos, &pos_info.info)?;

        // Send back force information
        let thread_index = self.plain_index_to_thread[&pos_info.index_sender];
        self.senders_force[&thread_index].send(
            ForceInformation{
                force,
                count: pos_info.count,
                index_sender: pos_info.index_sender,
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

        // Calculate forces between cells of own voxel
        self.voxels.iter_mut().map(|(_, vox)| vox.calculate_force_between_cells_internally()).collect::<Result<(),CalcError>>()?;

        // Calculate forces for all cells from neighbors
        // TODO can we do this without memory allocation?
        let key_iterator: Vec<_> = self.voxels.keys().map(|k| *k).collect();

        for voxel_index in key_iterator {
            for cell_count in 0..self.voxels[&voxel_index].cells.len() {
                let cell_pos = self.voxels[&voxel_index].cells[cell_count].0.pos();
                let cell_inf = self.voxels[&voxel_index].cells[cell_count].0.get_interaction_information();
                let mut force = For::zero();
                for neighbor_index in self.voxels[&voxel_index].neighbors.iter() {
                    match self.voxels.get(&neighbor_index) {
                        Some(vox) => Ok::<(), CalcError>(force += vox.calculate_force_from_cells_on_other_cell(&cell_pos, &cell_inf)?),
                        None => Ok(self.senders_pos[&self.plain_index_to_thread[&neighbor_index]].send(
                            PosInformation {
                                index_sender: voxel_index,
                                index_receiver: neighbor_index.clone(),
                                pos: cell_pos.clone(),
                                info: cell_inf.clone(),
                                count: cell_count,
                        })?),
                    }?;
                }
                self.voxels.get_mut(&voxel_index).unwrap().cells[cell_count].1.force += force;
            }
        }

        // Calculate custom force of voxel on cell
        self.voxels.iter_mut().map(|(_, vox)| vox.calculate_custom_force_on_cells()).collect::<Result<(),CalcError>>()?;

        // Wait for all threads to send PositionInformation
        self.barrier.wait();

        // Receive PositionInformation and send back ForceInformation
        for obt_pos in self.receiver_pos.try_iter() {
            self.calculate_forces_for_external_cells(obt_pos)?;
        }

        // Synchronize again such that every message reaches its receiver
        self.barrier.wait();
        
        // Update position and velocity of all cells with new information
        for obt_forces in self.receiver_force.try_iter() {
            let vox = self.voxels.get_mut(&obt_forces.index_sender).ok_or(IndexError { message: format!("EngineError: Sender with plain index {} was ended up in location where index is not present anymore", obt_forces.index_sender)})?;
            match vox.cells.get_mut(obt_forces.count) {
                Some((_, aux_storage)) => Ok(aux_storage.force+=obt_forces.force),
                None => Err(IndexError { message: format!("EngineError: Force Information with sender index {:?} and cell at vector position {} could not be matched", obt_forces.index_sender, obt_forces.count)}),
            }?;
        }

        // Update position and velocity of cells
        for (_, vox) in self.voxels.iter_mut() {
            for (cell, aux_storage) in vox.cells.iter_mut() {
                // Update cell position and velocity
                let (dx, dv) = cell.calculate_increment(aux_storage.force.clone())?;
                // Afterwards set force to 0.0 again
                aux_storage.force = For::zero();
                // TODO This is currently an Euler Stepper only. We should be able to use something like Verlet Integration instead: https://en.wikipedia.org/wiki/Verlet_integration
                cell.set_pos(&(cell.pos() + dx * *dt));
                cell.set_velocity(&(cell.velocity() + dv * *dt));
            }
        }
        Ok(())
    }

    pub fn sort_cells_in_voxels(&mut self) -> Result<(), SimulationError> {
        // Store all cells which need to find a new home in this variable
        let mut find_new_home_cells = Vec::<_>::new();
        
        for (voxel_index, vox) in self.voxels.iter_mut() {
            // Drain every cell which is currently not in the correct voxel
            let new_voxel_cells = vox.cells.drain_filter(|(c, _)| match self.index_to_plain_index.get(&self.domain.get_voxel_index(&c)) {
                Some(ind) => ind,
                None => panic!("Cannot find index {:?}", self.domain.get_voxel_index(&c)),
            }!=voxel_index);
            // Check if the cell needs to be sent to another multivoxelcontainer
            find_new_home_cells.append(&mut new_voxel_cells.collect::<Vec<_>>());
        }

        // Send cells to other multivoxelcontainer or keep them here
        for (cell, aux_storage) in find_new_home_cells {
            let ind = self.domain.get_voxel_index(&cell);
            let new_thread_index = self.index_to_thread[&ind];
            let cell_index = self.index_to_plain_index[&ind];
            match self.voxels.get_mut(&cell_index) {
                // If new voxel is in current multivoxelcontainer then save them there
                Some(vox) => {
                    vox.cells.push((cell, aux_storage));
                    Ok(())
                },
                // Otherwise send them to the correct other multivoxelcontainer
                None => {
                    match self.senders_cell.get(&new_thread_index) {
                        Some(sender) => {
                            // println!("Everything fine: Old: {:?} New: {:?}", self.mvc_id, new_thread_index);
                            // println!("Other threads {:?}", self.senders_cell.keys());
                            sender.send(cell)?;
                            Ok(())
                        }
                        None => Err(IndexError {message: format!("Could not correctly send cell with uuid {}", cell.get_uuid())})
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
        let cells = self.voxels.iter().map(|(_, vox)| vox.cells.clone().into_iter().map(|(c, _)| c))
            .flatten()
            .collect::<Vec<_>>();

        #[cfg(feature = "db_sled")]
        store_cells_in_database(self.database_cells.clone(), *iteration, cells)?;

        Ok(())
    }


    pub fn insert_cell(&mut self, iteration: &u32, cell: C) -> Option<C> {
        match self.voxels.get_mut(&self.index_to_plain_index[&self.domain.domain_raw.get_voxel_index(&cell)]) {
            Some(vox) => {
                let cellagentbox = CellAgentBox::new(*iteration, vox.plain_index, vox.uuid_counter, cell);
                vox.cells.push((cellagentbox, AuxiliaryCellPropertyStorage::default()));
                vox.uuid_counter += 1;
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
