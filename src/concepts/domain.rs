use crate::concepts::errors::*;
use crate::concepts::cell::*;

use std::collections::{HashMap,VecDeque};

use core::marker::PhantomData;
use core::fmt::Debug;
use core::hash::Hash;
use core::cmp::Eq;
use core::ops::{Add,AddAssign,Sub,SubAssign};
use num::Zero;

use crossbeam_channel::{Sender,Receiver,SendError};
use hurdles::Barrier;

use uuid::Uuid;


pub trait Domain<Cell, Index, V>
{
    fn apply_boundary(&self, cell: &mut Cell) -> Result<(), BoundaryError>;
    fn get_neighbor_voxel_indices(&self, index: &Index) -> Vec<Index>;
    fn get_voxel_index(&self, cell: &Cell) -> Index;
    fn distribute_voxels_for_threads(&self, n_threads: usize) -> Result<(usize, Vec<Vec<(Index, V)>>), CalcError>;
}


pub trait Voxel<Index, Pos, Force>
where
    Index: Hash,
{
    fn custom_force_on_cell(&self, cell: &Pos) -> Option<Result<Force, CalcError>> {
        None
    }

    fn get_index(&self) -> Index;
}


struct PosInformation<Index, Pos> {
    pos: Pos,
    uuid: Uuid,
    index: Index,
}


struct ForceInformation<Force> {
    force: Force,
    uuid: Uuid,
}


// This object has multiple voxels and runs on a single thread.
// It can communicate with other containers via channels.
pub struct MultiVoxelContainer<Index, Pos, Force, Velocity, V, D, C>
where
    Index: Hash,
    V: Voxel<Index, Pos, Force>,
    C: Cell<Pos, Force, Velocity>,
    D: Domain<C, Index, V>,
{
    voxels: HashMap<Index, V>,
    voxel_cells: HashMap<Index, VecDeque<C>>,

    // TODO
    // Maybe we need to implement this somewhere else since
    // it is currently not simple to change this variable on the fly.
    // However, maybe we should be thinking about specifying an interface to use this function
    // Something like:
    // fn update_domain(&mut self, domain: Domain) -> Result<(), BoundaryError>
    // And then automatically have the ability to change cell positions if the domain shrinks/grows for example
    // but then we might also want to change the number of voxels and redistribute cells accordingly
    // This needs much more though!
    domain: D,

    // Where do we want to send cells, positions and forces
    senders_cell: HashMap<Index, Sender<C>>,
    senders_pos: HashMap<Index, Sender<PosInformation<Index,Pos>>>,
    senders_force: HashMap<Index, Sender<ForceInformation<Force>>>,

    // Same for receiving
    receiver_cell: Receiver<C>,
    receiver_pos: Receiver<PosInformation<Index,Pos>>,
    receiver_force: Receiver<ForceInformation<Force>>,

    // TODO store datastructures for forces and neighboring voxels such that
    // memory allocation is minimized
    cell_forces: HashMap<Uuid, VecDeque<Force>>,
    neighbor_indices: HashMap<Index, Vec<Index>>,

    // Global barrier to synchronize threads and make sure every information is sent before further processing
    sending_barrier: Barrier,

    // Phantom data for velocity
    phantom_vel: PhantomData<Velocity>,
}


impl<Index, Pos, Force, Velocity, V, D, C> MultiVoxelContainer<Index, Pos, Force, Velocity, V, D, C>
where
    // TODO abstract away these trait bounds to more abstract traits
    // these traits should be defined when specifying the individual cell components
    // (eg. mechanics, interaction, etc...)
    Index: Hash + Eq + Copy + Debug,
    V: Voxel<Index, Pos, Force>,
    D: Domain<C, Index, V>,
    Velocity: Add + AddAssign + Sub + SubAssign + Zero,
    Force: Add + AddAssign + Sub + SubAssign + Zero,
    Pos: Add + AddAssign + Sub + SubAssign + Clone,
    C: Cell<Pos, Force, Velocity>,
{
    fn update_cell_cycle(&mut self, dt: f64) {
        self.voxel_cells
            .iter_mut()
            .for_each(|(_, cs)| cs
                .iter_mut()
                .for_each(|c| C::update_cycle(&dt, c))
            );
    }

    fn apply_boundaries(&mut self, domain: &D) -> Result<(), BoundaryError> {
        self.voxel_cells
            .iter_mut()
            .map(|(_, cells)| cells.iter_mut())
            .flatten()
            .map(|cell| domain.apply_boundary(cell))
            .collect()
    }

    fn insert_cells(&mut self, index: Index, new_cells: &mut VecDeque<C>) -> Result<(), CalcError>
    {
        self.voxel_cells.get_mut(&index)
            .ok_or(CalcError{ message: "New cell has incorrect index".to_owned()})?
            .append(new_cells);
        Ok(())
    }

    // TODO add functionality
    fn sort_cell_in_voxel(&mut self, cell: C) -> Result<(), ErrorVariant>
    {
        let index = self.domain.get_voxel_index(&cell);

        match self.voxel_cells.get_mut(&index) {
            Some(cells) => cells.push_back(cell),
            None => {
                match self.senders_cell.get(&index) {
                    Some(sender) => sender.send(cell),
                    None => Err(SendError(cell)),
                }?;
            },
        }
        Ok(())
    }

    fn update_mechanics(&mut self, dt: &f64) -> Result<(), ErrorVariant> {
        for (index, cells) in self.voxel_cells.iter() {
            for cell in cells.iter() {
                let id = cell.get_uuid();
                let position = cell.pos();

                // Calculate the force from this voxel
                let force = match self.voxels[index].custom_force_on_cell(&position) {
                    Some(force) => force,
                    None => Ok(Force::zero()),
                }?;

                // Store force
                match self.cell_forces.get_mut(&id) {
                    Some(forces) => forces.push_back(force),
                    None => (),
                };

                // Calculate forces from other voxels
                for neighbor_index in &self.neighbor_indices[&index] {
                    // If the desired voxel is in the current MultiVoxelContainer, we do not have to send any messages
                    if self.voxels.contains_key(&neighbor_index) {
                        
                        // Calculate the force
                        let force = match self.voxels[&neighbor_index].custom_force_on_cell(&position) {
                            Some(force) => force,
                            None => Ok(Force::zero()),
                        }?;

                        // Append to other forces acting on the same cell
                        match self.cell_forces.get_mut(&id) {
                            Some(forces) => forces.push_back(force),
                            None => (),// TODO Return an error since this should succeed always!
                        };

                    // Otherwise send information to other voxels to obtain value this way
                    } else {
                        // Send information on how to calculate the force and where to send back the result
                        let spi = PosInformation::<Index, Pos> {pos: position.clone(), uuid: id, index: *index};
                        self.senders_pos[&neighbor_index].send(spi)?;
                    }
                }
            }
        }

        // Wait for synchronization of threads. Every information should be sent such that receiving does not lose any information
        self.sending_barrier.wait();

        // Receive the positons for which to calculate forces for the other voxels
        let mut obtained_positions: Vec<PosInformation<Index,Pos>> = self.receiver_pos.try_iter().collect();


        for obt_position in obtained_positions.drain(..) {
            // Calculate the force acting on the cell from this voxel
            let force = match self.voxels[&obt_position.index].custom_force_on_cell(&obt_position.pos) {
                Some(force) => force,
                None => Ok(Force::zero()),
            }?;

            // Send back the force information to that MultiVoxelContainer
            let sfi = ForceInformation::<Force> {force: force, uuid: obt_position.uuid};
            self.senders_force[&obt_position.index].send(sfi)?;
        }

        // Wait for all forces which have been calculated in all threads
        self.sending_barrier.wait();

        // Receive forces
        let mut obtained_forces: Vec<ForceInformation<Force>> = self.receiver_force.try_iter().collect();

        // Store the obtained forces to the correct cells
        for obtained_force in obtained_forces.drain(..) {
            match self.cell_forces.get_mut(&obtained_force.uuid) {
                Some(forces) => forces.push_back(obtained_force.force),
                None => (),// TODO Return an error since this should succeed always!
            }
        }

        // TODO we actually need to use the forces to calculate the new velocities and positions together with the time step dt

        Ok(())
    }
}

// These should be sharable between threads
// unsafe impl<Index: Hash, Cell, Pos, Force, V:Voxel, D:Domain<Cell,Index>, C> Send for MultiVoxelContainer<Index, Cell, Pos, Force, V, D, C> {}
// unsafe impl<Index: Hash, Cell, Pos, Force, V:Voxel, D:Domain<Cell,Index>, C> Sync for MultiVoxelContainer<Index, Cell, Pos, Force, V, D, C> {}


// TODO automatically create many MultiVoxelContainers depending on how many threads are being used
// possibly use the builder pattern (check online resources first)
