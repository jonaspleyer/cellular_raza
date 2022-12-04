use crate::concepts::domain::*;
use crate::cell_properties::cell_model::*;
use crate::concepts::mechanics::*;
use crate::concepts::errors::*;
use crate::cell_properties::cycle::CycleModel;
use crate::concepts::cycle::Cycle;
use crate::concepts::interaction::Interaction;

use std::collections::HashMap;

use nalgebra::Vector3;

use hurdles::Barrier;
use crossbeam_channel::{Sender, Receiver, SendError};

use core::fmt::Display;


#[derive(Debug)]
pub enum ErrorVariant {
    SendCellErorr(SendError<CellModel>),
    SendPosError(SendError<(IndexType, PositionType, IdType)>),
    SendForceError(SendError<(IndexType, ForceType, IdType)>),
    CalcError(CalcError),
    BoundaryError(BoundaryError),
}

impl Display for ErrorVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorVariant::SendCellErorr(send_cell_error) => 
                write!(f, "{}", send_cell_error),
            ErrorVariant::SendPosError(send_pos_error) => 
                write!(f, "{}", send_pos_error),
            ErrorVariant::SendForceError(send_force_error) =>
                write!(f, "{}", send_force_error),
            ErrorVariant::CalcError(calc_error) =>
                write!(f, "{}", calc_error),
            ErrorVariant::BoundaryError(boundary_error) => 
                write!(f, "{}", boundary_error),
        }
    }
}

impl From<SendError<CellModel>> for ErrorVariant {
    fn from(err: SendError<CellModel>) -> Self {
        ErrorVariant::SendCellErorr(err)
    }
}

impl From<SendError<(IndexType, Vector3<f64>, u32)>> for ErrorVariant {
    fn from(err: SendError<(IndexType, Vector3<f64>, u32)>) -> Self {
        ErrorVariant::SendPosError(err)
    }
}

impl From<CalcError> for ErrorVariant {
    fn from(err: CalcError) -> Self {
        ErrorVariant::CalcError(err)
    }
}

impl From<BoundaryError> for ErrorVariant {
    fn from(err: BoundaryError) -> Self {
        ErrorVariant::BoundaryError(err)
    }
}


impl std::error::Error for ErrorVariant {}


pub type IndexType = [usize; 3];
pub type PositionType = Vector3<f64>;
pub type ForceType = Vector3<f64>;
pub type IdType = u32;


#[derive(Clone)]
pub struct Voxel {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub cells: Vec<CellModel>,
    pub cell_senders: HashMap<IndexType, Sender<CellModel>>,
    pub cell_receiver: Receiver<CellModel>,
    pub pos_senders: HashMap<IndexType, Sender<(IndexType, PositionType, IdType)>>,
    pub pos_receiver: Receiver<(IndexType, PositionType, IdType)>,
    pub force_senders: HashMap<IndexType, Sender<(IndexType, ForceType, IdType)>>,
    pub force_receiver: Receiver<(IndexType, ForceType, IdType)>,
    pub index: IndexType,
    pub domain: Cuboid,
}


pub struct VoxelContainer {
    pub voxels: Vec<Voxel>,
    pub barrier: Barrier,
}


#[derive(Clone)]
pub struct Cuboid {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub voxel_sizes: [f64; 3],
}


unsafe impl Send for Cuboid {}
unsafe impl Sync for Cuboid {}


impl Domain for Cuboid {
    fn apply_boundary(&self, cell: &mut CellModel) -> Result<(),BoundaryError> {
        let mut pos = cell.mechanics.pos();
        let mut velocity = cell.mechanics.velocity();

        // For each dimension (ie 3 in general)
        for i in 0..3 {
            // Check if the particle is below lower edge
            if pos[i] < self.min[i] {
                pos[i] = 2.0 * self.min[i] - pos[i];
                velocity[i] *= -1.0;
            }
            // Check if the particle is over the edge
            if pos[i] > self.max[i] {
                pos[i] = 2.0 * self.max[i] - pos[i];
                velocity[i] *= -1.0;
            }
        }
        // Set new position and velocity of particle
        cell.mechanics.set_pos(&pos);
        cell.mechanics.set_velocity(&velocity);

        // If new position is still out of boundary return error
        for i in 0..3 {
            if pos[i] < self.min[i] || pos[i] > self.max[i] {
                return Err(BoundaryError { message: format!("Particle with id {} is out of domain at position {:?}", cell.id, pos) });
            } else {
                return Ok(());
            }
        }
        Ok(())
    }
}


impl Cuboid {
    pub fn determine_voxel(&self, cell: &CellModel) -> IndexType {
        let p = cell.mechanics.pos();
        let q0 = (p[0] - self.min[0]) / self.voxel_sizes[0];
        let q1 = (p[1] - self.min[1]) / self.voxel_sizes[1];
        let q2 = (p[2] - self.min[2]) / self.voxel_sizes[2];
        return [q0 as usize, q1 as usize, q2 as usize];
    }
}


impl Voxel {
    pub fn update_cell_cycles(&mut self, dt: f64) -> Result<(), CalcError> {
        // Update the cycle status of cells
        self.cells.iter_mut().for_each(|cell| CycleModel::update(&dt, cell));

        // Remove cells which have been flagged
        self.cells.retain(|cell| !cell.flags.removal);
        Ok(())
    }

    // TODO use ODE integrator to solve for all forces
    // Calculate the forces beforehand and then just reuse in ODE rhs
    // This should not be a problem since there is no time dependence
    fn calculate_total_cell_forces(&self, x: &PositionType) -> Result<ForceType, CalcError> {
        let mut force = Vector3::from([0.0; 3]);
        for cell in &self.cells {
                if &cell.mechanics.pos() != x {
                    force += cell.interaction.potential(&cell.mechanics.pos(), x)?;
                }
        }
        Ok(force)
    }

    fn receive_cells(&mut self) -> Result<(), CalcError> {
        let new_cells: Vec<CellModel> = self.cell_receiver.try_iter().collect();
        for cell in new_cells {
            self.cells.push(cell);
        }
        Ok(())
    }

    fn send_pos_information(&mut self) -> Result<(), ErrorVariant> {
        // println!("{:?}", self.pos_senders.keys());
        // println!("{:?}\n", self.index);
        for index in self.pos_senders.keys() {
            for cell in self.cells.iter() {
                self.pos_senders[index].send((self.index, cell.mechanics.pos(), cell.id))?;
            }
        }
        Ok(())
    }

    fn receive_pos_and_send_force_informations(&self, barrier: &mut Barrier) -> Result<(), ErrorVariant> {
        let pos_combinations: Vec<(IndexType, PositionType, IdType)> = self.pos_receiver.try_iter().collect();

        // TODO WAIT HERE!
        barrier.wait();

        for (index, pos, id) in pos_combinations {
            let force = self.calculate_total_cell_forces(&pos)?;
            println!("Current Voxel {:?} Cell index {:?} Keys: {:?}", self.index, index, self.pos_senders.keys());
            self.pos_senders[&index].send((index, force, id))?;
        }
        Ok(())
    }

    // TODO seperate force and position sending channels
    // Also introduce different types for both of them

    fn receive_forces_and_update_mechanics(&mut self, dt: f64) -> Result<(), CalcError> {
        // TODO this is not correct and needs to be fixed
        let forces_other_voxels: HashMap<u32, ForceType> = self.pos_receiver
            .try_iter()
            .map(|(_, force, id)| {
                (id, force)
            }).collect();
        
        let forces_total: Vec<ForceType> = self.cells
            .iter()
            .map(|cell| {
                // Calculate the forces of cells in this voxel
                let res = self.calculate_total_cell_forces(&cell.mechanics.pos()).unwrap();
                
                // See if we have some result from other voxels and add it
                let add = match forces_other_voxels.get(&cell.id) {
                    Some(val) => val.clone(),
                    None => Vector3::from([0.0, 0.0, 0.0]),
                };
                res + add
            }).collect();

        for (cell, force) in self.cells.iter_mut().zip(forces_total) {
            let current_velocity = cell.mechanics.velocity();

            cell.mechanics.add_velocity(&(dt * force - cell.mechanics.dampening_constant * dt * current_velocity));
            cell.mechanics.add_pos(&(dt * cell.mechanics.velocity()));
        }

        Ok(())
    }

    fn apply_boundaries(&mut self) -> Result<(), BoundaryError> {
        for cell in self.cells.iter_mut() {
            self.domain.apply_boundary(cell)?;
        }
        Ok(())
    }

    fn send_cells(&mut self) -> Result<(), ErrorVariant> {
        // Send the updated cells to other threads if necessary
        for cell in self.cells.drain_filter(|cell| self.domain.determine_voxel(&cell)!=self.index) {
            let index = self.domain.determine_voxel(&cell);
            let sender = self.cell_senders.get(&index).ok_or(CalcError{message: "Cell is not in any voxel".to_owned()})?;
            sender.send(cell)?;
        }
        Ok(())
    }
}


impl VoxelContainer {
    pub fn update(&mut self, _t: f64, dt: f64) -> Result<(), ErrorVariant> {
        for vox in self.voxels.iter_mut() {
            // Updat the cell cycle. This also removes dead cells and creates new ones.
            vox.update_cell_cycles(dt)?;

            // Send information to calculate forces to other voxels
            vox.send_pos_information()?;
        }

        // Wait until all information is exchanged between voxels
        self.barrier.wait();

        let mut pos_info = HashMap::new();
        for vox in self.voxels.iter() {
            // Receive positions and calculate forces
            let pos_combinations: Vec<(IndexType, PositionType, IdType)> = vox.pos_receiver.try_iter().collect();
            pos_info.insert(vox.index, pos_combinations);
        }

        self.barrier.wait();

        for vox in self.voxels.iter() {
            let pos_combinations = pos_info.remove(&vox.index).unwrap();
            for (index, pos, id) in pos_combinations {
                let force = vox.calculate_total_cell_forces(&pos)?;
                // println!("Current Voxel {:?} Cell index {:?} Keys: {:?}", vox.index, index, vox.pos_senders.keys());
                vox.pos_senders[&index].send((index, force, id))?;
            }
        }

        // Wait until information exchange is complete
        self.barrier.wait();

        for vox in self.voxels.iter_mut() {
            vox.receive_forces_and_update_mechanics(dt)?;

            // Apply boundary conditions to the cell
            vox.apply_boundaries()?;

            // Send cells to new voxels if needed
            vox.send_cells()?;
        }

        // Wait for sending to finish
        self.barrier.wait();

        // Include the sent cells
        for vox in self.voxels.iter_mut() {
            vox.receive_cells()?;
        }

        self.barrier.wait();
        Ok(())
    }
}
