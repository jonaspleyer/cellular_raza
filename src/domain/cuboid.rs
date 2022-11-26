use crate::concepts::domain::*;
use crate::cell_properties::cell_model::*;
use crate::concepts::mechanics::*;
use crate::concepts::errors::*;
use crate::cell_properties::cycle::CycleModel;
use crate::concepts::cycle::Cycle;
use crate::concepts::interaction::Interaction;
use crate::plotting::cells_2d::*;

use std::collections::HashMap;

use ndarray::Array3;

use nalgebra::Vector3;

use std::sync::Arc;
use hurdles::Barrier;
use crossbeam_channel::{Sender, Receiver};



#[derive(Clone)]
pub struct Voxel {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub cells: Vec<CellModel>,
    pub cell_senders: HashMap<[usize; 3], Sender<CellModel>>,
    pub cell_receiver: Receiver<CellModel>,
    pub pos_senders: HashMap<[usize; 3], Sender<([usize; 3], Vector3<f64>, u32)>>,
    pub pos_receiver: Receiver<([usize; 3], Vector3<f64>, u32)>,
    pub id: [usize; 3],
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
    pub fn determine_voxel(&self, cell: &CellModel) -> [usize; 3] {
        let p = cell.mechanics.pos();
        let q0 = (p[0] - self.min[0]) / self.voxel_sizes[0];
        let q1 = (p[1] - self.min[1]) / self.voxel_sizes[1];
        let q2 = (p[2] - self.min[2]) / self.voxel_sizes[2];
        return [q0 as usize, q1 as usize, q2 as usize];
    }
}


impl Voxel {
    pub fn update_cell_cycles(&mut self, dt: f64) {
        // Update the cycle status of cells
        self.cells.iter_mut().for_each(|cell| CycleModel::update(&dt, cell));

        // Remove cells which have been flagged
        self.cells.retain(|cell| !cell.flags.removal);
    }

    // TODO use ODE integrator to solve for all forces
    // Calculate the forces beforehand and then just reuse in ODE rhs
    // This should not be a problem since there is no time dependence
    fn calculate_total_cell_forces(&self, x: &Vector3<f64>) -> Result<Vector3<f64>, CalcError> {
        let mut force = Vector3::from([0.0; 3]);
        for cell in &self.cells {
                if &cell.mechanics.pos() != x {
                    force += cell.interaction.potential(&cell.mechanics.pos(), x)?;
                }
        }
        Ok(force)
    }

    fn receive_cells(&mut self) {
        let new_cells: Vec<CellModel> = self.cell_receiver.try_iter().collect();
        for cell in new_cells {
            self.cells.push(cell);
        }
    }

    fn send_pos_information(&mut self) -> Result<(), CalcError> {
        for index in self.pos_senders.keys() {
            for cell in self.cells.iter() {
                self.pos_senders[index].send((*index, cell.mechanics.pos(), cell.id));
            }
        }
        Ok(())
    }

    fn receive_pos_and_send_force_informations(&self) -> Result<(), CalcError> {
        let pos_combinations: Vec<([usize; 3], Vector3<f64>, u32)> = self.pos_receiver.try_iter().collect();

        for (index, pos, id) in pos_combinations {
            let force = self.calculate_total_cell_forces(&pos)?;
            self.pos_senders[&index].send((index, force, id));
        }
        Ok(())
    }

    fn receive_forces_and_update_mechanics(&mut self, dt: f64) -> Result<(), CalcError> {
        let forces_other_voxels: HashMap<u32, Vector3<f64>> = self.pos_receiver
            .try_iter()
            .map(|(index, force, id)| {
                (id, force)
            }).collect();
        
        let forces_total: Vec<Vector3<f64>> = self.cells
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

    fn send_cells(&mut self) -> Result<(), CalcError> {
        // Send the updated cells to other threads if necessary
        for cell in self.cells.drain_filter(|cell| self.domain.determine_voxel(&cell)!=self.id) {
            let index = self.domain.determine_voxel(&cell);
            let sender = self.cell_senders.get(&index).ok_or(CalcError{message: "Cell is not in any voxel".to_owned()})?;
            sender.send(cell);
        }
        Ok(())
    }

    pub fn run_full_update(&mut self, t: f64, dt: f64) -> Result<(), CalcError> {
        

        // Receive calculated forces and update velocity and position
        // Calculate forces and update cell locations

        
        self.apply_boundaries();

        // Receive new cells in domain
        self.receive_cells();
        Ok(())
    }
}


impl VoxelContainer {
    pub fn update(&mut self, t: f64, dt: f64) -> Result<(), CalcError> {
        for vox in self.voxels.iter_mut() {
            // Updat the cell cycle. This also removes dead cells and creates new ones.
            vox.update_cell_cycles(dt);

            // Send information to calculate forces to other voxels
            vox.send_pos_information();
        }

        // Wait until all information is exchanged between voxels
        self.barrier.wait();

        for vox in self.voxels.iter_mut() {
            // Receive positions and calculate forces
            vox.receive_pos_and_send_force_informations()?;
        }

        // Wait until information exchange is complete
        self.barrier.wait();

        for vox in self.voxels.iter_mut() {
            vox.receive_forces_and_update_mechanics(dt)?;

            // Apply boundary conditions to the cell
            vox.apply_boundaries();

            // Send cells to new voxels if needed
            vox.send_cells();
        }

        // Wait for sending to finish
        self.barrier.wait();

        // Include the sent cells
        for vox in self.voxels.iter_mut() {
            vox.receive_cells();
        }

        self.barrier.wait();
        Ok(())
    }
}
