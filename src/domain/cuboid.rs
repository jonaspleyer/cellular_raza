use crate::concepts::domain::*;
use crate::cell_properties::cell_model::*;
use crate::concepts::mechanics::*;
use crate::concepts::errors::*;

use std::collections::{HashMap,HashSet};


pub struct Voxel<'a> {
    pub min: [f64; 3],
    pub max: [f64; 3],
    cells: HashSet<&'a CellModel>,
}


pub struct Cuboid {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub rebound: f64,
    pub voxel_sizes: [f64; 3],
    pub voxels: HashMap<[usize; 3],Vec<usize>>,
}


impl Domain for Cuboid {
    fn apply_boundary(&self, cell: &mut CellModel) -> Result<(),BoundaryError> {
        let mut pos = cell.mechanics.pos();
        let mut velocity = cell.mechanics.velocity();

        // For each dimension (ie 3 in general)
        for i in 0..3 {
            // Check if the particle is below lower edge
            if pos[i] < self.min[i] {
                pos[i] = 2.0 * self.min[i] - pos[i];
                velocity[i] *= -self.rebound;
            }
            // Check if the particle is over the edge
            if pos[i] > self.max[i] {
                pos[i] = 2.0 * self.max[i] - pos[i];
                velocity[i] *= -self.rebound;
            }
        }
        // Set new position and velocity of particle
        cell.mechanics.set_pos(&pos);
        cell.mechanics.set_velocity(&velocity);

        // If new position is still out of boundary return error
        let p = cell.mechanics.pos();
        for i in 0..3 {
            if p[i] < self.min[i] || p[i] > self.max[i] {
                return Err(BoundaryError { message: format!("Particle with id {} is out of domain at position {:?}", cell.id, p) });
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

    fn sort_in_voxels(&mut self, _cell: &CellModel) {
        // let index = self.determine_voxel(&cell);
        
    }

    /* fn get_interaction_partners(&self, cell: &CellModel) -> Vec<usize> {
        let index = self.determine_voxel(&cell);
        return self.voxels[&index].clone();
    }*/
}
