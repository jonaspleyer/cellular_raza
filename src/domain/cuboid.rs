use crate::concepts::domain::*;
use crate::cell_properties::cell_model::*;
use crate::concepts::mechanics::*;
use crate::concepts::errors::*;

use std::collections::{HashMap,HashSet};

use ndarray::Array3;


#[derive(Clone)]
pub struct Voxel<'a> {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub cells: Vec<&'a CellModel>,
}


pub struct Cuboid<'a> {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub voxel_sizes: [f64; 3],
    pub voxels: Array3<Voxel<'a>>,
}


impl<'a> Domain for Cuboid<'a> {
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


impl<'a> Cuboid<'a> {
    pub fn determine_voxel(&self, cell: &CellModel) -> [usize; 3] {
        let p = cell.mechanics.pos();
        let q0 = (p[0] - self.min[0]) / self.voxel_sizes[0];
        let q1 = (p[1] - self.min[1]) / self.voxel_sizes[1];
        let q2 = (p[2] - self.min[2]) / self.voxel_sizes[2];
        return [q0 as usize, q1 as usize, q2 as usize];
    }

    fn sort_in_voxels(&mut self, cell: &'a CellModel) {
        let vox = self.determine_voxel(cell);
        self.voxels[(vox[0], vox[1], vox[2])].cells.push(cell);
    }
}
