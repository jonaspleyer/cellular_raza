use crate::concepts::domain::*;
use crate::cell_properties::cell_model::*;
use crate::concepts::mechanics::*;


pub struct Cuboid {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub rebound: f64,
}


impl Domain for Cuboid {
    fn apply_boundary(&self, cell: &mut CellModel) {
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
    }
}
