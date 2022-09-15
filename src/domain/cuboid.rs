use crate::concepts::domain::*;


use nalgebra::Vector3;


pub struct Cuboid {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub rebound: f64,
}


impl Domain for Cuboid {
    fn apply_boundary(&self, _pos1: &Vector3<f64>, pos2: &mut Vector3<f64>, speed: &mut Vector3<f64>) {
        // Check if the particle is below lower edge
        for i in 0..3 {
            if pos2[i] < self.min[i] {
                pos2[i] = 2.0 * self.min[i] - pos2[i];
                speed[i] *= -self.rebound;
            }
            // Check if the particle is over the edge
            if pos2[i] > self.max[i] {
                pos2[i] = 2.0 * self.max[i] - pos2[i];
                speed[i] *= -self.rebound;
            }
        }
    }
}
