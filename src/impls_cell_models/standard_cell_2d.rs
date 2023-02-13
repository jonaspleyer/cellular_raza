use crate::prelude::*;

use nalgebra::Vector2;
use serde::{Serialize,Deserialize};

use core::fmt::Debug;


#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
pub struct StandardCell2D {
    pub pos: Vector2<f64>,
    pub velocity: Vector2<f64>,

    pub cell_radius: f64,
    pub potential_strength: f64,

    pub velocity_reduction: f64,

    pub maximum_age: f64,

    pub remove: bool,
    pub current_age: f64,
}


impl Cycle<StandardCell2D> for StandardCell2D {
    fn update_cycle(dt: &f64, cell: &mut StandardCell2D) {
        cell.current_age += dt;
        if cell.current_age > cell.maximum_age {
            cell.remove = true;
        }
    }
}

impl Interaction<Vector2<f64>, Vector2<f64>, ()> for StandardCell2D {
    fn get_interaction_information(&self) -> Option<()> {
        None
    }

    fn calculate_force_on(&self, own_pos: &Vector2<f64>, ext_pos: &Vector2<f64>, _ext_information: &Option<()>) -> Option<Result<Vector2<f64>, CalcError>> {
        let z = own_pos - ext_pos;
        let r = z.norm();
        let sigma = 2.0 * self.cell_radius;
        let spatial_cutoff = (1.0 + (2.0*sigma-r).signum())*0.5;
        let dir = z/r;
        let bound = 4.0 + sigma/r;
        Some(Ok(
            dir * self.potential_strength * ((sigma/r).powf(2.0) - (sigma/r).powf(4.0)).min(bound).max(-bound) * spatial_cutoff
        ))
    }
}

impl Mechanics<Vector2<f64>, Vector2<f64>, Vector2<f64>> for StandardCell2D {
    fn pos(&self) -> Vector2<f64> {
        self.pos
    }

    fn velocity(&self) -> Vector2<f64> {
        self.velocity
    }

    fn set_pos(&mut self, p: &Vector2<f64>) {
        self.pos = *p;
    }

    fn set_velocity(&mut self, v: &Vector2<f64>) {
        self.velocity = *v;
    }

    fn calculate_increment(&self, force: Vector2<f64>) -> Result<(Vector2<f64>, Vector2<f64>), CalcError> {
        let dx = self.velocity;
        let dv = force - self.velocity_reduction * self.velocity;
        Ok((dx, dv))
    }
}
