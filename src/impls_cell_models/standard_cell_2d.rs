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

impl Interaction<Vector2<f64>, Vector2<f64>> for StandardCell2D {
    fn force(&self, own_pos: &Vector2<f64>, ext_pos: &Vector2<f64>) -> Option<Result<Vector2<f64>, CalcError>> {
        let z = own_pos - ext_pos;
        let r = z.norm();
        let dir = z/r;
        Some(Ok(dir * 0.0_f64.max(self.potential_strength * (self.cell_radius - r))))
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
        let dv = force;
        Ok((dx, dv))
    }
}
