use cellular_control::prelude::*;

use nalgebra::Vector2;
use uuid::Uuid;

use core::fmt::Debug;


#[derive(Clone,Debug)]
pub struct StandardCell2D {
    pub pos: Vector2<f64>,
    pub velocity: Vector2<f64>,

    pub cell_radius: f64,
    pub potential_strength: f64,

    pub maximum_age: f64,

    pub remove: bool,
    pub current_age: f64,

    pub id: Uuid,
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
    fn potential(&self, x: &Vector2<f64>, y: &Vector2<f64>) -> Result<Vector2<f64>, CalcError> {
        let z = x - y;
        let r = z.norm();
        let dir = z/r;
        Ok(dir * 0.0_f64.max(self.potential_strength * (self.cell_radius - r)))
    }
}

impl Mechanics<Vector2<f64>, Vector2<f64>> for StandardCell2D {
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

    fn add_pos(&mut self, dp: &Vector2<f64>) {
        self.pos += dp;
    }

    fn add_velocity(&mut self, dv: &Vector2<f64>) {
        self.velocity += dv;
    }
}

impl Id for StandardCell2D {
    fn get_uuid(&self) -> Uuid {
        self.id
    }
}
