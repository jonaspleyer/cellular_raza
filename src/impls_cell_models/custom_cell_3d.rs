use crate::prelude::*;

use nalgebra::Vector3;
use uuid::Uuid;

use core::fmt::Debug;


#[derive(Clone,Debug)]
pub struct CustomCell3D {
    pub pos: Vector3<f64>,
    pub velocity: Vector3<f64>,

    pub cell_radius: f64,
    pub potential_strength: f64,

    pub maximum_age: f64,

    pub remove: bool,
    pub current_age: f64,

    pub id: Uuid,
}


impl Cycle<CustomCell3D> for CustomCell3D {
    fn update_cycle(dt: &f64, cell: &mut CustomCell3D) {
        cell.current_age += dt;
        if cell.current_age > cell.maximum_age {
            cell.remove = true;
        }
    }
}

impl Interaction<Vector3<f64>, Vector3<f64>> for CustomCell3D {
    fn force(&self, own_pos: &Vector3<f64>, ext_pos: &Vector3<f64>) -> Option<Result<Vector3<f64>, CalcError>> {
        let z = own_pos - ext_pos;
        let r = z.norm();
        let dir = z/r;
        Some(Ok(dir * 0.0_f64.max(self.potential_strength * (self.cell_radius - r))))
    }
}

impl Mechanics<Vector3<f64>, Vector3<f64>, Vector3<f64>> for CustomCell3D {
    fn pos(&self) -> Vector3<f64> {
        self.pos
    }

    fn velocity(&self) -> Vector3<f64> {
        self.velocity
    }

    fn set_pos(&mut self, p: &Vector3<f64>) {
        self.pos = *p;
    }

    fn set_velocity(&mut self, v: &Vector3<f64>) {
        self.velocity = *v;
    }

    fn calculate_increment(&self, force: Vector3<f64>) -> Result<(Vector3<f64>, Vector3<f64>), CalcError> {
        let dx = self.velocity;
        let dv = force;
        Ok((dx, dv))
    }
}
