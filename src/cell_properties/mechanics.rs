use crate::concepts::mechanics::*;


use std::ops::AddAssign;
use nalgebra::{Vector3,Vector6};


#[derive(Clone,Debug)]
pub struct MechanicsModel {
    pos_vel: Vector6<f64>,
    pub dampening_constant: f64,
}


impl From<(&Vector3<f64>, &Vector3<f64>, f64)> for MechanicsModel {
    fn from(pv: (&Vector3<f64>, &Vector3<f64>, f64)) -> Self {
        let p = pv.0;
        let v = pv.1;
        let x = Vector6::<f64>::new(p[0], p[1], p[2], v[0], v[1], v[2]);
        MechanicsModel {
            pos_vel: x,
            dampening_constant: pv.2,
        }
    }
}


impl Mechanics for MechanicsModel {
    fn pos(&self) -> Vector3<f64> {
        return Vector3::<f64>::from(self.pos_vel.fixed_rows::<3>(0));
    }

    fn velocity(&self) -> Vector3<f64> {
        return Vector3::<f64>::from(self.pos_vel.fixed_rows::<3>(3));
    }

    fn set_pos(&mut self, p: &Vector3<f64>) {
        self.pos_vel.fixed_rows_mut::<3>(0).set_column(0, p);
    }

    fn set_velocity(&mut self, v: &Vector3<f64>) {
        self.pos_vel.fixed_rows_mut::<3>(3).set_column(0, v);
    }

    fn add_pos(&mut self, dp: &Vector3<f64>) {
        self.pos_vel.fixed_rows_mut::<3>(0).add_assign(dp);
    }

    fn add_velocity(&mut self, dv: &Vector3<f64>) {
        self.pos_vel.fixed_rows_mut::<3>(3).add_assign(dv);
    }
}
