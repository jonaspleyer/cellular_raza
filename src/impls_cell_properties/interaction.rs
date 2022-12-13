use crate::concepts::errors::CalcError;
use crate::concepts::interaction::*;


use nalgebra::Vector3;


#[derive(Clone)]
pub struct LennardJones {
    pub epsilon: f64,
    pub sigma: f64,
}


impl Interaction<Vector3<f64>,Vector3<f64>> for LennardJones {
    fn force(&self, own_pos: &Vector3<f64>, ext_pos: &Vector3<f64>) -> Option<Result<Vector3<f64>, CalcError>> {
        let z = own_pos - ext_pos;
        let r = z.norm();
        let dir = z/r;
        let val = 4.0 * self.epsilon / r * (12.0 * (self.sigma/r).powf(12.0) - 1.0 * (self.sigma/r).powf(1.0));
        let max = 10.0 * self.epsilon / r;
        if val > max {
            return Some(Ok(dir * max));
        } else {
            return Some(Ok(dir * val));
        }
    }
}
