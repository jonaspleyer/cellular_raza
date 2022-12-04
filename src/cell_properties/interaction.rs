use crate::concepts::errors::CalcError;
use crate::concepts::interaction::*;


use nalgebra::Vector3;


#[derive(Clone)]
pub struct LennardJones {
    pub epsilon: f64,
    pub sigma: f64,
}


impl Interaction<Vector3<f64>,Vector3<f64>> for LennardJones {
    fn potential(&self, x: &Vector3<f64>, y: &Vector3<f64>) -> Result<Vector3<f64>, CalcError> {
        let r = (x - y).norm();
        let z = x - y;
        let dir = z/r;
        let val = 4.0 * self.epsilon / r * (12.0 * (self.sigma/r).powf(12.0) - 1.0 * (self.sigma/r).powf(1.0));
        let max = 10.0 * self.epsilon / r;
        if val > max {
            return Ok(dir * max);
        } else {
            return Ok(dir * val);
        }
    }
}
