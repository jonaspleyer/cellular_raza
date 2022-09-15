use crate::concepts::errors::CalcError;
use crate::concepts::interaction::*;


use nalgebra::Vector3;


#[derive(Clone)]
pub struct LennardJones {
    pub epsilon: f64,
    pub sigma: f64,
}


impl Interaction for LennardJones {
    fn potential(&self, x: &Vector3<f64>, y: &Vector3<f64>) -> Result<Vector3<f64>, CalcError> {
        let r = (x - y).norm();
        let z = x - y;
        Ok(z/r * 4.0 * self.epsilon / r * (12.0 * (self.sigma/r).powf(12.0) - 6.0 * (self.sigma/r).powf(6.0)))
    }
}
