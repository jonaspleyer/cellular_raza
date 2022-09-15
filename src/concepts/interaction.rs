use nalgebra::Vector3;
use crate::concepts::errors::CalcError;


pub trait Interaction {
    fn potential(&self, x: &Vector3<f64>, y: &Vector3<f64>) -> Result<Vector3<f64>, CalcError>;
}
