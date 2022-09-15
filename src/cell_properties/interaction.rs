use crate::concepts::errors::CalcError;


use nalgebra::Vector3;


#[derive(Clone)]
pub struct InteractionModel<'a, P> {
    pub potential: &'a dyn Fn(&Vector3<f64>, &Vector3<f64>, &P) -> Result<Vector3<f64>, CalcError>,
    pub parameter: P,
}
