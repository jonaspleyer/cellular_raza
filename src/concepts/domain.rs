use nalgebra::Vector3;


pub trait Domain  {
    fn apply_boundary(&self, pos1: &Vector3<f64>, pos2: &mut Vector3<f64>, speed: &mut Vector3<f64>);
}
