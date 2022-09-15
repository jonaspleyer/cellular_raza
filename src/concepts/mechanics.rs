use nalgebra::Vector3;


pub trait Mechanics {
    fn pos(&self) -> Vector3<f64>;
    fn velocity(&self) -> Vector3<f64>;

    fn set_pos(&mut self, p: &Vector3<f64>);
    fn set_velocity(&mut self, v: &Vector3<f64>);

    fn add_pos(&mut self, dp: &Vector3<f64>);
    fn add_velocity(&mut self, dv: &Vector3<f64>);
}
