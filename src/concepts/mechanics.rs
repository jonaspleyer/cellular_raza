

pub trait Mechanics<Pos,Velocity> {
    fn pos(&self) -> Pos;
    fn velocity(&self) -> Velocity;

    fn set_pos(&mut self, p: &Pos);
    fn set_velocity(&mut self, v: &Velocity);

    fn add_pos(&mut self, dp: &Pos);
    fn add_velocity(&mut self, dv: &Velocity);
}
