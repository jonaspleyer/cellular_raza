use crate::errors::{CalcError, RngError};

// TODO merge this with the interaction trait!
/// Describes the position of a cell-agent and allows to calculate increments and set/get information of the agent.
pub trait Mechanics<Pos, Vel, For, Float = f64> {
    /// Gets the cells current position.
    fn pos(&self) -> Pos;
    /// Gets the cells current velocity.
    fn velocity(&self) -> Vel;
    /// Sets the cells current position.
    fn set_pos(&mut self, pos: &Pos);
    /// Sets the cells current velocity.
    fn set_velocity(&mut self, velocity: &Vel);

    /// Define a new random variable in case that the mechanics type contains a random aspect to its motion.
    /// By default this function does nothing.
    #[allow(unused)]
    fn set_random_variable(
        &mut self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: Float,
    ) -> Result<Option<Float>, RngError> {
        Ok(None)
    }

    /// Calculate the time-derivative of force and velocity given all the forces that act on the cell.
    /// Simple dampening effects should be included in this trait if not explicitly given by the [Voxel](super::domain::Voxel).
    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError>;
}
