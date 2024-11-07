use crate::errors::{CalcError, RngError};

/// Methods for accessing the position of an agent.
pub trait Position<Pos> {
    /// Gets the cells current position.
    fn pos(&self) -> Pos;
    /// Gets the cells current velocity.
    fn set_pos(&mut self, position: &Pos);
}

/// Methods for accessing the velocity of an agent
pub trait Velocity<Vel> {
    /// Gets the cells current velocity.
    fn velocity(&self) -> Vel;
    /// Sets the cells current velocity.
    fn set_velocity(&mut self, velocity: &Vel);
}

/// Describes the position of a cell-agent and allows to calculate increments and set/get
/// information of the agent.
pub trait Mechanics<Pos, Vel, For, Float = f64> {
    /// Define a new random variable in case that the mechanics type contains a random aspect to
    /// its motion.
    /// By default this function does nothing.
    #[allow(unused)]
    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: Float,
    ) -> Result<(Pos, Vel), RngError>;

    /// Calculate the time-derivative of force and velocity given all the forces that act on the
    /// cell.
    /// Simple damping effects should be included in this trait if not explicitly given by the
    /// [SubDomainForce](super::SubDomainForce) trait.
    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError>;
}
