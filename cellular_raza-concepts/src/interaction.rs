use crate::errors::CalcError;

/// Trait describing force-interactions between cellular agents.
pub trait Interaction<Pos, Vel, Force, Inf = ()> {
    /// Get additional information of cellular properties (ie. for cell-specific interactions).
    /// For now, this can also be used to get the mass of the other cell-agent.
    /// In the future, we will probably provide a custom function for this.
    fn get_interaction_information(&self) -> Inf;

    /// Calculates the forces (velocity-derivative) on the corresponding external position given
    /// external velocity.
    /// By providing velocities, we can calculate terms that are related to friction.
    /// The function returns two forces, one acting on the current agent and the other on the
    /// external agent.
    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        own_vel: &Vel,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_info: &Inf,
    ) -> Result<(Force, Force), CalcError>;

    /// Checks if the other cell represented by position and information is a neighbor to the current one or not.
    #[allow(unused)]
    fn is_neighbour(&self, own_pos: &Pos, ext_pos: &Pos, ext_inf: &Inf) -> Result<bool, CalcError> {
        Ok(false)
    }

    /// Reacts to the results gathered by the [Interaction::is_neighbour]
    /// method and changes the state of the cell.
    #[allow(unused)]
    fn react_to_neighbours(&mut self, neighbours: usize) -> Result<(), CalcError> {
        Ok(())
    }
    // TODO
    // fn contact_function(&mut self, other_cell: &C, environment: &mut Env) -> Result<(), SimulationError>;
}

