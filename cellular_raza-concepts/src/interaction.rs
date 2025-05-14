use crate::errors::CalcError;

/// Exposes information to other cells or the domain for calculating interactions
pub trait InteractionInformation<Inf> {
    /// Get additional information of cellular properties (ie. for cell-specific interactions).
    /// For now, this can also be used to get the mass of the other cell-agent.
    /// In the future, we will probably provide a custom function for this.
    fn get_interaction_information(&self) -> Inf;
}

/// Trait describing force-interactions between cellular agents.
pub trait Interaction<Pos, Vel, Force, Inf = ()>: InteractionInformation<Inf> {
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
}

/// Allows cells to react to neighboring cells
pub trait NeighborInteraction<Pos, Inf>: InteractionInformation<Inf> {
    /// Checks if the other cell represented by position and information is a neighbor to the
    /// current one or not.
    #[allow(unused)]
    fn is_neighbor(&self, own_pos: &Pos, ext_pos: &Pos, ext_inf: &Inf) -> Result<bool, CalcError> {
        Ok(false)
    }

    /// Reacts to the results gathered by the [Interaction::is_neighbor]
    /// method and changes the state of the cell.
    #[allow(unused)]
    fn react_to_neighbors(&mut self, neighbors: usize) -> Result<(), CalcError> {
        Ok(())
    }
    // TODO
    // fn contact_function(&mut self, other_cell: &C, environment: &mut Env) -> Result<(), SimulationError>;
}

impl<Pos, Vel, For, Inf> Interaction<Pos, Vel, For, Inf>
    for Box<dyn Interaction<Pos, Vel, For, Inf>>
{
    fn get_interaction_information(&self) -> Inf {
        use core::ops::Deref;
        self.deref().get_interaction_information()
    }
    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        own_vel: &Vel,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_info: &Inf,
    ) -> Result<(For, For), CalcError> {
        use core::ops::Deref;
        self.deref()
            .calculate_force_between(own_pos, own_vel, ext_pos, ext_vel, ext_info)
    }
    fn is_neighbor(&self, own_pos: &Pos, ext_pos: &Pos, ext_inf: &Inf) -> Result<bool, CalcError> {
        use core::ops::Deref;
        self.deref().is_neighbor(own_pos, ext_pos, ext_inf)
    }
    fn react_to_neighbors(&mut self, neighbors: usize) -> Result<(), CalcError> {
        use core::ops::DerefMut;
        self.deref_mut().react_to_neighbors(neighbors)
    }
}
