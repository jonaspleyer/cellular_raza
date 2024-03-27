use crate::errors::CalcError;

/// Trait describing force-interactions between cellular agents.
pub trait Interaction<Pos, Vel, Force, Inf = ()> {
    /// Get additional information of cellular properties (ie. for cell-specific interactions).
    /// For now, this can also be used to get the mass of the other cell-agent. In the future, we will probably provide a custom function for this.
    fn get_interaction_information(&self) -> Inf;

    /// Calculates the force (velocity-derivative) on the corresponding external position given external velocity.
    /// By providing velocities, we can calculate terms that are related to friction.
    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        own_vel: &Vel,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_info: &Inf,
    ) -> Result<Force, CalcError>;

    /// Checks if the other cell represented by position and information is a neighbor to the current one or not.
    #[allow(unused)]
    fn is_neighbour(&self, own_pos: &Pos, ext_pos: &Pos, ext_inf: &Inf) -> Result<bool, CalcError> {
        Ok(false)
    }

    /// Reacts to the results gathered by the [Interaction::is_neighbour] method and changes the state of the cell.
    #[allow(unused)]
    fn react_to_neighbours(&mut self, neighbours: usize) -> Result<(), CalcError> {
        Ok(())
    }
    // TODO
    // fn contact_function(&mut self, other_cell: &C, environment: &mut Env) -> Result<(), SimulationError>;
}
// TODO we should not use the option here

/// Defines how the cell uses the extracellular gradient.
pub trait InteractionExtracellularGradient<Cell, ConcGradientExtracellular> {
    /// The cell-agent senses the gradient and changes the behaviour of the cell.
    fn sense_gradient(
        cell: &mut Cell,
        gradient: &ConcGradientExtracellular,
    ) -> Result<(), CalcError>;
}

/// Specify how cellular reactions are taking place.
pub trait CellularReactions<ConcVecIntracellular, ConcVecExtracellular = ConcVecIntracellular> {
    /// Retrives the current intracellular concentration.
    fn get_intracellular(&self) -> ConcVecIntracellular;

    /// Sets the intracellular concentration. This is used by the backend after values have been updated.
    fn set_intracellular(&mut self, concentration_vector: ConcVecIntracellular);

    /// Calculate the time-related change of the intracellular and extracellular concentrations.
    /// This is not the increment itself (thus no parameter `dt` was specified) but rather the time-derivativ.
    /// Such an approach can be useful when designing addaptive solvers.
    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        internal_concentration_vector: &ConcVecIntracellular,
        external_concentration_vector: &ConcVecExtracellular,
    ) -> Result<(ConcVecIntracellular, ConcVecExtracellular), CalcError>;
}

/// Obtain the current volume of the cell
///
/// This trait is used when updating extracellular reactions and processes.
/// For more details see [domain](crate::Domain).
pub trait Volume<F = f64> {
    /// Obtain the cells current volume.
    fn get_volume(&self) -> F;
}
