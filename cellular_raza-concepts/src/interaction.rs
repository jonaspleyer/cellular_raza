use crate::errors::CalcError;

/// Exposes information which is used to calculate interactions between cells and the domain.
pub trait InteractionInformation<Inf> {
    /// Get additional information of cellular properties (ie. for cell-specific interactions).
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

/// Allows reacting to multiple neighbors
///
/// Information of neighbors is first accumulated via the [NeighborSensing::accumulate_information]
/// function and finally, the cell reacts to the gathered data via the
/// [NeighborSensing::react_to_neighbors] function.
/// Finally, the accumulator is cleared with [NeighborSensing::clear_accumulator].
///
/// ## Neighbor Counting
/// ```
/// use cellular_raza_concepts::*;
///
/// struct Cell {/* .. */};
///
/// impl<Pos, Inf> NeighborSensing<Pos, usize, Inf> for Cell
/// where
///     Cell: InteractionInformation<Inf>,
/// {
///     fn accumulate_information(
///         &self,
///         _: &Pos,
///         _: &Pos,
///         _: &Inf,
///         neighbors: &mut usize
///     ) -> Result<(), CalcError> {
///         Ok(*neighbors += 1)
///     }
///
///     fn react_to_neighbors(&mut self, neighbors: &usize) -> Result<(), CalcError> {
///         /* .. */
///         Ok(())
///     }
///
///     fn clear_accumulator(neighbors: &mut usize) {
///         *neighbors = 0;
///     }
/// }
///
/// ```
pub trait NeighborSensing<Pos, Acc, Inf = ()>: InteractionInformation<Inf> {
    /// Checks if the other cell represented by position and information is a neighbor to the current one or not.
    fn accumulate_information(
        &self,
        own_pos: &Pos,
        ext_pos: &Pos,
        ext_inf: &Inf,
        accumulator: &mut Acc,
    ) -> Result<(), CalcError>;

    /// Reacts to the results gathered by the [Interaction::is_neighbor]
    /// method and changes the state of the cell.
    fn react_to_neighbors(&mut self, accumulator: &Acc) -> Result<(), CalcError>;

    /// Clears the accumulator
    fn clear_accumulator(accumulator: &mut Acc);
}
