use crate::CalcError;

/// Defines how the cell uses the extracellular gradient.
///
/// This trait can also be derived with the [CellAgent](crate::CellAgent) derive macro.
/// ```
/// use cellular_raza_concepts::{reactions_old::InteractionExtracellularGradient,
///     CalcError, CellAgent
/// };
///
/// struct DoNothingGradient;
///
/// impl<C, G> InteractionExtracellularGradient<C, G> for DoNothingGradient {
///     fn sense_gradient(
///         cell: &mut C,
///         gradient: &G,
///     ) -> Result<(), CalcError> {
///         Ok(())
///     }
/// }
///
/// #[derive(CellAgent)]
/// struct MyAgent {
///     #[ExtracellularGradient]
///     gradient: DoNothingGradient,
/// }
/// ```
pub trait InteractionExtracellularGradient<Cell, ConcGradientExtracellular> {
    /// The cell-agent senses the gradient and changes the behaviour of the cell.
    fn sense_gradient(
        cell: &mut Cell,
        gradient: &ConcGradientExtracellular,
    ) -> Result<(), CalcError>;
}

/// Specify how cellular reactions are taking place.
///
/// This trait can also be derived with the [CellAgent](crate::CellAgent) derive macro.
/// ```
/// use cellular_raza_concepts::{reactions_old::CellularReactions, CellAgent, CalcError};
/// struct MyReactions {
///     intracellular: f64,
///     half_time: f64,
/// }
///
/// impl CellularReactions<f64> for MyReactions {
///     fn get_intracellular(&self) -> f64 {
///         self.intracellular
///     }
///
///     fn set_intracellular(&mut self, intracellular: f64) {
///         self.intracellular = intracellular;
///     }
///
///     fn calculate_intra_and_extracellular_reaction_increment(
///         &self,
///         internal_concentration_vector: &f64,
///         external_concentration_vector: &f64,
///     ) -> Result<(f64, f64), CalcError> {
///         Ok((-self.half_time * self.intracellular, self.half_time * self.intracellular))
///     }
/// }
// #[derive(CellAgent)]
// struct MyAgent {
//     #[Reactions]
//     reactions: MyReactions,
// }
/// ```
pub trait CellularReactions<ConcVecIntracellular, ConcVecExtracellular = ConcVecIntracellular> {
    /// Retrieves the current intracellular concentration.
    fn get_intracellular(&self) -> ConcVecIntracellular;

    /// Sets the intracellular concentration. This is used by the backend after values have been updated.
    fn set_intracellular(&mut self, concentration_vector: ConcVecIntracellular);

    /// Calculate the time-related change of the intracellular and extracellular concentrations.
    /// This is not the increment itself (thus no parameter `dt` was specified) but rather the time-derivative.
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
