mod config;
mod domain_decomposition;
/// Error types which can arise during the simulation.
///
/// There is one main error type [SimulationError](errors::SimulationError) which should be derivable from errors
/// arising during the simulation process.
/// It is required for custom error types `MyCustomError` of the engine to implement the `From<MyCustomError> for SimulationError`.
/// Errors should be seperated by their ability to be recovered, ignored or handled otherwise.
/// Since this crate aims to provide an adaptive solving aproach, it is desired to have a fallback
/// mechanism which can be called for errors which may arise due to precision problems.
///
/// The following table shows a summary of the errors currently supported and their handling
/// aproach.
///
/// | ErrorType | Possible Error Reasons | Handling Strategy |
/// | --- | --- | --- |
/// | BoundaryError | Solver Accuracy, Domain Implementaion bug, Internal engine error | [RevertChangeAccuracy](errors::HandlingStrategies::RevertChangeAccuracy) |
/// | CalcError | Solver Accuracy, Bug by user implementation of corresponding function, Internal engine error | [RevertChangeAccuracy](errors::HandlingStrategies::RevertChangeAccuracy) |
// TODO implement the handling of these errors!
mod errors;
mod supervisor;
mod trait_bounds;

// Concepts for every simulation aspect
pub use cellular_raza_concepts::*;

// Implementation Details necessary
pub use config::*;
pub use domain_decomposition::*;
pub use errors::*;
pub use supervisor::*;
pub use trait_bounds::*;
