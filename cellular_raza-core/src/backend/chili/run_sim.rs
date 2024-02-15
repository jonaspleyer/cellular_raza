/// We do run simulations sometimes
///
/// General idea:
/// - Definitions
///     - Define objects (such as aux storage)
/// - Implement settings
///     - Build Storage Manager
///     - Build Time-Stepper
///     - (tracing handler?)
/// - Build thunk (thin function) which does single update step
/// - Call function parallelized
/// - Combine threads
///     - Return SimulationResult with Storage Manager
///     
#[macro_export]
#[doc(hidden)]
macro_rules! run_simulation {
    () => {};
}

#[doc(inline)]
pub use crate::run_simulation;
