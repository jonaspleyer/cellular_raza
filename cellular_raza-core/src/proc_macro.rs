pub use cellular_raza_core_derive::run_test_for_aspects;

/// Derives the [Communicator](cellular_raza_core::backend::chili::simulation_flow) trait.
///
/// Allows for the creation of a general AuxStorage struct which functions via the defined
/// Update traits in [simulation_flow](crate::backend::chili::simulation_flow).
///
/// This proc macros purpose is to support the [build_communicator!] macro.
pub use cellular_raza_core_derive::build_aux_storage;

/// Derives the [FromMap](crate::backend::chili::simulation_flow::FromMap) trait.
///  ```
/// use cellular_raza_core::backend::chili::simulation_flow::FromMap;
/// ```
pub use cellular_raza_core_derive::FromMap;

/// Derives the [UpdateCycle](cellular_raza_core::backend::chili::simulation_flow::UpdateCycle) and [UpdateMechanics](cellular_raza_core::backend::chili::simulation_flow::UpdateMechanics) trait automatically for the containing struct.
pub use cellular_raza_core_derive::AuxStorage;

/// 
pub use cellular_raza_core_derive::build_communicator;

pub use cellular_raza_core_derive::Communicator;
