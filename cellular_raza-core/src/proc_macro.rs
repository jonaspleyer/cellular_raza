pub use cellular_raza_core_proc_macro::run_test_for_aspects;

/// Derives the [Communicator](cellular_raza_core::backend::chili::simulation_flow) trait.
///
/// Allows for the creation of a general AuxStorage struct which functions via the defined
/// Update traits in [simulation_flow](crate::backend::chili::simulation_flow).
///
/// This proc macros purpose is to support the [build_communicator!] macro.
pub use cellular_raza_core_proc_macro::build_aux_storage;

/// Derives the [FromMap](crate::backend::chili::simulation_flow::FromMap) trait.
///  ```
/// use cellular_raza_core::backend::chili::simulation_flow::FromMap;
/// ```
pub use cellular_raza_core_proc_macro::FromMap;

/// Derives the [UpdateCycle](cellular_raza_core::backend::chili::simulation_flow::UpdateCycle) and [UpdateMechanics](cellular_raza_core::backend::chili::simulation_flow::UpdateMechanics) trait automatically for the containing struct.
pub use cellular_raza_core_proc_macro::AuxStorage;

/// 
pub use cellular_raza_core_proc_macro::build_communicator;

pub use cellular_raza_core_proc_macro::Communicator;
