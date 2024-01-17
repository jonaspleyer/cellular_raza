pub use cellular_raza_core_proc_macro::run_test_for_aspects;

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

/// Derives the [Communicator](crate::backend::chili::simulation_flow::Communicator) trait.
///
/// This macro supports the [build_communicator!] macro.
/// It is useful when a complex communicator struct should automatically be generated
/// at compile time.
/// By deriving existing functionality from fields, we can make sure to avoid code duplication.
/// Furthermore, this macro can be used in the future to manually construct communicators
/// for new backends.
///
/// We use the [ChannelComm](crate::backend::chili::simulation_flow::ChannelComm) struct which has
/// a working implementation of the [Communicator](crate::backend::chili::simulation_flow::Communicator)
/// trait.
/// ```
/// use cellular_raza_core::{
///     backend::chili::simulation_flow::ChannelComm,
///     proc_macro::Communicator
/// };
///
/// // Define a new struct from which we want
/// // to derive the Communicator functionality
/// // and use the #[derive(Communicator)] macro.
/// #[derive(Communicator)]
/// #[CommunicatorCorePath(cellular_raza_core)]
/// struct NewCommunicator {
///
///     // The #[Comm(I, T)] field symbolizes that this field
///     // as an implementation of the Communicator<I, T> trait
///     // which should be derived.
///     #[Comm(usize, String)]
///     old_comm: ChannelComm<usize, String>,
/// }
/// ```
/// Note that when importing from the `cellular_raza` crate, every `cellular_raza_core`
/// needs to be replaced by `cellular_raza::core`.
pub use cellular_raza_core_proc_macro::Communicator;
