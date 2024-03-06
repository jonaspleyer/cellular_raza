pub use cellular_raza_core_proc_macro::run_test_for_aspects;

/// Allows for the creation of a general AuxStorage struct which functions via the defined
/// Update traits in the [chili](crate::backend::chili) backend.
///
/// This proc macros purpose is to support the [build_communicator!] macro.
pub use cellular_raza_core_proc_macro::build_aux_storage;

/// Derives the [FromMap](crate::backend::chili::FromMap) trait.
///  ```
/// use cellular_raza_core::backend::chili::FromMap;
/// ```
pub use cellular_raza_core_proc_macro::FromMap;

/// Derives the [UpdateCycle](crate::backend::chili::UpdateCycle) and
/// [UpdateMechanics](crate::backend::chili::UpdateMechanics)
/// trait automatically for the containing struct.
pub use cellular_raza_core_proc_macro::AuxStorage;

/// Automatically build communicator struct depending on simulation aspects.
///
/// This macro internally constructs a new struct with fields for every given simulation aspect.
/// Each field is a [ChannelComm](crate::backend::chili::ChannelComm)
/// struct with different types.
///
/// It also automatically derives the
/// [FromMap](crate::backend::chili::FromMap) trait such that a
/// collection of communicators can be constructed from a given map.
/// ```
/// use cellular_raza_core::backend::chili::build_communicator;
///
/// build_communicator!(
///     // Define the name of the generated struct
///     name: MyCommunicator,
///
///     // Which simulation aspects and informatino exchange should be satisfied.
///     aspects: [Cycle],
///
///      // Path to the core library. Use `cellular_raza::core` when
///     // import from the `cellular_raza` crate.
///     core_path: cellular_raza_core
/// );
///
/// // Use the new struct in the following.
/// use cellular_raza_core::backend::chili::FromMap;
///
/// let new_map = std::collections::HashMap::from([
///     (0, vec![1,3]),
///     (1, vec![0,2]),
///     (2, vec![1,3]),
///     (3, vec![2,0]),
/// ]);
/// let communicators = MyCommunicator::from_map(&new_map).unwrap();
/// assert_eq!(communicators.len(), 4);
/// ```
pub use cellular_raza_core_proc_macro::build_communicator;

/// Derives the [Communicator](crate::backend::chili::Communicator) trait.
///
/// This macro supports the [build_communicator!] macro.
/// It is useful when a complex communicator struct should automatically be generated
/// at compile time.
/// By deriving existing functionality from fields, we can make sure to avoid code duplication.
/// Furthermore, this macro can be used in the future to manually construct communicators
/// for new backends.
///
/// We use the [ChannelComm](crate::backend::chili::ChannelComm) struct which has
/// a working implementation of the [Communicator](crate::backend::chili::Communicator)
/// trait.
/// ```
/// use cellular_raza_core::{
///     backend::chili::{ChannelComm, Communicator},
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

/// TODO
/// ```
/// ```
pub use cellular_raza_core_proc_macro::communicator_generics_placeholders;
