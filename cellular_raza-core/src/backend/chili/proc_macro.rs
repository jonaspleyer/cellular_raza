pub use cellular_raza_core_proc_macro::run_test_for_aspects;

/// Allows for the creation of a general AuxStorage struct which functions via the defined
/// Update traits in the [chili](crate::backend::chili) backend.
///
/// This proc macros purpose is to support the [build_communicator!] macro.
pub use cellular_raza_core_proc_macro::build_aux_storage;

/// Derives the [FromMap](crate::backend::chili::FromMap) trait.
///
/// Note that all fields of the struct need to implement the [FromMap] trait.
/// It is currently still necessary to specify the type parameter for the [FromMap] trait.
///
///  ```
/// use cellular_raza_core::backend::chili::{ChannelComm, FromMap};
/// use cellular_raza_concepts::IndexError;
///
/// #[derive(FromMap)]
/// #[FromMapIndex(usize)]
/// struct NewCommunicator {
///     channel_communicator_1: ChannelComm<usize, String>,
///     channel_communicator_2: ChannelComm<usize, bool>,
/// }
/// ```
pub use cellular_raza_core_proc_macro::FromMap;

/// Returns code which can be used to initialize a new empty AuxStorage.
///
/// The code produced by this macro is a closure.
/// Thus the most common usecase will be
/// ```
/// # use cellular_raza_core::backend::chili::*;
/// # use serde::{Serialize, Deserialize};
/// // This macro could also be executed by others such as run_simulation! or prepare_types!
/// build_aux_storage!(
///     aspects: [Cycle],
///     core_path: cellular_raza_core,
/// );
///
/// // This will create a new instance of the previously defined AuxStorage.
/// let aux_storage = (aux_storage_constructor!(
///     aspects: [Cycle],// and any other aspects desired
///     core_path: cellular_raza_core,
/// ))(()/* here should be the cell for which we construct the aux storage*/);
/// ```
pub use cellular_raza_core_proc_macro::aux_storage_constructor;

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
/// let new_map = std::collections::BTreeMap::from([
///     (0, std::collections::BTreeSet::from([1,3])),
///     (1, std::collections::BTreeSet::from([0,2])),
///     (2, std::collections::BTreeSet::from([1,3])),
///     (3, std::collections::BTreeSet::from([2,0])),
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

#[doc(inline)]
pub use cellular_raza_core_proc_macro::communicator_generics_placeholders;

#[doc(inline)]
pub use cellular_raza_core_proc_macro::run_simulation;

/// Prepare simulation types before executing the simulation via the [run_main] macro.
#[doc(inline)]
pub use cellular_raza_core_proc_macro::prepare_types;

#[doc(inline)]
pub use cellular_raza_core_proc_macro::test_compatibility;

/// Runs a with user-defined concepts. Assumes that types have been prepared with [prepare_types!].
///
/// ```ignore
/// run_main!(
///     // Mandatory Arguments
///     domain: $domain:ident,
///     agents: $agents:ident,
///     settings: $settings:ident,
///     aspects: [$($asp:ident),*],
///     // Optional Arguments
///     $(core_path: $path:path,)?
///     $(parallelizer: $parallelizer:ident,)?
/// )
/// ```
///
/// # Arguments
/// | Keyword | Description | Default |
/// | --- | --- | --- |
/// | `domain` | An object implementing the [Domain](cellular_raza_concepts::Domain) trait. | - |
/// | `agents` | Iterable of cell-agents compatible with [Domain](cellular_raza_concepts::Domain) | - |
/// | `settings` | [Settings](crate::backend::chili::Settings) | - |
/// | `aspects` | List of simulation aspects such as `[Mechanics, Interaction, ...]` See below. | - |
/// | `core_path` | Path that points to the core module of `cellular_raza` | `cellular_raza::core` |
/// | `parallelizer` | Method to parallelize the simulation (see below) | `OsThreads` |
///
/// # Simulation Aspects
/// | Aspect | Trait(s) |
/// | --- | --- |
/// | `Mechanics` | [`Mechanics`](cellular_raza_concepts::Mechanics),[`SubDomainMechanics`](cellular_raza_concepts::SubDomainMechanics)|
/// | `Interaction` | [`Interaction`](cellular_raza_concepts::Interaction) |
/// | `Cycle` | [`Cycle`](cellular_raza_concepts::Cycle) |
/// | `Reactions`¹ | [`CellularReactions`](cellular_raza_concepts::CellularReactions) |
/// | `DomainForce` | [`SubDomainForce`](cellular_raza_concepts::SubDomainForce) |
/// ¹Currently not working
///
/// ## Parallelization
/// ### OsThreads
/// This approach uses the provided threads from the standard library to parallelize execution of
/// the simulation.
/// We spawn multiple threads and store every storage instance when they have finished.
/// ```
/// # let n_threads = 3;
/// # type SimulationError = std::io::Error;
/// # fn main_code() -> Result<(), SimulationError> {Ok(())}
/// let mut handles = Vec::new();
///
/// for key in 0..n_threads {
///     let handle = std::thread::Builder::new()
///         .name(format!("cellular_raza-worker_thread-{:03.0}", key))
///         .spawn(move ||
///     -> Result<_, SimulationError> {main_code()})?;
///     handles.push(handle);
/// }
///
/// // Join them when the simulation has finished
/// let mut storage_accesses = vec![];
/// for handle in handles {
///     let result = handle
///         .join()
///         .expect("Could not join threads after simulation has finished")?;
///     storage_accesses.push(result);
/// }
/// # Ok::<(),Box<dyn std::error::Error>>(())
/// ```
///
/// ### Rayon
// TODO
/// This feature is currently not supported.
/// In the future, we plan on supporting additional parallelization strategies such as
/// [rayon](https://docs.rs/rayon/latest/rayon/).
pub use cellular_raza_core_proc_macro::run_main;
