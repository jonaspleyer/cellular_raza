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

/// Derives the update traits
/// [UpdateCycle](crate::backend::chili::UpdateCycle),
/// [UpdateMechanics](crate::backend::chili::UpdateMechanics),
/// [UpdateInteraction](crate::backend::chili::UpdateInteraction),
/// [UpdateReactions](crate::backend::chili::UpdateReactions) and
/// [UpdateReactionsContact](crate::backend::chili::UpdateReactionsContact)
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

/// Performs a complete numerical simulation.
///
/// ```ignore
/// run_simulation!(
///     // Arguments
///     domain: $domain:ident,
///     agents: $agents:ident,
///     settings: $settings:ident,
///     aspects: [$($asp:ident),*],
///
///     // Optional Arguments
///     $(core_path: $path:path,)?
///     $(parallelizer: $parallelizer:ident,)?
///     $(determinism: $determinism:bool,)?
///     $(aux_storage_name: $aux_storage_name:ident,)?
///     $(zero_force_default: $zero_force_default:closure,)?
///     $(zero_force_reactions_default: $zero_force_reactions_default:closure,)?
///     $(communicator_name: $communicator_name:ident,)?
///     $(mechanics_solver_order: $mechanics_solver_order:NonZeroUsize,)?
///     $(reactions_intra_solver_order: $reactions_intra_solver_order:NonZeroUsize,)?
///     $(reactions_contact_solver_order: $reactions_contact_solver_order:NonZeroUsize,)?
/// ) -> Result<StorageAccess<_, _>, SimulationError>;
/// ```
///
/// # Arguments
/// | Keyword | Description | Default |
/// | --- | --- | --- |
/// | `domain` | An object implementing the [Domain](cellular_raza_concepts::Domain) trait. | - |
/// | `agents` | Iterable of cell-agents | - |
/// | `settings` | See [Settings](crate::backend::chili::Settings) | - |
/// | `aspects` | List of simulation aspects such as `[Mechanics, Interaction, ...]` See below. | - |
/// | `core_path` | Path that points to the core module of `cellular_raza` | `cellular_raza::core` |
/// | `parallelizer` | Method to parallelize the simulation. Choose between `OsThreads` and `Rayon`. | `OsThreads` |
/// | `determinism` | Enforces sorting of values received from [step 2](super) | `false` |
/// | `aux_storage_name` | Name of helper struct to store cellular information. | `_CrAuxStorage` |
/// | `zero_force_default` | A closure returning the zero value of the force. | <code>&#124;c&#124; {num::Zero::zero()}</code> |
/// | `zero_force_reactions_default` | A closure returning the zero value of the reactions type. | <code>&#124;c&#124; {num::Zero::zero()}</code> |
/// | `communicator_name` | Name of the struct responsible for communication between threads. | `_CrCommunicator` |
/// | `mechanics_solver_order` | Order of the mechanics solver from `0` to `2` | `2` |
/// | `reactions_intra_solver_order` | Order of the intracellular reactions solver from `1` to `4` | `4` |
/// | `reactions_contact_solver_order` | Order of the contact reactions solver from `0` to `2` | `2` |
///
/// # Simulation Aspects
/// | Aspect | Trait(s) |
/// | --- | --- |
/// | `Mechanics` | [Mechanics](cellular_raza_concepts::Mechanics),[SubDomainMechanics](cellular_raza_concepts::SubDomainMechanics)|
/// | `Interaction` | [Interaction](cellular_raza_concepts::Interaction) |
/// | `Cycle` | [Cycle](cellular_raza_concepts::Cycle) |
/// | `Reactions` | [Reactions](cellular_raza_concepts::Reactions) |
/// | `ReactionsExtra` | [ReactionsExtra](cellular_raza_concepts::ReactionsExtra), [SubDomainReactions](cellular_raza_concepts::SubDomainReactions) |
/// | `ReactionsContact` | [ReactionsContact](cellular_raza_concepts::ReactionsContact) |
/// | `DomainForce` | [SubDomainForce](cellular_raza_concepts::SubDomainForce), [SubDomainMechanics](cellular_raza_concepts::SubDomainMechanics) |
///
/// # Returns
/// Returns a [StorageAccess](super::StorageAccess) to interoperate with calculted results.
///
/// # Comparison of Macro Keywords
///
/// This list shows which keyword can be provided in which macro.
///
/// <style>
///     .mytable th a {
///         writing-mode: vertical-lr;
///     }
/// </style>
/// <div class="mytable">
///
/// | Keyword | [run_simulation] | [run_main] | [test_compatibility] | [prepare_types] | [build_aux_storage] | [build_communicator] |
/// |:--------------------------------- | -- | -- | -- | -- | -- | -- |
//                                        rs   rm   tc   pt   ba   bc
/// | `domain`                          | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
/// | `agents`                          | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
/// | `settings`                        | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
/// | `aspects`                         | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
/// | `core_path`                       | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
/// | `parallelizer`                    | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
/// | `determinism`                     | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
/// | `aux_storage_name`                | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
/// | `zero_force_default`              | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
/// | `zero_force_reactions_default`    | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
/// | `communicator_name`               | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
/// | `mechanics_solver_order`          | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
/// | `reactions_intra_solver_order`    | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
/// | `reactions_contact_solver_order`  | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
///
/// </div>
#[doc(inline)]
pub use cellular_raza_core_proc_macro::run_simulation;

/// Prepare simulation types before executing the simulation via the [run_main] macro.
#[doc(inline)]
pub use cellular_raza_core_proc_macro::prepare_types;

#[doc(inline)]
pub use cellular_raza_core_proc_macro::test_compatibility;

/// Runs a with user-defined concepts. Assumes that types have been prepared with [prepare_types!].
///
/// See the documentation of the [run_simulation] macro.
pub use cellular_raza_core_proc_macro::run_main;
