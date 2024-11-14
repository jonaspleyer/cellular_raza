#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! This crate provides powerful macros to derive traits from the core crate.
//! It also provides macros to automatically construct AuxStorage structs
//! used to store intermediate data for running update steps and Communicator
//! struct to send messages between threads running the simulation.
//!
//! All macros are documented in the core crate unless their functionality can be
//! displayed without any additional dependencies.

#[macro_use]
mod kwargs;

mod aux_storage;
mod communicator;
mod from_map;
mod run_sim;
mod simulation_aspects;
mod testing;

#[allow(missing_docs)]
#[proc_macro_derive(
    AuxStorage,
    attributes(
        AuxStorageCorePath,
        UpdateCycle,
        UpdateMechanics,
        UpdateInteraction,
        UpdateReactions,
        UpdateReactionsContact,
    )
)]
pub fn _aux_storage(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    aux_storage::derive_aux_storage(input)
}

#[allow(missing_docs)]
#[proc_macro]
pub fn build_aux_storage(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let kwargs = syn::parse_macro_input!(input as aux_storage::KwargsAuxStorageParsed);
    let kwargs = aux_storage::KwargsAuxStorage::from(kwargs);
    aux_storage::construct_aux_storage(kwargs)
}

#[allow(missing_docs)]
#[proc_macro]
pub fn aux_storage_constructor(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let kwargs = syn::parse_macro_input!(input as aux_storage::KwargsAuxStorageParsed);
    let kwargs = aux_storage::KwargsAuxStorage::from(kwargs);
    let res = aux_storage::default_aux_storage_initializer(&kwargs);
    quote::quote!(#res).into()
}

#[allow(missing_docs)]
#[proc_macro_derive(Communicator, attributes(CommunicatorCorePath, Comm))]
pub fn _communicator(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    communicator::derive_communicator(input)
}

#[allow(missing_docs)]
#[proc_macro]
pub fn build_communicator(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    communicator::construct_communicator(input)
}

/// Inserts as many blanks as generics were used to create the communicator struct by
/// [build_communicator!].
#[proc_macro]
pub fn communicator_generics_placeholders(
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    communicator::communicator_generics_placeholders(input)
}

#[allow(missing_docs)]
#[proc_macro_derive(FromMap, attributes(FromMapCorePath, FromMapIndex))]
pub fn from_map(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    from_map::derive_from_map(input)
}

#[proc_macro]
/// Run a particularly structured test multiple times for combinations of aspects
///
/// The tests which we would like to run are macros that will
/// be given as one argument to this `proc_macro`.
/// These tests need to adhere to a strict format.
/// ```
/// macro_rules! some_test(
///     (
///         name:$test_name:ident,
///         aspects:[$($asp:ident),*]
///     ) => {
///         // Any code can be run here.
///         // For example, we can create a docstring test by using
///
///         /// ```
///         /// assert_eq!(0_usize, 10_usize - 10_usize);
///         $(#[doc = concat!("println!(\"", stringify!($asp), "\")")])*
///         /// ```
///         fn $test_name () {}
///     }
/// );
///
/// // This is how you would call the test by hand
///
/// some_test!(
///     name:my_test_name,
///     aspects: [Mechanics, Interaction]
/// );
/// ```
///
/// In the next step, we can use `run_test_for_aspects` to run this automatically generated
/// docstring test for every combination of aspects that we specify.
///
/// ```
/// # macro_rules! some_test(
/// #     (
/// #         name:$test_name:ident,
/// #         aspects:[$($asp:ident),*]
/// #     ) => {
/// #         // Any code can be run here.
/// #         // For example, we can create a docstring test by using
/// #
/// #         /// ```
/// #         /// assert_eq!(0_usize, 10_usize - 10_usize);
/// #         $(#[doc = concat!("println!(\"", stringify!($asp), "\")")])*
/// #         /// ```
/// #         fn $test_name () {}
/// #     }
/// # );
/// # use cellular_raza_core_proc_macro::run_test_for_aspects;
/// run_test_for_aspects!(
///     test: some_test,
///     aspects: [Mechanics, Interaction]
/// );
/// ```
/// This will have generated the following code:
/// ```
/// # macro_rules! some_test(
/// #     (
/// #         name:$test_name:ident,
/// #         aspects:[$($asp:ident),*]
/// #     ) => {
/// #         // Any code can be run here.
/// #         // For example, we can create a docstring test by using
/// #
/// #         /// ```
/// #         /// assert_eq!(0_usize, 10_usize - 10_usize);
/// #         $(#[doc = concat!("println!(\"", stringify!($asp), "\")")])*
/// #         /// ```
/// #         fn $test_name () {}
/// #     }
/// # );
/// some_test!(
///     name:mechanics,
///     aspects: [Mechanics]
/// );
/// some_test!(
///     name:interaction,
///     aspects: [Interaction]
/// );
/// some_test!(
///     name:mechanics_interaction,
///     aspects: [Mechanics, Interaction]
/// );
/// some_test!(
///     name:interaction_mechanics,
///     aspects: [Interaction, Mechanics]
/// );
/// ```
///
/// # Minimum Combinations
/// It is possible to specify a minimum number of combinations to test.
/// This means if we specify N aspects but only want to test combinations of M (where M<N)
/// different aspects, we can set the `min_combinations` variable of this macro.
///
/// ```
/// # macro_rules! some_test(
/// #     (
/// #         name:$test_name:ident,
/// #         aspects:[$($asp:ident),*]
/// #     ) => {
/// #         // Any code can be run here.
/// #         // For example, we can create a docstring test by using
/// #
/// #         /// ```
/// #         /// assert_eq!(0_usize, 10_usize - 10_usize);
/// #         $(#[doc = concat!("println!(\"", stringify!($asp), "\")")])*
/// #         /// ```
/// #         fn $test_name () {}
/// #     }
/// # );
/// # use cellular_raza_core_proc_macro::run_test_for_aspects;
/// run_test_for_aspects!(
///     test: some_test,
///     aspects: [Mechanics, Interaction, Cycle, Reactions],
///     min_combinations: 3,
/// );
/// ```
///
/// # Unsorted Combinations
/// By default all generated combinations of simulation aspects are sorted and will thus not
/// produce different tests when being reordered.
/// This means we assume that `aspects: [Mechanics, Interaction]` is identical to `aspects:
/// [Interaction, Mechanics]`.
/// In the case where we also want to test the unsorted cases, we can specify the `sorted` keyword.
///
/// ```
/// # macro_rules! some_test(
/// #     (
/// #         name:$test_name:ident,
/// #         aspects:[$($asp:ident),*]
/// #     ) => {
/// #         // Any code can be run here.
/// #         // For example, we can create a docstring test by using
/// #
/// #         /// ```
/// #         /// assert_eq!(0_usize, 10_usize - 10_usize);
/// #         $(#[doc = concat!("println!(\"", stringify!($asp), "\")")])*
/// #         /// ```
/// #         fn $test_name () {}
/// #     }
/// # );
/// # use cellular_raza_core_proc_macro::run_test_for_aspects;
/// run_test_for_aspects!(
///     test: some_test,
///     aspects: [Mechanics, Interaction],
///     sorted: false,
/// );
/// ```
pub fn run_test_for_aspects(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    testing::run_test_for_aspects(input)
}

/// Construct, test and run a full simulation.
///
/// # Arguments
// TODO use link when compiler error is fixed: https://github.com/rust-lang/rust/issues/123019
/// The `KwargsSim` struct contains all required and optional
/// arguments for this macro.
///
/// # Internals
/// This macro calls [prepare_types!], [test_compatibility!] and [run_main!] one after
/// another with identical arguments (where possible) and thus yields results for a full
/// simulation.
#[proc_macro]
pub fn run_simulation(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let kwargs = syn::parse_macro_input!(input as run_sim::KwargsSimParsed);
    let kwargs = run_sim::KwargsSim::from(kwargs);
    run_sim::run_simulation(kwargs).into()
}

/// Prepares communicator and auxiliary storage types with [build_communicator!] and
/// [build_aux_storage!].
#[proc_macro]
pub fn prepare_types(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let kwargs = syn::parse_macro_input!(input as run_sim::KwargsPrepareTypesParsed);
    let kwargs = run_sim::KwargsPrepareTypes::from(kwargs);
    run_sim::prepare_types(kwargs).into()
}

/// Checks if defined types and concepts are compatible before actually executing the simulation.
///
/// This macro only serves the purpose for easy-to-read compiler errors.
/// It has no runtime-overhead since it will be fully optimized away.
#[proc_macro]
pub fn test_compatibility(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let kwargs = syn::parse_macro_input!(input as run_sim::KwargsCompatibilityParsed);
    let kwargs = run_sim::KwargsCompatibility::from(kwargs);
    run_sim::test_compatibility(kwargs).into()
}

#[allow(missing_docs)]
#[proc_macro]
pub fn run_main(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let kwargs = syn::parse_macro_input!(input as run_sim::KwargsMainParsed);
    let kwargs = run_sim::KwargsMain::from(kwargs);
    run_sim::run_main(kwargs).into()
}
