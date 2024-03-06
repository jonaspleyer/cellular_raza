#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! This crate provides powerful macros to derive traits from the core crate.
//! It also provides macros to automatically construct AuxStorage structs
//! used to store intermediate data for running update steps and Communicator
//! struct to send messages between threads running the simulation.
//!
//! All macros are documented in the core crate unless their functionality can be
//! displayed without any additional dependencies.

mod aux_storage;
mod communicator;
mod from_map;
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
        UpdateReactions
    )
)]
pub fn _aux_storage(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    aux_storage::derive_aux_storage(input)
}

#[allow(missing_docs)]
#[proc_macro]
pub fn build_aux_storage(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    aux_storage::construct_aux_storage(input)
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

#[allow(missing_docs)]
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
pub fn run_test_for_aspects(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    testing::run_test_for_aspects(input)
}
