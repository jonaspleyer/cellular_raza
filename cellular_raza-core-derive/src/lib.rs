#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! This crate provides powerful derive macros to automatically implement the `UpdateCycle` and `UpdateMechanics` traits.
//! For the future, we are planning to have similar functionality with other concepts associated to CellAgents.

mod aux_storage;
mod communicator;
mod from_map;
mod simulation_aspects;
mod testing;

#[proc_macro_derive(
    AuxStorage,
    attributes(UpdateCycle, UpdateMechanics, UpdateInteraction, UpdateReactions)
)]
/// Derives the `UpdateCycle` and `UpdateMechanics` trait automatically for the containing struct.
pub fn _aux_storage(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    aux_storage::derive_aux_storage(input)
}

#[proc_macro]
pub fn build_aux_storage(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    aux_storage::construct_aux_storage(input)
}

/// Derives the [Communicator](cellular_raza_core::backend::chili::simulation_flow) trait.
///
/// This proc macros purpose is to support the [build_communicator] macro.
/// However, we still test individual derivation in the core crate.
#[proc_macro_derive(Communicator, attributes(CommunicatorCorePath, Comm))]
pub fn _communicator(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    communicator::derive_communicator(input)
}

#[proc_macro]
pub fn build_communicator(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    communicator::construct_communicator(input)
}

#[proc_macro_derive(FromMap, attributes(FromMapIndex))]
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
/// # use cellular_raza_core_derive::run_test_for_aspects;
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

/* #[derive(Clone, Debug)]
enum Aspect {
    Mechanics,
    Cycle,
    Interaction,
    CellularReactions,
}

fn element_to_aspect(element: syn::Expr) -> syn::Result<Aspect> {
    match element.clone() {
        syn::Expr::Path(path) => {
            let ident: syn::Ident = path.path.segments.into_iter().next().unwrap().ident;
            if ident == "Mechanics" {
                Ok(Aspect::Mechanics)
            } else if ident.to_string().to_lowercase() == "cycle" {
                Ok(Aspect::Cycle)
            } else if ident.to_string().to_lowercase() == "interaction" {
                Ok(Aspect::Interaction)
            } else if ident.to_string().to_lowercase() == "cellularreactions" {
                Ok(Aspect::CellularReactions)
            } else {
                Err(syn::Error::new(
                    element.span(),
                    format!("Expected one of [Mechanics, Cycle, Interaction, CellularReactions]"),
                ))
            }
        }
        _ => Err(syn::Error::new(element.span(), "Expected expression here.")),
    }
}

struct SimulationInformation {
    setup: syn::Ident,
    settings: syn::Ident,
    aspects: Vec<Aspect>,
}

impl syn::parse::Parse for SimulationInformation {
    fn parse(input: ParseStream) -> syn::parse::Result<Self> {
        let setup: syn::Ident = input.parse()?;
        let _: syn::token::Comma = input.parse()?;
        let settings: syn::Ident = input.parse()?;
        let _: syn::token::Comma = input.parse()?;
        let aspects: syn::ExprArray = input.parse()?;

        let aspects = aspects
            .elems
            .into_iter()
            .map(element_to_aspect)
            .collect::<syn::Result<Vec<_>>>()?;

        Ok(Self {
            setup,
            settings,
            aspects,
        })
    }
}

///
#[proc_macro]
pub fn run_full_simulation(input: TokenStream) -> TokenStream {
    // let mut tokens = parse_non_delimiter_tokens(input).into_iter();
    // let setup = tokens.next().unwrap();
    // let settings = tokens.next().unwrap();
    // let aspects = tokens.next().unwrap();

    let SimulationInformation {
        setup,
        settings,
        aspects,
    } = parse_macro_input!(input as SimulationInformation);

    TokenStream::from(quote!({
        // TODO construct the aux storage from the simulation aspects
        struct AuxStorage {}
        1_u8
    }))
}*/
