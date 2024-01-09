#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! This crate provides powerful derive macros to automatically implement the `UpdateCycle` and `UpdateMechanics` traits.
//! For the future, we are planning to have similar functionality with other concepts associated to CellAgents.

mod aux_storage;
mod communicator;

#[proc_macro_derive(
    AuxStorage,
    attributes(UpdateCycle, UpdateMechanics, UpdateInteraction, UpdateReactions)
)]
/// Derives the `UpdateCycle` and `UpdateMechanics` trait automatically for the containing struct.
pub fn aux_storage(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    aux_storage::derive_aux_storage(input)
}

#[proc_macro]
pub fn build_aux_storage(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    aux_storage::construct_aux_storage(input)
}

/// Derives the [Communicator](cellular_raza_core::backend::chili::simulation_flow) trait.
#[proc_macro_derive(Communicator, attributes(Comm))]
pub fn communicator(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    communicator::derive_communicator(input)
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
