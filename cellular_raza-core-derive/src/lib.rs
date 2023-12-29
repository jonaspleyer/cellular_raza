#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! This crate provides powerful derive macros to automatically implement the `UpdateCycle` and `UpdateMechanics` traits.
//! For the future, we are planning to have similar functionality with other concepts associated to CellAgents.

use proc_macro::{TokenStream, TokenTree};
use quote::{quote, ToTokens};
use syn::{parse::ParseStream, parse_macro_input, spanned::Spanned, DeriveInput};

#[proc_macro_derive(AuxStorage, attributes(UpdateCycle, UpdateMechanics))]
/// Derives the `UpdateCycle` and `UpdateMechanics` trait automatically for the containing struct.
pub fn aux_storage(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let ast = parse_macro_input!(input as DeriveInput);

    // Build the output, possibly using quasi-quotation
    let struct_name = ast.ident;
    let struct_generics = ast.generics.clone();
    let mut result = TokenStream::new();

    let data: syn::DataStruct = match ast.data {
        syn::Data::Struct(data) => data,
        _ => panic!("Usage of #[UpdateCycle] on a non-struct type"),
    };
    for field in data.fields.iter() {
        // Update Cycle
        if field.attrs.iter().any(|x| match &x.meta {
            syn::Meta::Path(path) => path.is_ident("UpdateCycle"),
            _ => false,
        }) {
            let name = &field.ident;
            let res2 = quote! {
                impl #struct_generics UpdateCycle for #struct_name #struct_generics {
                    fn set_cycle_events(&mut self, events: Vec<CycleEvent>) {
                        self.#name.set_cycle_events(events)
                    }

                    fn get_cycle_events(&self) -> Vec<CycleEvent> {
                        self.#name.get_cycle_events()
                    }

                    fn add_cycle_event(&mut self, event: CycleEvent) {
                        self.#name.add_cycle_event(event)
                    }
                }
            };
            result.extend(TokenStream::from(res2));
        }
        // Update Mechanics
        else if field.attrs.iter().any(|x| match &x.meta {
            syn::Meta::Path(path) => path.is_ident("UpdateMechanics"),
            _ => false,
        }) {
            let name = &field.ident;
            let generic_args = match &field.ty {
                syn::Type::Path(path) => {
                    path.path
                        .segments
                        .first()
                        .and_then(|segment| match &segment.arguments {
                            syn::PathArguments::AngleBracketed(arg) => {
                                Some(arg.args.clone().into_iter().collect::<Vec<_>>())
                            }
                            _ => None,
                        })
                }
                _ => None,
            }
            .or(Some(Vec::new()))
            .unwrap();

            let position_generic = generic_args[0].clone();
            let velocity_generic = generic_args[1].clone();
            let force_generic = generic_args[2].clone();

            let res2 = quote! {
                impl #struct_generics UpdateMechanics <#(#generic_args),*> for #struct_name #struct_generics
                where
                    F: Clone + core::ops::AddAssign<F> + num::Zero,
                {
                    fn set_last_position(&mut self, pos: #position_generic) {
                        self.#name.set_last_position(pos)
                    }
                    fn previous_positions(&self) -> std::collections::vec_deque::Iter<#position_generic> {
                        self.#name.previous_positions()
                    }
                    fn set_last_velocity(&mut self, vel: #velocity_generic) {
                        self.#name.set_last_velocity(vel)
                    }
                    fn previous_velocities(&self) -> std::collections::vec_deque::Iter<#velocity_generic> {
                        self.#name.previous_velocities()
                    }
                    fn add_force(&mut self, force: #force_generic) {
                        self.#name.add_force(force);
                    }
                    fn get_current_force(&self) -> #force_generic {
                        self.#name.get_current_force()
                    }
                    fn clear_forces(&mut self) {
                        self.#name.clear_forces()
                    }
                }
            };
            result.extend(TokenStream::from(res2));
        }
    }

    // Hand the output tokens back to the compiler
    TokenStream::from(result)
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
