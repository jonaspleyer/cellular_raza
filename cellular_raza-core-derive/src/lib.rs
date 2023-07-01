#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! This crate provides powerful derive macros to automatically implement the `UpdateCycle` and `UpdateMechanics` traits.
//! For the future, we are planning to have similar functionality with other concepts associated to CellAgents.

use proc_macro::{TokenStream, TokenTree};
use quote::{quote, ToTokens};
use syn::{parse_macro_input, DeriveInput};

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
            let generic_args_field = match &field.ty {
                syn::Type::Path(path) => {
                    path.path
                        .segments
                        .first()
                        .and_then(|segment| match &segment.arguments {
                            syn::PathArguments::AngleBracketed(arg) => {
                                Some(arg.args.clone().into_token_stream())
                            }
                            _ => None,
                        })
                }
                _ => None,
            }
            .or(Some("".into_token_stream()))
            .unwrap();

            let res2 = quote! {
                // TODO these generic parameters <P, V, N> should be inferred if possible
                // but it does not seem to be possible at this time.
                impl #struct_generics UpdateMechanics <#generic_args_field> for #struct_name #struct_generics {
                    fn set_last_position(&mut self, pos: P) {
                        self.#name.set_last_position(pos)
                    }
                    fn previous_positions(&self) -> std::collections::vec_deque::Iter<P> {
                        self.#name.previous_positions()
                    }
                    fn set_last_velocity(&mut self, vel: V) {
                        self.#name.set_last_velocity(vel)
                    }
                    fn previous_velocities(&self) -> std::collections::vec_deque::Iter<V> {
                        self.#name.previous_velocities()
                    }
                }
            };
            result.extend(TokenStream::from(res2));
        }
    }

    // Hand the output tokens back to the compiler
    TokenStream::from(result)
}

fn parse_non_delimiter_tokens(tokenstream: TokenStream) -> Vec<TokenStream> {
    use itertools::Itertools;
    tokenstream
        .into_iter()
        .group_by(|token| match token {
            TokenTree::Punct(p) => p.as_char() == char::from(','),
            _ => false,
        })
        .into_iter()
        .filter_map(|(is_comma, group)| if !is_comma { Some(group) } else { None })
        .map(|group| TokenStream::from_iter(group.into_iter()))
        .collect()
}

#[proc_macro]
/// Simple macro that checks if two supplied values are identical.
/// If this is the case it will insert the specified expression.
/// Otherwise it inserts nothing.
///
/// The macro can be used to omit code when two idents are not identical.
/// It can also be used to check if an identifier is contained in a range of identifiers.
/// ```
/// # use cellular_raza_core_derive::identical;
/// // Identifiers are not matching. This means that
/// // the last statement will never be inserted
/// identical!(MyFirstIdentifier, MySecondIdentifier, assert!(false));
///
/// // Identifiers are matching. The last statement
/// // (in this case `assert!(true)`) will be inserted into the code.
/// identical!(SameIdent, SameIdent, assert!(true));
///
/// // The String "hamster" is inserted here since both identifiers are matching.
/// assert_eq!("hamster", identical!(Id1, Id1, "hamster"));
///
/// // Identifiers are not equal if their capitalization does not match
/// identical!(caps, Caps, assert!(false));
///
/// // This works since 1_f64 is turned into a string and then
/// // compared to "1_f64" which is identically the same.
/// identical!(1_f64, "1_f64", assert!(true));
/// ```
///
/// The macro is not going to compile if given only two idents
/// ```compile_fail
/// identical!(Id1, Id2);
/// ```
///
/// The same holds true if we do not supply two identifiers.
/// ```compile_fail
/// identical!(Id1, println!("asdf"));
/// ```
pub fn identical(tokenstream: TokenStream) -> TokenStream {
    let tokens = parse_non_delimiter_tokens(tokenstream);
    let tokens_length = tokens.len();
    if tokens_length < 3 {
        panic!("Macro requires two identifiers to compare against each other and one expression to insert");
    } else {
        let m1 = tokens[0].clone();
        let m2 = tokens[1].clone();
        let expr = TokenStream::from_iter(tokens.into_iter().skip(2).into_iter());
        if m1.to_string() == m2.to_string() {
            expr
        } else {
            TokenStream::from(quote!())
        }
    }
}
