#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(CellAgent, attributes(Cycle, Mechanics))]
pub fn my_macro(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let ast = parse_macro_input!(input as DeriveInput);

    // Build the output, possibly using quasi-quotation
    let struct_name = ast.ident;
    let struct_generics = ast.generics;
    let mut result = TokenStream::new();

    let data: syn::DataStruct = match ast.data {
        syn::Data::Struct(data) => data,
        _ => panic!("Usage of #[Cycle] on a non-struct type"),
    };
    for field in data.fields.iter() {
        // Update Cycle
        if field.attrs.iter().any(|x| match &x.meta {
            syn::Meta::Path(path) => path.is_ident("Cycle"),
            _ => false,
        }) {
            let name = &field.ident;
            let res2 = quote! {
                impl #struct_generics Cycle<#struct_name> for #struct_name #struct_generics {
                    fn update_cycle(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &f64,
                        cell: &mut Cell,
                    ) -> Option<CycleEvent> {
                        self.#name.update_cycle(rng, dt, cell)
                    }

                    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Cell) -> Result<Cell, DivisionError> {
                        self.#name.divide(rng, cell)
                    }

                    fn update_conditional_phased_death(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &f64,
                        cell: &mut Cell,
                    ) -> Result<bool, DeathError> {
                        self.#name.update_conditional_phased_death(rng, dt, cell)
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
            let res2 = quote! {
                // TODO this needs! to be checked!!
                todo!();
                impl #struct_generics <Pos, Vel, For> Mechanics<Pos, Vel, For> for #struct_name #struct_generics
                where
                    #name: Mechanics<Pos, Vel, For>,
                {
                    fn pos(&self) -> Pos {self.#name.pos()}
                    fn velocity(&self) -> Vel {self.#name.velocity()}
                    fn set_pos(&mut self, pos: &Pos) {self.#name.set_pos(pos)}
                    fn set_velocity(&mut self, velocity: &Vel) {self.#name.set_velocity(velocity)}
                    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError> {
                        self.#name.calculate_increment(force)
                    }
                }
            };
            result.extend(TokenStream::from(res2));
        }
    }

    // Hand the output tokens back to the compiler
    TokenStream::from(result)
}
