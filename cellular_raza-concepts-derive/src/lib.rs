// #![warn(missing_docs)]
// #![warn(clippy::missing_docs_in_private_items)]

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(CellAgent, attributes(Cycle, Mechanics, Interaction, CellularReactions,InteractionExtracellularGradient))]
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
        if let Some(_) = field.attrs.iter().find_map(|x| match &x.meta {
            syn::Meta::Path(path) => {
                if path.is_ident("Cycle") {
                    Some(path)
                } else {
                    None
                }
            }
            _ => None,
        }) {
            let field_type = &field.ty;
            let res2 = quote! {
                impl #struct_generics Cycle<#struct_name> for #struct_name #struct_generics {
                    fn update_cycle(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &f64,
                        cell: &mut Self,
                    ) -> Option<CycleEvent> {
                        #field_type::update_cycle(rng, dt, cell)
                    }

                    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
                        #field_type::divide(rng, cell)
                    }

                    fn update_conditional_phased_death(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &f64,
                        cell: &mut Self,
                    ) -> Result<bool, DeathError> {
                        #field_type::update_conditional_phased_death(rng, dt, cell)
                    }
                }
            };
            result.extend(TokenStream::from(res2));
        }
        // Update Mechanics
        else if let Some(list) = field.attrs.iter().find_map(|x| match &x.meta {
            syn::Meta::List(list) => {
                if list.path.is_ident("Mechanics") {
                    Some(list)
                } else {
                    None
                }
            }
            _ => None,
        }) {
            let tokens = list.tokens.clone();
            let stream = quote!((#tokens));
            let attr: syn::TypeTuple = syn::parse2(stream).unwrap();
            let mut elems = attr.elems.into_iter();
            let position = elems.next().unwrap();
            let velocity = elems.next().unwrap();
            let force = elems.next().unwrap();

            let tokens = list.tokens.clone();
            let name = &field.ident;
            let res2 = quote! {
                impl #struct_generics Mechanics<#tokens> for #struct_name #struct_generics
                {
                    fn pos(&self) -> #position {self.#name.pos()}
                    fn velocity(&self) -> #velocity {self.#name.velocity()}
                    fn set_pos(&mut self, pos: &#position) {self.#name.set_pos(pos)}
                    fn set_velocity(&mut self, velocity: &#velocity) {self.#name.set_velocity(velocity)}
                    fn calculate_increment(&self, force: #force) -> Result<(#position, #velocity), CalcError> {
                        self.#name.calculate_increment(force)
                    }
                }
            };
            result.extend(TokenStream::from(res2));
        } else if let Some(list) = field.attrs.iter().find_map(|x| match &x.meta {
            syn::Meta::List(list) => {
                if list.path.is_ident("Interaction") {
                    Some(list)
                } else {
                    None
                }
            }
            _ => None,
        }) {
            let name = field.ident.clone();
            let tokens = list.tokens.clone();
            let stream = quote!((#tokens));
            let attr: syn::TypeTuple = syn::parse2(stream).unwrap();
            let mut elems = attr.elems.into_iter();
            let position = elems.next().unwrap();
            let velocity = elems.next().unwrap();
            let force = elems.next().unwrap();
            let inf = match elems.next() {
                Some(t) => t,
                None => syn::parse2(quote!(())).unwrap(),
            };
            let res = quote! {
                impl #struct_generics Interaction<#position, #velocity, #force, #inf> for #struct_name #struct_generics {
                    fn get_interaction_information(&self) -> #inf {
                        self.#name.get_interaction_information()
                    }

                    fn calculate_force_between(
                        &self,
                        own_pos: &#position,
                        own_vel: &#velocity,
                        ext_pos: &#position,
                        ext_vel: &#velocity,
                        ext_info: &#inf,
                    ) -> Option<Result<#force, CalcError>> {
                        self.#name.calculate_force_between(own_pos, own_vel, ext_pos, ext_vel, ext_info)
                    }

                    // fn contact_function(&mut self, other_cell: &C, environment: &mut Env) -> Result<(), SimulationError>;
                }
            };
            result.extend(TokenStream::from(res));
        } else if let Some(list) = field.attrs.iter().find_map(|x| match &x.meta {
            syn::Meta::List(list) => {
                if list.path.is_ident("CellularReactions") {
                    Some(list)
                } else {
                    None
                }
            }
            _ => None,
        }) {
            let name = field.ident.clone();
            let tokens = list.tokens.clone();
            let stream = quote!((#tokens));
            let attr: syn::TypeTuple = syn::parse2(stream).unwrap();
            let mut elems = attr.elems.into_iter();
            let concvecintracellular = elems.next().unwrap();
            let concvecextracellular = elems.next().or_else(|| Some(concvecintracellular.clone())).unwrap();
            let res = quote! {
                impl #struct_generics CellularReactions<#concvecintracellular, #concvecextracellular> for #struct_name #struct_generics {
                    fn get_intracellular(&self) -> #concvecintracellular {
                        self.#name.get_intracellular()
                    }

                    fn set_intracellular(&mut self, concentration_vector: #concvecintracellular) {
                        self.#name.set_intracellular(concentration_vector);
                    }

                    fn calculate_intra_and_extracellular_reaction_increment(
                        &self,
                        internal_concentration_vector: &#concvecintracellular,
                        external_concentration_vector: &#concvecextracellular,
                    ) -> Result<(#concvecintracellular, #concvecextracellular), CalcError> {
                        self.#name.calculate_intra_and_extracellular_reaction_increment(
                            internal_concentration_vector,
                            external_concentration_vector
                        )
                    }
                }
            };
            result.extend(TokenStream::from(res));
        } else if let Some(list) = field.attrs.iter().find_map(|x| match &x.meta {
            syn::Meta::List(list) => {
                if list.path.is_ident("InteractionExtracellularGradient") {
                    Some(list)
                } else {
                    None
                }
            }
            _ => None,
        }) {
            let field_type = &field.ty;
            let tokens = list.tokens.clone();
            let stream = quote!((#tokens));
            let attr: syn::TypeTuple = syn::parse2(stream).unwrap();
            let mut elems = attr.elems.into_iter();
            let concgradientextracellular = elems.next().unwrap();
            let res = quote! {
                impl InteractionExtracellularGradient<#struct_name #struct_generics, #concgradientextracellular> for #struct_name #struct_generics {
                    fn sense_gradient(
                        cell: &mut #struct_name #struct_generics,
                        gradient: &#concgradientextracellular,
                    ) -> Result<(), CalcError> {
                        #field_type::sense_gradient(cell, gradient)
                    }
                }
            };
            result.extend(TokenStream::from(res));
        }
    }

    // Hand the output tokens back to the compiler
    TokenStream::from(result)
}
