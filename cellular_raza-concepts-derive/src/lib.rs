// #![warn(missing_docs)]
// #![warn(clippy::missing_docs_in_private_items)]

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// Derive [concepts](cellular_raza_concepts)
///
/// This macro allows to simply derive already implemented concepts
/// from struct fields.
/// Currently the only allowed notation is by defining macros with curly braces.
/// ```ignore
/// #[derive(CellAgent)]
/// struct MyCell {
///     #[Cycle]
///     cycle: MyCycle,
///     ...
/// }
/// ```
/// Some attributes also require to specify types as well.
/// ```ignore
/// struct MyCell {
///     #[Mechanics([f64; 3], [f64; 3], [f64; 3])]
///     interaction: MyMechanics,
///     ...
/// }
/// ```
/// A summary can be seen in the following table
///
/// | Attribute | Type Arguments |
/// | --- | --- |
/// | `Cycle` | - |
/// | Mechanics | `(Pos, Vel, For)` The position, velocity and force types. |
/// | Interaction | `(Pos, Vel, For, Inf=())` The position, velocity, force and information types. The last one is optional. |
/// | CellularReactions | `(ConcVecIntracellular, ConcVecExtracellular)` The  types for intra- and extracellular concentraitons. |
/// | InteractionExtracellularGradient | `(ConcGradientExtracellular,)` The type of extracellular gradient. |
///
/// Notice that all type arguments need to be a list.
/// Thus we need to insert a comma at the end if we only have one entry.
/// See the `InteractionExtracellularGradient` attribute.
#[proc_macro_derive(
    CellAgent,
    attributes(
        Cycle,
        Mechanics,
        Interaction,
        CellularReactions,
        InteractionExtracellularGradient
    )
)]
pub fn derive_cell_agent(input: TokenStream) -> TokenStream {
    // TODO modularize this into multiple functions responsible for implementing the individual
    // aspects and document/comment them accordingly

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
                        <#field_type as Cycle<#struct_name>>::update_cycle(rng, dt, cell)
                    }

                    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
                        <#field_type as Cycle<#struct_name>>::divide(rng, cell)
                    }

                    fn update_conditional_phased_death(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &f64,
                        cell: &mut Self,
                    ) -> Result<bool, DeathError> {
                        <#field_type as Cycle<#struct_name>>::update_conditional_phased_death(rng, dt, cell)
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
            let field_type = &field.ty.clone();
            let res2 = quote! {
                impl #struct_generics Mechanics<#tokens> for #struct_name #struct_generics
                {
                    fn pos(&self) -> #position {
                        <#field_type as Mechanics<#tokens>>::pos(&self.#name)
                    }
                    fn velocity(&self) -> #velocity {
                        <#field_type as Mechanics<#tokens>>::velocity(&self.#name)
                    }
                    fn set_pos(&mut self, pos: &#position) {
                        <#field_type as Mechanics<#tokens>>::set_pos(&mut self.#name, pos)
                    }
                    fn set_velocity(&mut self, velocity: &#velocity) {
                        <#field_type as Mechanics<#tokens>>::set_velocity(&mut self.#name, velocity)
                    }
                    fn calculate_increment(&self, force: #force) -> Result<(#position, #velocity), CalcError> {
                        <#field_type as Mechanics<#tokens>>::calculate_increment(&self.#name, force)
                    }
                    fn set_random_variable(&mut self,
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: f64,
                    ) -> Result<Option<f64>, RngError> {
                        <#field_type as Mechanics<#tokens>>::set_random_variable(&mut self.#name, rng, dt)
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
            let field_type = field.ty.clone();
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
                        <#field_type as Interaction<#position, #velocity, #force, #inf>>::get_interaction_information(&self.#name)
                    }

                    fn calculate_force_between(
                        &self,
                        own_pos: &#position,
                        own_vel: &#velocity,
                        ext_pos: &#position,
                        ext_vel: &#velocity,
                        ext_info: &#inf,
                    ) -> Option<Result<#force, CalcError>> {
                        <#field_type as Interaction<#position, #velocity, #force, #inf>>::calculate_force_between(&self.#name, own_pos, own_vel, ext_pos, ext_vel, ext_info)
                    }

                    fn is_neighbour(&self, own_pos: &#position, ext_pos: &#position, ext_inf: &#inf) -> Result<bool, CalcError> {
                        <#field_type as Interaction<#position, #velocity, #force, #inf>>::is_neighbour(&self.#name, own_pos, ext_pos, ext_inf)
                    }

                    fn react_to_neighbours(&mut self, neighbours: usize) -> Result<(), CalcError> {
                        <#field_type as Interaction<#position, #velocity, #force, #inf>>::react_to_neighbours(&mut self.#name, neighbours)
                    }
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
            let field_type = field.ty.clone();
            let tokens = list.tokens.clone();
            let stream = quote!((#tokens));
            let attr: syn::TypeTuple = syn::parse2(stream).unwrap();
            let mut elems = attr.elems.into_iter();
            let concvecintracellular = elems.next().unwrap();
            let concvecextracellular = elems
                .next()
                .or_else(|| Some(concvecintracellular.clone()))
                .unwrap();
            let res = quote! {
                impl #struct_generics CellularReactions<#concvecintracellular, #concvecextracellular> for #struct_name #struct_generics {
                    fn get_intracellular(&self) -> #concvecintracellular {
                        <#field_type as CellularReactions<#concvecintracellular, #concvecextracellular>>::get_intracellular(&self.#name)
                    }

                    fn set_intracellular(&mut self, concentration_vector: #concvecintracellular) {
                        <#field_type as CellularReactions<#concvecintracellular, #concvecextracellular>>::set_intracellular(&mut self.#name, concentration_vector);
                    }

                    fn calculate_intra_and_extracellular_reaction_increment(
                        &self,
                        internal_concentration_vector: &#concvecintracellular,
                        external_concentration_vector: &#concvecextracellular,
                    ) -> Result<(#concvecintracellular, #concvecextracellular), CalcError> {
                        <#field_type as CellularReactions<#concvecintracellular, #concvecextracellular>>::calculate_intra_and_extracellular_reaction_increment(
                            &self.#name,
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
                        <#field_type as InteractionExtracellularGradient<#struct_name #struct_generics, #concgradientextracellular>>::sense_gradient(cell, gradient)
                    }
                }
            };
            result.extend(TokenStream::from(res));
        }
    }

    // Hand the output tokens back to the compiler
    TokenStream::from(result)
}
