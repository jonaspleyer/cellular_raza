use quote::quote;

use super::simulation_aspects::*;

// ##################################### DERIVE ######################################
// ##################################### PARSING #####################################
struct Communicator {
    struct_name: syn::Ident,
    generics: syn::Generics,
    comms: Vec<CommField>,
}

struct CommParser {
    _comm_ident: syn::Ident,
    index: syn::Type,
    _comma: syn::Token![,],
    message: syn::Type,
}

struct CommField {
    field_name: Option<syn::Ident>,
    field_type: syn::Type,
    index: syn::Type,
    message: syn::Type,
}

impl syn::parse::Parse for Communicator {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item_struct: syn::ItemStruct = input.parse()?;
        let comms = item_struct
            .fields
            .iter()
            .map(|field| field.attrs.iter().zip(std::iter::repeat(field)))
            .flatten()
            .map(|(attr, field)| {
                let s = &attr.meta;
                let stream: proc_macro::TokenStream = quote!(#s).into();
                let parsed: CommParser = syn::parse(stream)?;
                Ok(CommField {
                    field_name: field.ident.clone(),
                    field_type: field.ty.clone(),
                    index: parsed.index,
                    message: parsed.message,
                })
            })
            .collect::<syn::Result<Vec<_>>>()?;
        Ok(Self {
            struct_name: item_struct.ident,
            generics: item_struct.generics,
            comms,
        })
    }
}

impl syn::parse::Parse for CommParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _comm_ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            _comm_ident,
            index: content.parse()?,
            _comma: content.parse()?,
            message: content.parse()?,
        })
    }
}

// ################################### IMPLEMENTING ##################################
fn wrap_pre_flags(stream: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    quote!(
        #[allow(unused)]
        #[allow(non_camel_case_types)]
        const _: () = {
            extern crate cellular_raza_core as _crc;
            extern crate cellular_raza_core_derive as _crc_derive;

            use _crc::backend::chili::{
                errors::SimulationError,
                simulation_flow::Communicator
            };
            use _crc_derive::Communicator;

            #stream
        };
    )
}

impl Communicator {
    fn derive_communicator(&self) -> proc_macro2::TokenStream {
        let struct_name = &self.struct_name;
        let (impl_generics, ty_generics, where_clause) = &self.generics.split_for_impl();
        let addendum = quote!(I: Clone + core::hash::Hash + Eq + Ord,);
        let where_clause = match where_clause {
            Some(w) => quote!(where #(#w.predicates), #addendum),
            None => quote!(where #addendum),
        };

        let mut res = proc_macro2::TokenStream::new();
        res.extend(self.comms.iter().map(|comm| {
            let field_name = &comm.field_name;
            let field_type = &comm.field_type;

            let index = &comm.index;
            let message = &comm.message;

            quote!(
                impl #impl_generics Communicator<#index, #message>
                for #struct_name #ty_generics #where_clause

                {
                    fn send(&mut self, receiver: &#index, message: #message) -> Result<(), SimulationError> {
                        <#field_type as Communicator<#index, #message>>::send(&mut self.#field_name, receiver, message)
                    }
                    fn receive(&mut self) -> Vec<#message> {
                        <#field_type as Communicator<#index, #message>>::receive(&mut self.#field_name)
                    }
                }
            )
        }));
        wrap_pre_flags(res)
    }
}

pub fn derive_communicator(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let comm = syn::parse_macro_input!(input as Communicator);
    let stream = comm.derive_communicator();

    proc_macro::TokenStream::from(stream)
}

// ################################### CONSTRUCTING ##################################
struct ConstructInput {
    name_def: NameDefinition,
    _comma_1: syn::Token![,],
    aspects: SimulationAspects,
    _comma_2: syn::Token![,],
    path: SpecifiedPath,
}

impl syn::parse::Parse for ConstructInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            name_def: input.parse()?,
            _comma_1: input.parse()?,
            aspects: input.parse()?,
            _comma_2: input.parse()?,
            path: input.parse()?,
        })
    }
}

impl SimulationAspect {
    fn build_comm(&self, path: &syn::Path) -> (Vec<syn::Type>, Vec<proc_macro2::TokenStream>) {
        match self {
            SimulationAspect::Cycle => (vec![], vec![]),
            SimulationAspect::Reactions => (vec![], vec![]),
            SimulationAspect::Mechanics => (
                vec![
                    syn::parse2(quote!(I)).unwrap(),
                    syn::parse2(quote!(Cel)).unwrap(),
                    syn::parse2(quote!(Aux)).unwrap(),
                ],
                vec![quote!(
                    #[Comm(I, #path ::SendCell<Cel, Aux>)]
                    comm_cell: #path ::ChannelComm<I, #path ::SendCell<Cel, Aux>>
                )],
            ),
            SimulationAspect::Interaction => (
                vec![
                    syn::parse2(quote!(I)).unwrap(),
                    syn::parse2(quote!(Pos)).unwrap(),
                    syn::parse2(quote!(Vel)).unwrap(),
                    syn::parse2(quote!(For)).unwrap(),
                    syn::parse2(quote!(Inf)).unwrap(),
                ],
                vec![
                    quote!(
                        #[Comm(I, #path ::PosInformation<Pos, Vel, Inf>)]
                        comm_pos: #path ::ChannelComm<I, #path ::PosInformation<Pos, Vel, Inf>>
                    ),
                    quote!(
                        #[Comm(I, #path ::ForceInformation<For>)]
                        comm_force: #path ::ChannelComm<I, #path ::ForceInformation<For>>
                    ),
                ],
            ),
        }
    }
}

impl ConstructInput {
    fn build_communicator(self) -> proc_macro2::TokenStream {
        let struct_name = self.name_def.struct_name;
        let generics_fields: Vec<_> = self
            .aspects
            .items
            .into_iter()
            .map(|aspect| aspect.build_comm(&self.path.path))
            .collect();

        let mut generics = vec![];
        let mut fields = vec![];

        generics_fields.into_iter().for_each(|(g, f)| {
            g.into_iter().for_each(|gi| {
                if !generics.contains(&gi) {
                    generics.push(gi);
                }
            });
            fields.extend(f);
        });
        quote!(
            #[allow(non_camel_case_types)]
            #[derive(cellular_raza_core::derive::Communicator)]
            struct #struct_name <#(#generics),*> {
                #(#fields),*
            }
        )
    }
}

pub fn construct_communicator(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let constr = syn::parse_macro_input!(input as ConstructInput);
    let stream = constr.build_communicator();
    proc_macro::TokenStream::from(stream)
}
