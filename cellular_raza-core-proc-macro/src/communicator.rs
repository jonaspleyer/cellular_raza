use quote::quote;
use syn::spanned::Spanned;

use crate::run_sim::KwargsPrepareTypes;
use crate::run_sim::{KwargsSim, KwargsMain};

use super::simulation_aspects::*;

// ##################################### DERIVE ######################################
// ##################################### PARSING #####################################
struct Communicator {
    struct_name: syn::Ident,
    generics: syn::Generics,
    comms: Vec<CommField>,
    core_path: Option<syn::Path>,
}

struct CommParser {
    _comm_ident: syn::Ident,
    index: syn::Type,
    _comma: syn::Token![,],
    message: syn::Type,
}

#[allow(unused)]
struct CommCorePathParser {
    comm_core_path_token: syn::Ident,
    core_path: syn::Path,
}

impl syn::parse::Parse for CommCorePathParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let comm_core_path_token = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            comm_core_path_token,
            core_path: content.parse()?,
        })
    }
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
        let mut core_path_candidates = item_struct
            .attrs
            .iter()
            .filter(|attr| {
                attr.meta
                    .path()
                    .get_ident()
                    .is_some_and(|p| p == "CommunicatorCorePath")
            })
            .map(|attr| {
                let m = &attr.meta;
                let s = quote!(#m);
                let p: CommCorePathParser = syn::parse2(s)?;
                Ok(p.core_path)
            })
            .collect::<syn::Result<Vec<_>>>()?;
        if core_path_candidates.len() > 1 {
            return Err(syn::Error::new(
                core_path_candidates.last().unwrap().span(),
                "Expected only one or less #[CommCorePath(..)] fields",
            ));
        }
        let mut core_path = None;
        if core_path_candidates.len() == 1 {
            core_path = Some(core_path_candidates.remove(0));
        }
        Ok(Self {
            struct_name: item_struct.ident,
            generics: item_struct.generics,
            comms,
            core_path,
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
fn wrap_pre_flags(
    core_path: &proc_macro2::TokenStream,
    stream: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    quote!(
        #[allow(unused)]
        #[allow(non_camel_case_types)]
        const _: () = {
            use #core_path ::backend::chili::{SimulationError,Communicator};

            #stream
        };
    )
}

impl Communicator {
    fn derive_communicator(&self) -> proc_macro2::TokenStream {
        let struct_name = &self.struct_name;
        let core_path = &self.core_path;

        let (impl_generics, ty_generics, where_clause) = &self.generics.split_for_impl();

        let mut res = proc_macro2::TokenStream::new();
        res.extend(self.comms.iter().map(|comm| {
            let field_name = &comm.field_name;
            let field_type = &comm.field_type;

            let backend_path = quote!(#core_path ::backend::chili::);

            let index = &comm.index;
            let message = &comm.message;

            let addendum = quote!(#index: Clone + core::hash::Hash + Eq + Ord,);
            let where_clause = match where_clause {
                Some(w) => quote!(where #(#w.predicates), #addendum),
                None => quote!(where #addendum),
            };

            wrap_pre_flags(&quote!(#core_path), quote!(
                #[automatically_derived]
                impl #impl_generics #backend_path Communicator<#index, #message>
                for #struct_name #ty_generics #where_clause

                {
                    fn send(&mut self, receiver: &#index, message: #message) -> Result<(), #backend_path SimulationError> {
                        <#field_type as #backend_path Communicator<#index, #message>>::send(&mut self.#field_name, receiver, message)
                    }
                    fn receive(&mut self) -> Vec<#message> {
                        <#field_type as #backend_path Communicator<#index, #message>>::receive(&mut self.#field_name)
                    }
                }
            ))
        }));
        res
    }
}

pub fn derive_communicator(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let comm = syn::parse_macro_input!(input as Communicator);
    let stream = comm.derive_communicator();

    proc_macro::TokenStream::from(stream)
}

// ################################### CONSTRUCTING ##################################
define_kwargs!(
    KwargsCommunicator,
    KwargsCommunicatorParsed,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | crate::kwargs::convert_core_path(None),
    communicator_name: syn::Ident | crate::communicator::default_communicator_name(),
    @from
    KwargsSim,
    KwargsMain,
    KwargsPrepareTypes
);

fn index_type() -> syn::Type {
    syn::parse2(quote!(I)).unwrap()
}

impl SimulationAspect {
    fn build_comm(&self, core_path: &syn::Path) -> (Vec<syn::Type>, Vec<proc_macro2::TokenStream>) {
        let index_type = index_type();
        let backend_path = quote!(#core_path ::backend::chili::);
        match self {
            SimulationAspect::Cycle => (vec![], vec![]),
            SimulationAspect::Reactions => (vec![], vec![]),
            SimulationAspect::ReactionsExtra => (
                vec![
                    syn::parse2(quote!(Binfo)).unwrap(),
                    syn::parse2(quote!(NValue)).unwrap(),
                ],
                vec![
                    quote!(
                        #[Comm(
                            #index_type,
                            #backend_path ReactionsExtraBorderInfo<Binfo>
                        )]
                        comm_reactions_extra: #backend_path ChannelComm<
                            #index_type,
                            #backend_path ReactionsExtraBorderInfo<Binfo>
                        >
                    ),
                    quote!(
                        #[Comm(
                            #index_type,
                            #backend_path ReactionsExtraBorderReturn<NValue>
                        )]
                        comm_reactions_extra_return: #backend_path ChannelComm<
                            #index_type,
                            #backend_path ReactionsExtraBorderReturn<NValue>
                        >
                    ),
                ],
            ),
            SimulationAspect::ReactionsContact => (
                vec![
                    syn::parse2(quote!(Pos)).unwrap(),
                    syn::parse2(quote!(Ri)).unwrap(),
                    syn::parse2(quote!(RInf)).unwrap(),
                ],
                vec![
                    quote!(
                        #[Comm(I, #backend_path ReactionsContactInformation<Pos, Ri, RInf>)]
                        comm_reactions_contact: #backend_path ChannelComm<
                            #index_type,
                            #backend_path ReactionsContactInformation<Pos, Ri, RInf>
                        >
                    ),
                    quote!(
                        #[Comm(I, #backend_path ReactionsContactReturn<Ri>)]
                        comm_reactions_contact_return: #backend_path ChannelComm<
                            #index_type,
                            #backend_path ReactionsContactReturn<Ri>
                        >
                    ),
                ],
            ),
            SimulationAspect::Mechanics => (
                vec![
                    syn::parse2(quote!(Cel)).unwrap(),
                    syn::parse2(quote!(Aux)).unwrap(),
                ],
                vec![quote!(
                    #[Comm(I, #backend_path SendCell<Cel, Aux>)]
                    comm_cell: #backend_path ChannelComm<
                        #index_type,
                        #backend_path SendCell<Cel, Aux>
                    >
                )],
            ),
            SimulationAspect::Interaction => (
                vec![
                    syn::parse2(quote!(Pos)).unwrap(),
                    syn::parse2(quote!(Vel)).unwrap(),
                    syn::parse2(quote!(For)).unwrap(),
                    syn::parse2(quote!(Inf)).unwrap(),
                ],
                vec![
                    quote!(
                        #[Comm(I, #backend_path PosInformation<Pos, Vel, Inf>)]
                        comm_pos: #backend_path ChannelComm<
                            #index_type,
                            #backend_path PosInformation<Pos, Vel, Inf>
                        >
                    ),
                    quote!(
                        #[Comm(I, #backend_path ForceInformation<For>)]
                        comm_force: #backend_path ChannelComm<
                            #index_type,
                            #backend_path ForceInformation<For>
                        >
                    ),
                ],
            ),
            SimulationAspect::DomainForce => (vec![], vec![]),
        }
    }
}

fn generics_and_fields(
    simulation_aspects: &SimulationAspects,
    core_path: &syn::Path,
) -> (Vec<syn::Type>, Vec<proc_macro2::TokenStream>) {
    let index_type = index_type();
    let generics_fields: Vec<_> = simulation_aspects
        .items
        .iter()
        .map(|parsed_aspect| parsed_aspect.aspect.build_comm(&core_path))
        .collect();

    let mut generics = vec![index_type];
    let mut fields = vec![];

    generics_fields.into_iter().for_each(|(g, f)| {
        g.into_iter().for_each(|gi| {
            if !generics.contains(&gi) {
                generics.push(gi);
            }
        });
        fields.extend(f);
    });
    (generics, fields)
}

pub fn default_communicator_name() -> syn::Ident {
    syn::Ident::new("_CrCommunicator", proc_macro2::Span::call_site())
}

pub struct CommunicatorBuilder {
    pub struct_name: syn::Ident,
    pub core_path: syn::Path,
    pub aspects: SimulationAspects,
}

impl CommunicatorBuilder {
    pub fn build_communicator(self) -> proc_macro2::TokenStream {
        let struct_name = self.struct_name;
        let index_type = index_type();
        let core_path = &self.core_path;
        let (generics, fields) = generics_and_fields(&self.aspects, &core_path);
        // In the following code, we assume that I
        // is the index as implemented above in the build_comm function
        quote!(
            #[derive(#core_path ::backend::chili::Communicator)]
            #[CommunicatorCorePath(#core_path)]
            #[derive(#core_path ::backend::chili::FromMap)]
            #[FromMapIndex(#index_type)]
            #[FromMapCorePath(#core_path)]
            #[allow(non_camel_case_types)]
            struct #struct_name <#(#generics),*> {
                phantom_data: core::marker::PhantomData<#index_type>,
                #(#fields),*
            }
        )
    }
}

impl From<KwargsCommunicator> for CommunicatorBuilder {
    fn from(input: KwargsCommunicator) -> Self {
        Self {
            struct_name: input.communicator_name,
            core_path: input.core_path,
            aspects: input.aspects,
        }
    }
}

pub fn construct_communicator(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let constr = syn::parse_macro_input!(input as KwargsCommunicatorParsed);
    let constr: KwargsCommunicator = constr.into();
    let builder = CommunicatorBuilder::from(constr);
    let stream = builder.build_communicator();
    proc_macro::TokenStream::from(stream)
}

struct GenericsArguments {
    name_def: NameDefinition,
    _comma: syn::Token![,],
    aspects: SimulationAspects,
}

impl syn::parse::Parse for GenericsArguments {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            name_def: input.parse()?,
            _comma: input.parse()?,
            aspects: input.parse()?,
        })
    }
}

fn impl_generics(aspects: &SimulationAspects) -> Vec<proc_macro2::TokenStream> {
    let core_path: syn::Path = syn::parse2(quote!(cellular_raza::core)).expect(&format!(
        "{} {}",
        "Using dummy path in proc macro 'generics_and_fields' failed.",
        "This is an engine panic and should be reported!"
    ));
    let (generics, _) = generics_and_fields(&aspects, &core_path);
    let mut results = vec![];
    generics.into_iter().for_each(|_| results.push(quote!(_)));
    results
}

pub fn communicator_generics_placeholders(
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let arguments = syn::parse_macro_input!(input as GenericsArguments);
    let placeholders = impl_generics(&arguments.aspects)
        .into_iter()
        .map(|_| quote!(_));
    let name = arguments.name_def.struct_name;
    quote!(#name <#(#placeholders),*>).into()
}
