use quote::quote;
use syn::spanned::Spanned;

struct FromMapper {
    struct_name: syn::Ident,
    generics: syn::Generics,
    index_fields: Vec<MapField>,
    index: syn::Type,
}

#[derive(Clone)]
struct MapField {
    field_name: Option<syn::Ident>,
    field_type: syn::Type,
}

#[allow(unused)]
struct IndexParser {
    from_map_index_token: syn::Ident,
    index: syn::Type,
}

impl syn::parse::Parse for IndexParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        let from_map_index_token = input.parse()?;
        syn::parenthesized!(content in input);
        Ok(Self {
            from_map_index_token,
            index: content.parse()?,
        })
    }
}

fn index_from_attributes(
    attrs: &Vec<syn::Attribute>,
    span: proc_macro2::Span,
) -> syn::Result<syn::Type> {
    let candidates = attrs
        .iter()
        .filter(|attr| {
            attr.meta
                .path()
                .get_ident()
                .is_some_and(|ident| ident == "FromMapIndex")
        })
        .map(|attr| {
            let s = &attr.meta;
            let stream: proc_macro::TokenStream = quote!(#s).into();
            let parsed: IndexParser = syn::parse(stream)?;
            Ok(parsed)
        })
        .collect::<syn::Result<Vec<_>>>()?;
    if candidates.len() != 1 {
        Err(syn::Error::new(span, ""))
    } else {
        Ok(candidates[0].index.clone())
    }
}

impl syn::parse::Parse for FromMapper {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let struct_def: syn::ItemStruct = input.parse()?;
        let index = index_from_attributes(&struct_def.attrs, struct_def.span())?;
        Ok(Self {
            struct_name: struct_def.ident,
            generics: struct_def.generics,
            index_fields: struct_def
                .fields
                .into_iter()
                .map(|field| MapField {
                    field_name: field.ident,
                    field_type: field.ty,
                })
                .collect::<Vec<_>>(),
            index,
        })
    }
}

impl FromMapper {
    fn implement(&self, core_path: Option<proc_macro2::TokenStream>) -> proc_macro2::TokenStream {
        let struct_name = &self.struct_name;
        let field_names = self
            .index_fields
            .clone()
            .into_iter()
            .map(|field| field.field_name);
        let field_names_maps = field_names
            .clone()
            .map(|name| syn::Ident::new(&format!("{}_map", quote!(#name)), struct_name.span()))
            .collect::<Vec<_>>();
        let field_types = self
            .index_fields
            .clone()
            .into_iter()
            .map(|field| field.field_type);
        let index = &self.index;
        let (from_map_path, error_path) = match core_path {
            Some(p) => (
                quote!(#p ::backend::chili::simulation_flow::),
                quote!(#p ::backend::cpu_os_threads::prelude::),
            ),
            None => (quote!(), quote!()),
        };

        let (impl_generics, ty_generics, where_clause) = &self.generics.split_for_impl();

        let res = quote!(
            #[doc(hidden)]
            const _: () = {
                #[automatically_derived]
                impl #impl_generics #from_map_path FromMap<#index>
                for #struct_name #ty_generics #where_clause {
                    fn from_map(
                        map: &std::collections::HashMap<#index, Vec<#index>>
                    ) -> Result<
                        std::collections::HashMap<#index, Self>,
                        #error_path IndexError
                    >
                    where
                        #index: Eq + core::hash::Hash + Clone + Ord,
                    {
                        #(
                            let mut #field_names_maps = <#field_types as #from_map_path FromMap<#index>>::from_map(map)?;
                        )*
                        map.keys().into_iter().map(|key| {
                            Ok((
                                key.clone(),
                                Self {
                                    #(
                                        #field_names : #field_names_maps.remove(&key).ok_or(
                                            #error_path IndexError(format!("could not find index in map"))
                                        )?,
                                    )*
                                }
                            ))
                        }).collect::<Result<_, _>>()
                    }
                }
            };
        );
        res
    }
}

pub fn derive_from_map(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let from_mapper = syn::parse_macro_input!(input as FromMapper);
    let stream = from_mapper.implement(None);
    stream.into()
}
