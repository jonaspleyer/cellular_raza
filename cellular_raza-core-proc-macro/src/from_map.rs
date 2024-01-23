use quote::quote;
use syn::spanned::Spanned;

struct FromMapper {
    struct_name: syn::Ident,
    generics: syn::Generics,
    index_fields: Vec<MapField>,
    index: syn::Type,
    core_path: Option<syn::Path>,
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
            let stream = quote!(#s);
            let parsed: IndexParser = syn::parse2(stream)?;
            Ok(parsed)
        })
        .collect::<syn::Result<Vec<_>>>()?;
    if candidates.len() != 1 {
        Err(syn::Error::new(
            span,
            "Expected exactly one #[FromMapIndex(..)] attribute",
        ))
    } else {
        Ok(candidates[0].index.clone())
    }
}

pub struct CorePathParser {
    #[allow(unused)]
    pub from_map_core_path_token: syn::Ident,
    pub core_path: syn::Path,
}

impl syn::parse::Parse for CorePathParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let from_map_core_path_token = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            from_map_core_path_token,
            core_path: content.parse()?,
        })
    }
}

fn core_path_from_attributes(
    attrs: &Vec<syn::Attribute>,
    span: proc_macro2::Span,
) -> syn::Result<Option<syn::Path>> {
    let mut candidates = attrs
        .iter()
        .filter(|attr| {
            attr.meta
                .path()
                .get_ident()
                .is_some_and(|ident| ident == "FromMapCorePath")
        })
        .map(|attr| {
            let s = &attr.meta;
            let stream = quote!(#s);
            let parsed: CorePathParser = syn::parse2(stream)?;
            Ok(parsed)
        })
        .collect::<syn::Result<Vec<_>>>()?;
    if candidates.len() > 1 {
        return Err(syn::Error::new(
            span,
            "Expected at most one #[FromMapCorePath(..)] attribute",
        ));
    }
    if candidates.len() == 1 {
        return Ok(Some(candidates.remove(0).core_path));
    }
    Ok(None)
}

impl syn::parse::Parse for FromMapper {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let struct_def: syn::ItemStruct = input.parse()?;
        let index = index_from_attributes(&struct_def.attrs, struct_def.span())?;
        let core_path = core_path_from_attributes(&struct_def.attrs, struct_def.span())?;
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
            core_path,
        })
    }
}

impl FromMapper {
    fn implement(&self) -> proc_macro2::TokenStream {
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
        let backend_path = match &self.core_path {
            Some(p) => quote!(#p ::backend::chili::),
            None => quote!(),
        };

        let (impl_generics, ty_generics, where_clause) = &self.generics.split_for_impl();

        let res = quote!(
            #[doc(hidden)]
            const _: () = {
                #[automatically_derived]
                impl #impl_generics #backend_path FromMap<#index>
                for #struct_name #ty_generics #where_clause {
                    fn from_map(
                        map: &std::collections::HashMap<#index, Vec<#index>>
                    ) -> Result<
                        std::collections::HashMap<#index, Self>,
                        #backend_path IndexError
                    >
                    where
                        #index: Eq + core::hash::Hash + Clone + Ord,
                    {
                        #(
                            let mut #field_names_maps = <#field_types as #backend_path FromMap<#index>>::from_map(map)?;
                        )*
                        map.keys().into_iter().map(|key| {
                            Ok((
                                key.clone(),
                                Self {
                                    #(
                                        #field_names : #field_names_maps.remove(&key).ok_or(
                                            #backend_path IndexError(format!("could not find index in map"))
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
    let stream = from_mapper.implement();
    stream.into()
}
