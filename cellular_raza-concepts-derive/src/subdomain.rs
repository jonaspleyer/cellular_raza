pub enum DomainAspect {
    Base,
    Mechanics,
    Reactions,
}

impl DomainAspect {
    fn from_attribute(attr: &syn::Attribute) -> Option<Self> {
        let path = attr.meta.path().get_ident();
        if let Some(p) = path {
            let p_string = p.to_string();
            match p_string.as_str() {
                "Base" => Some(DomainAspect::Base),
                "Mechanics" => Some(DomainAspect::Mechanics),
                "Reactions" => Some(DomainAspect::Reactions),
                _ => None,
            }
        } else {
            None
        }
    }
}

pub struct DomainAspectField {
    aspects: Vec<DomainAspect>,
    field: syn::Field,
}

impl DomainAspectField {
    pub fn from_field(field: syn::Field) -> Self {
        let aspects = field
            .attrs
            .iter()
            .map(DomainAspect::from_attribute)
            .filter_map(|s| s)
            .collect::<Vec<_>>();
        Self { aspects, field }
    }
}

impl DomainAspect {
    pub fn from_fields(
        span: proc_macro2::Span,
        fields: syn::Fields,
    ) -> syn::Result<Vec<DomainAspectField>> {
        match fields {
            syn::Fields::Named(fields_named) => Ok(fields_named
                .named
                .into_iter()
                .map(|field| DomainAspectField::from_field(field))
                .collect::<Vec<_>>()),
            syn::Fields::Unnamed(fields_unnamed) => Ok(fields_unnamed
                .unnamed
                .into_iter()
                .map(|field| DomainAspectField::from_field(field))
                .collect::<Vec<_>>()),
            syn::Fields::Unit => Err(syn::Error::new(span, "Cannot derive from unit struct")),
        }
    }
}

#[allow(unused)]
pub struct DomainParser {
    attrs: Vec<syn::Attribute>,
    vis: syn::Visibility,
    struct_token: syn::Token![struct],
    name: syn::Ident,
    generics: syn::Generics,
    aspects: Vec<DomainAspectField>,
}

impl syn::parse::Parse for DomainParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item_struct: syn::ItemStruct = input.parse()?;
        let attrs = item_struct.attrs;
        let vis = item_struct.vis;
        let struct_token = item_struct.struct_token;
        let name = item_struct.ident;
        let generics = item_struct.generics;
        let aspects = DomainAspect::from_fields(name.span(), item_struct.fields)?;

        let res = Self {
            attrs,
            vis,
            struct_token,
            name,
            generics,
            aspects,
        };
        Ok(res)
    }
}

use super::cell_agent::FieldInfo;

pub struct DomainImplementer {
    name: syn::Ident,
    generics: syn::Generics,
    base: Option<FieldInfo>,
    mechanics: Option<FieldInfo>,
    reactions: Option<FieldInfo>,
}

impl From<DomainParser> for DomainImplementer {
    fn from(value: DomainParser) -> Self {
        let mut base = None;
        let mut mechanics = None;
        let mut reactions = None;

        value.aspects.into_iter().for_each(|aspect_field| {
            aspect_field.aspects.into_iter().for_each(|aspect| {
                let field_info = FieldInfo {
                    field_type: aspect_field.field.ty.clone(),
                    field_name: aspect_field.field.ident.clone(),
                };
                match aspect {
                    DomainAspect::Mechanics => mechanics = Some(field_info),
                    DomainAspect::Base => base = Some(field_info),
                    DomainAspect::Reactions => reactions = Some(field_info),
                }
            })
        });

        DomainImplementer {
            name: value.name,
            generics: value.generics,
            base,
            mechanics,
            reactions,
        }
    }
}

impl DomainImplementer {
    fn implement_base(&self) -> proc_macro2::TokenStream {
        let struct_name = &self.name;
        let (impl_generics, struct_ty_generics, where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.base {
            let field_type = &field_info.field_type;
            let field_name = &field_info.field_name;

            quote::quote!(
                impl #impl_generics SubDomain for #struct_name #struct_ty_generics #where_clause {
                    type VoxelIndex = <#field_type as SubDomain>::VoxelIndex;

                    fn get_neighbor_voxel_indices(
                        &self,
                        voxel_index: &Self::VoxelIndex
                    ) -> Vec<Self::VoxelIndex> {
                        <#field_type as SubDomain>::get_neighbor_voxel_indices(
                            &self.#field_name,
                            voxel_index,
                        )
                    }

                    fn get_all_indices(&self) -> Vec<Self::VoxelIndex> {
                        <#field_type as SubDomain>::get_all_indices(&self.#field_name)
                    }
                }
            )
        } else {
            proc_macro2::TokenStream::new()
        }
    }

            new_ident!(cell, "__cr_private_Cell");
            new_ident!(subdomain, "__cr_private_SubDomain");
            new_ident!(cell_iterator, "__cr_private_CellIterator");
            let tokens = quote::quote!(#cell, #subdomain, #cell_iterator);

            let where_clause =
                append_where_clause!(struct_where_clause, field_type, Domain, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, cell);
            push_ident!(generics, subdomain);
            push_ident!(generics, cell_iterator);
            let impl_generics = generics.split_for_impl().0;

            quote::quote!(
                impl #impl_generics Domain<#tokens>
                for #struct_name #struct_ty_generics #where_clause {
                    type SubDomainIndex = <#field_type as Domain<#tokens>>::SubDomainIndex;
                    type VoxelIndex = <#field_type as Domain<#tokens>>::VoxelIndex;

                    fn get_all_voxel_indices(&self) -> Vec<Self::VoxelIndex> {
                        <#field_type as Domain<#tokens>>::get_all_voxel_indices(&self.#field_name)
                    }

                    fn decompose(
                        self,
                        n_subdomains: core::num::NonZeroUsize,
                        cells: #cell_iterator
                    ) -> Result<DecomposedDomain<
                        Self::SubDomainIndex,
                        #subdomain,
                        #cell
                    >, DecomposeError>
                    where
                        #subdomain: SubDomain<C>
                    {
                        <#field_type as Domain<#tokens>>::decompose(
                            self.#field_name,
                            n_subdomains,
                            cells
                        )
                    }
                }
            )
        } else {
            proc_macro2::TokenStream::new()
        }
    }

    fn implement_mechanics(&self) -> proc_macro2::TokenStream {
        todo!()
    }

    fn implement_reactions(&self) -> proc_macro2::TokenStream {
        todo!()
    }
}

pub fn derive_domain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let domain_parser = syn::parse_macro_input!(input as DomainParser);
    let domain_implementer: DomainImplementer = domain_parser.into();

    let mut res = proc_macro2::TokenStream::new();
    res.extend(domain_implementer.implement_base());
    res.extend(domain_implementer.implement_mechanics());
    res.extend(domain_implementer.implement_reactions());
    super::cell_agent::wrap(res).into()
}
