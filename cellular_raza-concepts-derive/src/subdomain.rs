pub enum DomainAspect {
    Base,
    SortCells,
    Mechanics,
    Force,
    Reactions,
}

impl DomainAspect {
    fn from_attribute(attr: &syn::Attribute) -> Option<Self> {
        let path = attr.meta.path().get_ident();
        if let Some(p) = path {
            let p_string = p.to_string();
            match p_string.as_str() {
                "Base" => Some(DomainAspect::Base),
                "SortCells" => Some(DomainAspect::SortCells),
                "Mechanics" => Some(DomainAspect::Mechanics),
                "Force" => Some(DomainAspect::Force),
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
    sort_cells: Option<FieldInfo>,
    mechanics: Option<FieldInfo>,
    force: Option<FieldInfo>,
    reactions: Option<FieldInfo>,
}

impl From<DomainParser> for DomainImplementer {
    fn from(value: DomainParser) -> Self {
        let mut base = None;
        let mut sort_cells = None;
        let mut mechanics = None;
        let mut force = None;
        let mut reactions = None;

        value.aspects.into_iter().for_each(|aspect_field| {
            aspect_field.aspects.into_iter().for_each(|aspect| {
                let field_info = FieldInfo {
                    field_type: aspect_field.field.ty.clone(),
                    field_name: aspect_field.field.ident.clone(),
                };
                match aspect {
                    DomainAspect::Base => base = Some(field_info),
                    DomainAspect::SortCells => sort_cells = Some(field_info),
                    DomainAspect::Mechanics => mechanics = Some(field_info),
                    DomainAspect::Force => force = Some(field_info),
                    DomainAspect::Reactions => reactions = Some(field_info),
                }
            })
        });

        DomainImplementer {
            name: value.name,
            generics: value.generics,
            base,
            sort_cells,
            mechanics,
            force,
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

    fn implement_sort_cells(&self) -> proc_macro2::TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.sort_cells {
            let field_type = &field_info.field_type;
            let field_name = &field_info.field_name;
            new_ident!(cell, "__cr_private_Cell");
            let tokens = quote::quote!(#cell);

            let where_clause =
                append_where_clause!(struct_where_clause, field_type, SubDomainSortCells, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, cell);
            let impl_generics = generics.split_for_impl().0;

            quote::quote!(
                impl #impl_generics SubDomainSortCells<#cell>
                for #struct_name #struct_ty_generics #where_clause {
                    fn get_voxel_index_of(
                        &self,
                        cell: &#cell
                    ) -> Result<Self::VoxelIndex, BoundaryError> {
                        <#field_type as SubDomainSortCells<#cell>>::get_voxel_index_of(
                            &self.#field_name,
                            cell,
                        )
                    }
                }
            )
        } else {
            proc_macro2::TokenStream::new()
        }
    }

    fn implement_mechanics(&self) -> proc_macro2::TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.mechanics {
            let field_type = &field_info.field_type;
            let field_name = &field_info.field_name;
            new_ident!(position, "__cr_private_Pos");
            new_ident!(velocity, "__cr_private_Vel");
            let tokens = quote::quote!(#position, #velocity);

            let where_clause =
                append_where_clause!(struct_where_clause, field_type, SubDomainMechanics, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, position);
            push_ident!(generics, velocity);
            let impl_generics = generics.split_for_impl().0;

            quote::quote!(
                impl #impl_generics SubDomainMechanics<#position, #velocity>
                for #struct_name #struct_ty_generics #where_clause {
                    fn apply_boundary(
                        &self,
                        pos: &mut #position,
                        vel: &mut #velocity,
                    ) -> Result<(), BoundaryError> {
                        <#field_type as SubDomainMechanics<#position, #velocity>>::apply_boundary(
                            &self.#field_name,
                            pos,
                            vel,
                        )
                    }
                }
            )
        } else {
            proc_macro2::TokenStream::new()
        }
    }

    fn implement_reactions(&self) -> proc_macro2::TokenStream {
        if let Some(_) = &self.reactions {
            quote::quote!(
                unimplemented!("The Reactions traits are currently reworked and thus not\
                    accessible for derivation via the derive(SubDomain) macro at this point in\
                    time.")
            )
        } else {
            proc_macro2::TokenStream::new()
        }
    }

    fn implement_force(&self) -> proc_macro2::TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.force {
            let field_type = &field_info.field_type;
            let field_name = &field_info.field_name;
            new_ident!(position, "__cr_private_Pos");
            new_ident!(velocity, "__cr_private_Vel");
            new_ident!(force, "__cr_private_For");
            let tokens = quote::quote!(#position, #velocity, #force);

            let where_clause =
                append_where_clause!(struct_where_clause, field_type, SubDomainForce, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, position);
            push_ident!(generics, velocity);
            push_ident!(generics, force);
            let impl_generics = generics.split_for_impl().0;

            quote::quote!(
                impl #impl_generics SubDomainForce<#position, #velocity, #force>
                for #struct_name #struct_ty_generics #where_clause {
                    fn calculate_custom_force(
                        &self,
                        pos: &#position,
                        vel: &#velocity,
                    ) -> Result<#force, CalcError> {
                        <#field_type as SubDomainForce<#position, #velocity, #force>>
                            ::calculate_custom_force(
                                &self.#field_name,
                                pos,
                                vel,
                        )
                    }
                }
            )
        } else {
            proc_macro2::TokenStream::new()
        }
    }
}

pub fn derive_subdomain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let domain_parser = syn::parse_macro_input!(input as DomainParser);
    let domain_implementer: DomainImplementer = domain_parser.into();

    let mut res = proc_macro2::TokenStream::new();
    res.extend(domain_implementer.implement_base());
    res.extend(domain_implementer.implement_sort_cells());
    res.extend(domain_implementer.implement_mechanics());
    res.extend(domain_implementer.implement_force());
    res.extend(domain_implementer.implement_reactions());
    super::cell_agent::wrap(res).into()
}
