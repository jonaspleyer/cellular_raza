macro_rules! implement_parsing_of_derive_attributes(
    (
        $enum_name:ident,
        field_attributes: [$($name:ident),*],
        $field_struct:ident,
        struct_attributes: [$($name2:ident),*],
        $parser:ident
    ) => {
        pub enum $enum_name {
            $($name,)*
            $($name2,)*
        }

        pub struct $field_struct {
            elements: Vec<$enum_name>,
            field: syn::Field,
        }

        impl $enum_name {
            fn from_attribute(attr: &syn::Attribute) -> Option<Self> {
                let path = attr.meta.path().get_ident();
                path.and_then(|p| {
                    let p_string = p.to_string();
                    match p_string.as_str() {
                        $(
                            stringify!($name) => Some($enum_name::$name),
                        )*
                        $(
                            stringify!($name2) => Some($enum_name::$name2),
                        )*
                        _ => None,
                    }
                })
            }

            pub fn from_fields(span: proc_macro2::Span,
                fields: syn::Fields,
            ) -> syn::Result<Vec<$field_struct>> {
                match fields {
                    syn::Fields::Named(fields_named) => Ok(fields_named
                        .named
                        .into_iter()
                        .map(|field| $field_struct::from_field(field))
                        .collect::<Vec<_>>()),
                    syn::Fields::Unnamed(fields_unnamed) => Ok(fields_unnamed
                        .unnamed
                        .into_iter()
                        .map(|field| $field_struct::from_field(field))
                        .collect::<Vec<_>>()),
                    syn::Fields::Unit => Err(
                        syn::Error::new(span, "Cannot derive from unit struct")
                    ),
                }
            }
        }

        impl $field_struct {
            pub fn from_field(field: syn::Field) -> Self {
                let elements = field
                    .attrs
                    .iter()
                    .filter_map($enum_name::from_attribute)
                    .collect::<Vec<_>>();
                Self { elements, field }
            }
        }

        #[allow(unused)]
        pub struct $parser {
            attrs: Vec<$enum_name>,
            vis: syn::Visibility,
            struct_token: syn::Token![struct],
            name: syn::Ident,
            generics: syn::Generics,
            elements: Vec<$field_struct>,
        }

        impl syn::parse::Parse for $parser {
            fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
                let item_struct: syn::ItemStruct = input.parse()?;
                let attrs = item_struct.attrs
                    .into_iter()
                    .filter_map(|attr| $enum_name::from_attribute(&attr))
                    .collect();
                let vis = item_struct.vis;
                let struct_token = item_struct.struct_token;
                let name = item_struct.ident;
                let generics = item_struct.generics;
                let elements = $enum_name::from_fields(name.span(), item_struct.fields)?;

                let res = Self {
                    attrs,
                    vis,
                    struct_token,
                    name,
                    generics,
                    elements,
                };
                Ok(res)
            }
        }
    }
);

implement_parsing_of_derive_attributes!(
    SubDomainAspect,
    field_attributes: [Base, SortCells, Mechanics, Force, Reactions],
    SubDomainAspectField,
    struct_attributes: [],
    SubDomainParser
);

use super::cell_agent::FieldInfo;

pub struct SubDomainImplementer {
    name: syn::Ident,
    generics: syn::Generics,
    base: Option<FieldInfo>,
    sort_cells: Option<FieldInfo>,
    mechanics: Option<FieldInfo>,
    force: Option<FieldInfo>,
    reactions: Option<FieldInfo>,
}

impl From<SubDomainParser> for SubDomainImplementer {
    fn from(value: SubDomainParser) -> Self {
        let mut base = None;
        let mut sort_cells = None;
        let mut mechanics = None;
        let mut force = None;
        let mut reactions = None;

        value
            .elements
            .into_iter()
            .enumerate()
            .for_each(|(number, aspect_field)| {
                aspect_field.elements.into_iter().for_each(|aspect| {
                    let field_info = FieldInfo {
                        field_type: aspect_field.field.ty.clone(),
                        field_name: match aspect_field.field.ident.clone() {
                            Some(ident) => crate::cell_agent::FieldIdent::Ident(ident),
                            None => crate::cell_agent::FieldIdent::Int(
                                proc_macro2::Literal::usize_unsuffixed(number),
                            ),
                        },
                    };
                    match aspect {
                        SubDomainAspect::Base => base = Some(field_info),
                        SubDomainAspect::SortCells => sort_cells = Some(field_info),
                        SubDomainAspect::Mechanics => mechanics = Some(field_info),
                        SubDomainAspect::Force => force = Some(field_info),
                        SubDomainAspect::Reactions => reactions = Some(field_info),
                    }
                })
            });

        SubDomainImplementer {
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

impl SubDomainImplementer {
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
                append_where_clause!(struct_where_clause, field_type, SortCells, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, cell);
            let impl_generics = generics.split_for_impl().0;

            quote::quote!(
                impl #impl_generics SortCells<#cell>
                for #struct_name #struct_ty_generics #where_clause {
                    type VoxelIndex = <#field_type as SortCells<#cell>>::VoxelIndex;

                    fn get_voxel_index_of(
                        &self,
                        cell: &#cell
                    ) -> Result<Self::VoxelIndex, BoundaryError> {
                        <#field_type as SortCells<#cell>>::get_voxel_index_of(
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
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.reactions {
            let field_type = &field_info.field_type;
            let field_name = &field_info.field_name;
            new_ident!(position, "__cr_private_Pos");
            new_ident!(react_extra, "__cr__private_Re");
            new_ident!(float, "__cr__private_Float");
            let tokens = quote::quote!(#position, #react_extra, #float);

            let where_clause =
                append_where_clause!(struct_where_clause, field_type, SubDomainReactions, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, position);
            push_ident!(generics, react_extra);
            push_ident!(generics, float);
            let impl_generics = generics.split_for_impl().0;

            quote::quote!(
                impl #impl_generics SubDomainReactions<#position, #react_extra, #float>
                for #struct_name #struct_ty_generics #where_clause {
                    type NeighborValue = <
                        #field_type as SubDomainReactions<#position, #react_extra, #float>
                    >::NeighborValue;

                    type BorderInfo = <
                        #field_type as SubDomainReactions<#position, #react_extra, #float>
                    >::BorderInfo;

                    fn treat_increments<__cr_private_I, __cr_private_J>(
                        &mut self,
                        neighbors: __cr_private_I,
                        sources: __cr_private_J
                    ) -> Result<(), CalcError>
                    where
                        __cr_private_I: IntoIterator<Item = Self::NeighborValue>,
                        __cr_private_J: IntoIterator<Item = (#position, #react_extra)>,
                    {
                        <#field_type as SubDomainReactions<#position, #react_extra, #float>>::
                            treat_increments(&mut self.#field_name, neighbors, sources)
                    }

                    fn update_fluid_dynamics(
                        &mut self,
                        dt: #float,
                    ) -> Result<(), CalcError> {
                        <#field_type as SubDomainReactions<#position, #react_extra, #float>>::
                            update_fluid_dynamics(&mut self.#field_name, dt)
                    }

                    fn get_extracellular_at_pos(
                        &self,
                        pos: &#position
                    ) -> Result<#react_extra, CalcError> {
                        <#field_type as SubDomainReactions<#position, #react_extra, #float>>::
                            get_extracellular_at_pos(&self.#field_name, pos)
                    }

                    fn get_neighbor_values(
                        &self,
                        border_info: Self::BorderInfo
                    ) -> Self::NeighborValue {
                        <#field_type as SubDomainReactions<#position, #react_extra, #float>>::
                            get_neighbor_values(&self.#field_name, border_info)
                    }

                    fn get_border_info(&self) -> Self::BorderInfo {
                        <#field_type as SubDomainReactions<#position, #react_extra, #float>>::
                            get_border_info(&self.#field_name)
                    }
                }
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
    let subdomain_parser = syn::parse_macro_input!(input as SubDomainParser);
    let subdomain_implementer: SubDomainImplementer = subdomain_parser.into();

    let mut res = proc_macro2::TokenStream::new();
    res.extend(subdomain_implementer.implement_base());
    res.extend(subdomain_implementer.implement_sort_cells());
    res.extend(subdomain_implementer.implement_mechanics());
    res.extend(subdomain_implementer.implement_force());
    res.extend(subdomain_implementer.implement_reactions());
    super::cell_agent::wrap(res).into()
}
