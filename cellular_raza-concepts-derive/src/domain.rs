implement_parsing_of_derive_attributes!(
    DomainProperty,
    [Base, DomainRngSeed, DomainCreateSubDomains, SortCells],
    DomainPropertyField,
    DomainParser
);

use super::cell_agent::FieldInfo;

pub struct DomainImplementer {
    name: syn::Ident,
    generics: syn::Generics,
    base: Option<FieldInfo>,
    sort_cells: Option<FieldInfo>,
    rng_seed: Option<FieldInfo>,
    create_subdomains: Option<FieldInfo>,
}

impl From<DomainParser> for DomainImplementer {
    fn from(value: DomainParser) -> Self {
        let mut base = None;
        let mut sort_cells = None;
        let mut rng_seed = None;
        let mut create_subdomains = None;

        value
            .elements
            .into_iter()
            .for_each(|domain_property_field| {
                domain_property_field
                    .elements
                    .into_iter()
                    .for_each(|domain_property| {
                        let field_info = FieldInfo {
                            field_type: domain_property_field.field.ty.clone(),
                            field_name: domain_property_field.field.ident.clone(),
                        };
                        use DomainProperty::*;
                        match domain_property {
                            Base => base = Some(field_info),
                            DomainRngSeed => rng_seed = Some(field_info),
                            DomainCreateSubDomains => create_subdomains = Some(field_info),
                            SortCells => sort_cells = Some(field_info),
                        }
                    })
            });

        DomainImplementer {
            name: value.name,
            generics: value.generics,
            base,
            sort_cells,
            rng_seed,
            create_subdomains,
        }
    }
}

impl DomainImplementer {
    fn implement_base(&self) -> proc_macro2::TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.base {
            let field_type = &field_info.field_type;
            let field_name = &field_info.field_name;
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
                impl #impl_generics Domain<#tokens> for #struct_name #struct_ty_generics
                    #where_clause
                {
                    type SubDomainIndex = <#field_type as Domain<#tokens>>::SubDomainIndex;
                    type VoxelIndex = <#field_type as Domain<#tokens>>::VoxelIndex;

                    fn decompose(
                        self,
                        n_subdomains: core::num::NonZeroUsize,
                        cells: #cell_iterator
                    ) -> Result<
                        DecomposedDomain<Self::SubDomainIndex, #subdomain, #cell>,
                        DecomposeError
                    > {
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
                impl #impl_generics SortCells<#tokens> for #struct_name #struct_ty_generics
                    #where_clause
                {
                    type Index = <#field_type as SortCells<#tokens>>::Index;

                    fn get_index_of(&self, cell: &#cell) -> Result<Self::Index, BoundaryError> {
                        <#field_type as SortCells<#tokens>>::get_index_of(&self.#field_name, cell)
                    }
                }
            )
        } else {
            proc_macro2::TokenStream::new()
        }
    }

    fn implement_rng_seed(&self) -> proc_macro2::TokenStream {
        let struct_name = &self.name;
        let (impl_generics, struct_ty_generics, where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.rng_seed {
            let field_type = &field_info.field_type;
            let field_name = &field_info.field_name;

            quote::quote!(
                impl #impl_generics DomainRngSeed for #struct_name #struct_ty_generics
                    #where_clause
                {
                    fn get_rng_seed(&self) -> u64 {
                        <#field_type as DomainRngSeed>::get_rng_seed(&self.#field_name)
                    }
                }
            )
        } else {
            proc_macro2::TokenStream::new()
        }
    }

    fn implement_create_subdomains(&self) -> proc_macro2::TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.create_subdomains {
            let field_type = &field_info.field_type;
            let field_name = &field_info.field_name;
            new_ident!(subdomain, "__cr_private_SubDomain");
            let tokens = quote::quote!(#subdomain);

            let where_clause =
                append_where_clause!(struct_where_clause, field_type, Domain, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, subdomain);
            let impl_generics = generics.split_for_impl().0;

            quote::quote!(
                impl #impl_generics CreateSubDomains<#tokens> for #struct_name #struct_ty_generics
                    #where_clause
                {
                    type SubDomainIndex = <#field_type as CreateSubDomains<#tokens>>::SubDomainIndex;
                    type VoxelIndex = <#field_type as CreateSubDomains<#tokens>>::VoxelIndex;

                    fn create_subdomains(
                        &self,
                        n_subdomains: core::num::NonZeroUsize,
                    ) -> Result<
                        impl IntoIterator<Item = (Self::SubDOmainIndex, S, Vec<Self::VoxelIndex>)>,
                        DecomposeError,
                    > {
                        <#field_type as CreateSubDomains<#tokens>>::create_subdomains(
                            &self.#field_name,
                            n_subdomains,
                        )
                    }
                }
            )
        } else {
            proc_macro2::TokenStream::new()
        }
    }

    fn implement_derived_total(&self) -> proc_macro2::TokenStream {
        if self.base.is_none() {
            todo!();
            let struct_name = &self.name;
            let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

            new_ident!(cell, "__cr_private_Cell");
            new_ident!(subdomain, "__cr_private_SubDomain");
            new_ident!(cell_iterator, "__cr_private_CellIterator");
            let tokens = quote::quote!(#cell, #subdomain, #cell_iterator);

            let where_clause =
                append_where_clause!(struct_where_clause, struct_name, DomainRngSeed, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, cell);
            push_ident!(generics, subdomain);
            push_ident!(generics, cell_iterator);
            let impl_generics = generics.split_for_impl().0;

            quote::quote!(
            impl #impl_generics Domain<#tokens> for #struct_name
            // TODO #where_clause
            where
                Self: DomainRngSeed
                    + DomainCreateSubDomains<#subdomain>
                    + SortCells<#subdomain, Index = < as DomainCreateSubDomains<#subdomain>>::SubDomainIndex>,
                #subdomain: SubDomain<VoxelIndex = Self::VoxelIndex>,
                <Self as DomainCreateSubDomains<#subdomain>>::SubDomainIndex: Clone + core::hash::Hash + Eq,
                <Self as DomainCreateSubDomains<#subdomain>>::VoxelIndex: Clone + core::hash::Hash + Eq,
                #cell_iterator: IntoIterator<Item = #cell>,
            {
                type SubDomainIndex = <Self as DomainCreateSubDomains<#subdomain>>::SubDomainIndex;
                type VoxelIndex = <Self as DomainCreateSubDomains<#subdomain>>::VoxelIndex;

                fn decompose(
                    self,
                    n_subdomains: core::num::NonZeroUsize,
                    cells: Ci,
                ) -> Result<DecomposedDomain<Self::SubDomainIndex, #subdomain, #cell>, DecomposeError> {
                    // Get all subdomains
                    let subdomains = self.create_subdomains(n_subdomains)?;

                    // Build a map from a voxel_index to the subdomain_index in which it is
                    let mut voxel_index_to_subdomain_index =
                        std::collections::HashMap::<Self::VoxelIndex, Self::SubDomainIndex>::new();

                    // Build neighbor map
                    let mut neighbor_map: HashMap<Self::SubDomainIndex, Vec<Self::SubDomainIndex>> =
                        HashMap::new();

                    // Build index_subdomain_cells
                    let mut index_subdomain_cells = Vec::new();

                    // Sort cells into the subdomains
                    let mut index_to_cells = HashMap::<_, Vec<C>>::new();
                    for cell in cells {
                        let index = self.get_index_of(&cell)?;
                        let existing_cells = index_to_cells.get_mut(&index);
                        match existing_cells {
                            Some(cell_vec) => cell_vec.push(cell),
                            None => {
                                index_to_cells.insert(index, vec![cell]);
                            }
                        }
                    }

                    // Fill both with values
                    for (subdomain_index, subdomain, voxel_indices) in subdomains.into_iter() {
                        for voxel_index in voxel_indices.into_iter() {
                            voxel_index_to_subdomain_index.insert(
                                voxel_index.clone(),
                                subdomain_index.clone()
                            );
                            for neighbor_voxel_index in subdomain.get_neighbor_voxel_indices(
                                &voxel_index
                            ) {
                                let neighbor_subdomain = voxel_index_to_subdomain_index
                                    .get(&neighbor_voxel_index)
                                    .ok_or(DecomposeError::IndexError(crate::IndexError(format!(
                                        "TODO"
                                    ))))?;
                                let neighbors = neighbor_map.get_mut(&subdomain_index).ok_or(
                                    DecomposeError::IndexError(crate::IndexError(format!("TODO"))),
                                )?;
                                if neighbors.contains(neighbor_subdomain) {
                                    neighbors.push(neighbor_subdomain.clone());
                                }
                            }
                        }

                        // Obtain cells which are in this subdomain
                        let cells = index_to_cells
                            .remove(&subdomain_index)
                            .unwrap_or(Vec::new());
                        index_subdomain_cells.push((subdomain_index, subdomain, cells));
                    }

                    Ok(DecomposedDomain {
                        n_subdomains,
                        index_subdomain_cells,
                        neighbor_map,
                        rng_seed: self.get_rng_seed(),
                    })
                }
            }
            )
        } else {
            quote::quote!()
        }
    }
}

pub fn derive_domain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let domain_parser = syn::parse_macro_input!(input as DomainParser);
    let domain_implementer = DomainImplementer::from(domain_parser);

    let mut res = proc_macro2::TokenStream::new();
    res.extend(domain_implementer.implement_base());
    res.extend(domain_implementer.implement_sort_cells());
    res.extend(domain_implementer.implement_rng_seed());
    res.extend(domain_implementer.implement_create_subdomains());
    res.extend(domain_implementer.implement_derived_total());
    super::cell_agent::wrap(res).into()
}
