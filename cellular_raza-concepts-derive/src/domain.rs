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
        proc_macro2::TokenStream::new()
    }

    fn implement_sort_cells(&self) -> proc_macro2::TokenStream {
        proc_macro2::TokenStream::new()
    }

    fn implement_rng_seed(&self) -> proc_macro2::TokenStream {
        proc_macro2::TokenStream::new()
    }

    fn implement_create_subdomains(&self) -> proc_macro2::TokenStream {
        proc_macro2::TokenStream::new()
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
    super::cell_agent::wrap(res).into()
}
