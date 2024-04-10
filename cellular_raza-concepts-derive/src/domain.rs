implement_parsing_of_derive_attributes!(DomainProperty, [], DomainPropertyField, DomainParser);

use super::cell_agent::FieldInfo;

pub struct DomainImplementer {
    name: syn::Ident,
    generics: syn::Generics,
    // elements
}

impl From<DomainParser> for DomainImplementer {
    fn from(value: DomainParser) -> Self {
        value
            .elements
            .into_iter()
            .for_each(|domain_property_field| {
                domain_property_field
                    .elements
                    .into_iter()
                    .for_each(|domain_property| {
                        let _field_info = FieldInfo {
                            field_type: domain_property_field.field.ty.clone(),
                            field_name: domain_property_field.field.ident.clone(),
                        };
                        match domain_property {
                    // DomainProperty::
                }
                    })
            });

        DomainImplementer {
            name: value.name,
            generics: value.generics,
        }
    }
}

impl DomainImplementer {
    fn implement_decomposition(&self) -> proc_macro2::TokenStream {
        proc_macro2::TokenStream::new()
    }
}

pub fn derive_domain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let domain_parser = syn::parse_macro_input!(input as DomainParser);
    let domain_implementer = DomainImplementer::from(domain_parser);

    let mut res = proc_macro2::TokenStream::new();
    res.extend(domain_implementer.implement_decomposition());
    super::cell_agent::wrap(res).into()
}
