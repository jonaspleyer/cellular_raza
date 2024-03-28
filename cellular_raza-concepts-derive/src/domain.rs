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
                "Domain" => Some(DomainAspect::Base),
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

pub struct DomainImplementer {}

impl From<DomainParser> for DomainImplementer {
    fn from(value: DomainParser) -> Self {
        DomainImplementer {}
    }
}

impl DomainImplementer {
    fn derive_domain(self) -> proc_macro2::TokenStream {
        todo!()
    }
}

pub fn derive_domain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let domain_parser = syn::parse_macro_input!(input as DomainParser);
    let domain_implementer: DomainImplementer = domain_parser.into();
    domain_implementer.derive_domain().into()
}
