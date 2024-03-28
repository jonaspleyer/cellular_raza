pub enum DomainAspect {
    Base(DomainBaseParser),
    Mechanics(DomainMechanicsParser),
    Reactions(DomainReactionsParser),
}

struct DomainBaseParser {}

impl syn::parse::Parse for DomainBaseParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        todo!()
    }
}

struct DomainMechanicsParser {}

impl syn::parse::Parse for DomainMechanicsParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        todo!()
    }
}

struct DomainReactionsParser {}

impl syn::parse::Parse for DomainReactionsParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        todo!()
    }
}

impl DomainAspect {
    fn from_attribute(attr: &syn::Attribute) -> syn::Result<Option<Self>> {
        let path = attr.meta.path().get_ident();
        let cmp = |c: &str| path.is_some_and(|p| p.to_string() == c);

        let s = &attr.meta;
        let stream: proc_macro::TokenStream = quote::quote!(#s).into();

        if cmp("Domain") {
            let parsed: DomainBaseParser = syn::parse(stream)?;
            return Ok(Some(DomainAspect::Base(parsed)));
        }

        if cmp("Mechanics") {
            let parsed: DomainMechanicsParser = syn::parse(stream)?;
            return Ok(Some(DomainAspect::Mechanics(parsed)));
        }

        if cmp("Reactions") {
            let parsed: DomainReactionsParser = syn::parse(stream)?;
            return Ok(Some(DomainAspect::Reactions(parsed)));
        }

        Ok(None)
    }
}

pub struct DomainAspectField {
    aspects: Vec<DomainAspect>,
    field: syn::Field,
}

impl DomainAspectField {
    pub fn from_field(field: syn::Field) -> syn::Result<Self> {
        let mut errors = vec![];
        let aspects = field
            .attrs
            .iter()
            .map(DomainAspect::from_attribute)
            .filter_map(|r| r.map_err(|e| errors.push(e)).ok())
            .filter_map(|s| s)
            .collect::<Vec<_>>();
        for e in errors.into_iter() {
            return Err(e);
        }
        Ok(Self { aspects, field })
    }
}

impl DomainAspect {
    pub fn from_fields(
        span: proc_macro2::Span,
        fields: syn::Fields,
    ) -> syn::Result<Vec<DomainAspectField>> {
        todo!()
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
