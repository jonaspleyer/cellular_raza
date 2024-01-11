pub struct NameToken;

impl syn::parse::Parse for NameToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        match ident == "name" {
            true => return Ok(Self),
            _ => return Err(syn::Error::new(ident.span(), "Expected \"name\" token")),
        }
    }
}

pub struct AspectsToken;

impl syn::parse::Parse for AspectsToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        match ident == "aspects" {
            true => return Ok(Self),
            _ => return Err(syn::Error::new(ident.span(), "Expected \"aspects\" token")),
        }
    }
}

#[derive(Debug)]
pub enum SimulationAspect {
    Mechanics,
    Interaction,
    Cycle,
    Reactions,
}

impl syn::parse::Parse for SimulationAspect {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        for aspect in SimulationAspect::get_aspects() {
            if ident == format!("{:?}", aspect) {
                return Ok(aspect);
            }
        }
        Err(syn::Error::new(
            ident.span(),
            format!("Could not find simulation aspect {}", ident),
        ))
    }
}

pub struct NameDefinition {
    #[allow(unused)]
    name_token: NameToken,
    #[allow(unused)]
    double_colon_1: syn::Token![:],
    pub struct_name: syn::Ident,
}

impl syn::parse::Parse for NameDefinition {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            name_token: input.parse()?,
            double_colon_1: input.parse()?,
            struct_name: input.parse()?,
        })
    }
}

pub struct SimulationAspects {
    #[allow(unused)]
    pub aspects_token: AspectsToken,
    #[allow(unused)]
    double_colon_2: syn::Token![:],
    pub items: syn::punctuated::Punctuated<SimulationAspect, syn::token::Comma>,
}

impl syn::parse::Parse for SimulationAspects {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let aspects_token = input.parse()?;
        let double_colon_2 = input.parse()?;
        let content;
        syn::bracketed!(content in input);
        Ok(Self {
            aspects_token,
            double_colon_2,
            items: syn::punctuated::Punctuated::<SimulationAspect, syn::token::Comma>::parse_terminated(
                &content,
            )?,
        })
    }
}

impl SimulationAspect {
    pub fn get_aspects() -> Vec<SimulationAspect> {
        vec![
            SimulationAspect::Mechanics,
            SimulationAspect::Interaction,
            SimulationAspect::Cycle,
            SimulationAspect::Reactions,
        ]
    }
}
