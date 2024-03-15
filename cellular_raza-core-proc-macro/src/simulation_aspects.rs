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

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum SimulationAspect {
    // TODO add generic aspect which should always be present
    // None,
    Mechanics,
    Interaction,
    Cycle,
    Reactions,
}

pub struct ParsedSimulationAspect {
    pub aspect: SimulationAspect,
    pub ident: syn::Ident,
}

impl syn::parse::Parse for ParsedSimulationAspect {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        for aspect in SimulationAspect::get_aspects() {
            if ident == format!("{:?}", aspect) {
                return Ok(Self {
                    aspect,
                    ident,
                });
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
    pub items: syn::punctuated::Punctuated<ParsedSimulationAspect, syn::token::Comma>,
}

// TODO this macro does not give the correct error message
// at the correct place yet. Eg. when specifying aspects: [Mechanics, Mechanics]
impl syn::parse::Parse for SimulationAspects {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let aspects_token: AspectsToken = input.parse()?;
        let double_colon_2 = input.parse()?;
        let content;
        syn::bracketed!(content in input);
        let items = syn::punctuated::Punctuated::<ParsedSimulationAspect, syn::token::Comma>::parse_terminated(&content)?;
        use itertools::*;
        for duplicate in items.iter().duplicates_by(|pa| &pa.aspect).into_iter() {
            return Err(syn::Error::new(duplicate.ident.span(), format!("Found duplicate simulation aspect: {:?}", duplicate.aspect)));
        }
        Ok(Self {
            aspects_token,
            double_colon_2,
            items,
        })
    }
}

impl SimulationAspects {
    pub fn to_aspect_list(&self) -> Vec<SimulationAspect> {
        self.items
            .iter()
            .map(|aspect| aspect.clone())
            .collect::<Vec<_>>()
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

    pub fn get_aspects_strings() -> Vec<String> {
        SimulationAspect::get_aspects()
            .iter()
            .map(|aspect| aspect.into())
            .collect()
    }

    pub fn to_token_stream(&self) -> proc_macro2::TokenStream {
        match &self {
            SimulationAspect::Mechanics => quote::quote!(Mechanics),
            SimulationAspect::Interaction => quote::quote!(Interaction),
            SimulationAspect::Cycle => quote::quote!(Cycle),
            SimulationAspect::Reactions => quote::quote!(Reactions),
        }
    }

    pub fn to_token_stream_lowercase(&self) -> proc_macro2::TokenStream {
        match &self {
            SimulationAspect::Mechanics => quote::quote!(mechanics),
            SimulationAspect::Interaction => quote::quote!(interaction),
            SimulationAspect::Cycle => quote::quote!(cycle),
            SimulationAspect::Reactions => quote::quote!(reactions),
        }
    }
}

impl<'a> From<&'a SimulationAspect> for String {
    fn from(value: &'a SimulationAspect) -> Self {
        match value {
            SimulationAspect::Cycle => "Cycle",
            SimulationAspect::Interaction => "Interaction",
            SimulationAspect::Mechanics => "Mechanics",
            SimulationAspect::Reactions => "Reactions",
        }
        .to_owned()
    }
}

pub struct SimFlowPathToken;

impl syn::parse::Parse for SimFlowPathToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _path_token: syn::Ident = input.parse()?;
        if _path_token == "simulation_flow_path" {
            Ok(Self)
        } else {
            Err(syn::Error::new(_path_token.span(), "Expected \"path\""))
        }
    }
}

#[allow(unused)]
pub struct SimFlowPath {
    pub path_token: SimFlowPathToken,
    pub double_colon: syn::Token![:],
    pub path: syn::Path,
}

impl syn::parse::Parse for SimFlowPath {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            path_token: input.parse()?,
            double_colon: input.parse()?,
            path: input.parse()?,
        })
    }
}

pub struct CorePathtoken;

impl syn::parse::Parse for CorePathtoken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let core_path_ident: syn::Ident = input.parse()?;
        if core_path_ident != "core_path" {
            Err(syn::Error::new(
                core_path_ident.span(),
                "Expected core_path",
            ))
        } else {
            Ok(Self)
        }
    }
}

#[allow(unused)]
pub struct CorePath {
    pub core_path_token: CorePathtoken,
    pub double_colon: syn::Token![:],
    pub path: syn::Path,
}

impl syn::parse::Parse for CorePath {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            core_path_token: input.parse()?,
            double_colon: input.parse()?,
            path: input.parse()?,
        })
    }
}
