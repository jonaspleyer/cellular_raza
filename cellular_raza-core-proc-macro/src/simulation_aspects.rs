pub struct NameToken;

impl syn::parse::Parse for NameToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        match ident == "name" {
            true => Ok(Self),
            _ => Err(syn::Error::new(ident.span(), "Expected \"name\" token")),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AspectsToken(syn::Ident);

impl syn::parse::Parse for AspectsToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        match ident == "aspects" {
            true => Ok(Self(ident)),
            _ => Err(syn::Error::new(ident.span(), "Expected \"aspects\" token")),
        }
    }
}

/// Represents a property which can be numerically simulated.
///
/// ## Example Usage
/// ```ignore
/// my_macro!(Mechanics);
/// my_macro!(Interaction);
/// my_macro!(Cycle);
/// ```
/// See also [SimulationAspects].
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum SimulationAspect {
    // TODO add generic aspect which should always be present
    // None,
    Mechanics,
    Interaction,
    Cycle,
    DomainForce,
    Reactions,
    ReactionsExtra,
    ReactionsContact,
}

// TODO add option to specify type parameters for individual aspects
// ie. do
// aspects: [Mechanics(Vector3<f64>, Vector3<f64>, Vector3<f64>), ...]
// instead of
// aspects: [Mechanics, ... ]
// and use the specified types afterwards.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedSimulationAspect {
    pub aspect: SimulationAspect,
    pub ident: syn::Ident,
}

impl syn::parse::Parse for ParsedSimulationAspect {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        for aspect in SimulationAspect::get_aspects() {
            if ident == format!("{:?}", aspect) {
                return Ok(Self { aspect, ident });
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
    pub(crate) struct_name: syn::Ident,
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

/// A unique list of [SimulationAspect]s.
///
/// ## Example usage
/// ```ignore
/// my_macro!(
///     ...
///     aspects: [Mechanics, Interaction, Cycle, ...],
///     ...
/// );
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationAspects {
    #[allow(unused)]
    pub aspects_token: AspectsToken,
    #[allow(unused)]
    double_colon: syn::Token![:],
    pub items: syn::punctuated::Punctuated<ParsedSimulationAspect, syn::token::Comma>,
}

// TODO this macro does not give the correct error message
// at the correct place yet. Eg. when specifying aspects: [Mechanics, Mechanics]
impl syn::parse::Parse for SimulationAspects {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let aspects_token: AspectsToken = input.parse()?;
        Self::parse_give_initial_token(aspects_token.0, input)
    }
}

impl SimulationAspects {
    /// Map the parsed [SimulationAspect]s to a [Vec].
    pub fn to_aspect_list(&self) -> Vec<SimulationAspect> {
        self.items
            .iter()
            .map(|parsed_aspect| parsed_aspect.aspect.clone())
            .collect::<Vec<_>>()
    }

    /// Checks if the specified [SimulationAspect] is contained in this list.
    pub fn contains(&self, aspect: &SimulationAspect) -> bool {
        self.to_aspect_list().contains(aspect)
    }

    /// Checks if all of the specified [SimulationAspect]s are contained in this list.
    pub fn contains_multiple<'a>(
        &self,
        aspects: impl IntoIterator<Item = &'a SimulationAspect>,
    ) -> bool {
        for aspect in aspects.into_iter() {
            if !self.contains(aspect) {
                return false;
            }
        }
        true
    }

    /// Alternative implementation to the [syn::parse::Parse] trait.
    ///
    /// This function allows to provide a `aspects` token
    /// which was already previously parsed.
    pub fn parse_give_initial_token(
        aspects_token: syn::Ident,
        input: syn::parse::ParseStream,
    ) -> syn::Result<Self> {
        let double_colon = input.parse()?;
        let content;
        syn::bracketed!(content in input);
        let items = syn::punctuated::Punctuated::<ParsedSimulationAspect, syn::token::Comma>::parse_terminated(&content)?;
        use itertools::*;
        for duplicate in items.iter().duplicates_by(|pa| &pa.aspect).into_iter() {
            return Err(syn::Error::new(
                duplicate.ident.span(),
                format!("Found duplicate simulation aspect: {:?}", duplicate.aspect),
            ));
        }
        Ok(Self {
            aspects_token: AspectsToken(aspects_token),
            double_colon,
            items,
        })
    }
}

impl SimulationAspect {
    /// Obtain a [Vec] containing all possible [SimulationAspect]s.
    pub fn get_aspects() -> Vec<SimulationAspect> {
        vec![
            SimulationAspect::Mechanics,
            SimulationAspect::Interaction,
            SimulationAspect::Cycle,
            SimulationAspect::Reactions,
            SimulationAspect::ReactionsExtra,
            SimulationAspect::ReactionsContact,
            SimulationAspect::DomainForce,
        ]
    }

    /// Similar to [Self::get_aspects] but obtains [String]s instead.
    pub fn get_aspects_strings() -> Vec<String> {
        SimulationAspect::get_aspects()
            .iter()
            .map(|aspect| aspect.into())
            .collect()
    }

    /// Transform the current [SimulationAspect] into a [proc_macro2::TokenStream].
    pub fn to_token_stream(&self) -> proc_macro2::TokenStream {
        match &self {
            SimulationAspect::Mechanics => quote::quote!(Mechanics),
            SimulationAspect::Interaction => quote::quote!(Interaction),
            SimulationAspect::Cycle => quote::quote!(Cycle),
            SimulationAspect::Reactions => quote::quote!(Reactions),
            SimulationAspect::ReactionsExtra => quote::quote!(ReactionsExtra),
            SimulationAspect::ReactionsContact => quote::quote!(ReactionsContact),
            SimulationAspect::DomainForce => quote::quote!(DomainForce),
        }
    }

    /// Transform the current [SimulationAspect] into a [proc_macro2::TokenStream] with all
    /// lowercase letters.
    pub fn to_token_stream_lowercase(&self) -> proc_macro2::TokenStream {
        match &self {
            SimulationAspect::Mechanics => quote::quote!(mechanics),
            SimulationAspect::Interaction => quote::quote!(interaction),
            SimulationAspect::Cycle => quote::quote!(cycle),
            SimulationAspect::Reactions => quote::quote!(reactions),
            SimulationAspect::ReactionsExtra => quote::quote!(reactionsextra),
            SimulationAspect::ReactionsContact => quote::quote!(reactionscontact),
            SimulationAspect::DomainForce => quote::quote!(domainforce),
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
            SimulationAspect::ReactionsExtra => "ReactionsExtra",
            SimulationAspect::ReactionsContact => "ReactionsContact",
            SimulationAspect::DomainForce => "DomainForce",
        }
        .to_owned()
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
