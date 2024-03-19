use itertools::Itertools;

use crate::simulation_aspects::{SimulationAspect, SimulationAspects};

#[derive(Clone, PartialEq, Debug)]
enum Kwarg {
    DomainInput {
        #[allow(unused)]
        domain_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        domain: syn::Ident,
    },
    AgentsInput {
        #[allow(unused)]
        agents_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        #[allow(unused)]
        agents: syn::Ident,
    },
    SettingsInput {
        #[allow(unused)]
        settings_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        settings: syn::Ident,
    },
    AspectsInput {
        aspects: SimulationAspects,
    },
    CorePath {
        #[allow(unused)]
        settings_kw: syn::Ident,
        #[allow(unused)]
        double_colong: syn::Token![:],
        core_path: syn::Path,
    },
}

impl syn::parse::Parse for Kwarg {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let keyword: syn::Ident = input.parse()?;
        let keyword_string = keyword.clone().to_string();
        match keyword_string.as_str() {
            "domain" => Ok(Kwarg::DomainInput {
                domain_kw: keyword,
                double_colon: input.parse()?,
                domain: input.parse()?,
            }),
            "agents" => Ok(Kwarg::AgentsInput {
                agents_kw: keyword,
                double_colon: input.parse()?,
                agents: input.parse()?,
            }),
            "settings" => Ok(Kwarg::SettingsInput {
                settings_kw: keyword,
                double_colon: input.parse()?,
                settings: input.parse()?,
            }),
            "aspects" => Ok(Kwarg::AspectsInput {
                aspects: SimulationAspects::parse_give_initial_token(keyword, input)?,
            }),
            "core_path" => Ok(Kwarg::CorePath {
                settings_kw: keyword,
                double_colong: input.parse()?,
                core_path: input.parse()?,
            }),
            _ => Err(syn::Error::new(
                keyword.span(),
                format!("{keyword} is not a vaild keyword for this macro"),
            )),
        }
    }
}

impl quote::ToTokens for Kwarg {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        use quote::TokenStreamExt;
        match self {
            #[allow(unused)]
            Kwarg::DomainInput {
                domain_kw,
                double_colon,
                domain,
            } => {
                tokens.append(domain_kw.clone());
                // tokens.append(double_colon.clone());
                tokens.append(domain.clone());
            }
            #[allow(unused)]
            Kwarg::AgentsInput {
                agents_kw,
                double_colon,
                agents,
            } => {
                tokens.append(agents_kw.clone());
                // tokens.append(double_colon.clone());
                tokens.append(agents.clone());
            }
            #[allow(unused)]
            Kwarg::SettingsInput {
                settings_kw,
                double_colon,
                settings,
            } => {
                tokens.append(settings_kw.clone());
                // tokens.append(double_colon.clone());
                tokens.append(settings.clone());
            }
            _ => {}
        }
    }
}

#[derive(Clone)]
struct Kwargs {
    domain: syn::Ident,
    agents: syn::Ident,
    settings: syn::Ident,
    aspects: SimulationAspects,
    core_path: Option<syn::Path>,
}

impl syn::parse::Parse for Kwargs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kwargs: syn::punctuated::Punctuated<Kwarg, syn::Token![,]> =
            syn::punctuated::Punctuated::parse_terminated(input)?;
        let span = input.span();
        let core_path = kwargs
            .iter()
            .filter_map(|k| match k {
                #[allow(unused)]
                Kwarg::CorePath {
                    settings_kw,
                    double_colong,
                    core_path,
                } => Some(core_path.clone()),
                _ => None,
            })
            .next();

        // TODO do not unwrap! Rather construct nice error message
        let domain = kwargs
            .iter()
            .filter_map(|k| match k {
                #[allow(unused)]
                Kwarg::DomainInput {
                    domain_kw,
                    double_colon,
                    domain,
                } => Some(domain.clone()),
                _ => None,
            })
            .next()
            .ok_or(syn::Error::new(
                span,
                "macro is missing required information: domain",
            ))?;

        let agents = kwargs
            .iter()
            .filter_map(|k| match k {
                #[allow(unused)]
                Kwarg::AgentsInput {
                    agents_kw,
                    double_colon,
                    agents,
                } => Some(agents.clone()),
                _ => None,
            })
            .next()
            .ok_or(syn::Error::new(
                span,
                "macro is missing required information: agents",
            ))?;

        let settings = kwargs
            .iter()
            .filter_map(|k| match k {
                #[allow(unused)]
                Kwarg::SettingsInput {
                    settings_kw,
                    double_colon,
                    settings,
                } => Some(settings.clone()),
                _ => None,
            })
            .next()
            .ok_or(syn::Error::new(
                span,
                "macro is missing required information: settings",
            ))?;

        let aspects = kwargs
            .iter()
            .filter_map(|k| match k {
                #[allow(unused)]
                Kwarg::AspectsInput { aspects } => Some(aspects.clone()),
                _ => None,
            })
            .next()
            .ok_or(syn::Error::new(
                span,
                "macro is missing required information: aspects",
            ))?;

        Ok(Self {
            domain,
            agents,
            settings,
            aspects,
            core_path,
        })
    }
}

struct SimBuilder {
    domain: syn::Ident,
    agents: syn::Ident,
    settings: syn::Ident,
    aspects: SimulationAspects,
    core_path: syn::Path,
}

impl SimBuilder {
    fn initialize(kwargs: Kwargs) -> Self {
        let core_path = match kwargs.core_path {
            Some(p) => p,
            None => {
                let mut segments = syn::punctuated::Punctuated::new();
                segments.push(syn::PathSegment::from(syn::Ident::new(
                    "cellular_raza",
                    proc_macro2::Span::call_site(),
                )));
                segments.push(syn::PathSegment::from(syn::Ident::new(
                    "core",
                    proc_macro2::Span::call_site(),
                )));
                syn::Path {
                    leading_colon: None,
                    segments,
                }
            }
        };
        Self {
            domain: kwargs.domain,
            agents: kwargs.agents,
            settings: kwargs.settings,
            aspects: kwargs.aspects,
            core_path,
        }
    }

    /// Defines all types which will be used in the simulation
    fn prepare_types(&self) -> proc_macro2::TokenStream {
        // Build AuxStorage
        let aux_storage_builder = super::aux_storage::Builder {
            struct_name: syn::Ident::new("_CrAuxStorage", proc_macro2::Span::call_site()),
            core_path: self.core_path.clone(),
            aspects: self.aspects.clone(),
        };

        let mut output = aux_storage_builder.build_aux_storage();

        // Build Communicator
        let communicator_builder = super::communicator::CommunicatorBuilder {
            struct_name: syn::Ident::new("_CrCommunicator", proc_macro2::Span::call_site()),
            core_path: self.core_path.clone(),
            aspects: self.aspects.clone(),
        };
        output.extend(communicator_builder.build_communicator());

        output
    }

    /// Generate Zero-overhead functions that test compatibility between
    /// concepts before running the simulation, possibly reducing boilerplate
    /// in compiler errors
    fn test_compatibility(&self) -> proc_macro2::TokenStream {
        proc_macro2::TokenStream::new()
    }

    ///
    fn run_main(&self) -> proc_macro2::TokenStream {
        quote::quote!(Result::<usize, chili::SimulationError>::Ok(1_usize))
    }
}

pub fn run_simulation(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let kwargs = syn::parse_macro_input!(input as Kwargs);
    let mut output = proc_macro2::TokenStream::new();
    let sim_builder = SimBuilder::initialize(kwargs);
    output.extend(sim_builder.prepare_types());
    output.extend(sim_builder.test_compatibility());
    output.extend(sim_builder.run_main());
    quote::quote!({#output}).into()
}
