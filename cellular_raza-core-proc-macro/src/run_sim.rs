use itertools::Itertools;

use crate::simulation_aspects::{SimulationAspect, SimulationAspects};

#[derive(Clone, Eq, PartialEq, Debug)]
enum Parallelizer {
    OsThreads,
    Rayon,
}

impl syn::parse::Parse for Parallelizer {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        match ident.clone().to_string().as_str() {
            "OsThreads" => Ok(Self::OsThreads),
            "Rayon" => Ok(Self::Rayon),
            _ => Err(syn::Error::new(ident.span(), "Not a valid parallelizer")),
        }
    }
}

impl Parallelizer {
    fn parallelize_execution(
        &self,
        runner: syn::Ident,
        code: proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        match self {
            Self::OsThreads => (),
            Self::Rayon => (),
        }
        quote::quote!()
    }
}

/// # Important
#[derive(Clone, PartialEq, Debug)]
#[allow(non_camel_case_types)]
enum Kwarg {
    domain {
        #[allow(unused)]
        domain_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        domain: syn::Ident,
    },
    agents {
        #[allow(unused)]
        agents_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        #[allow(unused)]
        agents: syn::Ident,
    },
    settings {
        #[allow(unused)]
        settings_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        settings: syn::Ident,
    },
    aspects {
        aspects: SimulationAspects,
    },
    core_path {
        #[allow(unused)]
        core_path_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        core_path: syn::Path,
    },
    parallelizer {
        #[allow(unused)]
        parallelizer_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        parallelizer: Parallelizer,
    },
}

impl syn::parse::Parse for Kwarg {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let keyword: syn::Ident = input.parse()?;
        let keyword_string = keyword.clone().to_string();
        match keyword_string.as_str() {
            "domain" => Ok(Kwarg::domain {
                domain_kw: keyword,
                double_colon: input.parse()?,
                domain: input.parse()?,
            }),
            "agents" => Ok(Kwarg::agents {
                agents_kw: keyword,
                double_colon: input.parse()?,
                agents: input.parse()?,
            }),
            "settings" => Ok(Kwarg::settings {
                settings_kw: keyword,
                double_colon: input.parse()?,
                settings: input.parse()?,
            }),
            "aspects" => Ok(Kwarg::aspects {
                aspects: SimulationAspects::parse_give_initial_token(keyword, input)?,
            }),
            "core_path" => Ok(Kwarg::core_path {
                core_path_kw: keyword,
                double_colon: input.parse()?,
                core_path: input.parse()?,
            }),
            "parallelizer" => Ok(Kwarg::parallelizer {
                parallelizer_kw: keyword,
                double_colon: input.parse()?,
                parallelizer: input.parse()?,
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
            Kwarg::domain {
                domain_kw,
                double_colon,
                domain,
            } => {
                tokens.append(domain_kw.clone());
                // tokens.append(double_colon.clone());
                tokens.append(domain.clone());
            }
            #[allow(unused)]
            Kwarg::agents {
                agents_kw,
                double_colon,
                agents,
            } => {
                tokens.append(agents_kw.clone());
                // tokens.append(double_colon.clone());
                tokens.append(agents.clone());
            }
            #[allow(unused)]
            Kwarg::settings {
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
        let core_path = self.core_path.clone();
        let domain = self.domain.clone();
        let agents = self.agents.clone();
        let mut output = quote::quote!(
            #core_path::backend::chili::compatibility_tests::comp_domain_agents(
                &#domain,
                &#agents
            );
        );

        if self.aspects.contains_multiple(vec![
            &SimulationAspect::Mechanics,
            &SimulationAspect::Interaction,
        ]) {
            output.extend(quote::quote!(
                #core_path::backend::chili::compatibility_tests::comp_mechanics_interaction(
                    &#agents
                );
            ));
        }

        output
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
