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

/// Contains all possible keyword arguments for preparing,
/// testing and running a complete simulation.
///
/// # Important
/// The name of the variants of this enum are exactly identical
/// to the name of their fields which are of interest ultimately.
///
/// # Future
/// For the future, we plan on extending and changing this enum
/// to also include plain arguments. This means, that the following
/// to versions should both be accepted.
/// ```ignore
/// call_macro!(
///     aspects: aspects,
///     agents: agents,
/// );
/// call_macro!(
///     aspects,
///     agents,
/// );
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
                format!("{keyword} is not a valid keyword for this macro"),
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

macro_rules! parse_single_kwarg(
    (@base, $span:ident, $kwargs:ident, $field:ident) => {{
        let results = $kwargs
            .iter()
            .filter_map(|k| match k {
                #[allow(unused)]
                Kwarg::$field { $field, .. }
                => Some($field.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if results.len() > 1 {
            Err(syn::Error::new(
                $span,
                format!("multile entries for argument: {}", stringify!($field))
            ))
        } else {
            Ok(results)
        }
    }};
    (@with_error, $span:ident, $kwargs:ident, $field:ident) => {{
        let results = parse_single_kwarg!(@base, $span, $kwargs, $field)?;
        if results.len() != 1 {
            Err(syn::Error::new(
                $span,
                format!("macro is missing required argument: {}", stringify!($field))
            ))
        } else {
            Ok(results[0].clone())
        }
    }};
    (@optional, $span:ident, $kwargs:ident, $field:ident) => {{
        let results = parse_single_kwarg!(@base, $span, $kwargs, $field)?;
        if results.len() == 1 {
            Some(results[0].clone())
        } else {
            None
        }
    }};
);

macro_rules! define_kwargs(
    (
        $kwargs_name:ident,
        $kwargs_name_parsed:ident,
        $($kwarg:ident: $type:ty,)*
        @optionals
        $($kwarg_opt:ident: $type_opt:ty | $default:expr),*
    ) => {
        #[derive(Clone)]
        pub struct $kwargs_name_parsed {
            $($kwarg: $type,)*
            $($kwarg_opt: Option<$type_opt>,)*
        }

        #[derive(Clone)]
        pub struct $kwargs_name {
            $($kwarg: $type,)*
            $($kwarg_opt: $type_opt,)*
        }

        impl From<$kwargs_name_parsed> for $kwargs_name {
            fn from(value: $kwargs_name_parsed) -> Self {
                Self {
                    $($kwarg: value.$kwarg,)*
                    $($kwarg_opt: match value.$kwarg_opt {
                        Some(v) => v,
                        None => {$default},
                    },)*
                }
            }
        }

        impl syn::parse::Parse for $kwargs_name_parsed {
            fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
                let kwargs: syn::punctuated::Punctuated<Kwarg, syn::Token![,]> =
                    syn::punctuated::Punctuated::parse_terminated(input)?;
                let span = input.span();

                $(
                    let $kwarg = parse_single_kwarg!(@with_error, span, kwargs, $kwarg)?;
                )*
                $(
                    let $kwarg_opt = parse_single_kwarg!(@optional, span, kwargs, $kwarg_opt);
                )*

                Ok(Self {
                    $($kwarg,)*
                    $($kwarg_opt,)*
                })
            }
        }
    }
);

define_kwargs!(
    KwargsSim,
    KwargsSimParsed,
    domain: syn::Ident,
    agents: syn::Ident,
    settings: syn::Ident,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | convert_core_path(None),
    parallelizer: Parallelizer | Parallelizer::OsThreads
);

define_kwargs!(
    KwargsCompatibility,
    KwargsCompatibilityParsed,
    domain: syn::Ident,
    agents: syn::Ident,
    settings: syn::Ident,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | convert_core_path(None)
);

define_kwargs!(
    KwargsPrepareTypes,
    KwargsPrepareTypesParsed,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | convert_core_path(None)
);

            }
        }
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

define_kwargs!(
    KwargsMain,
    KwargsMainParsed,
    aspects: SimulationAspects,
    domain: syn::Ident,
    agents: syn::Ident,
    settings: syn::Ident,
    @optionals
    core_path: syn::Path | convert_core_path(None)
);

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
