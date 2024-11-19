use std::num::NonZeroUsize;

use crate::simulation_aspects::SimulationAspects;

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
pub enum Kwarg {
    domain {
        #[allow(unused)]
        domain_kw: syn::Ident,
        #[allow(unused)]
        double_colon: Option<syn::Token![:]>,
        domain: syn::Ident,
    },
    agents {
        #[allow(unused)]
        agents_kw: syn::Ident,
        #[allow(unused)]
        double_colon: Option<syn::Token![:]>,
        #[allow(unused)]
        agents: syn::Ident,
    },
    settings {
        #[allow(unused)]
        settings_kw: syn::Ident,
        #[allow(unused)]
        double_colon: Option<syn::Token![:]>,
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
        parallelizer: crate::run_sim::Parallelizer,
    },
    determinism {
        #[allow(unused)]
        determinism_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        determinism: bool,
    },
    aux_storage_name {
        #[allow(unused)]
        aux_storage_name_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        aux_storage_name: syn::Ident,
    },
    communicator_name {
        #[allow(unused)]
        communicator_name_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        communicator_name: syn::Ident,
    },
    mechanics_solver_order {
        #[allow(unused)]
        mechanics_solver_order_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        mechanics_solver_order: usize,
    },
    reactions_intra_solver_order {
        #[allow(unused)]
        reactions_intra_solver_order_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        reactions_intra_solver_order: usize,
    },
    reactions_contact_solver_order {
        #[allow(unused)]
        reactions_contact_solver_order_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        reactions_contact_solver_order: usize,
    },
    zero_force_default {
        #[allow(unused)]
        zero_force_default_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        zero_force_default: syn::ExprClosure,
    },
    zero_reactions_default {
        #[allow(unused)]
        zero_reactions_default_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        zero_reactions_default: syn::ExprClosure,
    },
}

macro_rules! parse_optional_kw(
    (
        $kwarg_variant:ident,
        $keyword:ident,
        $keyword_instance:ident,
        $value:ident,
        $input:ident
    ) => {{
        let keyword: syn::Ident = $keyword_instance;
        let (double_colon, value) = if $input.peek(syn::Token![,]) {
            (None, keyword.clone())
        } else {
            (Some($input.parse()?), $input.parse()?)
        };

        Ok(Kwarg:: $kwarg_variant {
            $keyword: keyword,
            double_colon,
            $value: value,
        })
    }};
);

impl syn::parse::Parse for Kwarg {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let keyword: syn::Ident = input.parse()?;
        let keyword_string = keyword.clone().to_string();
        match keyword_string.as_str() {
            "domain" => parse_optional_kw!(domain, domain_kw, keyword, domain, input),
            "agents" => parse_optional_kw!(agents, agents_kw, keyword, agents, input),
            "settings" => parse_optional_kw!(settings, settings_kw, keyword, settings, input),
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
            "determinism" => Ok(Kwarg::determinism {
                determinism_kw: keyword,
                double_colon: input.parse()?,
                determinism: input.parse::<syn::LitBool>()?.value,
            }),
            "aux_storage_name" => Ok(Kwarg::aux_storage_name {
                aux_storage_name_kw: keyword,
                double_colon: input.parse()?,
                aux_storage_name: input.parse()?,
            }),
            "communicator_name" => Ok(Kwarg::communicator_name {
                communicator_name_kw: keyword,
                double_colon: input.parse()?,
                communicator_name: input.parse()?,
            }),
            "mechanics_solver_order" => Ok(Kwarg::mechanics_solver_order {
                mechanics_solver_order_kw: keyword,
                double_colon: input.parse()?,
                // This is important. We need this since the AuxStorage
                // struct contains one entry less than the actual solver
                // order. Do not remove the -1 in the end.
                mechanics_solver_order: input
                    .parse::<syn::LitInt>()?
                    .base10_parse::<NonZeroUsize>()?
                    .get()
                    - 1,
            }),
            "reactions_intra_solver_order" => Ok(Kwarg::reactions_intra_solver_order {
                reactions_intra_solver_order_kw: keyword,
                double_colon: input.parse()?,
                reactions_intra_solver_order: input
                    .parse::<syn::LitInt>()?
                    .base10_parse::<NonZeroUsize>()?
                    .get(),
            }),
            "reactions_contact_solver_order" => Ok(Kwarg::reactions_contact_solver_order {
                reactions_contact_solver_order_kw: keyword,
                double_colon: input.parse()?,
                reactions_contact_solver_order: input
                    .parse::<syn::LitInt>()?
                    .base10_parse::<NonZeroUsize>()?
                    .get()
                    - 1,
            }),
            "zero_force_default" => Ok(Kwarg::zero_force_default {
                #[allow(unused)]
                zero_force_default_kw: keyword,
                #[allow(unused)]
                double_colon: input.parse()?,
                zero_force_default: input.parse()?,
            }),
            "zero_reactions_default" => Ok(Kwarg::zero_reactions_default {
                #[allow(unused)]
                zero_reactions_default_kw: keyword,
                #[allow(unused)]
                double_colon: input.parse()?,
                zero_reactions_default: input.parse()?,
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
                crate::kwargs::Kwarg::$field { $field, .. }
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
        $($kwarg_opt:ident: $type_opt:ty | $default:expr,)*
        @from
        $kwargs_from:ident$(,)?
    ) => {
        define_kwargs!(
            $kwargs_name,
            $kwargs_name_parsed,
            $($kwarg: $type,)*
            @optionals
            $($kwarg_opt: $type_opt | $default,)*
        );

        impl From<$kwargs_from> for $kwargs_name {
            fn from(parent: $kwargs_from) -> Self {
                Self {
                    $($kwarg: parent.$kwarg,)*
                    $($kwarg_opt: parent.$kwarg_opt,)*
                }
            }
        }
    };
    (
        $kwargs_name:ident,
        $kwargs_name_parsed:ident,
        $($kwarg:ident: $type:ty,)*
        @optionals
        $($kwarg_opt:ident: $type_opt:ty | $default:expr,)*
        @from
        $kwargs_from:ident,
        $($kwargs_from_more:ident),*
    ) => {
        define_kwargs!(
            $kwargs_name,
            $kwargs_name_parsed,
            $($kwarg: $type,)*
            @optionals
            $($kwarg_opt: $type_opt | $default,)*
            @from
            $($kwargs_from_more),*
        );

        impl From<$kwargs_from> for $kwargs_name {
            fn from(parent: $kwargs_from) -> Self {
                Self {
                    $($kwarg: parent.$kwarg,)*
                    $($kwarg_opt: parent.$kwarg_opt,)*
                }
            }
        }
    };
    (
        $kwargs_name:ident,
        $kwargs_name_parsed:ident,
        $($kwarg:ident: $type:ty,)*
        @optionals
        $($kwarg_opt:ident: $type_opt:ty | $default:expr,)*
    ) => {

        #[derive(Clone)]
        pub struct $kwargs_name_parsed {
            $(pub $kwarg: $type,)*
            $(pub $kwarg_opt: Option<$type_opt>,)*
        }

        #[derive(Clone)]
        pub struct $kwargs_name {
            $(pub $kwarg: $type,)*
            $(pub $kwarg_opt: $type_opt,)*
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
                let kwargs: syn::punctuated::Punctuated<crate::kwargs::Kwarg, syn::Token![,]> =
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

/// This function converts an `Option<syn::Path>` to a `syn::Path`
/// containing either the specified or default value
/// `cellular_raza::core` for the frequently used `core_path` argument.
pub fn convert_core_path(core_path: Option<syn::Path>) -> syn::Path {
    match core_path {
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
    }
}
