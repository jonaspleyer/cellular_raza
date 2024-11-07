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
