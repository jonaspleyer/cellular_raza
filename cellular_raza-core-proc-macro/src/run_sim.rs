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

// TODO complete this function
pub fn run_main_update(kwargs: KwargsMain) -> proc_macro2::TokenStream {
    use quote::quote;
    use SimulationAspect::*;

    let mut step_1 = proc_macro2::TokenStream::new();
    let mut step_2 = proc_macro2::TokenStream::new();
    let mut step_3 = proc_macro2::TokenStream::new();
    let mut step_4 = proc_macro2::TokenStream::new();

    if kwargs
        .aspects
        .contains_multiple(vec![&Mechanics, &Interaction])
    {
        step_1.extend(quote!(sbox.update_mechanics_step_1()?;));
        step_2.extend(quote!(sbox.update_mechanics_step_2()?;));
        step_3.extend(quote!(sbox.update_mechanics_step_3(&next_time_point.increment)?;));
    }

    if kwargs.aspects.contains(&Cycle) {
        step_3.extend(quote!(sbox.update_cell_cycle()?;));
    }

    if kwargs.aspects.contains(&Mechanics) {
        step_3.extend(quote!(sbox.sort_cells_in_voxels_step_1()?;));
        step_4.extend(quote!(sbox.sort_cells_in_voxels_step_2()?;));
    }

    let core_path = &kwargs.core_path;
    let settings = &kwargs.settings;

    quote!(
        #[allow(unused)]
        let _storage_manager: #core_path::storage::StorageManager<_, _> =
           #core_path::storage::StorageManager::construct(&#settings.storage, key as u64)?;

        // Set up the time stepper
        let mut _time_stepper = #settings.time.clone();
        use #core_path::time::TimeStepper;

        // Initialize the progress bar
        #[allow(unused)]
        let mut pb = match key {
            0 => Some(_time_stepper.initialize_bar()?),
            _ => None,
        };

        while let Some(next_time_point) = _time_stepper.advance()? {
            #step_1
            sbox.sync();
            #step_2
            sbox.sync();
            #step_3
            sbox.sync();
            #step_4

            match &mut pb {
                Some(bar) => _time_stepper.update_bar(bar)?,
                None => (),
            };
            sbox.save_voxels(&_storage_manager, &next_time_point)?;
        }
        Ok(())
    )
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
pub fn run_main(kwargs: KwargsMain) -> proc_macro2::TokenStream {
    let asp = &kwargs
        .aspects
        .items
        .iter()
        .map(|asp| asp.ident.clone())
        .collect::<Vec<_>>();
    let domain = &kwargs.domain;
    let agents = &kwargs.agents;
    let settings = &kwargs.settings;
    let core_path = &kwargs.core_path;

    let update_func = run_main_update(kwargs.clone());

    quote::quote!({
        type _Syncer = #core_path::backend::chili::BarrierSync;
        let _aux_storage = _CrAuxStorage::default();
        let mut runner = #core_path::backend::chili::construct_simulation_runner::<
            _,
            _,
            _,
            _,
            #core_path::backend::chili::communicator_generics_placeholders!(
                name: _CrCommunicator,
                aspects: [#(#asp),*]
            ),
            _Syncer,
            _
        >(
            #domain,
            #agents,
            #settings.n_threads,
            &_aux_storage,
        )?;

        runner.subdomain_boxes.iter_mut().map(|(&key, sbox)| {
            #update_func
        }).collect::<Result<Vec<_>, #core_path::backend::chili::SimulationError>>()?;
    })
}

pub fn prepare_types(kwargs_parsed: KwargsPrepareTypes) -> proc_macro2::TokenStream {
    // Build AuxStorage
    let kwargs: KwargsPrepareTypes = kwargs_parsed.into();
    let aux_storage_builder = super::aux_storage::Builder {
        struct_name: syn::Ident::new("_CrAuxStorage", proc_macro2::Span::call_site()),
        core_path: kwargs.core_path.clone(),
        aspects: kwargs.aspects.clone(),
    };

    let mut output = aux_storage_builder.build_aux_storage();

    // Build Communicator
    let communicator_builder = super::communicator::CommunicatorBuilder {
        struct_name: syn::Ident::new("_CrCommunicator", proc_macro2::Span::call_site()),
        core_path: kwargs.core_path.clone(),
        aspects: kwargs.aspects.clone(),
    };
    output.extend(communicator_builder.build_communicator());

    output
}

/// Generate Zero-overhead functions that test compatibility between
/// concepts before running the simulation, possibly reducing boilerplate
/// in compiler errors
pub fn test_compatibility(kwargs: KwargsCompatibility) -> proc_macro2::TokenStream {
    let core_path = &kwargs.core_path;
    let domain = &kwargs.domain;
    let agents = &kwargs.agents;
    let mut output = quote::quote!(
        #core_path::backend::chili::compatibility_tests::comp_domain_agents(
            &#domain,
            &#agents
        );
    );

    if kwargs.aspects.contains_multiple(vec![
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

pub fn run_simulation(kwargs: KwargsSim) -> proc_macro2::TokenStream {
    let types = prepare_types(KwargsPrepareTypes {
        aspects: kwargs.aspects.clone(),
        core_path: kwargs.core_path.clone(),
    });
    // let test_compat = kwargs.test_compatibility();
    // let run_main = kwargs.run_main();
    // let core_path = &kwargs.core_path;
    // quote::quote!({
    //     #types
    //     #test_compat
    //     #run_main
    //     Result::<(), #core_path::backend::chili::SimulationError>::Ok(())
    // })
    // .into()
    quote::quote!()
}
