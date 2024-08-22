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
        code: &proc_macro2::TokenStream,
        core_path: &syn::Path,
        settings: &syn::Ident,
    ) -> proc_macro2::TokenStream {
        let core_path = &core_path;
        match &self {
            Self::OsThreads => quote::quote!({
                let mut handles = vec![];
                for (key, mut sbox) in runner
                    .subdomain_boxes
                    .into_iter()
                {
                    let #settings = #settings.clone();
                    let handle = std::thread::Builder::new()
                        .name(format!("cellular_raza-worker_thread-{:03.0}", key))
                        .spawn(move ||
                            -> Result<_, #core_path::backend::chili::SimulationError> {#code})?;
                    handles.push(handle);
                }
                let mut storage_accesses = vec![];
                for handle in handles {
                    // TODO decide if we need to catch this error in the future
                    let result = handle
                        .join()
                        .expect("Could not join threads after simulation has finished")?;
                    storage_accesses.push(result);
                }
                let storage_access = storage_accesses.pop()
                    .ok_or(#core_path::storage::StorageError::InitError(
                        format!("Simulation Threads did not yield any storage managers")
                    ))?;
                Result::<_, #core_path::backend::chili::SimulationError>::Ok(storage_access)
            }),
            Self::Rayon => quote::quote!({
                // extern crate rayon;
                use rayon::prelude::*;
                let mut storage_accesses = runner.subdomain_boxes
                    .into_par_iter()
                    .map(|(key, mut sbox)| -> Result<
                        _,
                        #core_path::backend::chili::SimulationError
                    > {
                        #code
                    })
                    .collect::<Result<Vec<_>, #core_path::backend::chili::SimulationError>>()?;
                let storage_access = storage_accesses
                    .pop()
                    .ok_or(#core_path::storage::StorageError::InitError(
                        format!("Simulation threads did not yield any storage managers")
                    ))?;
                Result::<_, #core_path::backend::chili::SimulationError>::Ok(storage_access)
            }),
        }
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
    determinism {
        #[allow(unused)]
        determinism_kw: syn::Ident,
        #[allow(unused)]
        double_colon: syn::Token![:],
        determinism: bool,
    },
    // TODO add storage ie. to enable or disable storing of results entirely
    // figure out how this interacts with the TimeStepper trait in cellular_raza_core::time
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
            pub $($kwarg: $type,)*
            pub $($kwarg_opt: Option<$type_opt>,)*
        }

        #[derive(Clone)]
        pub struct $kwargs_name {
            pub $($kwarg: $type,)*
            pub $($kwarg_opt: $type_opt,)*
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
    parallelizer: Parallelizer | Parallelizer::OsThreads,
    determinism: bool | true,
);

define_kwargs!(
    KwargsCompatibility,
    KwargsCompatibilityParsed,
    domain: syn::Ident,
    agents: syn::Ident,
    settings: syn::Ident,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | convert_core_path(None),
    @from
    KwargsSim
);

define_kwargs!(
    KwargsPrepareTypes,
    KwargsPrepareTypesParsed,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | convert_core_path(None),
    @from
    KwargsSim,
    KwargsCompatibility
);

define_kwargs!(
    KwargsMain,
    KwargsMainParsed,
    domain: syn::Ident,
    agents: syn::Ident,
    settings: syn::Ident,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | convert_core_path(None),
    parallelizer: Parallelizer | Parallelizer::OsThreads,
    determinism: bool | true,
    @from
    KwargsSim
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

pub fn run_main_update(kwargs: KwargsMain) -> proc_macro2::TokenStream {
    use quote::quote;
    use SimulationAspect::*;

    let mut step_1 = proc_macro2::TokenStream::new();
    let mut step_2 = proc_macro2::TokenStream::new();
    let mut step_3 = proc_macro2::TokenStream::new();
    let mut step_4 = proc_macro2::TokenStream::new();
    let mut step_5 = proc_macro2::TokenStream::new();
    let mut local_func_names = Vec::<proc_macro2::TokenStream>::new();
    let mut local_subdomain_func_names = Vec::<proc_macro2::TokenStream>::new();

    let core_path = &kwargs.core_path;
    let settings = &kwargs.settings;
    let determinism = &kwargs.determinism;

    if kwargs
        .aspects
        .contains_multiple(vec![&Mechanics, &Interaction])
    {
        step_1.extend(quote!(sbox.update_mechanics_interaction_step_1()?;));
        step_2.extend(quote!(sbox.update_mechanics_interaction_step_2(#determinism)?;));
        step_3.extend(quote!(sbox.update_mechanics_interaction_step_3(#determinism)?;));
    }

    if kwargs.aspects.contains(&Mechanics) {
        local_func_names.push(quote!(#core_path::backend::chili::local_mechanics_update_step_3));
        step_4.extend(quote!(sbox.apply_boundary()?;));
    }

    if kwargs.aspects.contains(&Interaction) {
        local_func_names
            .push(quote!(#core_path::backend::chili::local_interaction_react_to_neighbors));
    }

    if kwargs.aspects.contains(&DomainForce) {
        step_1.extend(quote!(sbox.calculate_custom_domain_force()?;));
    }

    if kwargs.aspects.contains(&Cycle) {
        local_func_names.push(quote!(#core_path::backend::chili::local_cycle_update));
        step_4.extend(quote!(sbox.update_cell_cycle_4()?;));
    }

    if kwargs.aspects.contains(&Mechanics) {
        local_func_names
            .push(quote!(#core_path::backend::chili::local_mechanics_set_random_variable));
        step_4.extend(quote!(sbox.sort_cells_in_voxels_step_1()?;));
        step_5.extend(quote!(sbox.sort_cells_in_voxels_step_2(#determinism)?;));
    }

    if kwargs.aspects.contains(&Reactions) {
        local_func_names.push(quote!(#core_path::backend::chili::local_reactions_intracellular));
    }

    if kwargs.aspects.contains(&ReactionsContact) {
        step_1.extend(quote!(sbox.update_contact_reactions_step_1()?;));
        step_2.extend(quote!(sbox.update_contact_reactions_step_2(#determinism)?;));
        step_3.extend(quote!(sbox.update_contact_reactions_step_3(#determinism)?;));
        local_func_names
            .push(quote!(#core_path::backend::chili::local_update_contact_reactions_step_3));
    }

    if kwargs
        .aspects
        .contains_any([&Reactions, &ReactionsContact, &ReactionsExtra])
    {
        local_func_names.push(quote!(#core_path::backend::chili::local_reactions_use_increment));
    }

    if kwargs.aspects.contains(&ReactionsExtra) {
        step_1.extend(quote!(sbox.update_reactions_extra_step_1()?;));
        step_2.extend(quote!(sbox.update_reactions_extra_step_2(#determinism)?;));
        step_3.extend(quote!(sbox.update_reactions_extra_step_3(#determinism)?;));
        local_subdomain_func_names
            .push(quote!(#core_path::backend::chili::local_subdomain_update_reactions_extra));
    }

    let update_local_funcs = quote!(
        let __cr_private_combined_local_subdomain_funcs = |
            subdomain: &mut _,
            dt,
        | -> Result<(), #core_path::backend::chili::SimulationError> {
            #(
                #local_subdomain_func_names(subdomain, dt)?;
            )*
            Ok(())
        };
        sbox.run_local_subdomain_funcs(
            __cr_private_combined_local_subdomain_funcs,
            &next_time_point
        )?;
        let __cr_private_combined_local_cell_funcs = |
            cell: &mut _,
            aux_storage: &mut _,
            dt,
            rng: &mut rand_chacha::ChaCha8Rng
        | -> Result<(), #core_path::backend::chili::SimulationError> {
            #(
                #local_func_names(cell, aux_storage, dt, rng)?;
            )*
            Ok(())
        };
        sbox.run_local_cell_funcs(__cr_private_combined_local_cell_funcs, &next_time_point)?;
    );

    quote!(
        let builder = #settings.storage.clone().init();
        let builder_subdomains = builder.clone().suffix(builder.get_suffix().join("subdomains"));
        let builder_cells = builder.clone().suffix(builder.get_suffix().join("cells"));

        let mut _storage_manager_subdomains: #core_path::storage::StorageManager<
            #core_path::backend::chili::SubDomainPlainIndex,
            _
        > =
           #core_path::storage::StorageManager::open_or_create(builder_subdomains, key as u64)?;
        let mut _storage_manager_cells: #core_path::storage::StorageManager<_, _> =
           #core_path::storage::StorageManager::open_or_create(builder_cells, key as u64)?;

        // Set up the time stepper
        let mut _time_stepper = #settings.time.clone();
        use #core_path::time::TimeStepper;

        // Initialize the progress bar
        #[allow(unused)]
        let mut pb = match (key, #settings.show_progressbar) {
            (0, true) => Some(_time_stepper.initialize_bar()?),
            _ => None,
        };

        while let Some(next_time_point) = _time_stepper.advance()? {
            let mut f = || -> Result<(), #core_path::backend::chili::SimulationError> {
                #step_1
                sbox.sync()?;
                #step_2
                sbox.sync()?;
                #step_3
                #update_local_funcs
                #step_4
                sbox.sync()?;
                #step_5

                match (&mut pb, #settings.show_progressbar) {
                    (Some(bar), true) => _time_stepper.update_bar(bar)?,
                    _ => (),
                };
                sbox.save_subdomains(&mut _storage_manager_subdomains, &next_time_point)?;
                sbox.save_cells(&mut _storage_manager_cells, &next_time_point)?;
                Ok(())
            };
            let e = f();
            if sbox.store_error(e)? {break}
        }
        Ok(#core_path::backend::chili::StorageAccess {
            cells: _storage_manager_cells.clone(),
            subdomains: _storage_manager_subdomains.clone(),
        })
    )
}

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
    let parallelized_update_func =
        kwargs
            .parallelizer
            .parallelize_execution(&update_func, &core_path, settings);

    quote::quote!({
        type _Syncer = #core_path::backend::chili::BarrierSync;
        let _aux_storage = _CrAuxStorage::default();

        let __run_sim = || -> Result<
                #core_path::backend::chili::StorageAccess<_, _>,
                #core_path::backend::chili::SimulationError
        > {
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

            let res = #parallelized_update_func?;
            Result::<_, #core_path::backend::chili::SimulationError>::Ok(res)
        };
        __run_sim()
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
    let settings = &kwargs.settings;
    let mut output = quote::quote!(
        #core_path::backend::chili::compatibility_tests::domain_agents(
            &#domain,
            &#agents
        );
    );

    use SimulationAspect::*;
    if kwargs
        .aspects
        .contains_multiple(vec![&Mechanics, &Interaction])
    {
        output.extend(quote::quote!(
            #core_path::backend::chili::compatibility_tests::mechanics_interaction(
                &#agents
            );
        ));
    }

    if kwargs.aspects.contains(&Mechanics) {
        output.extend(quote::quote!(
            #core_path::backend::chili::compatibility_tests::time_stepper_mechanics(
                &#settings.time,
                &#agents,
            );
            #core_path::backend::chili::compatibility_tests::mechanics_implemented(
                &#agents,
            );
        ));
    }

    // TODO see comment at compatibility_tests function in chili backend.
    // if kwargs.aspects.contains(&Interaction) {
    //     output.extend(quote::quote!(
    //         #core_path::backend::chili::compatibility_tests::interaction_implemented(
    //             &#agents,
    //         );
    //     ));
    // }

    if kwargs.aspects.contains(&Cycle) {
        output.extend(quote::quote!(
            #core_path::backend::chili::compatibility_tests::cycle_implemented(
                &#agents,
            );
        ));
    }

    if kwargs.aspects.contains(&ReactionsContact) {
        output.extend(quote::quote!(
            #core_path::backend::chili::compatibility_tests::reactions_contact_implemented(
                &#agents,
            );
        ));
    }

    if kwargs.aspects.contains(&Reactions) {
        output.extend(quote::quote!(
            #core_path::backend::chili::compatibility_tests::reactions_implemented(
                &#agents,
            );
        ));
    }

    if kwargs.aspects.contains(&ReactionsExtra) {
        output.extend(quote::quote!(
            #core_path::backend::chili::compatibility_tests::subdomain_reactions_implemented(
                &#domain,
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
    let kwargs_compat = KwargsCompatibility::from(kwargs.clone());
    let test_compat = test_compatibility(kwargs_compat);
    let kwargs_main = KwargsMain::from(kwargs.clone());
    let run_main = run_main(kwargs_main);
    quote::quote!({
        #types
        #test_compat
        #run_main
    })
    .into()
}
