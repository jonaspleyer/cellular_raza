use crate::simulation_aspects::{SimulationAspect, SimulationAspects};

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Parallelizer {
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
                    let mut _storage_manager_subdomains = _storage_manager_subdomains_base
                        .clone_to_new_instance(key as u64);
                    let mut _storage_manager_cells = _storage_manager_cells_base
                        .clone_to_new_instance(key as u64);
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
                use #core_path::rayon::prelude::*;
                let mut storage_accesses = runner.subdomain_boxes
                    .into_par_iter()
                    .map(|(key, mut sbox)| -> Result<
                        _,
                        #core_path::backend::chili::SimulationError
                    > {
                let mut _storage_manager_subdomains = _storage_manager_subdomains_base
                    .clone_to_new_instance(key as u64);
                let mut _storage_manager_cells = _storage_manager_cells_base
                    .clone_to_new_instance(key as u64);
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

// IMPORTANT NOTICE: Just as done in the Kwargs enum,
// this value is actually the solver order minus one.
// This is due to the fact that the AuxStorage only
// needs to store one less value than the solver order
// and we need to specify the generic argument there
// with this variable.
pub const DEFAULT_MECHANICS_SOLVER_ORDER: usize = 2;
pub const DEFAULT_REACTIONS_SOLVER_ORDER_INTRA: usize = 4;
pub const DEFAULT_REACTIONS_SOLVER_ORDER_CONTACT: usize = 2;

pub fn default_update_mechanics_interaction_step_1_fn_name() -> syn::Ident {
    syn::Ident::new(
        "update_mechanics_interaction_step_1",
        proc_macro2::Span::call_site(),
    )
}

pub fn default_update_mechanics_interaction_step_2_fn_name() -> syn::Ident {
    syn::Ident::new(
        "update_mechanics_interaction_step_2",
        proc_macro2::Span::call_site(),
    )
}

pub fn default_update_mechanics_interaction_step_3_fn_name() -> syn::Ident {
    syn::Ident::new(
        "update_mechanics_interaction_step_3",
        proc_macro2::Span::call_site(),
    )
}

define_kwargs!(
    KwargsSim,
    KwargsSimParsed,
    domain: syn::Ident,
    agents: syn::Ident,
    settings: syn::Ident,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | crate::kwargs::convert_core_path(None),
    parallelizer: Parallelizer | Parallelizer::OsThreads,
    determinism: bool | true,
    aux_storage_name: syn::Ident | crate::aux_storage::default_aux_storage_name(),
    zero_force_default: syn::ExprClosure | crate::aux_storage::zero_force_default(),
    zero_reactions_default: syn::ExprClosure | crate::aux_storage::zero_reactions_default(),
    communicator_name: syn::Ident | crate::communicator::default_communicator_name(),
    mechanics_solver_order: usize | crate::run_sim::DEFAULT_MECHANICS_SOLVER_ORDER,
    reactions_intra_solver_order: usize | crate::run_sim::DEFAULT_REACTIONS_SOLVER_ORDER_INTRA,
    reactions_contact_solver_order: usize | crate::run_sim::DEFAULT_REACTIONS_SOLVER_ORDER_CONTACT,

    // Define functions to call for updates
    update_mechanics_interaction_step_1: syn::Ident |
        crate::run_sim::default_update_mechanics_interaction_step_1_fn_name(),
    update_mechanics_interaction_step_2: syn::Ident |
        crate::run_sim::default_update_mechanics_interaction_step_2_fn_name(),
    update_mechanics_interaction_step_3: syn::Ident |
        crate::run_sim::default_update_mechanics_interaction_step_3_fn_name(),
);

define_kwargs!(
    KwargsCompatibility,
    KwargsCompatibilityParsed,
    domain: syn::Ident,
    agents: syn::Ident,
    settings: syn::Ident,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | crate::kwargs::convert_core_path(None),
    @from
    KwargsSim
);

define_kwargs!(
    KwargsPrepareTypes,
    KwargsPrepareTypesParsed,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | crate::kwargs::convert_core_path(None),
    aux_storage_name: syn::Ident | crate::aux_storage::default_aux_storage_name(),
    zero_force_default: syn::ExprClosure | crate::aux_storage::zero_force_default(),
    zero_reactions_default: syn::ExprClosure | crate::aux_storage::zero_reactions_default(),
    communicator_name: syn::Ident | crate::communicator::default_communicator_name(),
    @from
    KwargsSim,
);

define_kwargs!(
    KwargsMain,
    KwargsMainParsed,
    domain: syn::Ident,
    agents: syn::Ident,
    settings: syn::Ident,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | crate::kwargs::convert_core_path(None),
    parallelizer: Parallelizer | Parallelizer::OsThreads,
    determinism: bool | true,
    aux_storage_name: syn::Ident | crate::aux_storage::default_aux_storage_name(),
    zero_force_default: syn::ExprClosure | crate::aux_storage::zero_force_default(),
    zero_reactions_default: syn::ExprClosure | crate::aux_storage::zero_reactions_default(),
    communicator_name: syn::Ident | crate::communicator::default_communicator_name(),
    mechanics_solver_order: usize | crate::run_sim::DEFAULT_MECHANICS_SOLVER_ORDER,
    reactions_intra_solver_order: usize | crate::run_sim::DEFAULT_REACTIONS_SOLVER_ORDER_INTRA,
    reactions_contact_solver_order: usize | crate::run_sim::DEFAULT_REACTIONS_SOLVER_ORDER_CONTACT,

    // Define functions to call for updates
    update_mechanics_interaction_step_1: syn::Ident |
        crate::run_sim::default_update_mechanics_interaction_step_1_fn_name(),
    update_mechanics_interaction_step_2: syn::Ident |
        crate::run_sim::default_update_mechanics_interaction_step_2_fn_name(),
    update_mechanics_interaction_step_3: syn::Ident |
        crate::run_sim::default_update_mechanics_interaction_step_3_fn_name(),
    @from
    KwargsSim
);

pub fn run_main_update(kwargs: KwargsMain) -> proc_macro2::TokenStream {
    use SimulationAspect::*;
    use quote::quote;

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

    let mechanics_solver_order = kwargs.mechanics_solver_order;
    let reactions_intra_solver_order = kwargs.reactions_intra_solver_order;
    let aux_storage_constructor = crate::aux_storage::default_aux_storage_initializer(&kwargs);

    if kwargs
        .aspects
        .contains_multiple(vec![&Mechanics, &Interaction])
    {
        let umis_fn_name_1 = &kwargs.update_mechanics_interaction_step_1;
        let umis_fn_name_2 = &kwargs.update_mechanics_interaction_step_2;
        let umis_fn_name_3 = &kwargs.update_mechanics_interaction_step_3;
        step_1.extend(quote!(sbox. #umis_fn_name_1 ()?;));
        step_2.extend(quote!(sbox. #umis_fn_name_2 (#determinism)?;));
        step_3.extend(quote!(sbox. #umis_fn_name_3 (#determinism)?;));
    }

    if kwargs.aspects.contains(&Mechanics) {
        local_func_names.push(quote!(
            #core_path::backend::chili::local_mechanics_update::<
                _,
                _,
                _,
                _,
                _,
                _,
                #mechanics_solver_order
            >));
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
        step_4.extend(quote!(sbox.update_cell_cycle_4(&#aux_storage_constructor)?;));
    }

    if kwargs.aspects.contains(&Mechanics) {
        step_4.extend(quote!(sbox.sort_cells_in_voxels_step_1()?;));
        step_5.extend(quote!(sbox.sort_cells_in_voxels_step_2(#determinism)?;));
    }

    if kwargs.aspects.contains(&Reactions) {
        local_func_names.push(
            quote!(#core_path::backend::chili::local_reactions_intracellular::<
            _,
            _,
            _,
            _,
            #reactions_intra_solver_order,
        >),
        );
    }

    if kwargs.aspects.contains(&ReactionsContact) {
        step_1.extend(quote!(sbox.update_contact_reactions_step_1()?;));
        step_2.extend(quote!(sbox.update_contact_reactions_step_2(#determinism)?;));
        step_3.extend(quote!(sbox.update_contact_reactions_step_3(#determinism)?;));
        local_func_names.push(quote!(#core_path::backend::chili::local_update_contact_reactions));
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
    let aux_storage_name = &kwargs.aux_storage_name;
    let communicator_name = &kwargs.communicator_name;
    let aux_storage_placeholders = crate::aux_storage::generics_placeholders(
        kwargs.clone(),
        kwargs.mechanics_solver_order,
        kwargs.reactions_contact_solver_order,
    );
    let aux_storage_constructor = crate::aux_storage::default_aux_storage_initializer(&kwargs);

    let update_func = run_main_update(kwargs.clone());
    let create_storage = quote::quote!(
        let builder = #settings.storage.clone().init();
        let builder_subdomains = builder.clone().suffix(builder.get_suffix().join("subdomains"));
        let builder_cells = builder.clone().suffix(builder.get_suffix().join("cells"));

        let _storage_manager_subdomains_base: #core_path::storage::StorageManager<
            #core_path::backend::chili::SubDomainPlainIndex,
            _
        > =
            #core_path::storage::StorageManager::open_or_create(builder_subdomains, 0)?;
        let _storage_manager_cells_base: #core_path::storage::StorageManager<_, _> =
            #core_path::storage::StorageManager::open_or_create(builder_cells, 0)?;
    );
    let parallelized_update_func =
        kwargs
            .parallelizer
            .parallelize_execution(&update_func, core_path, settings);

    quote::quote!({
        type _Syncer = #core_path::backend::chili::BarrierSync;
        let __run_sim = || -> Result<
                #core_path::backend::chili::StorageAccess<_, _>,
                #core_path::backend::chili::SimulationError
        > {
            let mut runner = #core_path::backend::chili::construct_simulation_runner::<
                _,
                _,
                _,
                #aux_storage_name<#(#aux_storage_placeholders),*>,
                #core_path::backend::chili::communicator_generics_placeholders!(
                    name: #communicator_name,
                    aspects: [#(#asp),*]
                ),
                _Syncer,
                _
            >(
                #domain,
                #agents,
                #settings.n_threads,
                #aux_storage_constructor,
            )?;

            #create_storage
            let res = #parallelized_update_func?;
            Result::<_, #core_path::backend::chili::SimulationError>::Ok(res)
        };
        __run_sim()
    })
}

pub fn prepare_types(kwargs: KwargsPrepareTypes) -> proc_macro2::TokenStream {
    // Build AuxStorage
    let aux_storage_builder = super::aux_storage::Builder {
        struct_name: kwargs.aux_storage_name.clone(),
        core_path: kwargs.core_path.clone(),
        aspects: kwargs.aspects.clone(),
    };
    let mut output = aux_storage_builder.build_aux_storage();

    // Build Communicator
    let communicator_builder = super::communicator::CommunicatorBuilder {
        struct_name: kwargs.communicator_name.clone(),
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
    let types = prepare_types(KwargsPrepareTypes::from(kwargs.clone()));

    let kwargs_compat = KwargsCompatibility::from(kwargs.clone());
    let test_compat = test_compatibility(kwargs_compat);

    let kwargs_main = KwargsMain::from(kwargs.clone());
    let run_main = run_main(kwargs_main);
    quote::quote!({
        #types
        #test_compat
        #run_main
    })
}
