#[doc(hidden)]
pub use crate::gen_step_1;

#[doc(hidden)]
#[macro_export]
macro_rules! gen_step_1(
    ($sbox:ident, Mechanics, Interaction) => {$sbox.update_mechanics_step_1()?;};
    // TODO this macro call variant should be removed in the future
    ($sbox:ident, $asp:ident) => {};
    ($sbox:ident, $asp1:ident, $($asp:ident),*) => {
        $crate::gen_step_1!($sbox, $asp1);
        $crate::gen_step_1!($sbox, $($asp),*);
    };
);

#[doc(hidden)]
pub use crate::gen_step_2;

#[doc(hidden)]
#[macro_export]
macro_rules! gen_step_2(
    ($sbox:ident, Mechanics, Interaction) => {$sbox.update_mechanics_step_2()?;};
    // TODO this macro call variant should be removed in the future
    ($sbox:ident, $asp:ident) => {};
    ($sbox:ident, $asp1:ident, $($asp:ident),*) => {
        $crate::gen_step_2!($sbox, $asp1);
        $crate::gen_step_2!($sbox, $($asp),*);
    };
);

#[doc(hidden)]
pub use crate::gen_step_3;

#[doc(hidden)]
#[macro_export]
macro_rules! gen_step_3(
    ($sbox:ident, $next_time_point:ident, Mechanics, Interaction) => {$sbox.update_mechanics_step_3(&$next_time_point.increment)?;};
    // TODO this macro call variant should be removed in the future
    ($sbox:ident, $next_time_point:ident, $asp:ident) => {};
    ($sbox:ident, $next_time_point:ident, $asp1:ident, $($asp:ident),*) => {
        $crate::gen_step_3!($sbox, $next_time_point, $asp1);
        $crate::gen_step_3!($sbox, $next_time_point, $($asp),*);
    };
);

#[doc(hidden)]
pub use crate::gen_step_4;

#[doc(hidden)]
#[macro_export]
macro_rules! gen_step_4(
    ($sbox:ident, $storage_manager:ident, $next_time_point:ident, $pb:ident, $time_stepper:ident, Mechanics, Interaction) => {
        $sbox.sort_cells_in_voxels_step_2()?;
        // TODO
        // $sbox.apply_boundary()?;
    };
    // TODO this macro call variant should be removed in the future
    ($sbox:ident, $storage_manager:ident, $next_time_point:ident, $pb:ident, $time_stepper:ident, $asp:ident) => {};
    ($sbox:ident, $storage_manager:ident, $next_time_point:ident, $pb:ident, $time_stepper:ident, $asp1:ident, $($asp:ident),*) => {
        $crate::gen_step_4!($sbox, $storage_manager, $next_time_point, $pb, $time_stepper, $asp1);
        $crate::gen_step_4!($sbox, $storage_manager, $next_time_point, $pb, $time_stepper, $($asp),*);
    };
);

#[doc(hidden)]
pub use crate::main_update;

#[doc(hidden)]
#[macro_export]
macro_rules! main_update(
    (
        subdomain: $sbox:ident,
        storage_manager: $storage_manager:ident,
        next_time_point: $next_time_point:ident,
        progress_bar: $pb:ident,
        time_stepper: $time_stepper:ident,
        aspects: [$($asp:ident),*],
        core_path: $core_path:path
    ) => {
        $crate::gen_step_1!($sbox, $($asp),*);
        $sbox.sync();
        $crate::gen_step_2!($sbox, $($asp),*);
        $sbox.sync();
        $crate::gen_step_3!($sbox, $next_time_point, $($asp),*);
        $sbox.sync();
        $crate::gen_step_4!($sbox, $storage_manager, $next_time_point, $pb, $time_stepper, $($asp),*);

        // Update progress bar
        match &mut $pb {
            Some(bar) => $time_stepper.update_bar(bar)?,
            None => (),
        };
        $sbox.save_voxels(&$storage_manager, &$next_time_point)?;
    }
);

#[doc(inline)]
pub use cellular_raza_core_proc_macro::run_simulation;

#[doc(inline)]
pub use cellular_raza_core_proc_macro::prepare_types;

#[doc(inline)]
pub use cellular_raza_core_proc_macro::test_compatibility;

#[doc(inline)]
pub use cellular_raza_core_proc_macro::run_main;

#[doc(inline)]
pub use crate::hidden_run_simulation;

///
#[doc(hidden)]
#[macro_export]
macro_rules! hidden_run_simulation(
    (@if_else {}                 , {$($default_value:tt)*}) => {$($default_value)*};
    (@if_else {$($opt_value:tt)*}, {$($default_value:tt)*}) => {$($opt_value)*};
    (
        domain: $domain:ident,
        agents: $agents:ident,
        settings: $settings:expr,
        aspects: [$($asp:ident),*]$(,)?
        // TODO add option to specify solvers
        $(mechanics_solver: $mechanics_solver:ident,)?
        // TODO implement the external controller
        $(external_controller: $ext_controller:ty,)?
        // TODO actually use this type
        $(syncer: $syncer:ty,)?
        $(parallelization: $parallel:ident,)?
    ) => {{
        use $crate::backend::chili::{AuxStorage, Communicator};
        $crate::backend::chili::build_communicator!(
            name: _CrCommunicator,
            aspects: [$($asp),*],
            core_path: $crate
        );
        $crate::backend::chili::build_aux_storage!(
            name: _CrAuxStorage,
            aspects: [$($asp),*],
            core_path: $crate
        );

        // Define syncer type
        $crate::run_simulation!(@if_else {$(type Syncer = $syncer;)?}, {type Syncer = $crate::backend::chili::BarrierSync;});

        let _aux_storage = _CrAuxStorage::default();
        let mut runner = $crate::backend::chili::construct_simulation_runner::<
            _,
            _,
            _,
            _,
            $crate::backend::chili::communicator_generics_placeholders!(
                name: _CrCommunicator,
                aspects: [$($asp),*]
            ),
            Syncer,
            _
        >(
            $domain,
            $agents,
            $settings.n_threads,
            &_aux_storage,
        )?;

        // TODO this should probably be replaced by os threads
        // or we simply make this an option in the macro signature
        runner
            .subdomain_boxes
            .iter_mut()
            .map(|(&key, sbox)| {
                // Set up the time stepper
                let mut _time_stepper = $settings.time.clone();
                use $crate::time::TimeStepper;

                // Initialize the progress bar
                #[allow(unused)]
                let mut pb = match key {
                    0 => Some(_time_stepper.initialize_bar()?),
                    _ => None,
                };

                // Initialize the storage manager
                #[allow(unused)]
                let storage_manager: $crate::storage::StorageManager<_, _> =
                    $crate::storage::StorageManager::construct(
                        &$settings.storage,
                        key as u64
                    )?;

                // Calls the main update macro which constructs the main update step
                #[allow(unused)]
                while let Some(next_time_point) = _time_stepper.advance()? {
                    $crate::backend::chili::main_update!(
                        subdomain: sbox,
                        storage_manager: storage_manager,
                        next_time_point: next_time_point,
                        progress_bar: pb,
                        time_stepper: _time_stepper,
                        aspects: [$($asp),*],
                        core_path: $crate
                    );
                }
                Ok(())
            })
            .collect::<Result<Vec<_>, $crate::backend::chili::SimulationError>>()?;
        Result::<(), $crate::backend::chili::SimulationError>::Ok(())
    }};
);
