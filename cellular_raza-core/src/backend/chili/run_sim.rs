#[doc(hidden)]
pub use crate::gen_step_1;

#[doc(hidden)]
#[macro_export]
macro_rules! gen_step_1(
    ($sbox:ident, Mechanics, Interaction) => {$sbox.update_mechanics_step_1()?;};
    // TODO this macro call variant should be removed in the future
    ($sbox:ident, $asp:ident) => {};
    ($sbox:ident, $asp1:ident, $($asp:ident),*) => {
        gen_step_1!($sbox, $asp1);
        gen_step_1!($sbox, $($asp),*);
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
        gen_step_2!($sbox, $asp1);
        gen_step_2!($sbox, $($asp),*);
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
        gen_step_3!($sbox, $next_time_point, $asp1);
        gen_step_3!($sbox, $next_time_point, $($asp),*);
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
        gen_step_4!($sbox, $storage_manager, $next_time_point, $pb, $time_stepper, $asp1);
        gen_step_4!($sbox, $storage_manager, $next_time_point, $pb, $time_stepper, $($asp),*);
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
        use $core_path as _core_path;
        _core_path::backend::chili::gen_step_1!($sbox, $($asp),*);
        $sbox.sync();
        _core_path::backend::chili::gen_step_2!($sbox, $($asp),*);
        $sbox.sync();
        _core_path::backend::chili::gen_step_3!($sbox, $next_time_point, $($asp),*);
        $sbox.sync();
        _core_path::backend::chili::gen_step_4!($sbox, $storage_manager, $next_time_point, $pb, $time_stepper, $($asp),*);

        // Update progress bar
        match &mut $pb {
            Some(bar) => $time_stepper.update_bar(bar)?,
            None => (),
        };
        $sbox.save_voxels(&$storage_manager, &$next_time_point)?;
    }
);

#[doc(inline)]
pub use crate::run_simulation;

///
#[doc(hidden)]
#[macro_export]
macro_rules! run_simulation(
    (@if_else                  , {$($default_value:tt)*}) => {$($default_value)*};
    (@if_else {$($opt_value:tt)*}, {$($default_value:tt)*}) => {$($opt_value)*};
    (
        // TODO can we combine all these values into some type of setup or similar?
        domain: $domain:ident,
        agents: $agents:ident,
        time: $time_stepper:ident,
        n_threads: $n_threads:expr,
        // TODO actually use this type
        $(syncer: $syncer:ty,)?
        storage: $storage_builder:ident,
        aspects: [$($asp:ident),*],
        $(core_path: $core_path:path,)?
        $(parallelization: $parallel:ident,)?
    ) => {{
        $crate::run_simulation!(@if_else {$(use $core_path as _core_path;)?}, {use cellular_raza::core as _core_path;});
        build_communicator!(
            name: _CrCommunicator,
            aspects: [$($asp),*],
            core_path: _core_path
        );
        build_aux_storage!(
            name: _CrAuxStorage,
            aspects: [$($asp),*],
            core_path: _core_path
        );

        let decomposed_domain = $domain
            .decompose($n_threads.try_into().unwrap(), $agents)?;

        let _aux_storage = _CrAuxStorage::default();
        let mut runner = chili::SimulationRunner::<_, chili::SubDomainBox<
            _,
            _,
            _,
            _,
            chili::communicator_generics_placeholders!(
                name: _CrCommunicator,
                aspects: [$($asp),*]
            ),
        >>::construct(
            decomposed_domain,
            &_aux_storage,
        );

        // TODO this should probably be replaced by os threads
        // or we simply make this an option in the macro signature
        use rayon::prelude::*;
        runner
            .subdomain_boxes
            .par_iter_mut()
            .map(|(key, sbox)| {
                // Set up the time stepper
                let mut _time_stepper = $time_stepper.clone();
                use cellular_raza::prelude::time::TimeStepper;

                // Initialize the progress bar
                #[allow(unused)]
                let mut pb = match key {
                    0 => Some(_time_stepper.initialize_bar()?),
                    _ => None,
                };

                // Initialize the storage manager
                #[allow(unused)]
                let storage_manager: cellular_raza::prelude::StorageManager<_, _> =
                    cellular_raza::prelude::StorageManager::construct(
                        &$storage_builder,
                        *key as u64
                    )?;

                // Calls the main update macro which constructs the main update step
                #[allow(unused)]
                while let Some(next_time_point) = _time_stepper.advance()? {
                    _core_path::backend::chili::main_update!(
                        subdomain: sbox,
                        storage_manager: storage_manager,
                        next_time_point: next_time_point,
                        progress_bar: pb,
                        time_stepper: _time_stepper,
                        aspects: [$($asp),*],
                        core_path: _core_path
                    );
                }
                Ok(())
            })
            .collect::<Result<Vec<_>, chili::SimulationError>>()?;
        Result::<(), chili::SimulationError>::Ok(())
    }};
);

