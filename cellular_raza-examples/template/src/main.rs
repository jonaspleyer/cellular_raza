use cellular_raza::building_blocks::{
    BoundLennardJonesF32, CartesianCuboid2NewF32, NewtonDamped2DF32,
};
use cellular_raza::concepts::domain_new::Domain;
use cellular_raza::concepts::{CalcError, CellAgent, Interaction, Mechanics, RngError, Volume};
use cellular_raza::core::backend::chili::{build_aux_storage, build_communicator};

use cellular_raza::core::backend::chili;

use nalgebra::Vector2;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

pub struct SimulationSettings {
    n_agents: usize,
    domain_size: f32,
    n_voxels: usize,
    n_threads: usize,
    dt: f32,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            n_agents: 200,
            domain_size: 30.0,
            n_voxels: 3,
            n_threads: 4,
            dt: 0.002,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct Vol(f64);

impl Volume for Vol {
    fn get_volume(&self) -> f64 {
        self.0
    }
}

#[derive(CellAgent, Clone, Deserialize, Serialize)]
struct Agent {
    #[Mechanics(Vector2<f32>, Vector2<f32>, Vector2<f32>, f32)]
    pub mechanics: NewtonDamped2DF32,
    #[Interaction(Vector2<f32>, Vector2<f32>, Vector2<f32>)]
    pub interaction: BoundLennardJonesF32,
    // #[Cycle]
    // pub cycle: NoCycle,
    // #[CellularReactions(Nothing, Nothing)]
    // pub reactions: NoCellularReactions,
    // #[ExtracellularGradient(nalgebra::SVector<Vector2<f64>, 2>)]
    // pub gradients: NoExtracellularGradientSensing,
    #[Volume]
    pub volume: Vol,
}

macro_rules! gen_step_1(
    ($sbox:ident, Mechanics) => {$sbox.update_mechanics_step_1()?;};
    ($sbox:ident, $asp:ident) => {};
    ($sbox:ident, $asp1:ident, $($asp:ident),*) => {
        gen_step_1!($sbox, $asp1);
        gen_step_1!($sbox, $($asp),*);
    };
);

macro_rules! gen_step_2(
    ($sbox:ident, Mechanics) => {$sbox.update_mechanics_step_2()?;};
    ($sbox:ident, $asp:ident) => {};
    ($sbox:ident, $asp1:ident, $($asp:ident),*) => {
        gen_step_2!($sbox, $asp1);
        gen_step_2!($sbox, $($asp),*);
    };
);

macro_rules! gen_step_3(
    ($sbox:ident, $next_time_point:ident, Mechanics) => {$sbox.update_mechanics_step_3(&$next_time_point.increment)?;};
    ($sbox:ident, $next_time_point:ident, $asp:ident) => {};
    ($sbox:ident, $next_time_point:ident, $asp1:ident, $($asp:ident),*) => {
        gen_step_3!($sbox, $next_time_point, $asp1);
        gen_step_3!($sbox, $next_time_point, $($asp),*);
    };
);

macro_rules! gen_step_4(
    ($sbox:ident, $storage_manager:ident, $next_time_point:ident, $pb:ident, $time_stepper:ident, Mechanics) => {
        $sbox.sort_cells_in_voxels_step_2()?;
           match &mut $pb {
               Some(bar) => $time_stepper.update_bar(bar)?,
               None => (),
           };
           // TODO
           // $sbox.apply_boundary()?;
           $sbox.save_voxels(&$storage_manager, &$next_time_point)?;
    };
    ($sbox:ident, $storage_manager:ident, $next_time_point:ident, $pb:ident, $time_stepper:ident, $asp:ident) => {};
    ($sbox:ident, $storage_manager:ident, $next_time_point:ident, $pb:ident, $time_stepper:ident, $asp1:ident, $($asp:ident),*) => {
        gen_step_4!($sbox, $storage_manager, $next_time_point, $pb, $time_stepper, $asp1);
        gen_step_4!($sbox, $storage_manager, $next_time_point, $pb, $time_stepper, $($asp),*);
    };
);

macro_rules! main_update(
    (
        subdomain: $sbox:ident,
        storage_manager: $storage_manager:ident,
        next_time_point: $next_time_point:ident,
        progress_bar: $pb:ident,
        time_stepper: $time_stepper:ident,
        aspects: [$($asp:ident),*]
    ) => {
            gen_step_1!($sbox, $($asp),*);
            $sbox.sync();
            gen_step_2!($sbox, $($asp),*);
            $sbox.sync();
            gen_step_3!($sbox, $next_time_point, $($asp),*);
            $sbox.sync();
            gen_step_4!($sbox, $storage_manager, $next_time_point, $pb, $time_stepper, $($asp),*);
    }
);

macro_rules! run_simulation(
    (
        domain: $domain:ident,
        agents: $agents:ident,
        time: $time_stepper:ident,
        n_threads: $n_threads:expr,
        syncer: $syncer:ty,
        storage: $storage_builder:ident,
        aspects: [$($asp:ident),*]
    ) => {{
        build_communicator!(
            name: _CrCommunicator,
            aspects: [$($asp),*],
            core_path: cellular_raza::core
        );
        build_aux_storage!(
            name: _CrAuxStorage,
            aspects: [$($asp),*],
            core_path: cellular_raza::core
        );

        // TODO this is not final and can not stay like this
        let decomposed_domain = $domain
            .decompose($n_threads.try_into().unwrap(), $agents)?;

        let mut runner: chili::SimulationRunner<
            _,
            chili::SubDomainBox<
                _,
                _,
                _,
                _CrAuxStorage<_, _, _, _, 2>,
                _CrCommunicator<_, _, _, _, _, _, _>,
                chili::BarrierSync,
            >,
        > = decomposed_domain.into();

        use rayon::prelude::*;
        runner
            .subdomain_boxes
            .par_iter_mut()
            .map(|(key, sbox)| {
                let mut time_stepper = $time_stepper.clone();
                use cellular_raza::prelude::time::TimeStepper;
                let mut pb = match key {
                    0 => Some(time_stepper.initialize_bar()?),
                    _ => None,
                };

                // Initialize the storage manager
                let storage_manager = cellular_raza::prelude::StorageManager::construct(&$storage_builder, *key as u64)?;
                while let Some(next_time_point) = time_stepper.advance()? {
                    main_update!(
                        subdomain: sbox,
                        storage_manager: storage_manager,
                        next_time_point: next_time_point,
                        progress_bar: pb,
                        time_stepper: time_stepper,
                        aspects: [Mechanics]
                    );
                    // update_subdomain!(name: sbox, aspects: [Mechanics, Interaction]);
                }
                Ok(())
            })
            .collect::<Result<Vec<_>, chili::SimulationError>>()?;
        Result::<(), chili::SimulationError>::Ok(())
    }}
);

fn run_simulation(
    simulation_settings: SimulationSettings,
    agents: Vec<Agent>,
) -> Result<(), chili::SimulationError> {
    // Domain Setup
    let domain = CartesianCuboid2NewF32::from_boundaries_and_n_voxels(
        [0.0; 2],
        [simulation_settings.domain_size; 2],
        [simulation_settings.n_voxels; 2],
    )?;

    // Storage Setup
    let location = std::path::Path::new("./out");
    let mut storage_priority = cellular_raza::prelude::UniqueVec::new();
    storage_priority.push(cellular_raza::prelude::StorageOption::SerdeJson);
    let storage_builder = cellular_raza::prelude::StorageBuilder::new()
        .priority(storage_priority)
        .location(location);

    // Time Setup
    let t0: f32 = 0.0;
    let dt = simulation_settings.dt;
    let save_points = vec![5.0, 10.0, 15.0, 20.0];
    let time_stepper = cellular_raza::prelude::time::FixedStepsize::from_partial_save_points(
        t0,
        dt,
        save_points.clone(),
    )?;

    run_simulation!(
        domain: domain,
        agents: agents,
        time: time_stepper,
        n_threads: simulation_settings.n_threads,
        // TODO make this optional
        syncer: chili::BarrierSync,
        storage: storage_builder,
        aspects: [Mechanics, Interaction, Cycle]
    )?;
    Ok(())
}

fn main() -> Result<(), chili::SimulationError> {
    use rand::Rng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let simulation_settings = SimulationSettings::default();

    // Create subscriber
    // Configure a custom event formatter
    let format = tracing_subscriber::fmt::format()
        .pretty()
        .with_thread_ids(true); // include the thread ID of the current thread

    // Create a `fmt` subscriber that uses our custom event format, and set it
    // as the default.
    tracing_subscriber::fmt().event_format(format).init();

    // Construct the corresponding AuxStorage
    let agent = Agent {
        mechanics: NewtonDamped2DF32 {
            pos: Vector2::from([0.0, 0.0]),
            vel: Vector2::from([0.0, 0.0]),
            damping_constant: 1.0,
            mass: 1.0,
        },
        interaction: BoundLennardJonesF32 {
            epsilon: 0.01,
            sigma: 1.0,
            bound: 0.1,
            cutoff: 1.0,
        },
        volume: Vol(1.0),
    };
    let agents = (0..simulation_settings.n_agents)
        .map(|_| {
            let mut new_agent = agent.clone();
            new_agent.set_pos(&Vector2::from([
                rng.gen_range(0.0..simulation_settings.domain_size),
                rng.gen_range(0.0..simulation_settings.domain_size),
            ]));
            new_agent
        })
        .collect();

    run_simulation(simulation_settings, agents)?;
    Ok(())
}
