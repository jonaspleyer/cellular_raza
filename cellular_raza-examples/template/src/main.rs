use cellular_raza::building_blocks::prelude::*;
use cellular_raza::concepts::domain_new::Domain;
use cellular_raza::concepts::*;
use cellular_raza::concepts_derive::*;
use cellular_raza::core::proc_macro::*;

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

#[derive(CellAgent, Clone)] // Deserialize, Serialize
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

build_aux_storage!(
    name: __cr_AuxStorage,
    aspects: [Mechanics, Interaction],
    core_path: cellular_raza::core
);
build_communicator!(
    name: MyCommunicator,
    aspects: [Mechanics, Interaction],
    core_path: cellular_raza::core
);

fn run_simulation(
    simulation_settings: SimulationSettings,
    agents: Vec<Agent>,
) -> Result<(), chili::SimulationError> {
    let domain = CartesianCuboid2NewF32::from_boundaries_and_n_voxels(
        [0.0; 2],
        [simulation_settings.domain_size; 2],
        [simulation_settings.n_voxels; 2],
    )?;
    let decomposed_domain = domain
        .decompose(simulation_settings.n_threads.try_into().unwrap(), agents)
        .unwrap();

    let mut supervisor: chili::SimulationSupervisor<
        _,
        chili::SubDomainBox<
            _,
            _,
            __cr_AuxStorage<_, _, _, _, 2>,
            MyCommunicator<_, _, _, _, _, _, _>,
            chili::BarrierSync,
        >,
    > = decomposed_domain.into();

    use kdam::{tqdm, BarExt};
    use rayon::prelude::*;
    let t0: f32 = 0.0;
    let dt = simulation_settings.dt;
    let save_points = vec![5.0, 10.0, 15.0, 20.0];
    let time_stepper = cellular_raza::prelude::time::FixedStepsize::from_partial_save_points(
        t0,
        dt,
        save_points.clone(),
    )?;
    supervisor
        .subdomain_boxes
        .par_iter_mut()
        .map(|(key, sbox)| {
            let mut time_stepper = time_stepper.clone();
            let mut pb = if key == &0 {
                Some(tqdm!(total = save_points.len()))
            } else {
                None
            };
            use cellular_raza::prelude::time::TimeStepper;
            while let Some(next_time_point) = time_stepper.advance()? {
                // update_subdomain!(name: sbox, aspects: [Mechanics, Interaction]);
                sbox.update_mechanics_step_1()?;

                sbox.sync();

                sbox.update_mechanics_step_2()?;

                sbox.sync();

                sbox.update_mechanics_step_3(&next_time_point.increment)?;

                sbox.sort_cells_in_voxels_step_1()?;

                sbox.sync();

                sbox.sort_cells_in_voxels_step_2()?;
                match (&mut pb, next_time_point.event) {
                    (Some(p), Some(_)) => p.update(1)?,
                    _ => true,
                };
            }
            Ok(())
        })
        .collect::<Result<Vec<_>, cellular_raza::core::backend::chili::SimulationError>>()?;
    Ok(())
}

fn main() -> Result<(), chili::SimulationError> {
    use rand::Rng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let simulation_settings = SimulationSettings::default();

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
