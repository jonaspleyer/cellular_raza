use cellular_raza::building_blocks::prelude::*;
use cellular_raza::concepts::domain_new::Domain;
use cellular_raza::concepts::prelude::*;
use cellular_raza::concepts_derive::*;
use cellular_raza::core::backend::chili::*;
use cellular_raza::core::proc_macro::*;

use nalgebra::Vector2;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

pub struct SimulationSettings {
    n_angents: usize,
    domain_size: f32,
    n_voxels: usize,
    n_threads: usize,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            n_angents: 100,
            domain_size: 30.0,
            n_voxels: 3,
            n_threads: 4,
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

use cellular_raza::core::backend::chili::aux_storage::{
    AuxStorageInteraction, AuxStorageMechanics, FixedSizeRingBufferIter, UpdateInteraction,
    UpdateMechanics,
};

build_aux_storage!(name: __cr_AuxStorage, aspects: [Mechanics, Interaction]);
build_communicator!(
    name: MyCommunicator,
    aspects: [Mechanics, Interaction],
    core_path: cellular_raza::core
);

use cellular_raza::core::backend::chili::simulation_flow::BarrierSync;

fn run_simulation(
    simulation_settings: SimulationSettings,
    agents: Vec<Agent>,
) -> Result<(), errors::SimulationError> {
    let domain = CartesianCuboid2NewF32::from_boundaries_and_n_voxels(
        [0.0; 2],
        [simulation_settings.domain_size; 2],
        [simulation_settings.n_voxels; 2],
    )?;
    let decomposed_domain = domain
        .decompose(simulation_settings.n_threads.try_into().unwrap(), agents)
        .unwrap();
    println!("{:#?}", decomposed_domain.neighbor_map);
    use cellular_raza::core::backend::chili::datastructures::*;

    let mut supervisor: SimulationSupervisor<
        _,
        SubDomainBox<
            _,
            _,
            __cr_AuxStorage<_, _, _, _, 0>,
            // TODO these two are just to fill in SOMETHING
            // The Agent will have to be replaced with the CellAgentBox<C> or something simlar
            // and the f64 by the __cr_AuxStorage<_, _, _, _, 4> from above
            MyCommunicator<_, _, _, _, _, _, _>,
            BarrierSync,
        >,
    > = decomposed_domain.into();

    use kdam::{tqdm, BarExt};
    use rayon::prelude::*;
    let dt = 0.002;
    supervisor
        .subdomain_boxes
        .par_iter_mut()
        .for_each(|(key, sbox)| {
            let mut pb = if key == &0 {
                Some(tqdm!(total = 1_000))
            } else {
                None
            };
            for _ in 0..1_000 {
                // update_subdomain!(name: sbox, aspects: [Mechanics, Interaction]);
                sbox.update_mechanics_step_1().unwrap();

                sbox.sync();

                sbox.update_mechanics_step_2().unwrap();

                sbox.sync();

                sbox.update_mechanics_step_3(&dt).unwrap();

                sbox.sort_cells_in_voxels_step_1().unwrap();

                sbox.sync();

                sbox.sort_cells_in_voxels_step_2().unwrap();
                match &mut pb {
                    Some(p) => p.update(1).unwrap(),
                    None => true,
                };
            }
        });
    Ok(())
}

fn main() -> Result<(), errors::SimulationError> {
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
    let agents = (0..simulation_settings.n_angents)
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
