use cellular_raza::prelude::*;

use nalgebra::Vector3;
use num::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const N_CELLS_1: usize = 800;
pub const N_CELLS_2: usize = 800;

pub const CELL_DAMPENING: f64 = 2.0;
pub const CELL_RADIUS: f64 = 6.0;

pub const CELL_MECHANICS_RELATIVE_INTERACTION_RANGE: f64 = 1.5;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 2.0;

pub const DT: f64 = 0.25;
pub const N_TIMES: usize = 3_000;
pub const SAVE_INTERVAL: usize = 10;

pub const N_THREADS: usize = 4;

pub const DOMAIN_SIZE: f64 = 110.0;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
enum Species {
    RedCell,
    BlueCell,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CellSpecificInteraction {
    species: Species,
    cell_radius: f64,
    potential_strength: f64,
    relative_interaction_range: f64,
}

impl Interaction<Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, Species)>
    for CellSpecificInteraction
{
    fn calculate_force_between(
        &self,
        own_pos: &Vector3<f64>,
        _own_vel: &Vector3<f64>,
        ext_pos: &Vector3<f64>,
        _ext_vel: &Vector3<f64>,
        ext_info: &(f64, Species),
    ) -> Option<Result<Vector3<f64>, CalcError>> {
        let min_relative_distance_to_center = 0.3162277660168379;
        let (r, dir) =
            match (own_pos - ext_pos).norm() < self.cell_radius * min_relative_distance_to_center {
                false => {
                    let z = own_pos - ext_pos;
                    let r = z.norm();
                    (r, z.normalize())
                }
                true => {
                    let dir = match own_pos == ext_pos {
                        true => {
                            return None;
                        }
                        false => (own_pos - ext_pos).normalize(),
                    };
                    let r = self.cell_radius * min_relative_distance_to_center;
                    (r, dir)
                }
            };
        let (ext_radius, species) = ext_info;
        // Introduce Non-dimensional length variable
        let sigma = r / (self.cell_radius + ext_radius);
        let bound = 4.0 + 1.0 / sigma;
        let spatial_cutoff = (1.0
            + (self.relative_interaction_range * (self.cell_radius + ext_radius) - r).signum())
            * 0.5;

        // Calculate the strength of the interaction with correct bounds
        let strength = self.potential_strength
            * ((1.0 / sigma).powf(2.0) - (1.0 / sigma).powf(4.0))
                .min(bound)
                .max(-bound);

        // Calculate only attracting and repelling forces
        let attracting_force = dir * strength.max(0.0) * spatial_cutoff;
        let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

        if *species == self.species {
            return Some(Ok(repelling_force + attracting_force));
        } else {
            return Some(Ok(repelling_force));
        }
    }

    fn get_interaction_information(&self) -> (f64, Species) {
        (self.cell_radius, self.species.clone())
    }
}

fn main() {
    // Define the seed
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let cells = (0..N_CELLS_1 + N_CELLS_2)
        .map(|n| {
            let pos = Vector3::from([
                rng.gen_range(0.0..DOMAIN_SIZE),
                rng.gen_range(0.0..DOMAIN_SIZE),
                rng.gen_range(0.0..DOMAIN_SIZE),
            ]);
            ModularCell {
                mechanics: NewtonDamped {
                    pos,
                    vel: Vector3::zero(),
                    damping_constant: CELL_DAMPENING,
                    mass: 1.0,
                },
                interaction: CellSpecificInteraction {
                    species: match n <= N_CELLS_1 {
                        true => Species::BlueCell,
                        false => Species::RedCell,
                    },
                    potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
                    relative_interaction_range: CELL_MECHANICS_RELATIVE_INTERACTION_RANGE,
                    cell_radius: CELL_RADIUS,
                },
                cycle: NoCycle {},
                interaction_extracellular: NoExtracellularGradientSensing,
                cellular_reactions: NoCellularreactions,
            }
        })
        .collect::<Vec<_>>();

    let domain = CartesianCuboid3::from_boundaries_and_interaction_ranges(
        [0.0; 3],
        [DOMAIN_SIZE; 3],
        [CELL_MECHANICS_RELATIVE_INTERACTION_RANGE * CELL_RADIUS * 2.0; 3],
    )
    .unwrap();

    let time = TimeSetup {
        t_start: 0.0,
        t_eval: (0..N_TIMES)
            .map(|n| (n as f64 * DT, n % SAVE_INTERVAL == 0))
            .collect(),
    };

    let meta_params = SimulationMetaParams {
        n_threads: N_THREADS,
    };

    let storage = StorageConfig::from_path("out/cell_sorting".into());

    let simulation_setup = SimulationSetup::new(domain, cells, time, meta_params, storage, ());

    let mut supervisor = SimulationSupervisor::initialize_from_setup(simulation_setup);

    supervisor.run_full_sim().unwrap();
}
