use cellular_raza::prelude::*;

use nalgebra::Vector2;
use num::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const N_CELLS_CARGO: usize = 10;
pub const N_CELLS_R11: usize = 100;
// pub const N_CELLS_Vac8: usize = 30;

pub const CELL_DAMPENING: f64 = 2.0;
pub const CARGO_CELL_RADIUS: f64 = 5.0;
pub const R11_CELL_RADIUS: f64 = 1.0;

pub const CELL_MECHANICS_RELATIVE_INTERACTION_RANGE: f64 = 1.5;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 2.0;

pub const DT: f64 = 0.25;
pub const N_TIMES: usize = 3_000;
pub const SAVE_INTERVAL: usize = 10;

pub const N_THREADS: usize = 4;

pub const DOMAIN_SIZE: f64 = 110.0;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
enum Species {
    Cargo,
    R11,
    // Vac8,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CellSpecificInteraction {
    species: Species,
    cell_radius: f64,
    potential_strength: f64,
    relative_interaction_range: f64,
}

impl Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, (f64, Species)>
    for CellSpecificInteraction
{
    fn calculate_force_between(
        &self,
        own_pos: &Vector2<f64>,
        _own_vel: &Vector2<f64>,
        ext_pos: &Vector2<f64>,
        _ext_vel: &Vector2<f64>,
        ext_info: &Option<(f64, Species)>,
    ) -> Option<Result<Vector2<f64>, CalcError>> {
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
        match ext_info {
            Some((ext_radius, species)) => {
                // Introduce Non-dimensional length variable
                let sigma = r / (self.cell_radius + ext_radius);
                let bound = 4.0 + 1.0 / sigma;
                let spatial_cutoff = (1.0
                    + (self.relative_interaction_range * (self.cell_radius + ext_radius) - r)
                        .signum())
                    * 0.5;

                // Calculate the strength of the interaction with correct bounds
                let strength = self.potential_strength
                    * ((1.0 / sigma).powf(2.0) - (1.0 / sigma).powf(4.0))
                        .min(bound)
                        .max(-bound);

                // Calculate only attracting and repelling forces
                let attracting_force = dir * strength.max(0.0) * spatial_cutoff;
                let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

                if *species != self.species {
                    return Some(Ok(repelling_force + attracting_force));
                } else {
                    return Some(Ok(repelling_force));
                }
            }
            None => None,
        }
    }

    fn get_interaction_information(&self) -> Option<(f64, Species)> {
        Some((self.cell_radius, self.species.clone()))
    }
}

fn main() {
    // Define the seed
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let cells = (0..N_CELLS_CARGO + N_CELLS_R11)
        .map(|n| {
            let pos = Vector2::from([
                rng.gen_range(0.0..DOMAIN_SIZE),
                rng.gen_range(0.0..DOMAIN_SIZE),
                // rng.gen_range(0.0..DOMAIN_SIZE),
            ]);
            ModularCell {
                mechanics: MechanicsModel2D {
                    pos,
                    vel: Vector2::zero(),
                    dampening_constant: CELL_DAMPENING,
                    mass: cell_radius,
                },
                interaction: CellSpecificInteraction {
                    species: match n <= N_CELLS_CARGO {
                        true => Species::Cargo,
                        false => Species::R11,
                    },
                    potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
                    relative_interaction_range: CELL_MECHANICS_RELATIVE_INTERACTION_RANGE,
                    cell_radius: if n <= N_CELLS_CARGO {
                        CARGO_CELL_RADIUS
                    } else {
                        R11_CELL_RADIUS
                    },
                },
                cycle: NoCycle {},
                interaction_extracellular: NoExtracellularGradientSensing {},
                cellular_reactions: NoCellularreactions {},
            }
        })
        .collect::<Vec<_>>();

    let domain = CartesianCuboid2::from_boundaries_and_interaction_ranges(
        [0.0; 2],
        [DOMAIN_SIZE; 2],
        [CELL_MECHANICS_RELATIVE_INTERACTION_RANGE * CARGO_CELL_RADIUS * 2.0; 2],
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

    let storage = StorageConfig::from_path("out/autophagy".into());
    // storage.export_formats = vec![ExportOptions::Vtk];

    let simulation_setup = create_simulation_setup!(
        Domain: domain,
        Cells: cells,
        Time: time,
        MetaParams: meta_params,
        Storage: storage
    );

    let mut supervisor = SimulationSupervisor::initialize_from_setup(simulation_setup);

    supervisor.run_full_sim().unwrap();
}
