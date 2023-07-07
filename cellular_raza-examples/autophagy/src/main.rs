use cellular_raza::prelude::*;

use nalgebra::Vector3;
use num::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const N_CELLS_CARGO: usize = 1;
pub const N_CELLS_R11: usize = 200;
pub const N_CELLS_ATG9: usize = 0;

pub const CELL_DAMPENING: f64 = 1.0;
pub const CELL_RADIUS_CARGO: f64 = 10.0;
pub const CELL_RADIUS_R11: f64 = 1.0;
pub const CELL_RADIUS_ATG9: f64 = 0.5;

pub const CELL_MECHANICS_INTERACTION_RANGE_CARGO: f64 = 5.0 * CELL_RADIUS_CARGO;
pub const CELL_MECHANICS_INTERACTION_RANGE_R11: f64 = 5.0 * CELL_RADIUS_R11;
pub const CELL_MECHANICS_INTERACTION_RANGE_ATG9: f64 = 2.0 * CELL_RADIUS_ATG9;

pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 2.0;
pub const CELL_MECHANICS_RELATIVE_CLUSTERING_STRENGTH: f64 = 0.03;

pub const DT: f64 = 0.02;
pub const N_TIMES: usize = 50_001;
pub const SAVE_INTERVAL: usize = 500;

pub const N_THREADS: usize = 4;

pub const DOMAIN_SIZE: f64 = 100.0;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
enum Species {
    Cargo,
    R11,
    ATG9,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CellSpecificInteraction {
    species: Species,
    cell_radius: f64,
    potential_strength: f64,
    interaction_range: f64,
    clustering_strength: f64,
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
        let spatial_cutoff = (1.0 + (self.interaction_range - r).signum()) * 0.5;

        // Calculate the strength of the interaction with correct bounds
        let strength = self.potential_strength
            * ((1.0 / sigma).powf(2.0) - (1.0 / sigma).powf(4.0))
                .min(bound)
                .max(-bound);

        // Calculate only attracting and repelling forces
        let attracting_force = dir * strength.max(0.0) * spatial_cutoff;
        let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

        match (species, &self.species) {
            // R11 will bind to cargo
            (Species::Cargo, Species::R11) => {
                return Some(Ok(repelling_force + attracting_force))
            }
            (Species::R11, Species::Cargo) => {
                return Some(Ok(repelling_force + attracting_force))
            }

            // R11 forms clusters
            (Species::R11, Species::R11) => {
                return Some(Ok(
                    repelling_force + self.clustering_strength * attracting_force
                ))
            }

            // ATG9 and R11 will bind
            (Species::R11, Species::ATG9) => Some(Ok(repelling_force + attracting_force)),
            (Species::ATG9, Species::R11) => Some(Ok(repelling_force + attracting_force)),

            // ATG9 forms clusters
            (Species::ATG9, Species::ATG9) => {
                return Some(Ok(
                    repelling_force + self.clustering_strength * attracting_force
                ))
            }

            (_, _) => return Some(Ok(repelling_force)),
        }
    }

    fn get_interaction_information(&self) -> (f64, Species) {
        (self.cell_radius, self.species.clone())
    }
}

fn main() -> Result<(), SimulationError> {
    // Define the seed
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let cells = (0..N_CELLS_CARGO + N_CELLS_R11 + N_CELLS_ATG9)
        .map(|n| {
            let pos = if n == 0 {
                Vector3::from([DOMAIN_SIZE / 2.0; 3])
            } else {
                Vector3::from([
                    rng.gen_range(0.0..DOMAIN_SIZE),
                    rng.gen_range(0.0..DOMAIN_SIZE),
                    rng.gen_range(0.0..DOMAIN_SIZE),
                ])
            };
            let vel = Vector3::zero();
            let (cell_radius, species, interaction_range) = if n < N_CELLS_CARGO {
                (
                    CELL_RADIUS_CARGO,
                    Species::Cargo,
                    CELL_MECHANICS_INTERACTION_RANGE_CARGO,
                )
            } else if n < N_CELLS_CARGO + N_CELLS_R11 {
                (
                    CELL_RADIUS_R11,
                    Species::R11,
                    CELL_MECHANICS_INTERACTION_RANGE_R11,
                )
            } else {
                (
                    CELL_RADIUS_ATG9,
                    Species::ATG9,
                    CELL_MECHANICS_INTERACTION_RANGE_ATG9,
                )
            };
            ModularCell {
                mechanics: MechanicsModel3D {
                    pos,
                    vel,
                    dampening_constant: CELL_DAMPENING,
                    mass: cell_radius,
                },
                interaction: CellSpecificInteraction {
                    species,
                    potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
                    interaction_range,
                    cell_radius,
                    clustering_strength: CELL_MECHANICS_RELATIVE_CLUSTERING_STRENGTH,
                },
                cycle: NoCycle {},
                interaction_extracellular: NoExtracellularGradientSensing {},
                cellular_reactions: NoCellularreactions {},
            }
        })
        .collect::<Vec<_>>();

    let domain =
        CartesianCuboid3::from_boundaries_and_n_voxels([0.0; 3], [DOMAIN_SIZE; 3], [2; 3])?;

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

    supervisor.run_full_sim()?;
    Ok(())
}
