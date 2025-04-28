use cellular_raza::prelude::*;

use nalgebra::Vector3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const N_CELLS_BLUE: usize = 600;
pub const N_CELLS_RED: usize = 1_000;

pub const CELL_DAMPING: f64 = 2.0;
pub const CELL_RADIUS: f64 = 6.0;
pub const CELL_KB_TEMP: f64 = 1.0;

pub const CELL_MECHANICS_CUTOFF: f64 = 20.0;
pub const CELL_MECHANICS_ATTRATION: f64 = 1.5;
pub const CELL_MECHANICS_REPULSION: f64 = 5.0;

pub const DT: f64 = 0.25;
pub const N_TIMES: u64 = 50_000;
pub const SAVE_INTERVAL: u64 = 1_000;

pub const N_THREADS: usize = 4;

pub const DOMAIN_SIZE: f64 = 200.0;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
enum Species {
    RedCell,
    BlueCell,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CellSpecificInteraction {
    species: Species,
    cell_radius: f64,
    attraction_rr: f64,
    attraction_bb: f64,
    attraction_rb: f64,
    repulsion: f64,
    cutoff: f64,
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
    ) -> Result<(Vector3<f64>, Vector3<f64>), CalcError> {
        let own_radius = self.cell_radius;
        let ext_radius = ext_info.0;
        let own_species = &self.species;
        let ext_species = &ext_info.1;

        let radius_combined = own_radius + ext_radius;

        let z = own_pos - ext_pos;
        let r = z.norm();
        let attraction = match (own_species, ext_species) {
            (Species::BlueCell, Species::RedCell) | (Species::RedCell, Species::BlueCell) => {
                self.attraction_rb
            }
            (Species::RedCell, Species::RedCell) => self.attraction_rr,
            (Species::BlueCell, Species::BlueCell) => self.attraction_bb,
        };

        // Make sure that no NaN values are calculated when the positions are matching
        let dir = if r == 0.0 {
            [0.0; 3].into()
        } else {
            z.normalize()
        };

        let force = if r <= radius_combined {
            self.repulsion * (radius_combined - r) * dir
        } else {
            if r <= self.cutoff {
                let sigma = r / radius_combined;
                -attraction * (r - radius_combined) * (self.cutoff - r) / self.cutoff
                    * (-1.0 * (sigma - 1.0).powf(2.0)).exp()
                    * dir
            } else {
                0.0 * dir
            }
        };
        Ok((-force, force))
    }

    fn get_interaction_information(&self) -> (f64, Species) {
        (self.cell_radius, self.species.clone())
    }
}

#[derive(CellAgent, Clone, Deserialize, Serialize)]
struct Cell {
    #[Interaction]
    interaction: CellSpecificInteraction,
    #[Mechanics]
    mechanics: Langevin3D,
}

fn main() -> Result<(), SimulationError> {
    // Define the seed
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let cells = (0..N_CELLS_BLUE + N_CELLS_RED)
        .map(|n| {
            let pos = [
                rng.random_range(0.0..DOMAIN_SIZE),
                rng.random_range(0.0..DOMAIN_SIZE),
                rng.random_range(0.0..DOMAIN_SIZE),
            ];
            Cell {
                mechanics: Langevin3D {
                    pos: pos.into(),
                    vel: [0.0; 3].into(),
                    mass: 10.0,
                    damping: CELL_DAMPING,
                    kb_temperature: CELL_KB_TEMP,
                },
                interaction: CellSpecificInteraction {
                    species: match n <= N_CELLS_BLUE {
                        true => Species::BlueCell,
                        false => Species::RedCell,
                    },
                    attraction_rr: CELL_MECHANICS_ATTRATION,
                    attraction_bb: 0.05 * CELL_MECHANICS_ATTRATION,
                    attraction_rb: 0.01 * CELL_MECHANICS_ATTRATION,
                    repulsion: CELL_MECHANICS_REPULSION,
                    cutoff: CELL_MECHANICS_CUTOFF,
                    cell_radius: CELL_RADIUS,
                },
            }
        })
        .collect::<Vec<_>>();

    let domain = CartesianCuboid::from_boundaries_and_interaction_range(
        [0.0; 3],
        [DOMAIN_SIZE; 3],
        CELL_MECHANICS_CUTOFF * 2.0,
    )?;

    let time = cellular_raza::core::time::FixedStepsize::from_partial_save_steps(
        0.0,
        DT,
        N_TIMES,
        SAVE_INTERVAL,
    )?;
    let storage_builder = StorageBuilder::new().location("out/cell_sorting");

    let settings = cellular_raza::core::backend::chili::Settings {
        n_threads: N_THREADS.try_into().unwrap(),
        time,
        storage: storage_builder,
        show_progressbar: true,
    };

    run_simulation!(
        domain: domain,
        agents: cells,
        settings: settings,
        aspects: [Mechanics, Interaction]
    )?;
    Ok(())
}
