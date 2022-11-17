// Imports from the cellular_control crate
use cellular_control::cell_properties::cell_model::*;
use cellular_control::cell_properties::cycle::*;
use cellular_control::cell_properties::death::*;
use cellular_control::cell_properties::interaction::*;
use cellular_control::cell_properties::mechanics::*;
use cellular_control::cell_properties::flags::*;

use cellular_control::domain::cuboid::*;

use cellular_control::concepts::mechanics::*;

// Imports from other crates
use nalgebra::Vector3;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use std::collections::HashMap;

use ndarray::Array3;

// Domain properties
pub const N_VOXEL: usize = 20;
pub const DOMAIN_SIZE: f64 = 100.0;

// Cell properties
pub const CELL_RADIUS: f64 = 3.0;
pub const CELL_LENNARD_JONES_STRENGTH: f64 = 0.1;
pub const CELL_INITIAL_VELOCITY: f64 = 100.0;
pub const CELL_CYCLE_LIFETIME1_LOW:  f64 = 0.0;
pub const CELL_CYCLE_LIFETIME1_HIGH: f64 = 1.0;
pub const CELL_CYCLE_LIFETIME2_LOW:  f64 = 0.0;
pub const CELL_CYCLE_LIFETIME2_HIGH: f64 = 1.0;
pub const CELL_CYCLE_LIFETIME3_LOW:  f64 = 0.0;
pub const CELL_CYCLE_LIFETIME3_HIGH: f64 = 1.0;
pub const CELL_CYCLE_LIFETIME4_LOW:  f64 = 1600.0;
pub const CELL_CYCLE_LIFETIME4_HIGH: f64 = 2400.0;
pub const CELL_VELOCITY_REDUCTION: f64 = 1.0;

// Number of cells initially in simulation
pub const N_CELLS: u32 = 1500;


pub fn insert_cells() -> Vec<CellModel> {
    let mut cells = Vec::new();

    for n in 0..N_CELLS {
        let mut rng = ChaCha8Rng::seed_from_u64(n.into());

        let de_model = DeathModel {};
        let in_model = LennardJones { epsilon: CELL_LENNARD_JONES_STRENGTH, sigma: CELL_RADIUS/2.0f64.powf(1.0/6.0) };
        let me_model = MechanicsModel::from((&Vector3::<f64>::from([0.0, 0.0, 0.0]), &Vector3::<f64>::from([0.0, 0.0, 0.0])));
        let fl_model = Flags { removal: false };

        let cy1 = CellCycle { lifetime: rng.gen_range(CELL_CYCLE_LIFETIME1_LOW..CELL_CYCLE_LIFETIME1_HIGH) };
        let cy2 = CellCycle { lifetime: rng.gen_range(CELL_CYCLE_LIFETIME2_LOW..CELL_CYCLE_LIFETIME2_HIGH) };
        let cy3 = CellCycle { lifetime: rng.gen_range(CELL_CYCLE_LIFETIME3_LOW..CELL_CYCLE_LIFETIME3_HIGH) };
        let cy4 = CellCycle { lifetime: rng.gen_range(CELL_CYCLE_LIFETIME4_LOW..CELL_CYCLE_LIFETIME4_HIGH) };
        let cy_model = CycleModel::from(&vec![cy1, cy2, cy3, cy4]);

        let mut cell = CellModel { mechanics: me_model, cell_cycle: cy_model, death_model: de_model, interaction: in_model, flags: fl_model, id: n };

        cell.mechanics.set_pos(&Vector3::<f64>::from([rng.gen_range(-DOMAIN_SIZE..DOMAIN_SIZE), rng.gen_range(-DOMAIN_SIZE..DOMAIN_SIZE), 0.0]));
        cell.mechanics.set_velocity(&Vector3::<f64>::from([rng.gen_range(-CELL_INITIAL_VELOCITY..CELL_INITIAL_VELOCITY), rng.gen_range(-CELL_INITIAL_VELOCITY..CELL_INITIAL_VELOCITY), 0.0]));
        cells.push(cell);
    }

    return cells;
}


pub fn define_domain() -> Cuboid<'static> {

    // Create a voxel
    let step = 2.0 * DOMAIN_SIZE/N_VOXEL as f64;
    let voxel_sizes = [step, step, step];

    let min = [-DOMAIN_SIZE, -DOMAIN_SIZE, -DOMAIN_SIZE];
    let max = [-DOMAIN_SIZE + step, -DOMAIN_SIZE + step, -DOMAIN_SIZE + step];
    let voxel = Voxel { min: min, max: max, cells: Vec::<&CellModel>::new(), };

    let mut voxels = Array3::<Voxel>::from_elem((N_VOXEL, N_VOXEL, N_VOXEL), voxel);
    Cuboid {
        min: [-DOMAIN_SIZE, -DOMAIN_SIZE, -DOMAIN_SIZE],
        max: [DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_SIZE],
        voxel_sizes: voxel_sizes,
        voxels: voxels,
    }
}
