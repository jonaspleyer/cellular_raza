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

use rand::distributions::Standard;
use rand::Rng;


pub fn insert_cells() -> Vec<CellModel> {
    let domain_size = 15.0;
    let velocity = 1.0;
    let radius = 1.5;

    let mut cells = Vec::new();

    for _ in 1..15 {
        let de_model = DeathModel { release_fluid: false, fluid_fraction: 0.0 };
        let in_model = LennardJones { epsilon: 0.1, sigma: radius/2.0f64.powf(1.0/6.0) };
        let me_model = MechanicsModel::from((&Vector3::<f64>::from([0.0, 0.0, 0.0]), &Vector3::<f64>::from([0.0, 0.0, 0.0])));
        let rn_model = Standard {};
        let fl_model = Flags { removal: false };

        let cy1 = CellCycle { lifetime: rand::thread_rng().gen_range(5.0..40.0) };
        let cy2 = CellCycle { lifetime: rand::thread_rng().gen_range(5.0..40.0) };
        let cy3 = CellCycle { lifetime: rand::thread_rng().gen_range(5.0..40.0) };
        let cy4 = CellCycle { lifetime: rand::thread_rng().gen_range(5.0..40.0) };
        let cy_model = CycleModel::from(&vec![cy1, cy2, cy3, cy4]);

        let mut cell = CellModel { mechanics: me_model, cell_cycle: cy_model, death_model: de_model, interaction: in_model, rng_model: rn_model, flags: fl_model };

        cell.mechanics.set_pos(&Vector3::<f64>::from([rand::thread_rng().gen_range(-domain_size..domain_size), rand::thread_rng().gen_range(-domain_size..domain_size), 0.0]));
        cell.mechanics.set_velocity(&Vector3::<f64>::from([rand::thread_rng().gen_range(-velocity..velocity), rand::thread_rng().gen_range(-velocity..velocity), 0.0]));
        cells.push(cell);
    }

    return cells;
}


pub fn define_domain() -> Cuboid {
    let size = 15.0;
    Cuboid {
        min: [-size, -size, -size],
        max: [size, size, size],
        rebound: 1.0,
    }
}
