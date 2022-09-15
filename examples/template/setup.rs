use cellular_control::concepts::errors::*;

use cellular_control::cell_properties::cell_model::*;
use cellular_control::cell_properties::cycle::*;
use cellular_control::cell_properties::death::*;
use cellular_control::cell_properties::interaction::*;
use cellular_control::cell_properties::spatial::*;

use cellular_control::domain::cuboid::*;


use nalgebra::Vector3;

use rand::distributions::Standard;
use rand::Rng;


fn lennard_jones_force(x1: &Vector3<f64>, x2: &Vector3<f64>, p: &[f64; 2]) -> Result<Vector3<f64>, CalcError> {
    let r = (x1 - x2).norm();
    let x = x1 - x2;
    Ok(x/r * 4.0 * p[0] / r * (12.0 * (p[1]/r).powf(12.0) - 6.0 * (p[1]/r).powf(6.0)))
}


pub fn insert_cells() -> Vec<CellModel<'static, [f64; 2]>> {
    let cy1 = CellCycle { lifetime: 20.0 };
    let cy2 = CellCycle { lifetime: 30.0 };
    let cy3 = CellCycle { lifetime: 30.0 };
    let cy4 = CellCycle { lifetime: 100.0 };

    let x_pos = 6.0;
    let y_pos = 6.0;
    let x_speed = 0.5;
    let y_speed = -0.6;

    let radius = 1.5;

    let cy_model = CycleModel { cycle1: cy1, cycle2: cy2, cycle3: cy3, cycle4: cy4 };
    let de_model = DeathModel { release_fluid: false, fluid_fraction: 0.0 };
    let in_model = InteractionModel { potential: &lennard_jones_force, parameter: [1.0, radius/2.0f64.powf(1.0/6.0)] };
    let sp_model = SpatialModel::from((&Vector3::<f64>::from([0.0, -y_pos, 0.0]), &Vector3::<f64>::from([0.0, y_speed, 0.0])));
    let rn_model = Standard {};

    let cell1 = CellModel { spatial: sp_model, cell_cycle: cy_model, death_model: de_model, interaction: in_model, rng_model: rn_model };
    let mut cell2 = cell1.clone();
    cell2.spatial.set_pos(Vector3::<f64>::from([0.0, y_pos, 0.0]));
    cell2.spatial.set_speed(Vector3::<f64>::from([0.0, -y_speed, 0.0]));
    
    let mut cell3 = cell1.clone();
    cell3.spatial.set_pos(Vector3::<f64>::from([x_pos, 0.0, 0.0]));
    cell3.spatial.set_speed(Vector3::<f64>::from([-x_speed, 0.0, 0.0]));

    let mut cell4 = cell1.clone();
    cell4.spatial.set_pos(Vector3::<f64>::from([-x_pos, 0.0, 0.0]));
    cell4.spatial.set_speed(Vector3::<f64>::from([x_speed, 0.0, 0.0]));

    let mut cells = vec![
        cell1,
        cell2,
        cell3,
        cell4
    ];

    for _ in 0..5 {
        let mut c = cells[0].clone();
        c.spatial.set_pos(Vector3::<f64>::from([rand::thread_rng().gen_range(-6.0..6.0), rand::thread_rng().gen_range(-6.0..6.0), 0.0]));
        cells.push(c);
    }

    return cells;
}


pub fn define_domain() -> Cuboid {
    let size = 6.0;
    Cuboid {
        min: [-size, -size, -size],
        max: [size, size, size],
        rebound: 1.0,
    }
}
