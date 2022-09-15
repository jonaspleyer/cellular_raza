// Imports from the cellular_control crate
use cellular_control::concepts::domain::Domain;

use cellular_control::cell_properties::cell_model::*;
use cellular_control::cell_properties::spatial::Spatial;


// Imports from other files for this example
mod setup;

// Imports from other crates
use nalgebra::{Vector3};

use ode_integrate::prelude::*;

use std::fs;
use std::io::prelude::*;


fn rhs(y: &[f64], dy: &mut [f64], _t: &f64, p: &(&Vec<CellModel<[f64; 2]>>, &f64)) -> Result<(), CalcError> {
    let cells = &p.0;
    let lambda = &p.1;

    for (x1, x2) in y.chunks(6).zip(dy.chunks_mut(6)) {
        let x = Vector3::<f64>::from([x1[0], x1[1], x1[2]]);
        let v = Vector3::<f64>::from([x1[3], x1[4], x1[5]]);

        let mut dx = Vector3::<f64>::from([x2[0], x2[1], x2[2]]);
        let mut dv = Vector3::<f64>::from([x2[3], x2[4], x2[5]]);

        for cell in cells.iter() {
            if (cell.spatial.pos() - x).norm() != 0.0 {
                match (cell.interaction.potential)(&cell.spatial.pos(), &x, &cell.interaction.parameter) {
                    Ok(dv1) => dv += dv1,
                    Err(error) => return Err(CalcError::from(format!("Error occured in interaction calculation: {:?}", error))),
                }
            }
        }
        for (x2i, x2_new) in x2.iter_mut().zip(dx.iter().chain(dv.iter()).map(|x| *x)) {
            *x2i = x2_new;
        }
        
    }
    Ok(())
}


fn main() {
    // Vector contains all cells currently present in simulation
    let mut cells = setup::insert_cells();

    // Spatial domain with boundaries to keep cells inside
    let domain = setup::define_domain();

    // How much speed does the particle loose for each timestep?
    let lambda: f64 = 0.1;

    // Time step of simulation
    let save_step = 1;
    let mut t = 0.0;
    let t_max = 100.0;
    let dt = 0.02;

    for i in 0..cells.len() {
        let data = format!("x0,x1,x2\n");
        let filename = format!("out/cell_{:010.0}.csv", i);
        fs::write(filename, data).expect("Unable to open file");
    }

    while t < t_max {
        // Calculate new positions of cells
        let mut updates = vec![vec![Vector3::<f64>::from([0.0, 0.0, 0.0]); 0]; cells.len()];
        for (update, c1) in updates.iter_mut().zip(cells.iter()) {
            
            for c2 in cells.iter() {
                
                if (c1.spatial.pos() - c2.spatial.pos()).norm() != 0.0 {
                    let ddx1 = (c1.interaction.potential)(&c1.spatial.pos(), &c2.spatial.pos(), &c1.interaction.parameter);
                    let dx1 = dt * ddx1.unwrap();
                    update.push(dx1);
                }
            }
        }

        let t_series: Vec<f64> = (0..save_step+1).map(|x| t + x as f64 * dt).collect();

        rhs(&pos_and_speed, &mut dx, &t, &(&cells, &lambda));

        
        
        // Set new positions and speed of cells
        /*for (cell, update) in cells.iter_mut().zip(updates.iter()) {
            for up in update.iter() {
                cell.spatial.set_speed(cell.spatial.speed() + *up);
            }
            let mut new_pos = cell.spatial.pos() + dt * cell.spatial.speed();
            cell.spatial.set_speed(cell.spatial.speed() - dt * lambda * cell.spatial.speed());

            domain.apply_boundary(&cell.spatial.pos(), &mut new_pos, &mut cell.spatial.speed());
            cell.spatial.set_pos(new_pos);
        }
        for (i, cell) in cells.iter().enumerate() {
            let data = format!("{},{},{}", cell.spatial.pos()[0], cell.spatial.pos()[1], cell.spatial.pos()[2]);
            let filename = format!("out/cell_{:010.0}.csv", i);
            let mut file = fs::OpenOptions::new()
                .append(true)
                .open(filename)
                .expect("Unable to open file");
            if let Err(e) = writeln!(file, "{}", data) {
                eprintln!("Could not write to file: {}", e);
            }
        }*/
        t += save_step as f64 * dt;
    }
}
