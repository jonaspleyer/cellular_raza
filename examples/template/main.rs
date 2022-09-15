// Imports from the cellular_control crate
use cellular_control::concepts::mechanics::*;
use cellular_control::concepts::interaction::*;
use cellular_control::concepts::domain::*;
use cellular_control::concepts::cycle::*;

use cellular_control::cell_properties::cycle::*;


// Imports from other files for this example
mod setup;

// Imports from other crates
use std::fs;
use std::io::Write;
use itertools::Itertools;
use nalgebra::Vector3;
use rayon::prelude::*;


fn main() {
    // Vector contains all cells currently present in simulation
    let mut cells = setup::insert_cells();

    // Spatial domain with boundaries to keep cells inside
    let domain = setup::define_domain();

    // How much speed do the particles lose
    let lambda = 0.03;

    // Time step of simulation
    let save_step = 5;
    let mut t = 0.0;
    let dt = 0.02;
    let t_max = 7500 as f64 * dt;

    for i in 0..cells.len() {
        let data = format!("t,x0,x1,x2\n");
        let filename = format!("out/cell_{:010.0}.csv", i);
        fs::write(filename, data).expect("Unable to open file");
    }

    while t < t_max {
        for _ in 0..save_step {
            // Update the cycle status of cells
            // cells.iter_mut().for_each(|cell| CellCycle::update(&dt, &mut cell));
            cells.iter_mut().for_each(|cell| 
                CycleModel::update(&dt, cell)
            );

            // Calculate interaction forces
            let mut velocities: Vec<Vector3<f64>> = cells.iter().map(|cell| cell.mechanics.velocity()).collect();
            for c in cells.iter().enumerate().combinations(2) {
                let (i, cell0) = c[0];
                let (j, cell1) = c[1];
                
                if cell0.mechanics.pos() != cell1.mechanics.pos() {
                    match cell1.interaction.potential(&cell0.mechanics.pos(), &cell1.mechanics.pos()) {
                        Ok(dv) => {
                            velocities[i] += dt * dv;
                            velocities[j] -= dt * dv;
                        },
                        Err(_) => panic!("Calculation failed!"),
                    }
                }
            }

            // Update positions and velocity
            for (cell, velocity) in cells.iter_mut().zip(velocities) {
                cell.mechanics.add_pos(&(dt * velocity));
                cell.mechanics.set_velocity(&(velocity - dt * lambda * cell.mechanics.velocity()));
            }

            // Apply boundary conditions
            cells.iter_mut().for_each(|cell|
                domain.apply_boundary(cell)
            );

            // Delete cells which are flagged for removal
            cells.retain(|cell| !cell.flags.removal);

            // Update time step
            t += dt;
        }
        // Write new positions to file
        for (i, cell) in cells.iter().enumerate() {
            let data = format!("{},{},{},{}", t, cell.mechanics.pos()[0], cell.mechanics.pos()[1], cell.mechanics.pos()[2]);
            let filename = format!("out/cell_{:010.0}.csv", i);
            let mut file = fs::OpenOptions::new()
                        .append(true)
                        .open(filename)
                        .expect("Unable to open file");
            if let Err(e) = writeln!(file, "{}", data) {
                        eprintln!("Could not write to file: {}", e);
            }
        }
    }
}
