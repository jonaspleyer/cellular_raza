// Imports from the cellular_control crate
use cellular_control::concepts::mechanics::*;
use cellular_control::concepts::interaction::*;
use cellular_control::concepts::cycle::*;
use cellular_control::concepts::domain::*;

use cellular_control::cell_properties::cycle::*;
use cellular_control::domain::cuboid::*;

// Imports from other files for this example
mod setup;
use crate::setup::*;

// Imports from other crates
use std::fs;
use std::io::Write;

use std::collections::HashMap;
use cellular_control::cell_properties::cell_model::*;

use ode_integrate::solvers::fixed_step::Rk4;
use ode_integrate::concepts::steppers::Stepper;
use ode_integrate::concepts::ode_def::OdeDefinition;
use ode_integrate::prelude::*;

use itertools::Itertools;
use std::cmp::{min,max};
use nalgebra::Vector3;
use std::time::{Instant};


fn generate_voxels(vox: &[usize; 3], n_vox: &[usize; 3]) -> Vec<[usize; 3]> {
    (0..3)
        .map(|i| (max(vox[i] as isize - 1, 0) as usize)..(min(vox[i] + 2, n_vox[i])))
        .multi_cartesian_product()
        .map(|w| [w[0], w[1], w[2]])
        .collect()
}


fn rhs(
    y: &Vec<f64>,
    dy: &mut Vec<f64>,
    _t: &f64,
    p: &(&Cuboid, &Vec<&mut CellModel>, &HashMap::<[usize; 3], Vec<&mut CellModel>>, &f64)
) -> Result<(), CalcError> {
    let domain = p.0;
    let voxel_cells = p.1;
    let voxel_index_to_cells = p.2;
    let lambda = p.3;

    for ((dyi, yi), cell0) in dy.chunks_mut(6).zip(y.chunks(6)).zip(voxel_cells) {
        // Set all values to zero
        dyi.iter_mut().for_each(|dyii| *dyii = 0.0);

        // Determine the voxel in which the current cells are living
        let vox = domain.determine_voxel(&cell0);

        // Get voxels with which the cells are interacting
        let voxels = generate_voxels(&vox, &[N_VOXEL; 3]);

        // Loop over all cells in all interaction voxels
        for voxel in voxels {
            match &voxel_index_to_cells.get(&voxel) {
                Some(cells) => cells.iter().for_each(|cell1| {
                    // Check that the position of the cells is not identical (ie. it is the same cell)
                    if cell1.mechanics.pos() != cell0.mechanics.pos() {
                        match cell1.interaction.potential(&cell0.mechanics.pos(), &cell1.mechanics.pos()) {
                            Ok(dv) => {
                                // Update velocities
                                dyi[3] += dv[0];
                                dyi[4] += dv[1];
                                dyi[5] += dv[2];
                            },
                            Err(_) => (),
                        }
                    }
                }),
                None => (),
            }
        }
        // Slow down particle movement
        dyi[3] -= lambda * yi[3];
        dyi[4] -= lambda * yi[4];
        dyi[5] -= lambda * yi[5];

        // Update positions
        dyi[0] += yi[3];
        dyi[1] += yi[4];
        dyi[2] += yi[5];
    }
    Ok(())
}


fn main() {
    // Vector contains all cells currently present in simulation
    let mut cells = setup::insert_cells();

    // Spatial domain with boundaries to keep cells inside
    let domain = setup::define_domain();

    // How much speed do the particles lose
    let lambda = 0.5;

    // Time step of simulation
    let save_step = 10;
    let mut t = 0.0;
    let dt = 0.1;
    let t_max = 1000.0;

    for cell in cells.iter() {
        let data = format!("t,x0,x1,x2\n{},{},{},{}\n", t, cell.mechanics.pos()[0], cell.mechanics.pos()[1], cell.mechanics.pos()[2]);
        let filename = format!("out/cell_{:010.0}.csv", cell.id);
        fs::write(filename, data).expect("Unable to open file");
    }

    let now = Instant::now();

    while t < t_max {
        for _ in 0..save_step {

            // Update the cycle status of cells
            cells.iter_mut().for_each(|cell| CycleModel::update(&dt, cell));
            
            // Sort into voxels
            let mut voxel_index_to_cells = HashMap::<[usize; 3], Vec<&mut CellModel>>::new();
            for cell in cells.iter_mut() {
                let index = domain.determine_voxel(&cell);
                match voxel_index_to_cells.get_mut(&index) {
                    Some(key) => key.push(cell),
                    None => match voxel_index_to_cells.insert(index, vec![cell]) {
                        Some(_) => panic!("Test"),
                        None => (),
                    },
                };
            }
            
            // Initialize values for solving with ode solver
            let mut pv_s: Vec<Vec<f64>> = voxel_index_to_cells.iter().map(|(_, b)| b).map(|voxel_cells| voxel_cells.iter().flat_map(|cell| {
                let p = cell.mechanics.pos();
                let v = cell.mechanics.velocity();
                vec![p[0], p[1], p[2], v[0], v[1], v[2]]
            }).collect()).collect();

            // Calculate new velocitites and positions by integrating with ode solver
            for ((_, voxel_cells), pv) in voxel_index_to_cells.iter().zip(pv_s.iter_mut()) {

                let ode_def = OdeDefinition {
                    y0: pv.clone(),
                    t0: t,
                    func: &rhs,
                };
                let mut rk4 = Rk4::from(ode_def);

                rk4.do_step_iter(pv, &t, &dt, &(&domain, &voxel_cells, &voxel_index_to_cells, &lambda)).unwrap();
            }
            
            // Set new positions for cells
            voxel_index_to_cells.iter_mut().zip(pv_s.iter()).for_each(|((_, voxel_cells), pv)| {
                voxel_cells.iter_mut().enumerate().for_each(|(i, cell)| {
                    cell.mechanics.set_pos(&Vector3::<f64>::from([pv[6*i + 0], pv[6*i + 1], pv[6*i + 2]]));
                    cell.mechanics.set_velocity(&Vector3::<f64>::from([pv[6*i + 3], pv[6*i + 4], pv[6*i + 5]]));
                });
            });

            // Apply boundary conditions
            cells.iter_mut().for_each(|cell| domain.apply_boundary(cell).unwrap());

            // Delete cells which are flagged for removal
            cells.retain(|cell| !cell.flags.removal);

            // Update time step
            t += dt;
        }
        // Write new positions to file
        for cell in cells.iter() {
            let data = format!("{},{},{},{}", t, cell.mechanics.pos()[0], cell.mechanics.pos()[1], cell.mechanics.pos()[2]);
            let filename = format!("out/cell_{:010.0}.csv", cell.id);
            let mut file = fs::OpenOptions::new()
                        .append(true)
                        .open(filename)
                        .expect("Unable to open file");
            if let Err(e) = writeln!(file, "{}", data) {
                        eprintln!("Could not write to file: {}", e);
            }
        }
        println!("t={:10.4} Saving", t);
    }
    println!("Elapsed time: {:10.2}s", now.elapsed().as_secs());
}
