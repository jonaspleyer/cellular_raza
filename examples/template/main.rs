// Imports from the cellular_control crate
use cellular_control::concepts::mechanics::*;
use cellular_control::concepts::interaction::*;
use cellular_control::cell_properties::cell_model::*;
use cellular_control::domain::cartesian_cuboid::*;
use cellular_control::plotting::cells_2d::plot_current_cells_2d;

// Imports from other files for this example
mod setup;
use crate::setup::*;

// Standard library imports
use std::fs;
use std::io::Write;
use std::collections::HashMap;
use std::cmp::{min,max};
use std::time::{Instant};
use std::sync::atomic::{AtomicBool,Ordering};
use std::sync::Arc;

use hurdles::Barrier;

use itertools::Itertools;
// use nalgebra::Vector3;

use ode_integrate::prelude::*;


fn _generate_voxels(vox: &[usize; 3], n_vox: &[usize; 3]) -> Vec<[usize; 3]> {
    (0..3)
        .map(|i| (max(vox[i] as isize - 1, 0) as usize)..(min(vox[i] + 2, n_vox[i])))
        .multi_cartesian_product()
        .map(|w| [w[0], w[1], w[2]])
        .collect()
}


fn _rhs(
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
        let voxels = _generate_voxels(&vox, &[N_VOXEL_X, N_VOXEL_Y, N_VOXEL_Z]);

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


pub const MULT_SAVE: usize = 40;
pub const DT: f64 = 0.02;
pub const T_MAX: f64 = 50.0;


fn save_to_file(t: f64, cells: &Vec<CellModel>) {
    // Write new positions to new file
    let filename = format!("out/snapshot_{:08.2}.csv", t);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(&filename)
        .expect("Unable to open file");

    let data = "id,t,x0,x1,x2";
    if let Err(e) = writeln!(file, "{}", data) {
        eprintln!("Could not write to file: {}", e);
    }

    for cell in cells.iter() {
        let data = format!("{},{},{},{},{}", cell.id, t, cell.mechanics.pos()[0], cell.mechanics.pos()[1], cell.mechanics.pos()[2]);

        if let Err(e) = writeln!(file, "{}", data) {
            eprintln!("Could not write to file: {}", e);
        }
    }
}


use crossbeam_channel::*;


fn main() {
    // Vector contains all cells currently present in simulation
    let cells = setup::insert_cells();

    // Spatial domain with boundaries to keep cells inside
    let (domain, mut voxels) = setup::define_domain();

    // Define sender receiver for cells when plotting results
    let (sender_plots, receiver_plots) = unbounded::<CellModel>();
    let save_now = Arc::new(AtomicBool::new(false));

    for cell in cells {
        let index = domain.determine_voxel(&cell);
        voxels[index].cells.push(cell);
    }

    let n_threads = min(N_THREADS, voxels.len());

    let chunk_size = max(1, ((voxels.len() as f64) / (n_threads as f64)).floor() as usize);

    let mut vox_vec: Vec<Voxel> = voxels.into_raw_vec();
    let mut handlers = Vec::new();

    let barrier_between_threads = Barrier::new(n_threads);
    let mut barrier_start = Barrier::new(n_threads+1);
    let mut barrier_end = Barrier::new(n_threads+1);

    let stop = Arc::new(AtomicBool::new(false));

    for _ in 0..n_threads {
        // Tell threads to wait until started from outside
        let stop_new = Arc::clone(&stop);

        // Barriers to start/stop threads between iterations
        let mut barrier_start_new = barrier_start.clone();
        let mut barrier_end_new = barrier_end.clone();

        // Channel to send cells for plotting and file storage
        let sender_plots_new = sender_plots.clone();
        let save_now_new = Arc::clone(&save_now);

        // Distribute the voxels onto the threads
        let vox_chunk: Vec<Voxel> = vox_vec.drain(..chunk_size).collect();
        let mut vox_cont = VoxelContainer {
            voxels: vox_chunk,
            barrier: barrier_between_threads.clone(),
        };

        // Spawn threads for our voxel containers
        let handler = std::thread::spawn(move || {
            let mut total_iter = 0;
            let mut t;

            loop {
                t = total_iter as f64 * DT;
                barrier_start_new.wait();

                vox_cont.update(t, DT).unwrap();

                if save_now_new.load(Ordering::Relaxed) {
                    let mut all_cells: Vec<CellModel> = vox_cont.voxels.iter().map(|v| v.cells.clone()).flatten().collect();
                    for cell in all_cells.drain(..) {
                        // TODO do not unwrap but catch nicely
                        sender_plots_new.send(cell).unwrap();
                    }
                }
                total_iter += 1;

                if stop_new.load(Ordering::Relaxed) {
                    barrier_end_new.wait();
                    break;
                }

                barrier_end_new.wait();
            }
        });
        handlers.push(handler);
    }

    let now = Instant::now();
    let total_steps = (T_MAX / DT).ceil() as usize + 1;
    let n_digits = T_MAX.to_string().len() as usize;
    let n_digits_after_decimal = max(0, n_digits as i64 - (T_MAX / DT).round().to_string().len() as i64) as usize;

    println!("Running loop for {} steps.", total_steps);
    println!("┌─{:─<w1$}─┬─{:─<8}─┐", "", "", w1=n_digits+n_digits_after_decimal);
    println!("│{:^w1$}│ {:^8} │", "Sim", "Wall", w1=n_digits+n_digits_after_decimal+2);
    println!("├─{:─<w1$}─┼─{:─<8}─┤", "", "", w1=n_digits+n_digits_after_decimal);

    // Run a loop 
    for loop_index in 0..total_steps {
        // Define the current time
        let t = loop_index as f64 * DT;

        // Stop the simulation if we reached the last step
        if loop_index==total_steps-1 {
            stop.store(true, Ordering::Relaxed);
        }

        // Tell worker threads to send results for saving
        if loop_index % MULT_SAVE == 0 {
            println!("│ {:w1$.w2$} │ {:8.2} │ Saving to file", t, now.elapsed().as_millis() as f64/1000.0, w1=n_digits+n_digits_after_decimal, w2=n_digits_after_decimal);
            save_now.store(true, Ordering::Relaxed);
        }

        // Tell the threads to start
        barrier_start.wait();

        barrier_end.wait();

        if save_now.load(Ordering::Relaxed) {
            save_now.store(false, Ordering::Relaxed);
            // let all_cells: Vec<CellModel> = receiver_plots.try_iter().collect();
            // std::thread::spawn(move || {
            //     save_to_file(t, &all_cells);
            //     plot_current_cells_2d(t, all_cells, [DOMAIN_SIZE_X, DOMAIN_SIZE_Y], [N_VOXEL_X, N_VOXEL_Y], CELL_RADIUS).unwrap();
            // });
        }

        // Sort into voxels
        /* let mut voxel_index_to_cells = HashMap::<[usize; 3], Vec<&mut CellModel>>::new();
        for cell in cells.iter_mut() {
            let index = domain.determine_voxel(&cell);
            match voxel_index_to_cells.get_mut(&index) {
                Some(key) => key.push(cell),
                None => match voxel_index_to_cells.insert(index, vec![cell]) {
                    Some(_) => panic!("Test"),
                    None => (),
                },
            };
        }*/
        
        // Initialize values for solving with ode solver
        /* let mut pv_s: Vec<Vec<f64>> = voxel_index_to_cells.iter().map(|(_, b)| b).map(|voxel_cells| voxel_cells.iter().flat_map(|cell| {
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

            rk4.do_step_iter(pv, &t, &DT, &(&domain, &voxel_cells, &voxel_index_to_cells, &CELL_VELOCITY_REDUCTION)).unwrap();
        }
        
        // Set new positions for cells
        voxel_index_to_cells.iter_mut().zip(pv_s.iter()).for_each(|((_, voxel_cells), pv)| {
            voxel_cells.iter_mut().enumerate().for_each(|(i, cell)| {
                cell.mechanics.set_pos(&Vector3::<f64>::from([pv[6*i + 0], pv[6*i + 1], pv[6*i + 2]]));
                cell.mechanics.set_velocity(&Vector3::<f64>::from([pv[6*i + 3], pv[6*i + 4], pv[6*i + 5]]));
            });
        });
        */

        // Additionally plot a nice 2d picture of all cells
        // plot_current_cells(save_index, t, &cells, DOMAIN_SIZE, N_VOXEL, CELL_RADIUS);
    }

    for handler in handlers {
        handler.join().unwrap();
    }
    println!("└─{:─<w1$}─┴─{:─<8}─┘", "", "", w1=n_digits+n_digits_after_decimal);
    println!("Simulation done");
}
