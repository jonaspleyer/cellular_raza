// Imports from other crates
use uuid::Uuid;
use nalgebra::Vector2;

use std::collections::HashMap;

// Imports from this crate
use cellular_control::prelude::*;

// Use the cell definition of the other file
mod cell;
use crate::cell::*;


// Constants of the simulation
pub const N_CELLS: u32 = 500;

pub const CELL_CYCLE_LIFETIME_LOW: f64 = 1000.0;
pub const CELL_CYCLE_LIFETIME_HIGH: f64 = 1200.0;

pub const CELL_RADIUS: f64 = 3.0;
pub const CELL_LENNARD_JONES_STRENGTH: f64 = 2.0;

pub const CELL_VELOCITY_REDUCTION: f64 = 1.0;

// Parameters for domain
pub const N_VOXEL_X: usize = 11;
pub const N_VOXEL_Y: usize = 3;
pub const DOMAIN_SIZE_X: f64 = 200.0;
pub const DOMAIN_SIZE_Y: f64 = 200.0;
pub const DX_VOXEL_X: f64 = 2.0 * DOMAIN_SIZE_X / N_VOXEL_X as f64;
pub const DX_VOXEL_Y: f64 = 2.0 * DOMAIN_SIZE_Y / N_VOXEL_Y as f64;


// Meta Parameters to control solving
pub const N_THREADS: usize = 4;


pub fn create_cells() -> Vec<StandardCell2D> {
    let mut cells = Vec::<StandardCell2D>::with_capacity(N_CELLS as usize);

    for k in 0..N_CELLS {
        let cell = StandardCell2D {
            pos: Vector2::<f64>::from([
                DX_VOXEL_X / 2.0 - DOMAIN_SIZE_X + 2.0 * k as f64 / N_CELLS as f64 * DOMAIN_SIZE_X,
                DX_VOXEL_Y / 2.0 - DOMAIN_SIZE_Y + 2.0 * k as f64 / N_CELLS as f64 * DOMAIN_SIZE_Y]),
            velocity: Vector2::<f64>::from([0.0, 0.0]),

            cell_radius: 3.0,
            potential_strength: 2.0,

            remove: false,
            age: 0.0,

            id: Uuid::from_u128(k as u128),
        };

        cells.push(cell);
    }
    
    cells
}


fn main() {
    let cells = create_cells();

    let domain = CartesianCuboid2 {
        min: [-DOMAIN_SIZE_X, -DOMAIN_SIZE_Y],
        max: [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
        n_vox: [N_VOXEL_X, N_VOXEL_Y],
        voxel_sizes: [DX_VOXEL_X, DX_VOXEL_Y],
    };

    let (n_threads, voxel_chunks) = <CartesianCuboid2 as Domain<StandardCell2D, [usize; 2], CartesianCuboidVoxel2>>::distribute_voxels_for_threads(&domain, N_THREADS).unwrap();

    // TODO use this to build the threads automatically with the implemented method
    // TODO also find good interface for main function
    let mut handles = Vec::new();
    for (l, chunk) in voxel_chunks.into_iter().enumerate() {
        let handle = std::thread::spawn(move || {
            println!("Thread {} executing", l);
            for (ind, vox) in chunk.iter() {
                println!("{:?}", ind);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join();
    }

    for cell in &cells[0..3] {
        let index = domain.get_voxel_index(cell);
        println!("Cell: {:?} with pos {:?} has index {:?}", cell.get_uuid(), cell.pos(), index);
    }
}
