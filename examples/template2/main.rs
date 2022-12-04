// Imports from other crates

// Imports from this crate
use cellular_control::prelude::*;

// Use the cell definition of the other file
mod cell;
mod domain;
use crate::cell::*;
use crate::domain::*;


// Constants of the simulation
pub const N_CELLS: u32 = 1000;

pub const CELL_CYCLE_LIFETIME_LOW: f64 = 1000.0;
pub const CELL_CYCLE_LIFETIME_HIGH: f64 = 1200.0;

pub const CELL_RADIUS: f64 = 3.0;
pub const CELL_LENNARD_JONES_STRENGTH: f64 = 2.0;

pub const CELL_VELOCITY_REDUCTION: f64 = 1.0;

pub const DOMAIN_SIZE_X: f64 = 200.0;
pub const DOMAIN_SIZE_Y: f64 = 200.0;
pub const DOMAIN_SIZE_Z: f64 = 10.0;


fn main() {
    let cells = insert_cells();

    println!("Cells : {:?}", cells.len());
}
