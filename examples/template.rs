// Imports from other crates
use uuid::Uuid;
use nalgebra::Vector2;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// Imports from this crate
use cellular_raza::prelude::*;


// Constants of the simulation
pub const N_CELLS: u32 = 200_000;

pub const CELL_CYCLE_LIFETIME_LOW: f64 = 1000.0;
pub const CELL_CYCLE_LIFETIME_HIGH: f64 = 1200.0;

pub const CELL_RADIUS: f64 = 3.0;
pub const CELL_LENNARD_JONES_STRENGTH: f64 = 0.5;

pub const CELL_INITIAL_VELOCITY: f64 = 0.2;

pub const CELL_VELOCITY_REDUCTION: f64 = 1.0;

// Parameters for domain
pub const N_VOXEL_X: usize = 100;
pub const N_VOXEL_Y: usize = 100;
pub const DOMAIN_SIZE_X: f64 = 16000.0;
pub const DOMAIN_SIZE_Y: f64 = 16000.0;
pub const DX_VOXEL_X: f64 = 2.0 * DOMAIN_SIZE_X / N_VOXEL_X as f64;
pub const DX_VOXEL_Y: f64 = 2.0 * DOMAIN_SIZE_Y / N_VOXEL_Y as f64;

// Time parameters
pub const N_TIMES: usize = 50;
pub const DT: f64 = 0.01;
pub const T_START: f64 = 0.0;


// Meta Parameters to control solving
pub const N_THREADS: usize = 1;


pub fn create_cells() -> Vec<StandardCell2D> {
    // Seed the rng
    let mut rng = ChaCha8Rng::seed_from_u64(2022);
    let mut cells = Vec::<StandardCell2D>::with_capacity(N_CELLS as usize);

    for k in 0..N_CELLS {
        let cell = StandardCell2D {
            pos: Vector2::<f64>::from([
                rng.gen_range(-DOMAIN_SIZE_X..DOMAIN_SIZE_X),
                rng.gen_range(-DOMAIN_SIZE_Y..DOMAIN_SIZE_Y)]),
            velocity: Vector2::<f64>::from([
                rng.gen_range(-CELL_INITIAL_VELOCITY..CELL_INITIAL_VELOCITY),
                rng.gen_range(-CELL_INITIAL_VELOCITY..CELL_INITIAL_VELOCITY)]),

            cell_radius: 3.0,
            potential_strength: 0.01,

            maximum_age: 1000.0,

            remove: false,
            current_age: 0.0,

            id: Uuid::from_u128(k as u128),
        };

        cells.push(cell);
    }
    
    cells
}


fn main() {
    let averaging_runs = 20;

    let mut descr = vec![[0_u128, 1_u128, 2_u128, 3_u128]];
    let mut times: Vec<_> = (1..18).map(|n_threads| {
        (0..averaging_runs).map(move |_| {
    
            let start = std::time::Instant::now();
            let cells = create_cells();
            let cell_create_time = start.elapsed().as_millis();

            let domain = CartesianCuboid2 {
                min: [-DOMAIN_SIZE_X, -DOMAIN_SIZE_Y],
                max: [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
                n_vox: [N_VOXEL_X, N_VOXEL_Y],
                voxel_sizes: [DX_VOXEL_X, DX_VOXEL_Y],
            };

            // TODO use this to build the threads automatically with the implemented method
            // TODO also find good interface for main function
            let setup = SimulationSetup {
                domain: domain,
                cells: cells,
                time: TimeSetup {
                    t_start: 0.0,
                    t_eval: (0..N_TIMES).map(|i| (T_START + DT * i as f64, true)).collect::<Vec<(f64, bool)>>(),
                },
                meta_params: SimulationMetaParams {
                    n_threads: n_threads
                }
            };

            let mut supervisor = Result::<SimulationSupervisor::<CartesianCuboid2, StandardCell2D, [usize; 2], Vector2<f64>, Vector2<f64>, Vector2<f64>, CartesianCuboidVoxel2>, Box<dyn std::error::Error>>::from(setup).unwrap();
            let supervisor_create_time = start.elapsed().as_millis();

            match supervisor.run_full_sim() {
                Ok(()) => (),
                Err(error) => println!("{error}"),
            }

            supervisor.end_simulation();
            let total_sim_time = start.elapsed().as_millis();
            println!("======= [Threads {:2.0}] =======", n_threads);
            println!("[x] Creating cells        {} ms", cell_create_time);
            println!("[x] Creating supervisor   {} ms", supervisor_create_time);
            println!("[x] Simulation            {} ms", total_sim_time);

            [n_threads as u128, cell_create_time, supervisor_create_time, total_sim_time]
        })
    }).flatten().collect();

    descr.append(&mut times);

    let data = format!("{:?}\n", descr).replace("], ", "\n").replace("[", "").replace("]]", "");
    std::fs::write("output.txt", data).expect("Unable to write file");
}
