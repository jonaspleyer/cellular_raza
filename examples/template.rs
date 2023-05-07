// Imports from other crates
use nalgebra::Vector2;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// Imports from this crate
use cellular_raza::backend::cpu_os_threads::prelude::*;


// Constants of the simulation
pub const N_CELLS: u32 = 20_000;

pub const CELL_CYCLE_LIFETIME_LOW: f64 = 1000.0;
pub const CELL_CYCLE_LIFETIME_HIGH: f64 = 1200.0;

pub const CELL_RADIUS: f64 = 3.0;
pub const CELL_LENNARD_JONES_STRENGTH: f64 = 0.5;

pub const CELL_INITIAL_VELOCITY: f64 = 0.2;

pub const CELL_VELOCITY_REDUCTION: f64 = 1.0;

// Parameters for domain
pub const N_VOXEL_X: usize = 30;
pub const N_VOXEL_Y: usize = 30;
pub const DOMAIN_SIZE_X: f64 = 1600.0;
pub const DOMAIN_SIZE_Y: f64 = 1600.0;
pub const DX_VOXEL_X: f64 = 2.0 * DOMAIN_SIZE_X / N_VOXEL_X as f64;
pub const DX_VOXEL_Y: f64 = 2.0 * DOMAIN_SIZE_Y / N_VOXEL_Y as f64;

// Time parameters
pub const N_TIMES: usize = 50;
pub const DT: f64 = 0.01;
pub const T_START: f64 = 0.0;


// Meta Parameters to control solving
pub const N_THREADS: usize = 1;

// Runs over which to calculate the average
pub const AVERAGING_RUNS: usize = 5;


pub fn create_cells() -> Vec<StandardCell2D> {
    // Seed the rng
    let mut rng = ChaCha8Rng::seed_from_u64(2022);
    let mut cells = Vec::<StandardCell2D>::with_capacity(N_CELLS as usize);

    for _ in 0..N_CELLS {
        let cell = StandardCell2D {
            pos: Vector2::<f64>::from([
                rng.gen_range(-DOMAIN_SIZE_X..DOMAIN_SIZE_X),
                rng.gen_range(-DOMAIN_SIZE_Y..DOMAIN_SIZE_Y)]),
            velocity: Vector2::<f64>::from([
                rng.gen_range(-CELL_INITIAL_VELOCITY..CELL_INITIAL_VELOCITY),
                rng.gen_range(-CELL_INITIAL_VELOCITY..CELL_INITIAL_VELOCITY)]),

            cell_radius: 3.0,
            potential_strength: 0.01,

            velocity_reduction: 0.0,

            maximum_age: 1000.0,

            remove: false,
            current_age: 0.0,
        };

        cells.push(cell);
    }
    
    cells
}


fn main() {
    let mut descr = vec![[0_u128, 1_u128, 2_u128, 3_u128]];
    let mut times: Vec<_> = (1..20).map(|n_threads| {
        (0..AVERAGING_RUNS).map(move |k| {
    
            let start = std::time::Instant::now();
            let cells = create_cells();
            let cell_create_time = start.elapsed().as_millis();

            let domain = CartesianCuboid2::from_boundaries_and_n_voxels(
                [-DOMAIN_SIZE_X, -DOMAIN_SIZE_Y],
                [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
                [N_VOXEL_X, N_VOXEL_Y]
            ).unwrap();

            let setup = SimulationSetup {
                domain,
                cells,
                time: TimeSetup {
                    t_start: 0.0,
                    t_eval: (0..N_TIMES).map(|i| (T_START + DT * i as f64, true, true)).collect::<Vec<(f64, bool, bool)>>(),
                },
                meta_params: SimulationMetaParams {
                    n_threads
                },
                database: SledDataBaseConfig {
                    name: format!("out/template_sim_{}", n_threads*AVERAGING_RUNS + k).into(),
                }
            };

            let mut supervisor = SimulationSupervisor::initialize_with_strategies(setup, Strategies {voxel_definition_strategies: None});
            let supervisor_create_time = start.elapsed().as_millis();

            match supervisor.run_full_sim() {
                Ok(()) => (),
                Err(error) => println!("{error}"),
            }

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
