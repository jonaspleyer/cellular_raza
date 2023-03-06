use cellular_raza::pipelines::cpu_os_threads::prelude::*;
use cellular_raza::*;

use nalgebra::Vector2;

use rand::Rng;


// Constants of the simulation
pub const N_CELLS: u32 = 200;

pub const CELL_MAXIMUM_AGE: f64 = 1000.0;

pub const CELL_RADIUS: f64 = 3.0;
pub const CELL_LENNARD_JONES_STRENGTH: f64 = 3.0;
pub const CELL_INITIAL_VELOCITY: f64 = 0.2;
pub const CELL_VELOCITY_REDUCTION: f64 = 1.0;

// Parameters for domain
pub const N_VOXEL_X: usize = 10;
pub const N_VOXEL_Y: usize = 10;
pub const DOMAIN_SIZE_X: f64 = 100.0;
pub const DOMAIN_SIZE_Y: f64 = 100.0;
pub const DX_VOXEL_X: f64 = 2.0 * DOMAIN_SIZE_X / N_VOXEL_X as f64;
pub const DX_VOXEL_Y: f64 = 2.0 * DOMAIN_SIZE_Y / N_VOXEL_Y as f64;

// Time parameters
pub const N_TIMES: usize = 1001;
pub const DT: f64 = 0.01;
pub const T_START: f64 = 0.0;


// Meta Parameters to control solving
pub const N_THREADS: usize = 4;


define_simulation_types!(
    Position:   Vector2<f64>,
    Force:      Vector2<f64>,
    Information:(),
    Velocity:   Vector2<f64>,
    CellTypes:  [StandardCell2D, CustomCell2D],
    Voxel:      CartesianCuboidVoxel2,
    Index:      [usize; 2],
    Domain:     CartesianCuboid2,
);


fn main() {
    // Define seed for initial random config
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // Define all cells in the simulation domain
    let mut cells: Vec<_> = (0..N_CELLS).map(|_| CellAgentType::StandardCell2D(StandardCell2D {
        pos: Vector2::<f64>::from([
            rng.gen_range(-DOMAIN_SIZE_X..DOMAIN_SIZE_X),
            rng.gen_range(-DOMAIN_SIZE_Y..DOMAIN_SIZE_Y)]),
        velocity: Vector2::<f64>::from([0.0, 0.0]),
        cell_radius: CELL_RADIUS,
        potential_strength: CELL_LENNARD_JONES_STRENGTH,
        velocity_reduction: CELL_VELOCITY_REDUCTION,
        maximum_age: CELL_MAXIMUM_AGE,
        remove: false,
        current_age: 0.0,
    })).collect();

    let mut c2 = (N_CELLS..2*N_CELLS).map(|_| CellAgentType::CustomCell2D(CustomCell2D {
        pos: Vector2::<f64>::from([
            rng.gen_range(-DOMAIN_SIZE_X..DOMAIN_SIZE_X),
            rng.gen_range(-DOMAIN_SIZE_Y..DOMAIN_SIZE_Y)]),
        velocity: Vector2::<f64>::from([0.0, 0.0]),
        cell_radius: CELL_RADIUS,
        potential_strength: CELL_LENNARD_JONES_STRENGTH,
        velocity_reduction: CELL_VELOCITY_REDUCTION,
        maximum_age: CELL_MAXIMUM_AGE,
        remove: false,
        current_age: 0.0,
    })).collect();

    cells.append(&mut c2);

    // Define the Simulation Domain itself
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
            t_eval: (0..N_TIMES).map(|i| (T_START + DT * i as f64, i % 10 == 0, i % 10 == 0)).collect::<Vec<(f64, bool, bool)>>(),
        },
        meta_params: SimulationMetaParams {
            n_threads: N_THREADS
        },
        database: SledDataBaseConfig {
            name: "out/simulation_multi_cells".to_owned().into(),
        }
    };

    // let mut supervisor = create_sim_supervisor!(setup);
    let mut supervisor = SimulationSupervisor::from(setup);

    match supervisor.run_full_sim() {
        Ok(()) => (),
        Err(error) => println!("{error}"),
    }

    supervisor.end_simulation().unwrap();

    supervisor.plot_cells_at_every_iter_bitmap().unwrap();
}
