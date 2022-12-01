// Imports from the cellular_control crate
use cellular_control::cell_properties::cell_model::*;
use cellular_control::cell_properties::cycle::*;
use cellular_control::cell_properties::death::*;
use cellular_control::cell_properties::interaction::*;
use cellular_control::cell_properties::mechanics::*;
use cellular_control::cell_properties::flags::*;

use cellular_control::domain::cartesian_cuboid::*;

use cellular_control::concepts::mechanics::*;

// Imports from other crates
use nalgebra::Vector3;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use std::collections::HashMap;
use itertools::iproduct;

use ndarray::{Array1,Array3};
use crossbeam_channel::unbounded;
use hurdles::Barrier;



// Domain properties
pub const N_VOXEL_X: usize = 10;
pub const N_VOXEL_Y: usize = 10;
pub const N_VOXEL_Z: usize = 1;

pub const DOMAIN_SIZE: f64 = 200.0;

// Cell properties
pub const CELL_RADIUS: f64 = 3.0;
pub const CELL_LENNARD_JONES_STRENGTH: f64 = 1.0;
pub const CELL_INITIAL_VELOCITY: f64 = 0.01;
pub const CELL_CYCLE_LIFETIME1_LOW:  f64 = 0.0;
pub const CELL_CYCLE_LIFETIME1_HIGH: f64 = 1.0;
pub const CELL_CYCLE_LIFETIME2_LOW:  f64 = 0.0;
pub const CELL_CYCLE_LIFETIME2_HIGH: f64 = 1.0;
pub const CELL_CYCLE_LIFETIME3_LOW:  f64 = 0.0;
pub const CELL_CYCLE_LIFETIME3_HIGH: f64 = 1.0;
pub const CELL_CYCLE_LIFETIME4_LOW:  f64 = 1600.0;
pub const CELL_CYCLE_LIFETIME4_HIGH: f64 = 2400.0;
pub const CELL_VELOCITY_REDUCTION: f64 = 1.0;

// Number of cells initially in simulation
pub const N_CELLS: u32 = 10000;

// Other parameters
pub const N_THREADS: usize = 1;


pub fn insert_cells() -> Vec<CellModel> {
    let mut cells = Vec::new();

    for n in 0..N_CELLS {
        let mut rng = ChaCha8Rng::seed_from_u64(n.into());

        let de_model = DeathModel {};
        let in_model = LennardJones { epsilon: CELL_LENNARD_JONES_STRENGTH, sigma: 0.1 * CELL_RADIUS/2.0f64.powf(1.0/6.0) };
        let me_model = MechanicsModel::from((&Vector3::<f64>::from([0.0, 0.0, 0.0]), &Vector3::<f64>::from([0.0, 0.0, 0.0]), CELL_VELOCITY_REDUCTION));
        let fl_model = Flags { removal: false };

        let cy1 = CellCycle { lifetime: rng.gen_range(CELL_CYCLE_LIFETIME1_LOW..CELL_CYCLE_LIFETIME1_HIGH) };
        let cy2 = CellCycle { lifetime: rng.gen_range(CELL_CYCLE_LIFETIME2_LOW..CELL_CYCLE_LIFETIME2_HIGH) };
        let cy3 = CellCycle { lifetime: rng.gen_range(CELL_CYCLE_LIFETIME3_LOW..CELL_CYCLE_LIFETIME3_HIGH) };
        let cy4 = CellCycle { lifetime: rng.gen_range(CELL_CYCLE_LIFETIME4_LOW..CELL_CYCLE_LIFETIME4_HIGH) };
        let cy_model = CycleModel::from(&vec![cy1, cy2, cy3, cy4]);

        let mut cell = CellModel { mechanics: me_model, cell_cycle: cy_model, death_model: de_model, interaction: in_model, flags: fl_model, id: n };

        let p1 = rng.gen_range(-DOMAIN_SIZE..DOMAIN_SIZE);
        let p2 = rng.gen_range(-DOMAIN_SIZE..DOMAIN_SIZE);
        cell.mechanics.set_pos(&Vector3::<f64>::from([p1, p2, 0.0]));
        let m1;
        let m2;
        if p1 > 0.0 {
            m1 = rng.gen_range(-CELL_INITIAL_VELOCITY..0.0);
        } else {
            m1 = rng.gen_range(-0.0..CELL_INITIAL_VELOCITY);
        }
        if p2 > 0.0 {
            m2 = rng.gen_range(-CELL_INITIAL_VELOCITY..0.0);
        } else {
            m2 = rng.gen_range(-0.0..CELL_INITIAL_VELOCITY);
        }
            
        cell.mechanics.set_velocity(&Vector3::<f64>::from([m1, m2, 0.0]));
        cells.push(cell);
    }

    return cells;
}


use std::time::Instant;
use crossbeam_channel::{Sender,Receiver};
use std::cmp::{min,max};


pub fn define_domain() -> (Cuboid, Array3<Voxel>) {
    // Create a voxel
    let steps: [f64; 3] = [N_VOXEL_X, N_VOXEL_Y, N_VOXEL_Z].map(|i| 2.0 * DOMAIN_SIZE/i as f64);
    let voxel_sizes = steps;

    // Define the cuboid domain
    let c = Cuboid {
        min: [-DOMAIN_SIZE, -DOMAIN_SIZE, -DOMAIN_SIZE],
        max: [DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_SIZE],
        voxel_sizes: voxel_sizes,
        // voxels: voxels,
    };

    let mi = [-DOMAIN_SIZE, -DOMAIN_SIZE, -DOMAIN_SIZE];
    let ma = [-DOMAIN_SIZE + steps[0], -DOMAIN_SIZE + steps[1], -DOMAIN_SIZE + steps[2]];

    // let senders_receivers = irpoduct!(0..N_VOXEL, 0..N_VOXEL, 0..N_VOXEL)

    // Iterate over all voxel combinations and insert sender and receivers
    let (_, cell_r) = unbounded::<CellModel>();
    let (_, pos_r) = unbounded::<([usize; 3], Vector3<f64>, u32)>();

    let voxel = Voxel {
        min: mi,
        max: ma,
        cells: Vec::<CellModel>::new(),
        cell_senders: HashMap::new(),
        cell_receiver: cell_r,
        pos_senders: HashMap::new(),
        pos_receiver: pos_r,
        id: [0, 0, 0],
        domain: c.clone(),
    };
    let mut voxels = Array3::<Voxel>::from_elem((N_VOXEL_X, N_VOXEL_Y, N_VOXEL_Z), voxel);

    for (i, j, k) in iproduct!(0..N_VOXEL_X, 0..N_VOXEL_Y, 0..N_VOXEL_Z) {
        voxels[(i, j, k)].min = [
            -DOMAIN_SIZE + steps[0] * i as f64,
            -DOMAIN_SIZE + steps[1] * j as f64,
            -DOMAIN_SIZE + steps[2] * k as f64
        ];
        voxels[(i, j, k)].max = [
            -DOMAIN_SIZE + steps[0] * (i+1) as f64,
            -DOMAIN_SIZE + steps[1] * (j+1) as f64,
            -DOMAIN_SIZE + steps[2] * (k+1) as f64
        ];
        voxels[(i, j, k)].id = [i, j, k];
    }

    let now = Instant::now();
    let channel_pairs = Array1::<(Sender<CellModel>, Receiver<CellModel>)>::from_iter(
        iproduct!(0..N_VOXEL_X, 0..N_VOXEL_Y, 0..N_VOXEL_Z).map(|_| unbounded::<CellModel>())
    ).into_shape((N_VOXEL_X, N_VOXEL_Y, N_VOXEL_Z)).unwrap();

    let channel_pairs_2 = Array1::<(Sender<([usize; 3], Vector3<f64>, u32)>, Receiver<([usize; 3], Vector3<f64>, u32)>)>::from_iter(
        iproduct!(0..N_VOXEL_X, 0..N_VOXEL_Y, 0..N_VOXEL_Z).map(|_| unbounded::<([usize; 3], Vector3<f64>, u32)>())
    ).into_shape((N_VOXEL_X, N_VOXEL_Y, N_VOXEL_Z)).unwrap();

    println!("Iterations 0 took {}", now.elapsed().as_millis());


    let now = Instant::now();
    for (m0, m1, m2) in iproduct!(0..N_VOXEL_X, 0..N_VOXEL_Y, 0..N_VOXEL_Z) {
        // Create a channel for voxel (m0, m1, m2)
        let (cell_s, cell_r) = &channel_pairs[[m0, m1, m2]];

        // Store the receiver at this voxel
        voxels[[m0, m1, m2]].cell_receiver = cell_r.clone();

        for (n0, n1, n2) in iproduct!(0..N_VOXEL_X, 0..N_VOXEL_Y, 0..N_VOXEL_Z) {
        // for (n0, n1, n2) in iproduct!(max(m0-1,0)..min(m0+1,N_VOXEL), max(m1-1,0)..min(m1+1,N_VOXEL), max(m2-1,0)..min(m2+1, N_VOXEL)) {
            if n0-2 <= m0 && m0 <= n0+2 && n1-2 <= m1 && m1 <= n1+2 && n2-2 <= m2 && m2 <= n2+2 {
                voxels[(n0, n1, n2)].cell_senders.insert([m0, m1, m2], cell_s.clone());
            }
        }

        // Create channels for positions
        let (pos_s, pos_r) = &channel_pairs_2[[m0, m1, m2]];

        // Store receiver and the senders
        voxels[[m0, m1, m2]].pos_receiver = pos_r.clone();

        // for (n0, n1, n2) in iproduct!(max(m0-1,0)..min(m0+1,N_VOXEL), max(m1-1,0)..min(m1+1,N_VOXEL), max(m2-1,0)..min(m2+1, N_VOXEL)) {
        for (n0, n1, n2) in iproduct!(0..N_VOXEL_X, 0..N_VOXEL_Y, 0..N_VOXEL_Z) {
            if n0 <= m0+2 && m0 <= n0+2 && n1 <= m1+2 && m1 <= n1+2 && n2 <= m2+2 && m2 <= n2+2 {
                voxels[(n0, n1, n2)].pos_senders.insert([m0, m1, m2], pos_s.clone());
            }
        }
    }
    println!("Iterations 1 took {}", now.elapsed().as_millis());

    (c, voxels)
}
