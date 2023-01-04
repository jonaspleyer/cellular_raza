use crate::concepts::cell::{CellAgent,CellAgentBox,Id};
use crate::concepts::domain::{Domain,Voxel,MultiVoxelContainer,PosInformation,ForceInformation};
use crate::concepts::domain::{DomainBox,Index};
use crate::concepts::mechanics::{Position,Force,Velocity};
use crate::concepts::errors::{CalcError,SimulationError};

use std::thread;
use std::collections::{HashMap};
use std::error::Error;

use core::marker::PhantomData;

use crossbeam_channel::unbounded;
use crossbeam_channel::{Sender,Receiver};

use hurdles::Barrier;

use std::sync::atomic::{AtomicBool,Ordering};
use std::sync::Arc;

use uuid::Uuid;

use rayon::prelude::*;


/// # Supervisor controlling simulation execution
/// 
pub struct SimulationSupervisor<Pos, For, Vel, Cel, Ind, Vox, Dom>
where
    Ind: Index,
    Pos: Position,
    For: Force,
    Vel: Velocity,
    Vox: Voxel<Ind, Pos, For>,
    Cel: CellAgent<Pos, For, Vel>,
    Dom: Domain<CellAgentBox<Pos, For, Vel, Cel>, Ind, Vox>,
{
    worker_threads: Vec<thread::JoinHandle<()>>,
    multivoxelcontainers: Vec<MultiVoxelContainer<Ind, Pos, For, Vel, Vox, Dom, Cel>>,

    time: TimeSetup,

    domain: DomainBox<CellAgentBox<Pos, For, Vel, Cel>, Ind, Vox, Dom>,

    // Variables controlling simulation flow
    save_now: Arc<AtomicBool>,
    stop_now: Arc<AtomicBool>,

    // PhantomData for template arguments
    phantom_cell: PhantomData<CellAgentBox<Pos, For, Vel, Cel>>,
    phantom_index: PhantomData<Ind>,
    phantom_voxel: PhantomData<Vox>,
    phantom_pos: PhantomData<Pos>,
    phantom_force: PhantomData<For>,
    phantom_velocity: PhantomData<Vel>,
}


/// # Store meta parameters for simulation
pub struct SimulationMetaParams {
    pub n_threads: usize,
}


pub struct TimeSetup {
    pub t_start: f64,
    pub t_eval: Vec<(f64, bool)>,
}


pub struct DataBaseConfig  {
    pub name: String,
}


/// # Complete Set of parameters controlling execution flow of simulation
pub struct SimulationSetup<Dom, C>
{
    pub domain: Dom,
    pub cells: Vec<C>,
    pub time: TimeSetup,
    pub meta_params: SimulationMetaParams,
    pub database: DataBaseConfig,
}


impl<Pos, For, Vel, Cel, Ind, Vox, Dom> From<SimulationSetup<Dom, Cel>> for Result<SimulationSupervisor<Pos, For, Vel, Cel, Ind, Vox, Dom>, Box<dyn Error>>
where
    Dom: Domain<CellAgentBox<Pos, For, Vel, Cel>, Ind, Vox> + Clone + 'static,
    Ind: Index + 'static,
    Pos: Position + 'static + std::fmt::Debug,
    For: Force + 'static,
    Vel: Velocity + 'static,
    Vox: Voxel<Ind, Pos, For> + Clone + 'static,
    Cel: CellAgent<Pos, For, Vel> + 'static,
{
    fn from(setup: SimulationSetup<Dom, Cel>) -> Result<SimulationSupervisor<Pos, For, Vel, Cel, Ind, Vox, Dom>, Box<dyn Error>> {
        // Create groups of voxels to put into our MultiVelContainers
        let (n_threads, voxel_chunks) = <Dom>::generate_contiguous_multi_voxel_regions(&setup.domain, setup.meta_params.n_threads).unwrap();

        // Create MultiVelContainer from voxel chunks
        let multivoxelcontainers;
        let cells = setup.cells
            .into_par_iter()
            .enumerate()
            .map(|(i, c)| CellAgentBox::<Pos, For, Vel, Cel>::from((i as u128, c)))
            .collect::<Vec<CellAgentBox<Pos, For, Vel, Cel>>>();

        // Create sender receiver pairs for all threads
        let sender_receiver_pairs_cell: Vec<(Sender<CellAgentBox::<Pos, For, Vel, Cel>>, Receiver<CellAgentBox::<Pos, For, Vel, Cel>>)> = (0..n_threads).map(|_| unbounded()).collect();
        let sender_receiver_pairs_pos: Vec<(Sender<PosInformation<Ind, Pos>>, Receiver<PosInformation<Ind, Pos>>)> = (0..n_threads).map(|_| unbounded()).collect();
        let sender_receiver_pairs_force: Vec<(Sender<ForceInformation<For>>, Receiver<ForceInformation<For>>)> = (0..n_threads).map(|_| unbounded()).collect();

        // Create a barrier to synchronize all threads
        let barrier = Barrier::new(n_threads);

        // Create an intermediate mapping just for this setup
        // Map voxel index to thread number
        let mut index_to_thread: HashMap<Ind,usize> = HashMap::new();
        for (i, chunk) in voxel_chunks.iter().enumerate() {
            for (index, _) in chunk {
                index_to_thread.insert(index.clone(), i);
            }
        }

        let n_chunks = voxel_chunks.len();
        let chunk_size = (cells.len() as f64/ n_threads as f64).ceil() as usize;
        let mut sorted_cells = cells
            .into_par_iter()
            .chunks(chunk_size)
            .map(|cell_chunk| {
                let mut cs = HashMap::<usize, HashMap<Ind, Vec<CellAgentBox<Pos, For, Vel, Cel>>>>::new();
                for cell in cell_chunk.into_iter() {
                    let index = setup.domain.get_voxel_index(&cell);
                    let id_thread = index_to_thread[&index];
                    match cs.get_mut(&id_thread) {
                        Some(index_to_cells) => match index_to_cells.get_mut(&index) {
                            Some(cs) => cs.push(cell),
                            None => {index_to_cells.insert(index, vec![cell]);},
                        },
                        None => {cs.insert(id_thread, HashMap::from([(index, vec![cell])]));},
                    }
                }
                cs
            })
            // .reduce(|| HashMap::<usize, HashMap<Ind, Vec<Cel>>>::new(), |mut acc, x| {
            .reduce(|| (0..n_chunks).map(|i| (i, HashMap::new())).collect::<HashMap<usize, HashMap<Ind, Vec<CellAgentBox<Pos, For, Vel, Cel>>>>>(), |mut acc, x| {
                for (id_thread, idc) in x.into_iter() {
                    for (index, mut cells) in idc.into_iter() {
                        match acc.get_mut(&id_thread) {
                            Some(index_to_cells) => match index_to_cells.get_mut(&index) {
                                Some(cs) => cs.append(&mut cells),
                                None => {index_to_cells.insert(index, cells);},
                            },
                            None => {acc.insert(id_thread, HashMap::from([(index, cells)]));},
                        }
                    }
                }
                return acc;
            });

        let voxel_and_cells: Vec<(Vec<(Ind, Vox)>, HashMap<Ind, Vec<CellAgentBox<Pos, For, Vel, Cel>>>)> = voxel_chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| (chunk, sorted_cells.remove(&i).unwrap()))
            .collect();

        // Create an instance to communicate with the database
        let tree = sled::open(setup.database.name).unwrap();

        multivoxelcontainers = voxel_and_cells.into_par_iter().enumerate().map(|(i, (chunk, mut index_to_cells))| {
            // TODO insert all variables correctly into this container here
            let voxels: HashMap<Ind,Vox> = chunk.clone().into_iter().collect();

            // Fill index_to_cells with non-occupied indices of voxels
            for (ind, _) in voxels.iter() {
                match index_to_cells.get(&ind) {
                    Some(_) => (),
                    None => {index_to_cells.insert(ind.clone(), Vec::new());},
                }
            }

            // Quick macro to create senders
            macro_rules! create_senders {
                ($sr_pairs: expr) => {
                    chunk.clone()
                        .into_iter()
                        .map(|(index, _)| setup.domain
                            .get_neighbor_voxel_indices(&index)
                            .into_iter()
                            .map(|ind| 
                                (ind.clone(), $sr_pairs[index_to_thread[&ind]].0.clone())))
                        .flatten()
                        .collect::<HashMap<Ind,_>>()
                }
            }

            let senders_cell = create_senders!(sender_receiver_pairs_cell);
            let senders_pos = create_senders!(sender_receiver_pairs_pos);
            let senders_force = create_senders!(sender_receiver_pairs_force);

            // Clone database instance
            let tree_new = tree.clone();

            // Get all cells which belong into this voxel_chunk
            // let cells_filt: Vec<Cel> = cells.drain_filter(|c| any(chunk.iter(), |(i, _)| *i==setup.domain.get_voxel_index(&c))).collect();
            let cell_forces_empty: HashMap<Uuid, Vec<Result<For, CalcError>>> = index_to_cells
                .iter()
                .map(|(_, cs)| cs
                    .iter()
                    .map(|c| (c.get_uuid(), Vec::new()))
                ).flatten()
                .collect();

            let neighbor_indices: HashMap<Ind, Vec<Ind>> = chunk
                .clone()
                .into_iter()
                .map(|(i, _)| (i.clone(), setup.domain.get_neighbor_voxel_indices(&i)))
                .collect();

            // Define the container for many voxels
            let cont = MultiVoxelContainer {
                voxels: voxels,
                voxel_cells: index_to_cells,
                
                domain: setup.domain.clone(),

                senders_cell: senders_cell,
                senders_pos: senders_pos,
                senders_force: senders_force,
                
                receiver_cell: sender_receiver_pairs_cell[i].1.clone(),
                receiver_pos: sender_receiver_pairs_pos[i].1.clone(),
                receiver_force: sender_receiver_pairs_force[i].1.clone(),

                cell_forces: cell_forces_empty,
                neighbor_indices: neighbor_indices,

                barrier: barrier.clone(),

                database: tree_new,
            };

            // multivoxelcontainers.push(cont);
            return cont;
        }).collect();

        let save_now = Arc::new(AtomicBool::new(false));
        let stop_now = Arc::new(AtomicBool::new(false));

        Ok(SimulationSupervisor {
            worker_threads: Vec::new(),
            multivoxelcontainers: multivoxelcontainers,

            time: setup.time,

            domain: setup.domain.into(),

            // Variables controlling simulation flow
            save_now: save_now,
            stop_now: stop_now,

            phantom_cell: PhantomData,
            phantom_index: PhantomData,
            phantom_voxel: PhantomData,
            phantom_pos: PhantomData,
            phantom_force: PhantomData,
            phantom_velocity: PhantomData,
        })
    }
}


#[macro_export]
macro_rules! implement_cell_types {
    ($pos:ty, $force:ty, $velocity:ty, [$($celltype:ident),+]) => {
        use serde::{Serialize,Deserialize};

        #[derive(Debug,Clone,Serialize,Deserialize)]
        pub enum CellAgentType {
            $($celltype($celltype)),+
        }

        impl Interaction<$pos, $force> for CellAgentType {
            fn force(&self, own_pos: &$pos, ext_pos: &$pos) -> Option<Result<$force, CalcError>> {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.force(own_pos, ext_pos),)+
                }
            }
        }

        impl Cycle<CellAgentType> for CellAgentType {
            fn update_cycle(dt: &f64, c: &mut CellAgentType) {
                match c {
                    $(CellAgentType::$celltype(cell) => $celltype::update_cycle(dt, cell),)+
                }
            }
        }

        impl Mechanics<$pos, $force, $velocity> for CellAgentType {
            fn pos(&self) -> $pos {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.pos(),)+
                }
            }

            fn velocity(&self) -> $velocity {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.velocity(),)+
                }
            }

            fn set_pos(&mut self, pos: &$pos) {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.set_pos(pos),)+
                }
            }

            fn set_velocity(&mut self, velocity: &$velocity) {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.set_velocity(velocity),)+
                }
            }

            fn calculate_increment(&self, force: $force) -> Result<($pos, $velocity), CalcError> {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.calculate_increment(force),)+
                }
            }
        }

        unsafe impl Send for CellAgentType {}
        unsafe impl Sync for CellAgentType {}
    }
}


#[macro_export]
macro_rules! define_simulation_types {
    (
        Position:   $position:ty,
        Force:      $force:ty,
        Velocity:   $velocity:ty,
        CellTypes:  [$($celltype:ident),+],
        Voxel:      $voxel:ty,
        Index:      $index:ty,
        Domain:     $domain:ty,
    ) => {
        // Create an enum containing all cell types
        implement_cell_types!($position, $force, $velocity, [$($celltype),+]);

        pub type SimTypePosition = $position;
        pub type SimTypeForce = $force;
        pub type SimTypeVelocity = $velocity;
        pub type SimTypeVoxel = $voxel;
        pub type SimTypeIndex = $index;
        pub type SimTypeDomain = $domain;
    }
}


#[macro_export]
macro_rules! create_sim_supervisor {
    ($setup:expr) => {
        Result::<SimulationSupervisor::<SimTypePosition, SimTypeForce, SimTypeVelocity, CellAgentType, SimTypeIndex, SimTypeVoxel, SimTypeDomain>, Box<dyn std::error::Error>>::from($setup).unwrap()
    }
}


impl<Pos, For, Vel, Cel, Ind, Vox, Dom> SimulationSupervisor<Pos, For, Vel, Cel, Ind, Vox, Dom>
where
    Dom: Domain<CellAgentBox<Pos, For, Vel, Cel>, Ind, Vox> + Clone + 'static,
    Ind: Index + 'static,
    Pos: Position + 'static,
    For: Force + 'static,
    Vel: Velocity + 'static,
    Vox: Voxel<Ind, Pos, For> + Clone + 'static,
    Cel: CellAgent<Pos, For, Vel> + 'static,
{
    fn spawn_worker_threads_and_run_sim(&mut self) -> Result<(), SimulationError> {
        let mut handles = Vec::new();
        let mut start_barrier = Barrier::new(self.multivoxelcontainers.len()+1);

        for (l, mut cont) in self.multivoxelcontainers.drain(..).enumerate() {
            // Clone barriers to use them for synchronization in threads
            let mut new_start_barrier = start_barrier.clone();

            // See if we need to save
            let save_now_new = Arc::clone(&self.save_now);
            let stop_now_new = Arc::clone(&self.stop_now);

            // Copy time evaluation points
            let t_start = self.time.t_start;
            let t_eval = self.time.t_eval.clone();

            // Spawn a thread for each multivoxelcontainer that is running
            let handle = thread::Builder::new().name(format!("worker_thread_{:03.0}", l)).spawn(move || {
                new_start_barrier.wait();

                let mut time = t_start;
                for (t, save) in t_eval {
                    let dt = t - time;
                    time = t;

                    match cont.run_full_update(&t, &dt) {
                        Ok(()) => (),
                        Err(error) => {
                            println!("Encountered error in update_mechanics: {}. Stopping simulation.", error);
                            // Make sure to stop all threads after this iteration.
                            stop_now_new.store(true, Ordering::Relaxed);
                        },
                    }

                    // if save_now_new.load(Ordering::Relaxed) {
                    if save {
                        // let mut all_cells: Vec<Cel> = cont.voxel_cells.iter().map(|(_, cells)| cells.clone()).flatten().collect();
                        // for cell in all_cells.drain(..) {
                        //     // TODO do not unwrap but catch nicely
                        //     // sender_plots_new.send(cell).unwrap();
                        // }
                        // if l==0 {
                        //     println!("Saving Simulation at time {}", t);
                        // }

                        // Save cells to database
                        cont.save_cells_to_database(&t).unwrap();
                    }

                    // Check if we are stopping the simulation now
                    if stop_now_new.load(Ordering::Relaxed) {
                        // new_barrier.wait();
                        break;
                    }
                }

                // println!("thread {} stopping with voxels {} and cells {}", l, cont.voxel_cells.len(), cont.voxel_cells.iter().map(|(_, cells)| cells.len()).sum::<usize>());

                // new_barrier.wait();
            })?;
            handles.push(handle);
        }

        // Store worker threads in supervisor
        self.worker_threads = handles;

        // This starts all threads simultanously
        start_barrier.wait();
        Ok(())
    }

    pub fn run_full_sim(&mut self) -> Result<(), SimulationError> {
        self.spawn_worker_threads_and_run_sim()?;
        
        Ok(())
    }

    pub fn run_until(&mut self, end_time: f64) -> Result<(), SimulationError> {
        self.time.t_eval.drain_filter(|(t, _)| *t <= end_time);
        self.run_full_sim()?;
        Ok(())
    }

    /* TODO figure this out
    pub fn get_current_cells(&self) -> Result<Vec<Cel>, SimulationError> {

    }*/

    pub fn end_simulation(self) {
        for thread in self.worker_threads {
            match thread.join() {
                Ok(_) => (),
                Err(err) => println!("{:?}", err),
            }
        }
    }
}
