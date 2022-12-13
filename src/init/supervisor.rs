use crate::concepts::cell::Cell;
use crate::concepts::domain::{Domain,Voxel,MultiVoxelContainer,PosInformation,ForceInformation};
use crate::concepts::domain::Index;
use crate::concepts::mechanics::{Position,Force,Velocity};
use crate::concepts::errors::CalcError;

use std::thread;
use std::collections::{HashMap,LinkedList};
use std::error::Error;

use core::marker::PhantomData;

use crossbeam_channel::unbounded;
use crossbeam_channel::{Sender,Receiver};

use hurdles::Barrier;

use std::sync::atomic::{AtomicBool,Ordering};
use std::sync::Arc;

use itertools::any;

use uuid::Uuid;


/// # Supervisor controlling simulation execution
/// 
pub struct SimulationSupervisor<Dom, Cel, Ind, Pos, For, Vel, Vox>
where
    Dom: Domain<Cel, Ind, Vox>,
    Ind: Index,
    Vox: Voxel<Ind, Pos, For>,
    Cel: Cell<Pos, For, Vel>,
{
    worker_threads: Vec<thread::JoinHandle<()>>,
    multivoxelcontainers: Vec<MultiVoxelContainer<Ind, Pos, For, Vel, Vox, Dom, Cel>>,

    time: TimeSetup,

    domain: Dom,

    // Variables controlling simulation flow
    save_now: Arc<AtomicBool>,
    stop_now: Arc<AtomicBool>,

    // PhantomData for template arguments
    phantom_cell: PhantomData<Cel>,
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
    pub t_eval: Vec<f64>,
}


/// # Complete Set of parameters controlling execution flow of simulation
pub struct SimulationSetup<Dom, C>
{
    pub domain: Dom,
    pub cells: Vec<C>,
    pub time: TimeSetup,
    pub meta_params: SimulationMetaParams,
}


impl<Dom, Cel, Ind, Pos, For, Vel, Vox> From<SimulationSetup<Dom, Cel>> for Result<SimulationSupervisor<Dom, Cel, Ind, Pos, For, Vel, Vox>, Box<dyn Error>>
where
    Dom: Domain<Cel, Ind, Vox> + Clone + 'static,
    Ind: Index + 'static,
    Pos: Position + 'static + std::fmt::Debug,
    For: Force + 'static,
    Vel: Velocity + 'static,
    Vox: Voxel<Ind, Pos, For> + Clone + 'static,
    Cel: Cell<Pos, For, Vel> + 'static,
{
    fn from(setup: SimulationSetup<Dom, Cel>) -> Result<SimulationSupervisor<Dom, Cel, Ind, Pos, For, Vel, Vox>, Box<dyn Error>> {
        // Create groups of voxels to put into our MultiVelContainers
        let (n_threads, mut voxel_chunks) = <Dom>::generate_contiguous_multi_voxel_regions(&setup.domain, setup.meta_params.n_threads).unwrap();

        // Create MultiVelContainer from voxel chunks
        let mut multivoxelcontainers = Vec::new();
        let mut cells = setup.cells;

        // Create sender receiver pairs for all threads
        let sender_receiver_pairs_cell: Vec<(Sender<Cel>, Receiver<Cel>)> = (0..n_threads).map(|_| unbounded()).collect();
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

        for (i, chunk) in voxel_chunks.drain(..).enumerate() {
            // TODO insert all variables correctly into this container here
            let voxels: HashMap<Ind,Vox> = chunk.clone().into_iter().collect();
            let voxel_cells_empty: HashMap<Ind, LinkedList<Cel>> = chunk.clone().into_iter().map(|(i, _)| (i, LinkedList::new())).collect();

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

            // Get all cells which belong into this voxel_chunk
            let cells_filt: Vec<Cel> = cells.drain_filter(|c| any(chunk.iter(), |(i, _)| *i==setup.domain.get_voxel_index(&c))).collect();
            let cell_forces_empty: HashMap<Uuid, LinkedList<Result<For, CalcError>>> = cells.iter().map(|c| (c.get_uuid(), LinkedList::new())).collect();
            let neighbor_indices: HashMap<Ind, Vec<Ind>> = chunk
                .clone()
                .into_iter()
                .map(|(i, _)| (i.clone(), setup.domain.get_neighbor_voxel_indices(&i)))
                .collect();

            // Define the container for many voxels
            let mut cont = MultiVoxelContainer {
                voxels: voxels,
                voxel_cells: voxel_cells_empty,
                
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

                phantom_vel: PhantomData,
            };
            
            // Now filter again for the different voxels
            for cell in cells_filt {
                cont.sort_cell_in_voxel(cell)?;
            }

            multivoxelcontainers.push(cont);
        }

        let save_now = Arc::new(AtomicBool::new(false));
        let stop_now = Arc::new(AtomicBool::new(false));

        Ok(SimulationSupervisor {
            worker_threads: Vec::new(),
            multivoxelcontainers: multivoxelcontainers,

            time: setup.time,

            domain: setup.domain,

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


// TODO give constants in setup
pub const DT: f64 = 0.01;


impl<Dom, Cel, Ind, Pos, For, Vel, Vox> SimulationSupervisor<Dom, Cel, Ind, Pos, For, Vel, Vox>
where
    Dom: Domain<Cel, Ind, Vox> + Clone + 'static,
    Ind: Index + 'static,
    Pos: Position + 'static,
    For: Force + 'static,
    Vel: Velocity + 'static,
    Vox: Voxel<Ind, Pos, For> + Clone + 'static,
    Cel: Cell<Pos, For, Vel> + 'static,
{
    fn spawn_worker_threads_and_run_sim(&mut self) -> Result<(), Box<dyn Error>> {
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
                println!("thread {} executing with voxels {} and cells {}", l, cont.voxel_cells.len(), cont.voxel_cells.iter().map(|(_, cells)| cells.len()).sum::<usize>());

                let mut time = t_start;
                for t in t_eval {
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

                    if save_now_new.load(Ordering::Relaxed) {
                        // let mut all_cells: Vec<CellModel> = vox_cont.voxels.iter().map(|v| v.cells.clone()).flatten().collect();
                        // for cell in all_cells.drain(..) {
                        //     // TODO do not unwrap but catch nicely
                        //     sender_plots_new.send(cell).unwrap();
                        // }
                    }

                    // Check if we are stopping the simulation now
                    if stop_now_new.load(Ordering::Relaxed) {
                        // new_barrier.wait();
                        break;
                    }
                }

                println!("thread {} stopping with voxels {} and cells {}", l, cont.voxel_cells.len(), cont.voxel_cells.iter().map(|(_, cells)| cells.len()).sum::<usize>());

                // new_barrier.wait();
            })?;
            handles.push(handle);
        }

        // Store worker threads in supervisor
        self.worker_threads = handles;

        // This starts all threads simultanously
        println!("Before starting");
        start_barrier.wait();
        Ok(())
    }

    pub fn run_full_sim(&mut self) -> Result<(), Box<dyn Error>> {
        self.spawn_worker_threads_and_run_sim()?;
        
        Ok(())
    }

    pub fn end_simulation(self) {
        for thread in self.worker_threads {
            match thread.join() {
                Ok(_) => (),
                Err(err) => println!("{:?}", err),
            }
        }
    }
}
