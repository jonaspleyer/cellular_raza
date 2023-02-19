use crate::concepts::cell::{CellAgent,CellAgentBox};
use crate::concepts::domain::{Domain,Voxel,MultiVoxelContainer,PosInformation,ForceInformation,PlainIndex};
use crate::concepts::domain::{DomainBox,Index};
use crate::concepts::mechanics::{Position,Force,Velocity};
use crate::concepts::errors::SimulationError;

use std::thread;
use std::collections::{HashMap,BTreeMap};

use core::marker::PhantomData;

use crossbeam_channel::unbounded;
use crossbeam_channel::{Sender,Receiver};

use hurdles::Barrier;

use std::sync::atomic::{AtomicBool,Ordering};
use std::sync::Arc;

use uuid::Uuid;

use rayon::prelude::*;

use serde::{Serialize,Deserialize};

use plotters::{
    prelude::{BitMapBackend,Cartesian2d,DrawingArea},
    coord::types::RangedCoordf64,
};

use indicatif::{ProgressBar,ProgressStyle};


#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct SimulationConfig {
    pub show_progressbar: bool,
}


impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            show_progressbar: true,
        }
    }
}


/// # Supervisor controlling simulation execution
/// 
pub struct SimulationSupervisor<Pos, For, Inf, Vel, Cel, Ind, Vox, Dom>
where
    Ind: Index,
    Pos: Position,
    For: Force + Serialize + for<'a> Deserialize<'a>,
    Vel: Velocity,
    Vox: Voxel<Ind, Pos, For>,
    Cel: CellAgent<Pos, For, Inf, Vel>,
    Dom: Domain<Cel, Ind, Vox>,
{
    worker_threads: Vec<thread::JoinHandle<Result<MultiVoxelContainer<Ind, Pos, For, Inf, Vox, Dom, Cel>, SimulationError>>>,
    multivoxelcontainers: Vec<MultiVoxelContainer<Ind, Pos, For, Inf, Vox, Dom, Cel>>,

    time: TimeSetup,
    meta_params: SimulationMetaParams,
    #[cfg(feature = "db_sled")]
    database: SledDataBaseConfig,

    domain: DomainBox<Dom>,

    pub config: SimulationConfig,
    pub plotting_config: PlottingConfig,

    // Variables controlling simulation flow
    stop_now: Arc<AtomicBool>,

    // Tree of database
    #[cfg(not(feature = "no_db"))]
    tree_cells: typed_sled::Tree<String, Vec<u8>>,
    #[cfg(not(feature = "no_db"))]
    meta_infos: typed_sled::Tree<String, Vec<u8>>,

    // PhantomData for template arguments
    phantom_velocity: PhantomData<Vel>,
}


/// # Store meta parameters for simulation
#[derive(Clone,Serialize,Deserialize)]
pub struct SimulationMetaParams {
    pub n_threads: usize,
}


// TODO rethink how to specify time points to save
// we need to frequently save cells and environment
// Sometimes we need full snapshots for recovery purposes
#[derive(Clone,Serialize,Deserialize)]
pub struct TimeSetup {
    pub t_start: f64,
    pub t_eval: Vec<(f64, bool, bool)>,
}


#[cfg(feature = "db_sled")]
#[derive(Clone,Serialize,Deserialize)]
pub struct SledDataBaseConfig  {
    pub name: std::path::PathBuf,
}


/// # Complete Set of parameters controlling execution flow of simulation
#[derive(Clone,Serialize,Deserialize)]
pub struct SimulationSetup<Dom, C>
{
    pub domain: Dom,
    pub cells: Vec<C>,
    pub time: TimeSetup,
    pub meta_params: SimulationMetaParams,
    #[cfg(feature = "db_sled")]
    pub database: SledDataBaseConfig,
    // pub save_folder: std::path::Path,
}


impl<Pos, For, Inf, Vel, Cel, Ind, Vox, Dom> From<SimulationSetup<Dom, Cel>> for SimulationSupervisor<Pos, For, Inf, Vel, Cel, Ind, Vox, Dom>
where
    Dom: Domain<Cel, Ind, Vox> + Clone + 'static,
    Ind: Index + 'static,
    Pos: Position + 'static + std::fmt::Debug,
    For: Force + Serialize + for<'a> Deserialize<'a> + 'static,
    Vel: Velocity + 'static,
    Vox: Voxel<Ind, Pos, For> + Clone + 'static,
    Cel: CellAgent<Pos, For, Inf, Vel> + 'static,
{
    fn from(setup: SimulationSetup<Dom, Cel>) -> SimulationSupervisor<Pos, For, Inf, Vel, Cel, Ind, Vox, Dom> {
        // Create groups of voxels to put into our MultiVelContainers
        let (n_threads, voxel_chunks) = <Dom>::generate_contiguous_multi_voxel_regions(&setup.domain, setup.meta_params.n_threads).unwrap();

        let convert_to_plain_index: BTreeMap<Ind, PlainIndex> = setup.domain.get_all_indices().into_iter().enumerate().map(|(count, ind)| (ind, count as PlainIndex)).collect();
        let convert_to_index: BTreeMap<PlainIndex, Ind> = convert_to_plain_index.iter().map(|(i, j)| (j.clone(), i.clone())).collect();

        let mut index_to_thread = BTreeMap::<Ind, usize>::new();
        let mut plain_index_to_thread = BTreeMap::<PlainIndex, usize>::new();
        for (n_thread, chunk) in voxel_chunks.iter().enumerate() {
            for (ind, _) in chunk.iter() {
                index_to_thread.insert(ind.clone(), n_thread);
                plain_index_to_thread.insert(convert_to_plain_index[&ind], n_thread);
            }
        }

        // Create MultiVelContainer from voxel chunks
        let multivoxelcontainers;

        // Create sender receiver pairs for all threads
        let sender_receiver_pairs_cell: Vec<(Sender<CellAgentBox::<Cel>>, Receiver<CellAgentBox::<Cel>>)> = (0..n_threads).map(|_| unbounded()).collect();
        let sender_receiver_pairs_pos: Vec<(Sender<PosInformation<Pos, Inf>>, Receiver<PosInformation<Pos, Inf>>)> = (0..n_threads).map(|_| unbounded()).collect();
        let sender_receiver_pairs_force: Vec<(Sender<ForceInformation<For>>, Receiver<ForceInformation<For>>)> = (0..n_threads).map(|_| unbounded()).collect();

        // Create a barrier to synchronize all threads
        let barrier = Barrier::new(n_threads);

        // Create an intermediate mapping just for this setup
        // Map voxel index to thread number
        let mut plain_index_to_thread: BTreeMap<PlainIndex,usize> = BTreeMap::new();
        for (i, chunk) in voxel_chunks.iter().enumerate() {
            for (index, _) in chunk {
                plain_index_to_thread.insert(convert_to_plain_index[&index], i);
            }
        }

        // Sort cells into correct voxels
        let n_chunks = voxel_chunks.len();
        let chunk_size = (setup.cells.len() as f64/ n_threads as f64).ceil() as usize;

        let mut sorted_cells = setup.cells
            .into_par_iter()
            .enumerate()
            .chunks(chunk_size)
            .map(|cell_chunk| {
                let mut cs = BTreeMap::<usize, BTreeMap<PlainIndex, Vec<(usize, Cel)>>>::new();
                for cell in cell_chunk.into_iter() {
                    let index = setup.domain.get_voxel_index(&cell.1);
                    let plain_index = convert_to_plain_index[&index];
                    let id_thread = plain_index_to_thread[&plain_index];
                    match cs.get_mut(&id_thread) {
                        Some(index_to_cells) => match index_to_cells.get_mut(&plain_index) {
                            Some(cs) => cs.push(cell),
                            None => {index_to_cells.insert(plain_index, vec![cell]);},
                        },
                        None => {cs.insert(id_thread, BTreeMap::from([(plain_index, vec![cell])]));},
                    }
                }
                cs
            })
            .reduce(|| (0..n_chunks).map(|i| (i, BTreeMap::new())).collect::<BTreeMap<usize, BTreeMap<PlainIndex, Vec<(usize, Cel)>>>>(), |mut acc, x| {
                for (id_thread, idc) in x.into_iter() {
                    for (index, mut cells) in idc.into_iter() {
                        match acc.get_mut(&id_thread) {
                            Some(index_to_cells) => match index_to_cells.get_mut(&index) {
                                Some(cs) => cs.append(&mut cells),
                                None => {index_to_cells.insert(index, cells);},
                            },
                            None => {acc.insert(id_thread, BTreeMap::from([(index, cells)]));},
                        }
                    }
                }
                return acc;
            });

        let voxel_and_raw_cells: Vec<(Vec<(PlainIndex, Vox)>, BTreeMap<PlainIndex, Vec<(usize, Cel)>>)> = voxel_chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| (chunk.into_iter().map(|(ind, vox)| (convert_to_plain_index[&ind], vox)).collect(), sorted_cells.remove(&i).unwrap()))
            .collect();

        let voxel_and_cell_boxes: Vec<(Vec<(PlainIndex, Vox)>, BTreeMap<PlainIndex, Vec<CellAgentBox<Cel>>>)> = voxel_and_raw_cells
            .into_iter()
            .map(|(chunk, sorted_cells)| {
                let res = (chunk, sorted_cells
                    .into_iter()
                    .map(|(ind, mut cells)| {
                        cells.sort_by(|(i, _), (j, _)| i.cmp(&j));
                        (ind, cells.into_iter().enumerate().map(|(n_cell, (_, cell))| CellAgentBox::new(ind, 0, n_cell as u64, cell)).collect()
                    )
                }).collect());
                res
            })
            .collect();

        // Create an instance to communicate with the database
        #[cfg(not(feature = "no_db"))]// TODO fix this if feature "no_db" is active!
        let date = chrono::Local::now().format("%Y-%m-%d:%H-%M-%S");
        let filename = match setup.database.name.file_name() {
            Some(name) => format!("{}_{}", date, name.to_str().unwrap()),
            None => format!("{}", date),
        };
        let mut complete_path = setup.database.name.clone();
        complete_path.set_file_name(filename);
        let db = sled::Config::new().path(std::path::Path::new(&complete_path)).open().unwrap();
        let tree_cells = typed_sled::Tree::<String, Vec<u8>>::open(&db, "cell_storage");
        let meta_infos = typed_sled::Tree::<String, Vec<u8>>::open(&db, "meta_infos");

        // Create all multivoxelcontainers
        multivoxelcontainers = voxel_and_cell_boxes.into_iter().enumerate().map(|(i, (chunk, mut index_to_cells))| {
            // TODO insert all variables correctly into this container here
            let voxels: BTreeMap::<PlainIndex, crate::concepts::domain::VoxelBox<Ind,Vox,Cel,For>> = chunk.clone()
                .into_iter()
                .map(|(plain_index, voxel)| {
                    let cells = match index_to_cells.remove(&plain_index) {
                        Some(cs) => cs,
                        None => Vec::new(),
                    };
                    let neighbors = setup.domain.get_neighbor_voxel_indices(&convert_to_index[&plain_index]).into_iter().map(|i| convert_to_plain_index[&i]).collect::<Vec<_>>();
                    let vbox = crate::concepts::domain::VoxelBox::<Ind,Vox,Cel,For>::new(
                        plain_index,
                        convert_to_index[&plain_index].clone(),
                        voxel,
                        neighbors,
                        cells,
                    );
                    (plain_index, vbox)
                }).collect();

            // Create non-occupied voxels
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
                        .map(|(plain_index, _)| {
                            setup.domain
                                .get_neighbor_voxel_indices(&convert_to_index[&plain_index])
                                .into_iter()
                                .map(|ind| (index_to_thread[&ind], $sr_pairs[index_to_thread[&ind]].0.clone()))
                        })
                        .flatten()
                        .collect::<HashMap<usize,_>>()
                }
            }

            let senders_cell = create_senders!(sender_receiver_pairs_cell);
            let senders_pos = create_senders!(sender_receiver_pairs_pos);
            let senders_force = create_senders!(sender_receiver_pairs_force);

            // Clone database instance
            #[cfg(not(feature = "no_db"))]
            let tree_cells_new = tree_cells.clone();

            // Define the container for many voxels
            let cont = MultiVoxelContainer {
                voxels,
                index_to_plain_index: convert_to_plain_index.clone(),
                
                domain: DomainBox::from(setup.domain.clone()),

                index_to_thread: index_to_thread.clone(),
                plain_index_to_thread: plain_index_to_thread.clone(),

                senders_cell,
                senders_pos,
                senders_force,
                
                receiver_cell: sender_receiver_pairs_cell[i].1.clone(),
                receiver_pos: sender_receiver_pairs_pos[i].1.clone(),
                receiver_force: sender_receiver_pairs_force[i].1.clone(),

                barrier: barrier.clone(),

                #[cfg(not(feature = "no_db"))]
                database_cells: tree_cells_new,

                mvc_id: i as u16,
            };

            return cont;
        }).collect();

        let stop_now = Arc::new(AtomicBool::new(false));

        SimulationSupervisor {
            worker_threads: Vec::new(),
            multivoxelcontainers,

            time: setup.time,
            meta_params: setup.meta_params,
            #[cfg(feature = "db_sled")]
            database: setup.database,

            domain: setup.domain.into(),

            config: SimulationConfig::default(),
            plotting_config: PlottingConfig::default(),

            // Variables controlling simulation flow
            stop_now,

            #[cfg(not(feature = "no_db"))]
            tree_cells,
            #[cfg(not(feature = "no_db"))]
            meta_infos,

            phantom_velocity: PhantomData,
        }
    }
}


#[macro_export]
macro_rules! implement_cell_types {
    ($pos:ty, $force:ty, $information:ty, $velocity:ty, $voxel:ty, $index:ty, [$($celltype:ident),+]) => {
        use serde::{Serialize,Deserialize};

        #[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
        pub enum CellAgentType {
            $($celltype($celltype)),+
        }

        impl Interaction<$pos, $force, $information> for CellAgentType {
            fn get_interaction_information(&self) -> Option<$information> {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.get_interaction_information(),)+
                }
            }

            fn calculate_force_on(&self, own_pos: &$pos, ext_pos: &$pos, ext_information: &Option<$information>) -> Option<Result<$force, CalcError>> {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.calculate_force_on(own_pos, ext_pos, ext_information),)+
                }
            }
        }

        impl Cycle<CellAgentType> for CellAgentType {
            fn update_cycle(rng: &mut rand_chacha::ChaCha8Rng, dt: &f64, c: &mut CellAgentType) -> Option<crate::concepts::cycle::CycleEvent> {
                match c {
                    $(CellAgentType::$celltype(cell) => $celltype::update_cycle(rng, dt, cell),)+
                }
            }

            fn divide(rng: &mut rand_chacha::ChaCha8Rng, c: &mut CellAgentType) -> Result<Option<CellAgentType>, crate::concepts::errors::DivisionError> {
                match c {
                    $(CellAgentType::$celltype(cell) => match $celltype::divide(rng, cell)? {
                        Some(new_cell) => Ok(Some(CellAgentType::$celltype(new_cell))),
                        None => Ok(None),
                    },)+
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

        impl crate::plotting::spatial::PlotSelf for CellAgentType
        {
            fn plot_self<Db>(&self, root: &mut plotters::prelude::DrawingArea<Db, plotters::coord::cartesian::Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>>) -> Result<(), DrawingError>
            where
                Db: plotters::backend::DrawingBackend,
            {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.plot_self(root),)+
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
        Information:$information:ty,
        Velocity:   $velocity:ty,
        CellTypes:  [$($celltype:ident),+],
        Voxel:      $voxel:ty,
        Index:      $index:ty,
        Domain:     $domain:ty,
    ) => {
        // Create an enum containing all cell types
        implement_cell_types!($position, $force, $information, $velocity, $voxel, $index, [$($celltype),+]);

        pub type SimTypePosition = $position;
        pub type SimTypeForce = $force;
        pub type SimTypeVelocity = $velocity;
        pub type SimTypeVoxel = $voxel;
        pub type SimTypeIndex = $index;
        pub type SimTypeDomain = $domain;
    }
}


pub const PROGRESS_BAR_STYLE: &str = "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}";


#[macro_export]
macro_rules! create_sim_supervisor {
    ($setup:expr) => {
        Result::<SimulationSupervisor::<SimTypePosition, SimTypeForce, SimTypeVelocity, CellAgentType, SimTypeIndex, SimTypeVoxel, SimTypeDomain>, Box<dyn std::error::Error>>::from($setup).unwrap()
    }
}


pub struct PlottingConfig//<PlotCellsFunc, PlotDomainFunc, Cel, Dom>
{
    pub image_size: u32,
    pub n_threads: Option<usize>,
}


impl Default for PlottingConfig
{
    fn default() -> Self {
        PlottingConfig {
            image_size: 1000,
            n_threads: None,
        }
    }
}


impl<Pos, For, Inf, Vel, Cel, Ind, Vox, Dom> SimulationSupervisor<Pos, For, Inf, Vel, Cel, Ind, Vox, Dom>
where
    Dom: Domain<Cel, Ind, Vox> + Clone + 'static,
    Ind: Index + 'static,
    Pos: Position + 'static,
    For: Force + Serialize + for<'a> Deserialize<'a> + 'static,
    Inf: crate::concepts::interaction::InteractionInformation + 'static,
    Vel: Velocity + 'static,
    Vox: Voxel<Ind, Pos, For> + Clone + 'static,
    Cel: CellAgent<Pos, For, Inf, Vel> + 'static,
{
    fn spawn_worker_threads_and_run_sim(&mut self) -> Result<(), SimulationError> {
        let mut handles = Vec::new();
        let mut start_barrier = Barrier::new(self.multivoxelcontainers.len()+1);

        // Create progress bar and define style
        println!("Running Simulation");

        for (l, mut cont) in self.multivoxelcontainers.drain(..).enumerate() {
            // Clone barriers to use them for synchronization in threads
            let mut new_start_barrier = start_barrier.clone();

            // See if we need to save
            let stop_now_new = Arc::clone(&self.stop_now);

            // Copy time evaluation points
            let t_start = self.time.t_start;
            let t_eval = self.time.t_eval.clone();

            // Add bars
            // if self.config.show_progressbar && cont.mvc_id == 0 {
            let show_progressbar = self.config.show_progressbar;
            let style = ProgressStyle::with_template(PROGRESS_BAR_STYLE)?;
            let bar = ProgressBar::new(t_eval.len() as u64);
            bar.set_style(style);

            // Spawn a thread for each multivoxelcontainer that is running
            let handle = thread::Builder::new().name(format!("worker_thread_{:03.0}", l)).spawn(move || -> Result<_, SimulationError> {
                new_start_barrier.wait();

                let mut time = t_start;
                #[allow(unused)]
                let mut iteration = 0u32;
                for (t, _save, save_full) in t_eval {
                    let dt = t - time;
                    time = t;

                    match cont.run_full_update(&t, &dt) {
                        Ok(()) => (),
                        Err(error) => {
                            // TODO this is not always an error in update_mechanics!
                            println!("Encountered error in update_mechanics: {:?}. Stopping simulation.", error);
                            // Make sure to stop all threads after this iteration.
                            stop_now_new.store(true, Ordering::Relaxed);
                        },
                    }

                    if show_progressbar && cont.mvc_id == 0 {
                        bar.inc(1);
                    }

                    // if save_now_new.load(Ordering::Relaxed) {
                    #[cfg(not(feature = "no_db"))]
                    if _save {
                        // Save cells to database
                        cont.save_cells_to_database(&iteration)?;
                    }

                    if save_full {}

                    // Check if we are stopping the simulation now
                    if stop_now_new.load(Ordering::Relaxed) {
                        // new_barrier.wait();
                        break;
                    }
                    iteration += 1;
                }
                if show_progressbar && cont.mvc_id == 0 {
                    bar.finish();
                }
                return Ok(cont);
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
        self.time.t_eval.drain_filter(|(t, _, _)| *t <= end_time);
        self.run_full_sim()?;
        Ok(())
    }

    #[cfg(not(feature = "no_db"))]
    pub fn save_current_setup(&self, iteration: &Option<u32>) -> Result<(), SimulationError> {
        let setup_current = SimulationSetup {
            domain: self.domain.clone(),
            cells: Vec::<Cel>::new(),
            time: TimeSetup { t_start: 0.0, t_eval: Vec::new() },
            meta_params: SimulationMetaParams { n_threads: self.worker_threads.len() },
            #[cfg(feature = "db_sled")]
            database: self.database.clone(),
        };

        let setup_serialized = bincode::serialize(&setup_current)?;
        let key;
        match iteration {
            Some(iter) => key = format!("setup_{:10.0}", iter),
            None => key = "setup_last".to_owned(),
        };
        self.meta_infos.insert(&key, &setup_serialized)?;
        Ok(())
    }

    // TODO find a way to pause the simulation without destroying the threads and
    // send/retrieve information to from the threads to the main thread where
    // the program was executed

    pub fn end_simulation(&mut self) -> Result<(), SimulationError> {
        for thread in self.worker_threads.drain(..) {
            // TODO introduce new error type to gain a error message here!
            // Do not use unwrap anymore
            let t = thread.join().unwrap()?;
            self.multivoxelcontainers.push(t);
        }
        #[cfg(not(feature = "no_db"))]
        self.save_current_setup(&None)?;
        Ok(())
    }

    // ########################################
    // #### DATABASE RELATED FUNCTIONALITY ####
    // ########################################
    #[cfg(feature = "db_sled")]
    pub fn get_cell_uuids_at_iter(&self, iter: &u32) -> Result<Vec<Uuid>, SimulationError> {
        crate::storage::sled_database::io::get_cell_uuids_at_iter::<Cel>(&self.tree_cells, iter)
    }

    #[cfg(feature = "db_sled")]
    pub fn get_cells_at_iter(&self, iter: &u32) -> Result<Vec<CellAgentBox<Cel>>, SimulationError> {
        crate::storage::sled_database::io::get_cells_at_iter::<Cel>(&self.tree_cells, iter)
    }

    #[cfg(feature = "db_sled")]
    pub fn get_cell_history_from_database(&self, uuid: &Uuid) -> Result<Vec<(u32, CellAgentBox<Cel>)>, SimulationError> {
        crate::storage::sled_database::io::get_cell_history_from_database(&self.tree_cells, uuid)
    }

    #[cfg(feature = "db_sled")]
    pub fn get_all_cell_histories(&self) -> Result<HashMap<Uuid, Vec<(u32, CellAgentBox<Cel>)>>, SimulationError> {
        crate::storage::sled_database::io::get_all_cell_histories(&self.tree_cells)
    }

    // ########################################
    // #### PLOTTING RELATED FUNCTIONALITY ####
    // ########################################
    #[cfg(not(feature = "no_db"))]
    pub fn plot_cells_at_iter_bitmap(&self, iteration: u32) -> Result<(), SimulationError>
    where
        Dom: crate::plotting::spatial::CreatePlottingRoot,
        Cel: crate::plotting::spatial::PlotSelf,
    {
        let current_cells = self.get_cells_at_iter(&(iteration as u32))?;

        // Create a plotting root
        let filename = format!("out/cells_at_iter_{:010.0}.png", iteration);
        let image_size = 1000;

        let mut chart = self.domain.domain_raw.create_bitmap_root(image_size as u32, &filename)?;

        use crate::plotting::spatial::*;
        for cell in current_cells {
            // TODO catch this error
            cell.plot_self(&mut chart)?;
        }

        // TODO catch this error
        chart.present()?;
        Ok(())
    }

    #[cfg(not(feature = "no_db"))]
    pub fn plot_cells_at_iter_bitmap_with_cell_plotting_func<PlotCellsFunc>(&self, iteration: u32, cell_plotting_func: PlotCellsFunc) -> Result<(), SimulationError>
    where
        PlotCellsFunc: Fn(&Cel, &mut DrawingArea<BitMapBackend<'_>, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), SimulationError>,
        Dom: crate::plotting::spatial::CreatePlottingRoot,
    {
        let current_cells = self.get_cells_at_iter(&(iteration as u32))?;

        // Create a plotting root
        let filename = format!("out/cells_at_iter_{:010.0}.png", iteration);

        let mut chart = self.domain.domain_raw.create_bitmap_root(self.plotting_config.image_size, &filename)?;

        current_cells.iter().map(|cell| cell_plotting_func(&cell.cell, &mut chart)).collect::<Result<(), SimulationError>>()?;

        chart.present()?;
        Ok(())
    }

    #[cfg(not(feature = "no_db"))]
    pub fn plot_cells_at_every_iter_bitmap(&self) -> Result<(), SimulationError>
    where
        Dom: crate::plotting::spatial::CreatePlottingRoot,
        Cel: crate::plotting::spatial::PlotSelf,
    {
        // Install the pool
        let n_threads = match self.plotting_config.n_threads {
            Some(threads) => threads,
            None => self.meta_params.n_threads,
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build()?;
        pool.install(|| -> Result<(), SimulationError> {
            // Create progress bar for tree deserialization
            let style = ProgressStyle::with_template(PROGRESS_BAR_STYLE)?;
            let cells_at_iter = crate::storage::sled_database::io::deserialize_tree::<Cel>(&self.tree_cells, Some(style.clone()))?;

            // Create progress bar for image generation
            let bar = ProgressBar::new(cells_at_iter.len() as u64);
            bar.set_style(style);

            println!("Generating Images");
            cells_at_iter.into_par_iter().map(|(iteration, current_cells)| -> Result<(), SimulationError> {
                // Create a plotting root
                let filename = format!("out/cells_at_iter_{:010.0}.png", iteration);

                let mut chart = self.domain.domain_raw.create_bitmap_root(self.plotting_config.image_size, &filename)?;

                current_cells
                    .iter()
                    .map(|cell| cell.cell.plot_self(&mut chart))
                    .collect::<Result<(), _>>()?;

                chart.present()?;
                bar.inc(1);
                Ok(())
            }).collect::<Result<Vec<_>, SimulationError>>()?;

            bar.finish();
            Ok(())
        })?;
        Ok(())
    }

    // TODO rework this to accept a general configuration struct or something
    // Idea: have different functions for when a plotting Trait was already implemented or when a configuration is supplied
    // have the configuration provided override existing implementations of the domain
    // ie: The domain has trait implementation how to plot it and Plotting config has a function with same functionality (but possibly different result)
    // then use the provided function in the plotting config
    #[cfg(not(feature = "no_db"))]
    pub fn plot_cells_at_every_iter_bitmap_with_cell_plotting_func<Func>(&self, plotting_function: &Func) -> Result<(), SimulationError>
    where
        Dom: crate::plotting::spatial::CreatePlottingRoot,
        Func: Fn(&Cel, &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), SimulationError> + Send + Sync,
    {
        // Install the pool
        let n_threads = match self.plotting_config.n_threads {
            Some(threads) => threads,
            None => self.meta_params.n_threads,
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build()?;
        pool.install(|| -> Result<(), SimulationError> {
            // Create progress bar for tree deserialization
            let style = ProgressStyle::with_template(PROGRESS_BAR_STYLE)?;
            // Deserialize the database tree
            let cells_at_iter = crate::storage::sled_database::io::deserialize_tree::<Cel>(&self.tree_cells, Some(style.clone()))?;

            // Create progress bar for image generation
            let bar = ProgressBar::new(cells_at_iter.len() as u64);
            bar.set_style(style);

            println!("Generating Images");
            cells_at_iter.into_par_iter().map(|(iteration, current_cells)| -> Result<(), SimulationError> {
                // Create a plotting root
                let filename = format!("out/cells_at_iter_{:010.0}.png", iteration);

                let mut chart = self.domain.domain_raw.create_bitmap_root(self.plotting_config.image_size, &filename)?;

                current_cells
                    .iter()
                    .map(|cell| plotting_function(&cell.cell, &mut chart))
                    .collect::<Result<(), SimulationError>>()?;

                chart.present()?;
                bar.inc(1);
                Ok(())
            }).collect::<Result<Vec<_>, SimulationError>>()?;

            bar.finish();
            Ok(())
        })?;
        Ok(())
    }
}
