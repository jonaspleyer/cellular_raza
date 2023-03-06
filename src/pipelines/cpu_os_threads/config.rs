use crate::concepts::cell::{CellAgent,CellAgentBox};
use crate::concepts::domain::{Index,Domain,Voxel};
use crate::concepts::mechanics::{Position,Force,Velocity};

use super::domain_decomposition::{VoxelBox,DomainBox,MultiVoxelContainer,PosInformation,ForceInformation,PlainIndex,ConcentrationBoundaryInformation,IndexBoundaryInformation};
use super::supervisor::SimulationSupervisor;

use std::collections::{BTreeMap,HashMap};

use crossbeam_channel::{Sender,Receiver,unbounded};
use hurdles::Barrier;

use serde::{Serialize,Deserialize};

use rayon::prelude::*;


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
}


pub const PROGRESS_BAR_STYLE: &str = "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}";


#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct PlottingConfig
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


#[derive(Clone)]
pub struct Strategies<Vox>
where
    Vox: Clone,
{
    pub voxel_definition_strategies: Option<fn(&mut Vox)>,
}


impl<Pos, For, Inf, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular, Cel, Ind, Vox, Dom> SimulationSupervisor<MultiVoxelContainer<Ind, Pos, For, Inf, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular, Vox, Dom, Cel>, Dom>
where
    Dom: Domain<Cel, Ind, Vox> + Clone + 'static,
    Ind: Index + 'static,
    Pos: Serialize + for<'a> Deserialize<'a> + Position + 'static + std::fmt::Debug,
    For: Serialize + for<'a> Deserialize<'a> + Force + 'static,
    Vel: Serialize + for<'a> Deserialize<'a> + Velocity + 'static,
    ConcVecExtracellular: Serialize + for<'a>Deserialize<'a>,
    ConcBoundaryExtracellular: Serialize + for<'a>Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a> + num::Zero,
    Vox: Voxel<Ind, Pos, For> + Clone + 'static,
    Cel: CellAgent<Pos, For, Inf, Vel> + 'static,
{
    pub fn new_with_strategies(setup: SimulationSetup<Dom, Cel>, strategies: Strategies<Vox>) -> SimulationSupervisor<MultiVoxelContainer<Ind, Pos, For, Inf, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular, Vox, Dom, Cel>, Dom>
    where
        Cel: Sized,
    {
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

        let sender_receiver_pairs_boundary_concentrations: Vec<(Sender<ConcentrationBoundaryInformation<ConcBoundaryExtracellular,Ind>>, Receiver<ConcentrationBoundaryInformation<ConcBoundaryExtracellular,Ind>>)> = (0..n_threads).map(|_| unbounded()).collect();
        let sender_receiver_pairs_boundary_index: Vec<(Sender<IndexBoundaryInformation<Ind>>, Receiver<IndexBoundaryInformation<Ind>>)> = (0..n_threads).map(|_| unbounded()).collect();

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
        let tree_voxels = typed_sled::Tree::<String, Vec<u8>>::open(&db, "voxel_storage");
        let meta_infos = typed_sled::Tree::<String, Vec<u8>>::open(&db, "meta_infos");

        // Create all multivoxelcontainers
        multivoxelcontainers = voxel_and_cell_boxes.into_iter().enumerate().map(|(i, (chunk, mut index_to_cells))| {
            // TODO insert all variables correctly into this container here
            let mut voxels: BTreeMap::<PlainIndex, VoxelBox<Ind,Vox,Cel,Pos,For,Vel,ConcVecExtracellular, ConcBoundaryExtracellular,ConcVecIntracellular>> = chunk.clone()
                .into_iter()
                .map(|(plain_index, voxel)| {
                    let cells = match index_to_cells.remove(&plain_index) {
                        Some(cs) => cs,
                        None => Vec::new(),
                    };
                    let neighbors = setup.domain.get_neighbor_voxel_indices(&convert_to_index[&plain_index]).into_iter().map(|i| convert_to_plain_index[&i]).collect::<Vec<_>>();
                    let vbox = VoxelBox::<Ind,Vox,Cel,Pos,For,Vel,ConcVecExtracellular, ConcBoundaryExtracellular,ConcVecIntracellular>::new(
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
            let senders_boundary_index = create_senders!(sender_receiver_pairs_boundary_index);
            let senders_boundary_concentrations = create_senders!(sender_receiver_pairs_boundary_concentrations);

            // Clone database instance
            #[cfg(not(feature = "no_db"))]
            let tree_cells_new = tree_cells.clone();
            let tree_voxels_new = tree_voxels.clone();

            voxels.iter_mut().for_each(|(_, voxelbox)| {
                match strategies.voxel_definition_strategies {
                    Some(strategy) => strategy(&mut voxelbox.voxel),
                    None => (),
                };
            });

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

                senders_boundary_index,
                senders_boundary_concentrations,
                
                receiver_cell: sender_receiver_pairs_cell[i].1.clone(),
                receiver_pos: sender_receiver_pairs_pos[i].1.clone(),
                receiver_force: sender_receiver_pairs_force[i].1.clone(),

                receiver_index: sender_receiver_pairs_boundary_index[i].1.clone(),
                receiver_concentrations: sender_receiver_pairs_boundary_concentrations[i].1.clone(),

                barrier: barrier.clone(),

                #[cfg(not(feature = "no_db"))]
                database_cells: tree_cells_new,
                database_voxels: tree_voxels_new,

                mvc_id: i as u16,
            };

            return cont;
        }).collect();

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

            #[cfg(not(feature = "no_db"))]
            tree_cells,
            #[cfg(not(feature = "no_db"))]
            tree_voxels,
            #[cfg(not(feature = "no_db"))]
            meta_infos,
        }
    }
}
