use crate::concepts::cell::{CellAgent, CellAgentBox};
use crate::concepts::domain::{Domain, Index, Voxel};
use crate::concepts::mechanics::{Force, Position, Velocity};
use crate::storage::concepts::{StorageInterface, StorageManager};

use super::domain_decomposition::{
    ConcentrationBoundaryInformation, DomainBox, ForceInformation, IndexBoundaryInformation,
    MultiVoxelContainer, PlainIndex, PosInformation, VoxelBox,
};
use super::supervisor::SimulationSupervisor;
#[cfg(feature = "controller")]
use super::supervisor::ControllerBox;
use crate::concepts::cell::CellularIdentifier;

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

use crossbeam_channel::{unbounded, Receiver, Sender};
use hurdles::Barrier;

use serde::{Deserialize, Serialize};

use rayon::prelude::*;

#[derive(Serialize, Deserialize, Clone, Debug)]
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
#[derive(Clone, Serialize, Deserialize)]
pub struct SimulationMetaParams {
    pub n_threads: usize,
}

// TODO rethink how to specify time points to save
// we need to frequently save cells and environment
// Sometimes we need full snapshots for recovery purposes
#[derive(Clone, Serialize, Deserialize)]
pub struct TimeSetup {
    pub t_start: f64,
    pub t_eval: Vec<(f64, bool)>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub location: std::path::PathBuf,
}

/// # Complete Set of parameters controlling execution flow of simulation
#[derive(Clone, Serialize, Deserialize)]
pub struct SimulationSetup<Dom, Cel, Cont = ()> {
    pub(crate) domain: Dom,
    pub(crate) cells: Vec<Cel>,
    pub(crate) time: TimeSetup,
    pub(crate) meta_params: SimulationMetaParams,
    pub(crate) storage: StorageConfig,
    #[cfg(feature = "controller")]
    pub(crate) controller: Cont,
    pub(crate) phantom_cont: PhantomData<Cont>,
}

// TODO rework this do not write two differenct functions
#[cfg(not(feature = "controller"))]
impl<Dom, Cel> SimulationSetup<Dom, Cel> {
    pub fn new<V>(
        domain: Dom,
        cells: V,
        time: TimeSetup,
        meta_params: SimulationMetaParams,
        storage: StorageConfig,
    ) -> SimulationSetup<Dom, Cel>
    where
        V: IntoIterator<Item = Cel>,
    {
        SimulationSetup {
            domain,
            cells: cells.into_iter().collect(),
            time,
            meta_params,
            storage,
            phantom_cont: PhantomData,
        }
    }
}

#[cfg(feature = "controller")]
impl<Dom, Cel, Cont> SimulationSetup<Dom, Cel, Cont> {
    pub fn new<V>(
        domain: Dom,
        cells: V,
        time: TimeSetup,
        meta_params: SimulationMetaParams,
        storage: StorageConfig,
        controller: Cont,
    ) -> SimulationSetup<Dom, Cel, Cont>
    where
        V: IntoIterator<Item = Cel>,
    {
        SimulationSetup {
            domain,
            cells: cells.into_iter().collect(),
            time,
            meta_params,
            storage,
            controller,
            phantom_cont: PhantomData,
        }
    }
}

pub const PROGRESS_BAR_STYLE: &str =
    "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ImageType {
    BitMap,
    // TODO
    // Svg,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PlottingConfig {
    pub image_size: u32,
    pub n_threads: Option<usize>,
    pub image_type: ImageType,
    pub show_progressbar: bool,
}

impl Default for PlottingConfig {
    fn default() -> Self {
        PlottingConfig {
            image_size: 1000,
            n_threads: None,
            image_type: ImageType::BitMap,
            show_progressbar: true,
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

impl<
        Pos,
        For,
        Inf,
        Vel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
        Cel,
        Ind,
        Vox,
        Dom,
        Cont,
        Obs,
    >
    SimulationSupervisor<
        MultiVoxelContainer<
            Ind,
            Pos,
            Vel,
            For,
            Inf,
            Vox,
            Dom,
            Cel,
            ConcVecExtracellular,
            ConcBoundaryExtracellular,
            ConcVecIntracellular,
        >,
        Dom,
        Cel,
        Cont,
        Obs,
    >
where
    Dom: Domain<Cel, Ind, Vox> + Clone + 'static,
    Ind: Index + 'static,
    Pos: Serialize + for<'a> Deserialize<'a> + Position + 'static + std::fmt::Debug,
    For: Serialize + for<'a> Deserialize<'a> + Force + 'static,
    Vel: Serialize + for<'a> Deserialize<'a> + Velocity + 'static,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a> + num::Zero,
    Vox: Voxel<Ind, Pos, Vel, For> + Clone + 'static,
    Cel: CellAgent<Pos, Vel, For, Inf> + 'static,
    VoxelBox<
        Ind,
        Pos,
        Vel,
        For,
        Vox,
        Cel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
    >: Clone,
    Cont: Serialize + for<'a> Deserialize<'a>,
{
    pub fn initialize_from_setup(
        setup: SimulationSetup<Dom, Cel, Cont>,
    ) -> SimulationSupervisor<
        MultiVoxelContainer<
            Ind,
            Pos,
            Vel,
            For,
            Inf,
            Vox,
            Dom,
            Cel,
            ConcVecExtracellular,
            ConcBoundaryExtracellular,
            ConcVecIntracellular,
        >,
        Dom,
        Cel,
        Cont,
        Obs,
    >
    where
        Cel: Sized,
    {
        Self::initialize_with_strategies(
            setup,
            Strategies {
                voxel_definition_strategies: None,
            },
        )
    }

    pub fn initialize_with_strategies(
        mut setup: SimulationSetup<Dom, Cel, Cont>,
        strategies: Strategies<Vox>,
    ) -> SimulationSupervisor<
        MultiVoxelContainer<
            Ind,
            Pos,
            Vel,
            For,
            Inf,
            Vox,
            Dom,
            Cel,
            ConcVecExtracellular,
            ConcBoundaryExtracellular,
            ConcVecIntracellular,
        >,
        Dom,
        Cel,
        Cont,
        Obs,
    >
    where
        Cel: Sized,
    {
        // Create groups of voxels to put into our MultiVelContainers
        let (n_threads, voxel_chunks) = <Dom>::generate_contiguous_multi_voxel_regions(
            &setup.domain,
            setup.meta_params.n_threads,
        )
        .unwrap();

        let convert_to_plain_index: BTreeMap<Ind, PlainIndex> = setup
            .domain
            .get_all_indices()
            .into_iter()
            .enumerate()
            .map(|(count, ind)| (ind, count as PlainIndex))
            .collect();
        let convert_to_index: BTreeMap<PlainIndex, Ind> = convert_to_plain_index
            .iter()
            .map(|(i, j)| (j.clone(), i.clone()))
            .collect();

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
        let sender_receiver_pairs_cell: Vec<(
            Sender<CellAgentBox<Cel>>,
            Receiver<CellAgentBox<Cel>>,
        )> = (0..n_threads).map(|_| unbounded()).collect();
        let sender_receiver_pairs_pos: Vec<(
            Sender<PosInformation<Pos, Vel, Inf>>,
            Receiver<PosInformation<Pos, Vel, Inf>>,
        )> = (0..n_threads).map(|_| unbounded()).collect();
        let sender_receiver_pairs_force: Vec<(
            Sender<ForceInformation<For>>,
            Receiver<ForceInformation<For>>,
        )> = (0..n_threads).map(|_| unbounded()).collect();

        let sender_receiver_pairs_boundary_concentrations: Vec<(
            Sender<ConcentrationBoundaryInformation<ConcBoundaryExtracellular, Ind>>,
            Receiver<ConcentrationBoundaryInformation<ConcBoundaryExtracellular, Ind>>,
        )> = (0..n_threads).map(|_| unbounded()).collect();
        let sender_receiver_pairs_boundary_index: Vec<(
            Sender<IndexBoundaryInformation<Ind>>,
            Receiver<IndexBoundaryInformation<Ind>>,
        )> = (0..n_threads).map(|_| unbounded()).collect();

        // Create a barrier to synchronize all threads
        let barrier = Barrier::new(n_threads);

        // Create an intermediate mapping just for this setup
        // Map voxel index to thread number
        let mut plain_index_to_thread: BTreeMap<PlainIndex, usize> = BTreeMap::new();
        for (i, chunk) in voxel_chunks.iter().enumerate() {
            for (index, _) in chunk {
                plain_index_to_thread.insert(convert_to_plain_index[&index], i);
            }
        }

        // Sort cells into correct voxels
        let n_chunks = voxel_chunks.len();
        let chunk_size = (setup.cells.len() as f64 / n_threads as f64).ceil() as usize;

        let mut sorted_cells = setup
            .cells
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
                            None => {
                                index_to_cells.insert(plain_index, vec![cell]);
                            }
                        },
                        None => {
                            cs.insert(id_thread, BTreeMap::from([(plain_index, vec![cell])]));
                        }
                    }
                }
                cs
            })
            .reduce(
                || {
                    (0..n_chunks)
                        .map(|i| (i, BTreeMap::new()))
                        .collect::<BTreeMap<usize, BTreeMap<PlainIndex, Vec<(usize, Cel)>>>>()
                },
                |mut acc, x| {
                    for (id_thread, idc) in x.into_iter() {
                        for (index, mut cells) in idc.into_iter() {
                            match acc.get_mut(&id_thread) {
                                Some(index_to_cells) => match index_to_cells.get_mut(&index) {
                                    Some(cs) => cs.append(&mut cells),
                                    None => {
                                        index_to_cells.insert(index, cells);
                                    }
                                },
                                None => {
                                    acc.insert(id_thread, BTreeMap::from([(index, cells)]));
                                }
                            }
                        }
                    }
                    return acc;
                },
            );

        let voxel_and_raw_cells: Vec<(
            Vec<(PlainIndex, Vox)>,
            BTreeMap<PlainIndex, Vec<(usize, Cel)>>,
        )> = voxel_chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                (
                    chunk
                        .into_iter()
                        .map(|(ind, vox)| (convert_to_plain_index[&ind], vox))
                        .collect(),
                    sorted_cells.remove(&i).unwrap(),
                )
            })
            .collect();

        let voxel_and_cell_boxes: Vec<(
            Vec<(PlainIndex, Vox)>,
            BTreeMap<PlainIndex, Vec<CellAgentBox<Cel>>>,
        )> = voxel_and_raw_cells
            .into_iter()
            .map(|(chunk, sorted_cells)| {
                let res = (
                    chunk,
                    sorted_cells
                        .into_iter()
                        .map(|(ind, mut cells)| {
                            cells.sort_by(|(i, _), (j, _)| i.cmp(&j));
                            (
                                ind,
                                cells
                                    .into_iter()
                                    .enumerate()
                                    .map(|(n_cell, (_, cell))| {
                                        CellAgentBox::new(ind, n_cell as u64, cell, None)
                                    })
                                    .collect(),
                            )
                        })
                        .collect(),
                );
                res
            })
            .collect();

        // Format the current time
        let date = chrono::Local::now().format("%Y-%m-%d-%H:%M:%S");
        let location_with_time = setup.storage.location.join(format!("{}", date));
        setup.storage.location = location_with_time.into();

        // Create
        let meta_infos_path = setup.storage.location.clone().join("meta_infos");
        let meta_infos =
            StorageManager::<(), SimulationSetup<DomainBox<Dom>, Cel, Cont>>::open_or_create(
                &meta_infos_path,
                0,
            )
            .unwrap();

        // Create all multivoxelcontainers
        multivoxelcontainers = voxel_and_cell_boxes
            .into_iter()
            .enumerate()
            .map(|(i, (chunk, mut index_to_cells))| {
                let mut voxels: BTreeMap<
                    PlainIndex,
                    VoxelBox<
                        Ind,
                        Pos,
                        Vel,
                        For,
                        Vox,
                        Cel,
                        ConcVecExtracellular,
                        ConcBoundaryExtracellular,
                        ConcVecIntracellular,
                    >,
                > = chunk
                    .clone()
                    .into_iter()
                    .map(|(plain_index, voxel)| {
                        let cells = match index_to_cells.remove(&plain_index) {
                            Some(cs) => cs,
                            None => Vec::new(),
                        };
                        let neighbors = setup
                            .domain
                            .get_neighbor_voxel_indices(&convert_to_index[&plain_index])
                            .into_iter()
                            .map(|i| convert_to_plain_index[&i])
                            .collect::<Vec<_>>();
                        let vbox = VoxelBox::<
                            Ind,
                            Pos,
                            Vel,
                            For,
                            Vox,
                            Cel,
                            ConcVecExtracellular,
                            ConcBoundaryExtracellular,
                            ConcVecIntracellular,
                        >::new(
                            plain_index,
                            convert_to_index[&plain_index].clone(),
                            voxel,
                            neighbors,
                            cells,
                        );
                        (plain_index, vbox)
                    })
                    .collect();

                // Create non-occupied voxels
                for (ind, _) in voxels.iter() {
                    match index_to_cells.get(&ind) {
                        Some(_) => (),
                        None => {
                            index_to_cells.insert(ind.clone(), Vec::new());
                        }
                    }
                }

                // Quick macro to create senders
                macro_rules! create_senders {
                    ($sr_pairs: expr) => {
                        chunk
                            .clone()
                            .into_iter()
                            .map(|(plain_index, _)| {
                                setup
                                    .domain
                                    .get_neighbor_voxel_indices(&convert_to_index[&plain_index])
                                    .into_iter()
                                    .map(|ind| {
                                        (
                                            index_to_thread[&ind],
                                            $sr_pairs[index_to_thread[&ind]].0.clone(),
                                        )
                                    })
                            })
                            .flatten()
                            .collect::<HashMap<usize, _>>()
                    };
                }

                let senders_cell = create_senders!(sender_receiver_pairs_cell);
                let senders_pos = create_senders!(sender_receiver_pairs_pos);
                let senders_force = create_senders!(sender_receiver_pairs_force);
                let senders_boundary_index = create_senders!(sender_receiver_pairs_boundary_index);
                let senders_boundary_concentrations =
                    create_senders!(sender_receiver_pairs_boundary_concentrations);

                let storage_cells_path = setup.storage.location.clone().join("cell_storage");
                let storage_voxels_path = setup.storage.location.clone().join("voxel_storage");

                // TODO catch these errors!
                let storage_cells =
                    StorageManager::<CellularIdentifier, CellAgentBox<Cel>>::open_or_create(
                        &storage_cells_path,
                        i as u64,
                    )
                    .unwrap();
                let storage_voxels =
                    StorageManager::<
                        PlainIndex,
                        VoxelBox<
                            Ind,
                            Pos,
                            Vel,
                            For,
                            Vox,
                            Cel,
                            ConcVecExtracellular,
                            ConcBoundaryExtracellular,
                            ConcVecIntracellular,
                        >,
                    >::open_or_create(&storage_voxels_path, i as u64)
                    .unwrap();

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
                    receiver_concentrations: sender_receiver_pairs_boundary_concentrations[i]
                        .1
                        .clone(),

                    barrier: barrier.clone(),

                    storage_cells,
                    storage_voxels,

                    mvc_id: i as u32,
                };

                return cont;
            })
            .collect();

        SimulationSupervisor {
            worker_threads: Vec::new(),
            multivoxelcontainers,

            time: setup.time,
            meta_params: setup.meta_params,
            storage: setup.storage,

            domain: setup.domain.into(),

            config: SimulationConfig::default(),

            meta_infos,

            #[cfg(feature = "controller")]
            controller_box: std::sync::Arc::new(std::sync::Mutex::new(ControllerBox {
                controller: setup.controller,
                measurements: HashMap::new(),
            })),
            phantom_cont: PhantomData,
            phantom_obs: PhantomData,
        }
    }
}
