use crate::concepts::cell::{CellAgent,CellAgentBox};
use crate::concepts::interaction::{CellularReactions,InteractionExtracellularGRadient};
use crate::concepts::domain::{AuxiliaryCellPropertyStorage,Domain,ExtracellularMechanics,Voxel,Concentration,MultiVoxelContainer,VoxelBox};
use crate::concepts::domain::{DomainBox,Index};
use crate::concepts::mechanics::{Position,Force,Velocity};
use crate::concepts::errors::SimulationError;

use super::config::{PROGRESS_BAR_STYLE,PlottingConfig,SimulationConfig,SimulationMetaParams,SimulationSetup,SledDataBaseConfig,TimeSetup};

use std::thread;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool,Ordering};
use std::sync::Arc;

use core::ops::{Mul,Add,AddAssign};
use core::marker::PhantomData;

use hurdles::Barrier;

use uuid::Uuid;

use rayon::prelude::*;

use serde::{Serialize,Deserialize};

use plotters::{
    prelude::{BitMapBackend,Cartesian2d,DrawingArea},
    coord::types::RangedCoordf64,
};

use indicatif::{ProgressBar,ProgressStyle};


/// # Supervisor controlling simulation execution
/// 
pub struct SimulationSupervisor<Pos, For, Inf, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular, Cel, Ind, Vox, Dom>
where
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a> + 'static,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a> + 'static,
    Cel: Serialize + for<'a> Deserialize<'a>,
    Dom: Serialize + for<'a> Deserialize<'a>,
{
    pub(super) worker_threads: Vec<thread::JoinHandle<Result<MultiVoxelContainer<Ind, Pos, For, Inf, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular, Vox, Dom, Cel>, SimulationError>>>,
    pub(super) multivoxelcontainers: Vec<MultiVoxelContainer<Ind, Pos, For, Inf, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular, Vox, Dom, Cel>>,

    pub(super) time: TimeSetup,
    pub(super) meta_params: SimulationMetaParams,
    #[cfg(feature = "db_sled")]
    pub(super) database: SledDataBaseConfig,

    pub(super) domain: DomainBox<Dom>,

    pub config: SimulationConfig,
    pub plotting_config: PlottingConfig,

    // Tree of database
    #[cfg(not(feature = "no_db"))]
    pub(super) tree_cells: typed_sled::Tree<String, Vec<u8>>,
    #[cfg(not(feature = "no_db"))]
    pub(super) tree_voxels: typed_sled::Tree<String, Vec<u8>>,
    #[cfg(not(feature = "no_db"))]
    pub(super) meta_infos: typed_sled::Tree<String, Vec<u8>>,

    // PhantomData for template arguments
    pub(super) phantom_velocity: PhantomData<Vel>,
}


impl<Pos, For, Inf, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular, Cel, Ind, Vox, Dom> SimulationSupervisor<Pos, For, Inf, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular, Cel, Ind, Vox, Dom>
where
    Dom: Serialize + for<'a> Deserialize<'a> + Clone,
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a>Deserialize<'a>,
    ConcBoundaryExtracellular: Serialize + for<'a>Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a>,
    Cel: Serialize + for<'a> Deserialize<'a>,
    Ind: Serialize + for<'a> Deserialize<'a>,
    Vox: Serialize + for<'a> Deserialize<'a>,
{
    fn spawn_worker_threads_and_run_sim<ConcGradientExtracellular,ConcTotalExtracellular>(&mut self) -> Result<(), SimulationError>
    where
        Dom: 'static + Domain<Cel, Ind, Vox>,
        Pos: 'static + Position,
        For: 'static + Force,
        Inf: 'static + crate::concepts::interaction::InteractionInformation,
        Vel: 'static + Velocity,
        ConcVecExtracellular: 'static + Concentration,
        ConcTotalExtracellular: 'static + Concentration,
        ConcVecIntracellular: 'static + Concentration,
        ConcBoundaryExtracellular: 'static + Send + Sync,
        ConcVecIntracellular: Mul<f64,Output=ConcVecIntracellular> + Add<ConcVecIntracellular,Output=ConcVecIntracellular> + AddAssign<ConcVecIntracellular>,
        Ind: 'static + Index,
        Vox: 'static + Voxel<Ind, Pos, For>,
        Vox: ExtracellularMechanics<Ind,Pos,ConcVecExtracellular,ConcGradientExtracellular,ConcTotalExtracellular,ConcBoundaryExtracellular>,
        Cel: 'static + CellAgent<Pos, For, Inf, Vel> + CellularReactions<ConcVecIntracellular, ConcVecExtracellular> + InteractionExtracellularGRadient<Cel, ConcGradientExtracellular>,
        VoxelBox<Ind, Vox, Cel, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>: Clone,
        AuxiliaryCellPropertyStorage<Pos, For, Vel, ConcVecIntracellular>: Clone
    {
        let mut handles = Vec::new();
        let mut start_barrier = Barrier::new(self.multivoxelcontainers.len()+1);

        // Create progress bar and define style
        println!("Running Simulation");

        for (l, mut cont) in self.multivoxelcontainers.drain(..).enumerate() {
            // Clone barriers to use them for synchronization in threads
            let mut new_start_barrier = start_barrier.clone();

            // See if we need to save
            let stop_now_new = Arc::new(AtomicBool::new(false));

            // Copy time evaluation points
            let t_start = self.time.t_start;
            let t_eval = self.time.t_eval.clone();

            // Add bar
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

                    match cont.run_full_update(&dt) {
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

                    if save_full {
                        cont.save_voxels_to_database(&iteration)?;
                    }

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

    pub fn run_full_sim<ConcGradientExtracellular,ConcTotalExtracellular>(&mut self) -> Result<(), SimulationError>
    where
        Dom: 'static + Domain<Cel, Ind, Vox>,
        Pos: 'static + Position,
        For: 'static + Force,
        Inf: 'static + crate::concepts::interaction::InteractionInformation,
        Vel: 'static + Velocity,
        ConcVecExtracellular: 'static + Concentration,
        ConcTotalExtracellular: 'static + Concentration,
        ConcBoundaryExtracellular: Send + Sync + 'static,
        ConcVecIntracellular: Send + Sync + 'static + Concentration,
        ConcVecIntracellular: Mul<f64,Output=ConcVecIntracellular> + Add<ConcVecIntracellular,Output=ConcVecIntracellular> + AddAssign<ConcVecIntracellular>,
        Ind: 'static + Index,
        Vox: 'static + Voxel<Ind, Pos, For>,
        Vox: ExtracellularMechanics<Ind,Pos,ConcVecExtracellular,ConcGradientExtracellular,ConcTotalExtracellular,ConcBoundaryExtracellular>,
        Cel: 'static + CellAgent<Pos, For, Inf, Vel> + CellularReactions<ConcVecIntracellular, ConcVecExtracellular> + InteractionExtracellularGRadient<Cel, ConcGradientExtracellular>,
        VoxelBox<Ind, Vox, Cel, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>: Clone,
        AuxiliaryCellPropertyStorage<Pos, For, Vel, ConcVecIntracellular>: Clone
    {
        self.spawn_worker_threads_and_run_sim()?;
        
        Ok(())
    }

    pub fn run_until<ConcTotalExtracellular,ConcGradientExtracellular>(&mut self, end_time: f64) -> Result<(), SimulationError>
    where
        Dom: 'static + Domain<Cel, Ind, Vox>,
        Pos: 'static + Position,
        For: 'static + Force,
        Inf: 'static + crate::concepts::interaction::InteractionInformation,
        Vel: 'static + Velocity,
        ConcVecExtracellular: 'static + Concentration,
        ConcTotalExtracellular: 'static + Concentration,
        ConcBoundaryExtracellular: Send + Sync + 'static,
        ConcVecIntracellular: 'static + Concentration,
        ConcVecIntracellular: Mul<f64,Output=ConcVecIntracellular> + Add<ConcVecIntracellular,Output=ConcVecIntracellular> + AddAssign<ConcVecIntracellular>,
        Ind: 'static + Index,
        Vox: 'static + Voxel<Ind, Pos, For>,
        Vox: ExtracellularMechanics<Ind,Pos,ConcVecExtracellular,ConcGradientExtracellular,ConcTotalExtracellular,ConcBoundaryExtracellular>,
        Cel: 'static + CellAgent<Pos, For, Inf, Vel> + CellularReactions<ConcVecIntracellular, ConcVecExtracellular> + InteractionExtracellularGRadient<Cel, ConcGradientExtracellular>,
        VoxelBox<Ind, Vox, Cel, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>: Clone,
        AuxiliaryCellPropertyStorage<Pos, For, Vel, ConcVecIntracellular>: Clone
    {
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
    pub fn get_all_cell_histories(&self) -> Result<HashMap<Uuid, Vec<(u32, CellAgentBox<Cel>)>>, SimulationError>
    where
        Cel: Clone,
    {
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
        Pos: Send + Sync,
        For: Send + Sync,
        Inf: Send + Sync,
        Vel: Send + Sync,
        ConcVecExtracellular: Send + Sync,
        ConcBoundaryExtracellular: Send + Sync,
        ConcVecIntracellular: Send + Sync,
        Ind: Send + Sync,
        Dom: crate::plotting::spatial::CreatePlottingRoot + Send + Sync,
        Vox: Send + Sync,
        Cel: crate::plotting::spatial::PlotSelf + Send + Sync,
        CellAgentBox<Cel>: Send + Sync,
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
        Pos: Send + Sync,
        For: Send + Sync,
        Inf: Send + Sync,
        Vel: Send + Sync,
        ConcVecExtracellular: Send + Sync,
        ConcBoundaryExtracellular: Send + Sync,
        ConcVecIntracellular: Send + Sync,
        Ind: Send + Sync,
        Dom: crate::plotting::spatial::CreatePlottingRoot + Send + Sync,
        Vox: Send + Sync,
        Cel: Send + Sync,
        CellAgentBox<Cel>: Send + Sync,
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

    #[cfg(not(feature = "no_db"))]
    pub fn plot_cells_at_every_iter_bitmap_with_cell_plotting_func_and_voxel_plotting_func<FuncCells,FuncVoxels>(&self, plotting_function_cells: &FuncCells, plotting_function_voxels: &FuncVoxels) -> Result<(), SimulationError>
    where
        Pos: Send + Sync,
        For: Send + Sync,
        Inf: Send + Sync,
        Vel: Send + Sync,
        ConcVecExtracellular: Send + Sync,
        ConcBoundaryExtracellular: Send + Sync + 'static,
        ConcVecIntracellular: Send + Sync,
        Ind: Send + Sync,
        Dom: crate::plotting::spatial::CreatePlottingRoot + Send + Sync,
        Vox: Send + Sync,
        Cel: Send + Sync,
        CellAgentBox<Cel>: Send + Sync,
        // VoxelBox<Ind, Vox, Cel, Pos, For, Vel, Conc>: Send + Sync,
        FuncCells: Fn(&Cel, &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), SimulationError> + Send + Sync,
        FuncVoxels: Fn(&Vox, &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), SimulationError> + Send + Sync,
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
            let voxels_at_iter = crate::storage::sled_database::io::voxels_deserialize_tree::<Ind, Vox, Cel, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>(&self.tree_voxels, Some(style.clone()))?;

            // Create progress bar for image generation
            let bar = ProgressBar::new(voxels_at_iter.len() as u64);
            bar.set_style(style);

            println!("Generating Images");
            use rayon::prelude::*;

            // TODO this is not a parallel iterator!
            voxels_at_iter.into_par_iter()
                .map(|(iteration, voxel_boxes)| -> Result<(), SimulationError> {
                // Create a plotting root
                let filename = format!("out/cells_at_iter_{:010.0}.png", iteration);

                let mut chart = self.domain.domain_raw.create_bitmap_root(self.plotting_config.image_size, &filename)?;

                voxel_boxes
                    .iter()
                    .map(|voxelbox| plotting_function_voxels(&voxelbox.voxel, &mut chart))
                    .collect::<Result<(), SimulationError>>()?;

                voxel_boxes
                    .iter()
                    .map(|voxelbox| voxelbox.cells.iter())
                    .flatten()
                    .map(|(cellbox, _)| plotting_function_cells(&cellbox.cell, &mut chart))
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
