use crate::concepts::cell::{CellAgent,CellAgentBox,CellularIdentifier};
use crate::concepts::interaction::{CellularReactions,InteractionExtracellularGradient};
use crate::concepts::domain::{Domain,ExtracellularMechanics,Voxel,Concentration};
use crate::concepts::domain::{Index};
use crate::concepts::mechanics::{Position,Force,Velocity};
use crate::concepts::errors::SimulationError;

use crate::storage::sled_database::SledStorageInterface;

use super::domain_decomposition::{AuxiliaryCellPropertyStorage,MultiVoxelContainer,VoxelBox,DomainBox};

use super::config::{PROGRESS_BAR_STYLE,PlottingConfig,SimulationConfig,SimulationMetaParams,SimulationSetup,StorageConfig,TimeSetup};

use std::thread;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool,Ordering};
use std::sync::Arc;

use core::ops::{Mul,Add,AddAssign};

use hurdles::Barrier;

use rayon::prelude::*;

use serde::{Serialize,Deserialize};

use plotters::{
    prelude::{BitMapBackend,Cartesian2d,DrawingArea},
    coord::types::RangedCoordf64,
};

use indicatif::{ProgressBar,ProgressStyle};


/// # Supervisor controlling simulation execution
/// 
pub struct SimulationSupervisor<MVC, Dom, C>
where
    Dom: Serialize + for<'a> Deserialize<'a>,
    C: Serialize + for<'a>Deserialize<'a>,
{
    pub worker_threads: Vec<thread::JoinHandle<Result<MVC, SimulationError>>>,
    pub multivoxelcontainers: Vec<MVC>,

    pub time: TimeSetup,
    pub meta_params: SimulationMetaParams,
    pub storage: StorageConfig,

    pub domain: DomainBox<Dom>,

    pub config: SimulationConfig,
    pub plotting_config: PlottingConfig,

    #[cfg(feature = "db_sled")]
    pub meta_infos: SledStorageInterface<(), SimulationSetup<DomainBox<Dom>, C>>,
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
    Dom
>
SimulationSupervisor<
    MultiVoxelContainer<
        Ind,
        Pos,
        For,
        Inf,
        Vel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
        Vox,
        Dom,
        Cel
    >,
Dom,
Cel>

where
    Dom: 'static + Serialize + for<'a> Deserialize<'a> + Clone,
    Pos: 'static + Serialize + for<'a> Deserialize<'a>,
    For: 'static + Serialize + for<'a> Deserialize<'a>,
    Inf: 'static,
    Vel: 'static + Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: 'static + Serialize + for<'a>Deserialize<'a>,
    ConcBoundaryExtracellular: 'static + Serialize + for<'a>Deserialize<'a>,
    ConcVecIntracellular: 'static + Serialize + for<'a> Deserialize<'a>,
    Cel: 'static + Serialize + for<'a> Deserialize<'a>,
    Ind: 'static + Serialize + for<'a> Deserialize<'a>,
    Vox: 'static + Serialize + for<'a> Deserialize<'a>,
{
    fn spawn_worker_threads_and_run_sim<ConcGradientExtracellular,ConcTotalExtracellular>(&mut self) -> Result<(), SimulationError>
    where
        Dom: Domain<Cel, Ind, Vox>,
        Pos: Position,
        For: Force,
        Inf: crate::concepts::interaction::InteractionInformation,
        Vel: Velocity,
        ConcVecExtracellular: Concentration,
        ConcTotalExtracellular: Concentration,
        ConcVecIntracellular: Concentration,
        ConcBoundaryExtracellular: Send + Sync,
        ConcVecIntracellular: Mul<f64,Output=ConcVecIntracellular> + Add<ConcVecIntracellular,Output=ConcVecIntracellular> + AddAssign<ConcVecIntracellular>,
        Ind: Index,
        Vox: Voxel<Ind, Pos, For>,
        Vox: ExtracellularMechanics<Ind,Pos,ConcVecExtracellular,ConcGradientExtracellular,ConcTotalExtracellular,ConcBoundaryExtracellular>,
        Cel: CellAgent<Pos, For, Inf, Vel> + CellularReactions<ConcVecIntracellular, ConcVecExtracellular> + InteractionExtracellularGradient<Cel, ConcGradientExtracellular>,
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
                let mut iteration = 0u64;
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
        Dom: Domain<Cel, Ind, Vox>,
        Pos: Position,
        For: Force,
        Inf: crate::concepts::interaction::InteractionInformation,
        Vel: Velocity,
        ConcVecExtracellular: Concentration,
        ConcTotalExtracellular: Concentration,
        ConcBoundaryExtracellular: Send + Sync + 'static,
        ConcVecIntracellular: Send + Sync + Concentration,
        ConcVecIntracellular: Mul<f64,Output=ConcVecIntracellular> + Add<ConcVecIntracellular,Output=ConcVecIntracellular> + AddAssign<ConcVecIntracellular>,
        Ind: Index,
        Vox: Voxel<Ind, Pos, For>,
        Vox: ExtracellularMechanics<Ind,Pos,ConcVecExtracellular,ConcGradientExtracellular,ConcTotalExtracellular,ConcBoundaryExtracellular>,
        Cel: CellAgent<Pos, For, Inf, Vel> + CellularReactions<ConcVecIntracellular, ConcVecExtracellular> + InteractionExtracellularGradient<Cel, ConcGradientExtracellular>,
        VoxelBox<Ind, Vox, Cel, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>: Clone,
        AuxiliaryCellPropertyStorage<Pos, For, Vel, ConcVecIntracellular>: Clone
    {
        self.spawn_worker_threads_and_run_sim()?;
        
        Ok(())
    }

    pub fn run_until<ConcTotalExtracellular,ConcGradientExtracellular>(&mut self, end_time: f64) -> Result<(), SimulationError>
    where
        Dom: Domain<Cel, Ind, Vox>,
        Pos: Position,
        For: Force,
        Inf: crate::concepts::interaction::InteractionInformation,
        Vel: Velocity,
        ConcVecExtracellular: Concentration,
        ConcTotalExtracellular: Concentration,
        ConcBoundaryExtracellular: Send + Sync + 'static,
        ConcVecIntracellular: Concentration,
        ConcVecIntracellular: Mul<f64,Output=ConcVecIntracellular> + Add<ConcVecIntracellular,Output=ConcVecIntracellular> + AddAssign<ConcVecIntracellular>,
        Ind: Index,
        Vox: Voxel<Ind, Pos, For>,
        Vox: ExtracellularMechanics<Ind,Pos,ConcVecExtracellular,ConcGradientExtracellular,ConcTotalExtracellular,ConcBoundaryExtracellular>,
        Cel: CellAgent<Pos, For, Inf, Vel> + CellularReactions<ConcVecIntracellular, ConcVecExtracellular> + InteractionExtracellularGRadient<Cel, ConcGradientExtracellular>,
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
            storage: self.storage.clone(),
        };

        self.meta_infos.store_single_element(iteration, (), setup_current)?;
        Ok(())
    }

    // ########################################
    // #### DATABASE RELATED FUNCTIONALITY ####
    // ########################################
    /* #[cfg(feature = "db_sled")]
    pub fn get_cells_at_iter(&self, iter: u64) -> Result<Vec<CellAgentBox<Cel>>, SimulationError>
    where
        Cel: Clone,
    {
        // super::storage_interface::get_cells_at_iter::<CellAgentBox<Cel>>(&self.tree_cells, iter, None, None)
    }

    #[cfg(feature = "db_sled")]
    pub fn get_cell_history(&self, id: &CellularIdentifier) -> Result<Option<Vec<(u64, CellAgentBox<Cel>)>>, SimulationError>
    where
        Cel: Clone,
    {
        super::storage_interface::get_cell_history::<CellAgentBox<Cel>>(&self.tree_cells, id, None, None)
    }

    #[cfg(feature = "db_sled")]
    pub fn get_all_cell_histories(&self) -> Result<HashMap<CellularIdentifier, Vec<(u64, CellAgentBox<Cel>)>>, SimulationError>
    where
        Cel: Clone,
    {
        super::storage_interface::get_all_cell_histories::<CellAgentBox<Cel>>(&self.tree_cells, None, None)
    }

    // ########################################
    // #### PLOTTING RELATED FUNCTIONALITY ####
    // ########################################
    #[cfg(not(feature = "no_db"))]
    pub fn plot_cells_at_iter_bitmap(&self, iteration: u64) -> Result<(), SimulationError>
    where
        Dom: crate::plotting::spatial::CreatePlottingRoot,
        Cel: crate::plotting::spatial::PlotSelf + Clone,
    {
        let current_cells = self.get_cells_at_iter(iteration)?;

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
    pub fn plot_cells_at_iter_bitmap_with_cell_plotting_func<PlotCellsFunc>(&self, iteration: u64, cell_plotting_func: PlotCellsFunc) -> Result<(), SimulationError>
    where
        PlotCellsFunc: Fn(&Cel, &mut DrawingArea<BitMapBackend<'_>, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), SimulationError>,
        Dom: crate::plotting::spatial::CreatePlottingRoot,
        Cel: Clone,
    {
        let current_cells = self.get_cells_at_iter(iteration)?;

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
        Cel: crate::plotting::spatial::PlotSelf + Send + Sync + Clone,
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
            let all_cells = super::storage_interface::get_all_cells::<CellAgentBox<Cel>>(&self.tree_cells, None, Some(style.clone()))?;

            // Create progress bar for image generation
            let bar = ProgressBar::new(all_cells.len() as u64);
            bar.set_style(style);

            println!("Generating Images");
            all_cells.into_par_iter().map(|(iteration, current_cells)| -> Result<(), SimulationError> {
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
        Cel: Send + Sync + Clone,
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
            let all_cells = super::storage_interface::get_all_cells::<CellAgentBox<Cel>>(&self.tree_cells, None, Some(style.clone()))?;

            // Create progress bar for image generation
            let bar = ProgressBar::new(all_cells.len() as u64);
            bar.set_style(style);

            println!("Generating Images");
            all_cells.into_par_iter().map(|(iteration, current_cells)| -> Result<(), SimulationError> {
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
        Pos: Send + Sync + Clone,
        For: Send + Sync + Clone,
        Inf: Send + Sync + Clone,
        Vel: Send + Sync + Clone,
        ConcVecExtracellular: Send + Sync + Clone,
        ConcBoundaryExtracellular: Send + Sync + 'static + Clone,
        ConcVecIntracellular: Send + Sync + Clone,
        Ind: Send + Sync + Clone,
        Dom: crate::plotting::spatial::CreatePlottingRoot + Send + Sync + Clone,
        Vox: Send + Sync + Clone,
        Cel: Send + Sync + Clone,
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
            let all_voxels = super::storage_interface::get_all_voxels::<VoxelBox<Ind, Vox, Cel, Pos, For, Vel, ConcVecExtracellular, ConcBoundaryExtracellular, ConcVecIntracellular>>(&self.tree_voxels, None, Some(style.clone()))?;

            // Create progress bar for image generation
            let bar = ProgressBar::new(all_voxels.len() as u64);
            bar.set_style(style);

            println!("Generating Images");
            use rayon::prelude::*;

            // TODO this is not a parallel iterator!
            all_voxels.into_par_iter()
                .map(|(iteration, voxel_boxes)| -> Result<(), SimulationError> {
                // Create a plotting root
                // TODO make this correct and much nicer
                // let filename = format!("out/cells_at_iter_{:010.0}.png", iteration);
                let mut file_path = self.storage.location.clone();
                file_path.push(format!("cells_at_iter{:010.0}.png", iteration));
                let filename = file_path.into_os_string().into_string().unwrap();

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
    }*/
}
