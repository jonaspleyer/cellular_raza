use cellular_raza_concepts::cell::{CellAgent, CellAgentBox, CellularIdentifier};
use cellular_raza_concepts::domain::Index;
use cellular_raza_concepts::domain::{
    Concentration, Controller, Domain, ExtracellularMechanics, Voxel,
};
use cellular_raza_concepts::errors::{ControllerError, DrawingError, RequestError};
use cellular_raza_concepts::interaction::{CellularReactions, InteractionExtracellularGradient};
use cellular_raza_concepts::mechanics::{Force, Position, Velocity};
use cellular_raza_concepts::plotting::{CreatePlottingRoot, PlotSelf};

use super::errors::*;
use crate::storage::concepts::{StorageInterface, StorageManager};

use super::domain_decomposition::{
    AuxiliaryCellPropertyStorage, DomainBox, MultiVoxelContainer, VoxelBox,
};

use super::config::{
    ImageType, PlottingConfig, SimulationConfig, SimulationMetaParams, SimulationSetup, TimeSetup,
    PROGRESS_BAR_STYLE,
};

use super::config::StorageConfig;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use core::ops::{Add, AddAssign, Mul};

use core::marker::PhantomData;

use hurdles::Barrier;

use serde::{Deserialize, Serialize};

use plotters::{
    coord::types::RangedCoordf64,
    prelude::{BitMapBackend, Cartesian2d, DrawingArea},
};

use indicatif::{ProgressBar, ProgressStyle};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct ControllerBox<Cont, Obs> {
    pub controller: Cont,
    pub measurements: std::collections::BTreeMap<u64, std::collections::BTreeMap<u32, Obs>>,
}

impl<Cont, Obs> ControllerBox<Cont, Obs> {
    fn measure<'a, I, Cel>(
        &mut self,
        iteration: u64,
        thread_index: u32,
        cells: I,
    ) -> Result<(), SimulationError>
    where
        Cel: 'a + Serialize + for<'b> Deserialize<'b>,
        I: Iterator<Item = &'a CellAgentBox<Cel>> + Clone,
        Cont: Controller<Cel, Obs>,
    {
        let obs = self.controller.measure(cells)?;
        let entry = self
            .measurements
            .entry(iteration)
            .or_insert(std::collections::BTreeMap::new());
        entry.insert(thread_index, obs);

        // If the number of entries is above the limit defined by Controller::N, we omit the first results
        while self.measurements.len() > Cont::N_SAVE {
            self.measurements.pop_first();
        }
        Ok(())
    }

    fn adjust<'a, Cel, J>(&mut self, iteration: u64, cells: J) -> Result<(), ControllerError>
    where
        Cel: 'a + Serialize + for<'b> Deserialize<'b>,
        J: Iterator<
            Item = (
                &'a mut CellAgentBox<Cel>,
                &'a mut Vec<cellular_raza_concepts::cycle::CycleEvent>,
            ),
        >,
        Cont: Controller<Cel, Obs>,
    {
        match self.measurements.get_mut(&iteration) {
            Some(measurements) => self.controller.adjust(measurements.values(), cells),
            None => Err(ControllerError {
                message: format!("could not find measurements at iteration {}", iteration),
            }),
        }
    }
}

/// # Supervisor controlling simulation execution
///
pub struct SimulationSupervisor<MVC, Dom, Cel, Cont = (), Obs = ()>
where
    Cel: Serialize + for<'a> Deserialize<'a>,
    Dom: Serialize + for<'a> Deserialize<'a>,
    Cont: Serialize + for<'a> Deserialize<'a>,
{
    pub(crate) worker_threads: Vec<thread::JoinHandle<Result<MVC, SimulationError>>>,
    pub(crate) multivoxelcontainers: Vec<MVC>,

    pub(crate) time: TimeSetup,
    pub(crate) meta_params: SimulationMetaParams,
    pub(crate) storage: StorageConfig,

    pub(crate) domain: DomainBox<Dom>,

    pub config: SimulationConfig,

    pub(crate) meta_infos: StorageManager<(), SimulationSetup<DomainBox<Dom>, Cel, Cont>>,

    pub(crate) controller_box: Arc<std::sync::Mutex<ControllerBox<Cont, Obs>>>,
    pub(crate) phantom_obs: PhantomData<Obs>,
    pub(crate) phantom_cont: PhantomData<Cont>,
}

impl<
        Pos,
        Vel,
        For,
        Inf,
        Cel,
        Ind,
        Vox,
        Dom,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
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
    Dom: 'static + Serialize + for<'a> Deserialize<'a>,
    Pos: 'static + Serialize + for<'a> Deserialize<'a>,
    For: 'static + Serialize + for<'a> Deserialize<'a>,
    Inf: 'static,
    Vel: 'static + Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: 'static + Serialize + for<'a> Deserialize<'a>,
    ConcBoundaryExtracellular: 'static + Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: 'static + Serialize + for<'a> Deserialize<'a>,
    Cel: 'static + Serialize + for<'a> Deserialize<'a>,
    Ind: 'static + Serialize + for<'a> Deserialize<'a>,
    Vox: 'static + Serialize + for<'a> Deserialize<'a>,
    Cont: 'static + Serialize + for<'a> Deserialize<'a> + Send + Sync,
    Obs: 'static + Send + Sync,
{
    fn spawn_worker_threads_and_run_sim<ConcGradientExtracellular, ConcTotalExtracellular>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Dom: Domain<Cel, Ind, Vox>,
        Pos: Position,
        For: Force,
        Inf: cellular_raza_concepts::interaction::InteractionInformation,
        Vel: Velocity,
        ConcVecExtracellular: Concentration,
        ConcTotalExtracellular: Concentration,
        ConcVecIntracellular: Concentration,
        ConcBoundaryExtracellular: Send + Sync,
        ConcVecIntracellular: Mul<f64, Output = ConcVecIntracellular>
            + Add<ConcVecIntracellular, Output = ConcVecIntracellular>
            + AddAssign<ConcVecIntracellular>,
        Ind: Index,
        Vox: Voxel<Ind, Pos, Vel, For>,
        Vox: ExtracellularMechanics<
            Ind,
            Pos,
            ConcVecExtracellular,
            ConcGradientExtracellular,
            ConcTotalExtracellular,
            ConcBoundaryExtracellular,
        >,
        Cel: CellAgent<Pos, Vel, For, Inf>
            + CellularReactions<ConcVecIntracellular, ConcVecExtracellular>
            + InteractionExtracellularGradient<Cel, ConcGradientExtracellular>,
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
        AuxiliaryCellPropertyStorage<Pos, Vel, For, ConcVecIntracellular>: Clone,
        Cont: Controller<Cel, Obs>,
    {
        let mut handles = Vec::new();
        let mut start_barrier = Barrier::new(self.multivoxelcontainers.len() + 1);
        let controller_barrier = Barrier::new(self.multivoxelcontainers.len());

        // Create progress bar and define style
        if self.config.show_progressbar {
            println!("Running Simulation");
        }

        for (l, mut cont) in self.multivoxelcontainers.drain(..).enumerate() {
            // Clone barriers to use them for synchronization in threads
            let mut new_start_barrier = start_barrier.clone();
            let mut controller_barrier_new = controller_barrier.clone();

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

            let controller_box = self.controller_box.clone();

            // Spawn a thread for each multivoxelcontainer that is running
            let handle = thread::Builder::new().name(format!("worker_thread_{:03.0}", l)).spawn(move || -> Result<_, SimulationError> {
                new_start_barrier.wait();

                let mut time = t_start;
                #[allow(unused)]
                let mut iteration = 0u64;
                for (t, save) in t_eval {
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
                    if save {
                        cont.save_voxels_to_database(&iteration)?;
                        cont.save_cells_to_database(&iteration)?;
                    }

                    // TODO Make sure to only call this if the controller type is not ()
                    {
                        controller_box.lock().unwrap().measure(
                            iteration,
                            l as u32,
                            cont.voxels
                            .iter()
                            .flat_map(|vox| vox.1.cells
                                .iter()
                                .map(|(cbox, _)| cbox)
                            )
                        )?;
                        controller_barrier_new.wait();
                        controller_box.lock().unwrap().adjust(iteration,
                            cont.voxels
                            .iter_mut()
                            .flat_map(|vox| vox.1.cells
                                .iter_mut()
                                .map(|(cbox, aux_storage)| (cbox, &mut aux_storage.cycle_events))
                            )
                        )?;
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

    pub fn run_full_sim<ConcGradientExtracellular, ConcTotalExtracellular>(
        &mut self,
    ) -> Result<
        SimulationResult<
            Ind,
            Pos,
            For,
            Vel,
            ConcVecExtracellular,
            ConcBoundaryExtracellular,
            ConcVecIntracellular,
            Vox,
            Dom,
            Cel,
        >,
        SimulationError,
    >
    where
        Dom: Domain<Cel, Ind, Vox>,
        Pos: Position,
        For: Force,
        Inf: cellular_raza_concepts::interaction::InteractionInformation,
        Vel: Velocity,
        ConcVecExtracellular: Concentration,
        ConcTotalExtracellular: Concentration,
        ConcBoundaryExtracellular: Send + Sync + 'static,
        ConcVecIntracellular: Send + Sync + Concentration,
        ConcVecIntracellular: Mul<f64, Output = ConcVecIntracellular>
            + Add<ConcVecIntracellular, Output = ConcVecIntracellular>
            + AddAssign<ConcVecIntracellular>,
        Ind: Index,
        Vox: Voxel<Ind, Pos, Vel, For>,
        Vox: ExtracellularMechanics<
            Ind,
            Pos,
            ConcVecExtracellular,
            ConcGradientExtracellular,
            ConcTotalExtracellular,
            ConcBoundaryExtracellular,
        >,
        Cel: CellAgent<Pos, Vel, For, Inf>
            + CellularReactions<ConcVecIntracellular, ConcVecExtracellular>
            + InteractionExtracellularGradient<Cel, ConcGradientExtracellular>,
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
        AuxiliaryCellPropertyStorage<Pos, Vel, For, ConcVecIntracellular>: Clone,
        Cont: Controller<Cel, Obs>,
    {
        // Run the simulation
        self.spawn_worker_threads_and_run_sim()?;

        // Collect all threads
        let mut databases = Vec::new();
        for thread in self.worker_threads.drain(..) {
            let t = thread
                .join()
                .expect("Could not join threads after Simulation has finished")?;
            databases.push((t.storage_cells, t.storage_voxels, t.domain))
        }

        // Create a simulationresult which can then be used to further plot and analyze results
        let (storage_cells, storage_voxels, domain) = databases.pop().ok_or(RequestError {
            message: format!("The threads of the simulation did not yield any handles"),
        })?;

        let simulation_result = SimulationResult {
            storage: self.storage.clone(),
            domain,
            storage_cells,
            storage_voxels,
            plotting_config: PlottingConfig::default(),
        };

        Ok(simulation_result)
    }

    pub fn save_current_setup(&self, iteration: u64) -> Result<(), SimulationError>
    where
        Dom: Clone,
        Cont: Clone,
        Obs: Clone,
    {
        let setup_current = SimulationSetup {
            domain: self.domain.clone(),
            cells: Vec::<Cel>::new(),
            time: TimeSetup {
                t_start: 0.0,
                t_eval: Vec::new(),
            },
            meta_params: self.meta_params.clone(),
            storage: self.storage.clone(),
            controller: self.controller_box.lock().unwrap().controller.clone(),
        };

        self.meta_infos
            .store_single_element(iteration, &(), &setup_current)?;
        Ok(())
    }
}

use super::domain_decomposition::PlainIndex;

pub struct SimulationResult<
    Ind,
    Pos,
    For,
    Vel,
    ConcVecExtracellular,
    ConcBoundaryExtracellular,
    ConcVecIntracellular,
    Vox,
    Dom,
    Cel,
> where
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    Cel: Serialize + for<'a> Deserialize<'a>,
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
    >: for<'a> Deserialize<'a> + Serialize,
    Dom: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a> + 'static,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a>,
{
    pub storage: StorageConfig,

    pub domain: DomainBox<Dom>,
    pub storage_cells: StorageManager<CellularIdentifier, CellAgentBox<Cel>>,
    pub storage_voxels: StorageManager<
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
    >,
    pub plotting_config: PlottingConfig,
}

impl<
        Ind,
        Pos,
        For,
        Vel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
        Vox,
        Dom,
        Cel,
    >
    SimulationResult<
        Ind,
        Pos,
        For,
        Vel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
        Vox,
        Dom,
        Cel,
    >
where
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    Cel: Serialize + for<'a> Deserialize<'a>,
    Dom: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a> + 'static,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a>,
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
    >: for<'a> Deserialize<'a> + Serialize,
{
    fn plot_spatial_at_iteration_with_functions<Cpf, Vpf, Dpf>(
        &self,
        iteration: u64,
        cell_plotting_func: Cpf,
        voxel_plotting_func: Vpf,
        domain_plotting_func: Dpf,
        progress_bar: Option<indicatif::ProgressBar>,
    ) -> Result<(), SimulationError>
    where
        Dpf: for<'a> Fn(
            &Dom,
            u32,
            &'a String,
        ) -> Result<
            DrawingArea<BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            DrawingError,
        >,
        Cpf: Fn(
                &Cel,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        Vpf: Fn(
                &Vox,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
    {
        // Obtain the voxels from the database
        let voxel_boxes = self
            .storage_voxels
            .load_all_elements_at_iteration(iteration)?
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();

        // Choose the correct file path
        let mut file_path = self.storage.get_location().clone();
        file_path.push("images");
        match std::fs::create_dir(&file_path) {
            Ok(()) => (),
            Err(_) => (),
        }
        file_path.push(format!("cells_at_iter_{:010.0}.png", iteration));
        let filename = file_path.into_os_string().into_string().unwrap();

        let mut chart = domain_plotting_func(
            &self.domain.domain_raw,
            self.plotting_config.image_size,
            &filename,
        )?;

        voxel_boxes
            .iter()
            .map(|voxelbox| voxel_plotting_func(&voxelbox.voxel, &mut chart))
            .collect::<Result<(), DrawingError>>()?;

        voxel_boxes
            .iter()
            .map(|voxelbox| voxelbox.cells.iter())
            .flatten()
            .map(|(cellbox, _)| cell_plotting_func(&cellbox.cell, &mut chart))
            .collect::<Result<(), DrawingError>>()?;

        chart.present()?;

        match progress_bar {
            Some(bar) => bar.inc(1),
            None => (),
        }

        Ok(())
    }

    pub fn plot_spatial_at_iteration(&self, iteration: u64) -> Result<(), SimulationError>
    where
        Dom: CreatePlottingRoot,
        Cel: PlotSelf,
        Vox: PlotSelf,
    {
        match self.plotting_config.image_type {
            ImageType::BitMap => self.plot_spatial_at_iteration_with_functions(
                iteration,
                Cel::plot_self_bitmap,
                Vox::plot_self_bitmap,
                Dom::create_bitmap_root,
                None,
            ),
            // ImageType::Svg => self.plot_spatial_at_iteration_with_functions(iteration, C::plot_self::<BitMapBackend>, V::plot_self, D::create_svg_root),
        }
    }

    pub fn plot_spatial_at_iteration_custom_functions<Cpf, Vpf, Dpf>(
        &self,
        iteration: u64,
        cell_plotting_func: Cpf,
        voxel_plotting_func: Vpf,
        domain_plotting_func: Dpf,
    ) -> Result<(), SimulationError>
    where
        Dpf: for<'a> Fn(
            &Dom,
            u32,
            &'a String,
        ) -> Result<
            DrawingArea<BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            DrawingError,
        >,
        Cpf: Fn(
                &Cel,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        Vpf: Fn(
                &Vox,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
    {
        self.plot_spatial_at_iteration_with_functions(
            iteration,
            cell_plotting_func,
            voxel_plotting_func,
            domain_plotting_func,
            None,
        )
    }

    pub fn plot_spatial_at_iteration_custom_cell_voxel_functions<Cpf, Vpf>(
        &self,
        iteration: u64,
        cell_plotting_func: Cpf,
        voxel_plotting_func: Vpf,
    ) -> Result<(), SimulationError>
    where
        Dom: CreatePlottingRoot,
        Cpf: Fn(
                &Cel,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        Vpf: Fn(
                &Vox,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
    {
        self.plot_spatial_at_iteration_with_functions(
            iteration,
            cell_plotting_func,
            voxel_plotting_func,
            Dom::create_bitmap_root,
            None,
        )
    }

    pub fn plot_spatial_at_iteration_custom_cell_funtion<Cpf>(
        &self,
        iteration: u64,
        cell_plotting_func: Cpf,
    ) -> Result<(), SimulationError>
    where
        Vox: PlotSelf,
        Dom: CreatePlottingRoot,
        Cpf: Fn(
                &Cel,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
    {
        match self.plotting_config.image_type {
            ImageType::BitMap => self.plot_spatial_at_iteration_with_functions(
                iteration,
                cell_plotting_func,
                Vox::plot_self_bitmap,
                Dom::create_bitmap_root,
                None,
            ),
        }
    }

    fn build_thread_pool(&self) -> Result<rayon::ThreadPool, SimulationError> {
        // Build a thread pool
        let mut builder = rayon::ThreadPoolBuilder::new();
        // Set the number of threads
        builder = match self.plotting_config.n_threads {
            Some(n) => builder.num_threads(n),
            // If not threads were supplied, we use only one
            _ => builder.num_threads(1),
        };
        Ok(builder.build()?)
    }

    fn build_progress_bar(
        &self,
        n_iterations: u64,
    ) -> Result<Option<indicatif::ProgressBar>, SimulationError> {
        let mut progress_bar = None;
        if self.plotting_config.show_progressbar {
            let style = ProgressStyle::with_template(PROGRESS_BAR_STYLE)?;
            let bar = ProgressBar::new(n_iterations);
            bar.set_style(style);
            progress_bar = Some(bar);
        }
        Ok(progress_bar)
    }

    pub fn plot_spatial_all_iterations_with_functions<Cpf, Vpf, Dpf>(
        &self,
        cell_plotting_func: &Cpf,
        voxel_plotting_func: &Vpf,
        domain_plotting_func: &Dpf,
    ) -> Result<(), SimulationError>
    where
        Dpf: for<'a> Fn(
                &Dom,
                u32,
                &'a String,
            ) -> Result<
                DrawingArea<BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
                DrawingError,
            > + Send
            + Sync,
        Cpf: Fn(
                &Cel,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        Vpf: Fn(
                &Vox,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        CellAgentBox<Cel>: Send + Sync,
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
        >: Send + Sync,
        DomainBox<Dom>: Send + Sync,
    {
        let pool = self.build_thread_pool()?;

        // Generate all images by calling the pool
        pool.install(move || -> Result<(), SimulationError> {
            use rayon::prelude::*;
            let all_iterations = self.storage_voxels.get_all_iterations()?;
            let progress_bar = self.build_progress_bar(all_iterations.len() as u64)?;
            match progress_bar {
                Some(_) => println!("Generating Images"),
                None => (),
            }
            all_iterations
                .into_par_iter()
                .map(|iteration| {
                    self.plot_spatial_at_iteration_with_functions(
                        iteration,
                        &cell_plotting_func,
                        &voxel_plotting_func,
                        &domain_plotting_func,
                        progress_bar.clone(),
                    )
                })
                .collect::<Result<(), SimulationError>>()?;

            match progress_bar {
                Some(bar) => bar.finish(),
                None => (),
            }
            Ok(())
        })
    }

    pub fn plot_spatial_all_iterations_custom_cell_voxel_functions<Cpf, Vpf>(
        &self,
        cell_plotting_func: Cpf,
        voxel_plotting_func: Vpf,
    ) -> Result<(), SimulationError>
    where
        Dom: CreatePlottingRoot,
        Cpf: Fn(
                &Cel,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        Vpf: Fn(
                &Vox,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        CellAgentBox<Cel>: Send + Sync,
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
        >: Send + Sync,
        DomainBox<Dom>: Send + Sync,
    {
        self.plot_spatial_all_iterations_with_functions(
            &cell_plotting_func,
            &voxel_plotting_func,
            &Dom::create_bitmap_root,
        )
    }
}
