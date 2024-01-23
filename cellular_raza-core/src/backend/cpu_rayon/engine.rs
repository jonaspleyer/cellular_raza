
use cellular_raza_concepts::{CellAgent, CellAgentBox, CellularIdentifier};
use cellular_raza_concepts::domain::Index;
use cellular_raza_concepts::domain::{
    Concentration, Controller, Domain, ExtracellularMechanics, Voxel,
};
use cellular_raza_concepts::{ControllerError, DrawingError, RequestError};
use cellular_raza_concepts::{CellularReactions, InteractionExtracellularGradient};
use cellular_raza_concepts::mechanics::{Force, Position, Velocity};
use cellular_raza_concepts::{CreatePlottingRoot, PlotSelf};

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
pub struct ControllerBox<Cont, Obs> {
    pub controller: Cont,
    pub measurements: std::collections::BTreeMap<u32, Obs>,
}

impl<Cont, Obs> ControllerBox<Cont, Obs> {
    pub fn measure<'a, I, Cel>(&mut self, thread_index: u32, cells: I) -> Result<(), SimulationError>
    where
        Cel: 'a + Serialize + for<'b> Deserialize<'b>,
        I: Iterator<Item = &'a CellAgentBox<Cel>> + Clone,
        Cont: Controller<Cel, Obs>,
    {
        let obs = self.controller.measure(cells)?;
        self.measurements.insert(thread_index, obs);
        Ok(())
    }

    pub fn adjust<'a, Cel, J>(&mut self, cells: J) -> Result<(), ControllerError>
    where
        Cel: 'a + Serialize + for<'b> Deserialize<'b>,
        J: Iterator<
            Item = (
                &'a mut CellAgentBox<Cel>,
                &'a mut Vec<cellular_raza_concepts::CycleEvent>,
            ),
        >,
        Cont: Controller<Cel, Obs>,
    {
        self.controller.adjust(self.measurements.values(), cells)
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
    pub worker_threads: Vec<thread::JoinHandle<Result<MVC, SimulationError>>>,
    pub multivoxelcontainers: Vec<MVC>,

    pub time: TimeSetup,
    pub meta_params: SimulationMetaParams,
    pub storage: StorageConfig,

    pub domain: DomainBox<Dom>,

    pub config: SimulationConfig,

    pub meta_infos: StorageManager<(), SimulationSetup<DomainBox<Dom>, Cel, Cont>>,

    pub controller_box: Arc<std::sync::Mutex<ControllerBox<Cont, Obs>>>,
    pub phantom_obs: PhantomData<Obs>,
    pub phantom_cont: PhantomData<Cont>,
}

#[macro_export]
macro_rules! run_step_1(
    ($cont:expr, Mechanics) => {
        $cont.update_cellular_mechanics_step_1().unwrap();
    };

    ($cont:expr, FluidMechanics) => {
        $cont.update_fluid_mechanics_step_1().unwrap();
    };

    ($cont:expr, $a:ident) => {};

    ($cont:expr, $a:ident, $($ai:ident),+) => {
        run_step_1!($cont, $a);
        run_step_1!($cont $(, $ai)+);
    };
);

#[macro_export]
macro_rules! run_step_2(
    ($cont:expr, Mechanics) => {
        $cont.update_cellular_mechanics_step_2().unwrap();
    };

    ($cont:expr, FluidMechanics) => {
        $cont.update_fluid_mechanics_step_2().unwrap();
    };

    ($cont:expr, $a:ident) => {};

    ($cont:expr, $a:ident, $($ai:ident),+) => {
        run_step_2!($cont, $a);
        run_step_2!($cont $(, $ai)+);
    };
);

#[macro_export]
macro_rules! run_step_3(
    ($cont:expr, $dt:expr, Mechanics) => {
        $cont.update_cellular_mechanics_step_3($dt).unwrap();
    };

    ($cont:expr, $dt:expr, FluidMechanics) => {
        $cont.update_fluid_mechanics_step_3($dt).unwrap();
    };

    ($cont:expr, $dt:expr, Reactions) => {
        $cont.update_cellular_reactions($dt).unwrap();
    };

    ($cont:expr, $dt:expr, Cycle) => {
        $cont.update_local_functions($dt).unwrap();
    };

    ($cont:expr, $dt:expr, $a:ident) => {};

    ($cont:expr, $dt:expr, $a:ident, $($ai:ident),+) => {
        run_step_3!($cont, $dt, $a);
        run_step_3!($cont, $dt $(, $ai)+);
    };
);

#[macro_export]
macro_rules! run_full_sim {
    ($supervisor:expr, CellFeatures=[$($feature:ident),+ $(,)?]) => {
        // Run the simulation
        let mut handles = Vec::new();
        let (mut start_barrier, controller_barrier) = $supervisor.__private_initialize();

        if $supervisor.config.show_progressbar {
            println!("Running Simulation");
        }

        for (l, mut cont) in $supervisor.multivoxelcontainers.drain(..).enumerate() {
            // Clone barriers to use them for synchronization in threads
            let mut new_start_barrier = start_barrier.clone();
            let mut controller_barrier_new = controller_barrier.clone();

            // See if we need to save
            let stop_now_new = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

            // Copy time evaluation points
            let t_start = $supervisor.time.t_start;
            let t_eval = $supervisor.time.t_eval.clone();

            // Add bar
            // let show_progressbar = $supervisor.config.show_progressbar;
            // let style = ProgressStyle::with_template(PROGRESS_BAR_STYLE)?;
            // let bar = ProgressBar::new(t_eval.len() as u64);
            // bar.set_style(style);

            let controller_box = $supervisor.controller_box.clone();

            // Spawn a thread for each multivoxelcontainer that is running
            let handle = std::thread::Builder::new().name(format!("worker_thread_{:03.0}", l)).spawn(move || -> Result<_, SimulationError> {
                new_start_barrier.wait();

                let mut time = t_start;
                #[allow(unused)]
                let mut iteration = 0u64;
                for (t, save) in t_eval {
                    let dt = t - time;
                    time = t;

                    // These methods are used for sending requests and gathering information in general
                    // This gathers information of forces acting between cells and send between threads
                    // cont.update_fluid_mechanics_step_1().unwrap();

                    // Gather boundary conditions between voxels and domain boundaries and send between threads
                    run_step_1!(cont, $($feature),+);//$(,$feature)*);

                    // Wait for all threads to synchronize.
                    // The goal is to have as few as possible synchronizations
                    cont.barrier.wait();

                    // 
                    // cont.update_fluid_mechanics_step_2().unwrap();

                    // cont.update_cellular_mechanics_step_2().unwrap();
                    run_step_2!(cont, $($feature),+);

                    cont.barrier.wait();

                    // cont.update_cellular_reactions(&dt).unwrap();

                    // These are the true update steps where cell agents are modified the order here may play a role!
                    // 
                    // cont.update_fluid_mechanics_step_3(dt).unwrap();

                    // cont.update_cellular_mechanics_step_3(&dt).unwrap();

                    // TODO this currently also does application of domain boundaries and inclusion of new cells which is wrong in general!
                    // cont.update_local_functions(&dt).unwrap();
                    run_step_3!(cont, &dt, $($feature),+);

                    // This function needs an additional synchronization step which cannot correctly be done in between the other ones
                    cont.sort_cells_in_voxels_step_1().unwrap();

                    cont.barrier.wait();

                    cont.sort_cells_in_voxels_step_2().unwrap();

                    // if show_progressbar && cont.mvc_id == 0 {
                    //     bar.inc(1);
                    // }
                    if cont.mvc_id == 0 {
                        print!("\r{}", iteration);
                    }

                    // if save_now_new.load(Ordering::Relaxed) {
                    if save {
                        cont.save_voxels_to_database(&iteration).unwrap();
                        cont.save_cells_to_database(&iteration).unwrap();
                    }

                    // TODO Make sure to only call this if the controller type is not ()
                    {
                        controller_box.lock().unwrap().measure(
                            l as u32,
                            cont.voxels
                            .iter()
                            .flat_map(|vox| vox.1.cells
                                .iter()
                                .map(|(cbox, _)| cbox)
                            )
                        ).unwrap();
                        controller_barrier_new.wait();
                        controller_box.lock().unwrap().adjust(
                            cont.voxels
                            .iter_mut()
                            .flat_map(|vox| vox.1.cells
                                .iter_mut()
                                .map(|(cbox, aux_storage)| (cbox, &mut aux_storage.cycle_events))
                            )
                        ).unwrap();
                    }

                    // Check if we are stopping the simulation now
                    // if stop_now_new.load(Ordering::Relaxed) {
                    //     // new_barrier.wait();
                    //     break;
                    // }
                    iteration += 1;
                }
                // if show_progressbar && cont.mvc_id == 0 {
                //     bar.finish();
                // }
                return Ok(cont);
            }).unwrap();
            handles.push(handle);
        }

        $supervisor.worker_threads = handles;

        // This starts all threads simultanously
        start_barrier.wait();

        // Collect all threads
        let mut databases = Vec::new();
        for thread in $supervisor.worker_threads.drain(..) {
            let t = thread
                .join()
                .expect("Could not join threads after Simulation has finished").unwrap();
            databases.push((t.storage_cells, t.storage_voxels, t.domain))
        }

        // Create a simulationresult which can then be used to further plot and analyze results
        let (storage_cells, storage_voxels, domain) = databases.pop().ok_or(RequestError {
            message: format!("The threads of the simulation did not yield any handles"),
        }).unwrap();

        let simulation_result = SimulationResult {
            storage: $supervisor.storage.clone(),
            domain,
            storage_cells,
            storage_voxels,
            plotting_config: PlottingConfig::default(),
        };
    }
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
    #[doc(hidden)]
    pub fn __private_initialize(&self) -> (Barrier, Barrier) {
        (Barrier::new(self.multivoxelcontainers.len()+1), Barrier::new(self.multivoxelcontainers.len()))
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
