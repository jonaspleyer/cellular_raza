use crate::concepts::cell::{CellAgent, CellAgentBox, CellularIdentifier};
use crate::concepts::domain::Index;
use crate::concepts::domain::{Concentration, Domain, ExtracellularMechanics, Voxel};
use crate::concepts::errors::{DrawingError, RequestError, SimulationError};
use crate::concepts::interaction::{CellularReactions, InteractionExtracellularGradient};
use crate::concepts::mechanics::{Force, Position, Velocity};

use crate::plotting::spatial::{CreatePlottingRoot, PlotSelf};
use crate::storage::sled_database::SledStorageInterface;

use super::domain_decomposition::{
    AuxiliaryCellPropertyStorage, DomainBox, MultiVoxelContainer, VoxelBox,
};

use super::config::{
    ImageType, PlottingConfig, SimulationConfig, SimulationMetaParams, SimulationSetup,
    StorageConfig, TimeSetup, PROGRESS_BAR_STYLE,
};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use core::ops::{Add, AddAssign, Mul};

use hurdles::Barrier;

use serde::{Deserialize, Serialize};

use plotters::{
    coord::types::RangedCoordf64,
    prelude::{BitMapBackend, Cartesian2d, DrawingArea},
};

use indicatif::{ProgressBar, ProgressStyle};

/// # Supervisor controlling simulation execution
///
pub struct SimulationSupervisor<MVC, Dom, C>
where
    Dom: Serialize + for<'a> Deserialize<'a>,
    C: Serialize + for<'a> Deserialize<'a>,
{
    pub worker_threads: Vec<thread::JoinHandle<Result<MVC, SimulationError>>>,
    pub multivoxelcontainers: Vec<MVC>,

    pub time: TimeSetup,
    pub meta_params: SimulationMetaParams,
    pub storage: StorageConfig,

    pub domain: DomainBox<Dom>,

    pub config: SimulationConfig,
    pub plotting_config: PlottingConfig,

    #[cfg(feature = "sled")]
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
        Dom,
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
            Cel,
        >,
        Dom,
        Cel,
    >
where
    Dom: 'static + Serialize + for<'a> Deserialize<'a> + Clone,
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
{
    fn spawn_worker_threads_and_run_sim<ConcGradientExtracellular, ConcTotalExtracellular>(
        &mut self,
    ) -> Result<(), SimulationError>
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
        ConcVecIntracellular: Mul<f64, Output = ConcVecIntracellular>
            + Add<ConcVecIntracellular, Output = ConcVecIntracellular>
            + AddAssign<ConcVecIntracellular>,
        Ind: Index,
        Vox: Voxel<Ind, Pos, For>,
        Vox: ExtracellularMechanics<
            Ind,
            Pos,
            ConcVecExtracellular,
            ConcGradientExtracellular,
            ConcTotalExtracellular,
            ConcBoundaryExtracellular,
        >,
        Cel: CellAgent<Pos, For, Inf, Vel>
            + CellularReactions<ConcVecIntracellular, ConcVecExtracellular>
            + InteractionExtracellularGradient<Cel, ConcGradientExtracellular>,
        VoxelBox<
            Ind,
            Vox,
            Cel,
            Pos,
            For,
            Vel,
            ConcVecExtracellular,
            ConcBoundaryExtracellular,
            ConcVecIntracellular,
        >: Clone,
        AuxiliaryCellPropertyStorage<Pos, For, Vel, ConcVecIntracellular>: Clone,
    {
        let mut handles = Vec::new();
        let mut start_barrier = Barrier::new(self.multivoxelcontainers.len() + 1);

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
        Inf: crate::concepts::interaction::InteractionInformation,
        Vel: Velocity,
        ConcVecExtracellular: Concentration,
        ConcTotalExtracellular: Concentration,
        ConcBoundaryExtracellular: Send + Sync + 'static,
        ConcVecIntracellular: Send + Sync + Concentration,
        ConcVecIntracellular: Mul<f64, Output = ConcVecIntracellular>
            + Add<ConcVecIntracellular, Output = ConcVecIntracellular>
            + AddAssign<ConcVecIntracellular>,
        Ind: Index,
        Vox: Voxel<Ind, Pos, For>,
        Vox: ExtracellularMechanics<
            Ind,
            Pos,
            ConcVecExtracellular,
            ConcGradientExtracellular,
            ConcTotalExtracellular,
            ConcBoundaryExtracellular,
        >,
        Cel: CellAgent<Pos, For, Inf, Vel>
            + CellularReactions<ConcVecIntracellular, ConcVecExtracellular>
            + InteractionExtracellularGradient<Cel, ConcGradientExtracellular>,
        VoxelBox<
            Ind,
            Vox,
            Cel,
            Pos,
            For,
            Vel,
            ConcVecExtracellular,
            ConcBoundaryExtracellular,
            ConcVecIntracellular,
        >: Clone,
        AuxiliaryCellPropertyStorage<Pos, For, Vel, ConcVecIntracellular>: Clone,
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
            #[cfg(feature = "sled")]
            storage_cells,
            #[cfg(feature = "sled")]
            storage_voxels,
            plotting_config: PlottingConfig::default(),
        };

        Ok(simulation_result)
    }

    #[cfg(any(feature = "sled", feature = "serde_json"))]
    pub fn save_current_setup(&self, iteration: u64) -> Result<(), SimulationError> {
        let setup_current = SimulationSetup {
            domain: self.domain.clone(),
            cells: Vec::<Cel>::new(),
            time: TimeSetup {
                t_start: 0.0,
                t_eval: Vec::new(),
            },
            meta_params: SimulationMetaParams {
                n_threads: self.worker_threads.len(),
            },
            #[cfg(feature = "sled")]
            storage: self.storage.clone(),
        };

        self.meta_infos
            .store_single_element(iteration, (), setup_current)?;
        Ok(())
    }
}

use super::domain_decomposition::PlainIndex;

pub struct SimulationResult<
    I,
    Pos,
    For,
    Vel,
    ConcVecExtracellular,
    ConcBoundaryExtracellular,
    ConcVecIntracellular,
    V,
    D,
    C,
> where
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    C: Serialize + for<'a> Deserialize<'a>,
    VoxelBox<
        I,
        V,
        C,
        Pos,
        For,
        Vel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
    >: for<'a> Deserialize<'a> + Serialize,
    D: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a> + 'static,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a>,
{
    pub storage: StorageConfig,

    pub domain: DomainBox<D>,
    #[cfg(feature = "sled")]
    pub storage_cells: SledStorageInterface<CellularIdentifier, CellAgentBox<C>>,
    #[cfg(feature = "sled")]
    pub storage_voxels: SledStorageInterface<
        PlainIndex,
        VoxelBox<
            I,
            V,
            C,
            Pos,
            For,
            Vel,
            ConcVecExtracellular,
            ConcBoundaryExtracellular,
            ConcVecIntracellular,
        >,
    >,
    pub plotting_config: PlottingConfig,
}

impl<
        I,
        Pos,
        For,
        Vel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
        V,
        D,
        C,
    >
    SimulationResult<
        I,
        Pos,
        For,
        Vel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
        V,
        D,
        C,
    >
where
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    C: Serialize + for<'a> Deserialize<'a>,
    D: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a> + 'static,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a>,
    VoxelBox<
        I,
        V,
        C,
        Pos,
        For,
        Vel,
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
    ) -> Result<(), SimulationError>
    where
        Dpf: for<'a> Fn(
            &D,
            u32,
            &'a String,
        ) -> Result<
            DrawingArea<BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            DrawingError,
        >,
        Cpf: Fn(
                &C,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        Vpf: Fn(
                &V,
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
        let mut file_path = self.storage.location.clone();
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

        Ok(())
    }

    pub fn plot_spatial_at_iteration(&self, iteration: u64) -> Result<(), SimulationError>
    where
        D: CreatePlottingRoot,
        C: PlotSelf,
        V: PlotSelf,
    {
        match self.plotting_config.image_type {
            ImageType::BitMap => self.plot_spatial_at_iteration_with_functions(
                iteration,
                C::plot_self_bitmap,
                V::plot_self_bitmap,
                D::create_bitmap_root,
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
            &D,
            u32,
            &'a String,
        ) -> Result<
            DrawingArea<BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            DrawingError,
        >,
        Cpf: Fn(
                &C,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        Vpf: Fn(
                &V,
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
        )
    }

    pub fn plot_spatial_at_iteration_custom_cell_voxel_functions<Cpf, Vpf>(
        &self,
        iteration: u64,
        cell_plotting_func: Cpf,
        voxel_plotting_func: Vpf,
    ) -> Result<(), SimulationError>
    where
        D: CreatePlottingRoot,
        Cpf: Fn(
                &C,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        Vpf: Fn(
                &V,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
    {
        self.plot_spatial_at_iteration_with_functions(
            iteration,
            cell_plotting_func,
            voxel_plotting_func,
            D::create_bitmap_root,
        )
    }

    pub fn plot_spatial_at_iteration_custom_cell_funtion<Cpf>(
        &self,
        iteration: u64,
        cell_plotting_func: Cpf,
    ) -> Result<(), SimulationError>
    where
        V: PlotSelf,
        D: CreatePlottingRoot,
        Cpf: Fn(
                &C,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
    {
        match self.plotting_config.image_type {
            ImageType::BitMap => self.plot_spatial_at_iteration_with_functions(
                iteration,
                cell_plotting_func,
                V::plot_self_bitmap,
                D::create_bitmap_root,
            ),
        }
    }

    pub fn plot_spatial_all_iterations_with_functions<Cpf, Vpf, Dpf>(
        &self,
        cell_plotting_func: Cpf,
        voxel_plotting_func: Vpf,
        domain_plotting_func: Dpf,
    ) -> Result<(), SimulationError>
    where
        Dpf: for<'a> Fn(
            &D,
            u32,
            &'a String,
        ) -> Result<
            DrawingArea<BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            DrawingError,
        >,
        Cpf: Fn(
                &C,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        Vpf: Fn(
                &V,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
    {
        for iteration in self.storage_voxels.get_all_iterations()?.into_iter() {
            self.plot_spatial_at_iteration_with_functions(
                iteration,
                &cell_plotting_func,
                &voxel_plotting_func,
                &domain_plotting_func,
            )?;
        }
        Ok(())
    }

    pub fn plot_spatial_all_iterations_custom_cell_voxel_functions<Cpf, Vpf>(
        &self,
        cell_plotting_func: Cpf,
        voxel_plotting_func: Vpf,
    ) -> Result<(), SimulationError>
    where
        D: CreatePlottingRoot,
        Cpf: Fn(
                &C,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
        Vpf: Fn(
                &V,
                &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
            ) -> Result<(), DrawingError>
            + Send
            + Sync,
    {
        for iteration in self.storage_voxels.get_all_iterations()?.into_iter() {
            self.plot_spatial_at_iteration_with_functions(
                iteration,
                &cell_plotting_func,
                &voxel_plotting_func,
                D::create_bitmap_root,
            )?;
        }
        Ok(())
    }
}
