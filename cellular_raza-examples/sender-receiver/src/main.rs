use cellular_raza::prelude::*;

use nalgebra::Vector2;

use num::Zero;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ###########
// ## UNITS ##
// ###########

pub const MICRO_METRE: f64 = 1.0;
pub const MINUTE: f64 = 60.0;
pub const SECOND: f64 = 1.0;
pub const MOLAR: f64 = 1.0;

// Number of cells to put into simulation in the Beginning
pub const N_CELLS_INITIAL_SENDER: u32 = 20;
pub const N_CELLS_INITIAL_RECEIVER: u32 = 20;

// Mechanical parameters
pub const CELL_MECHANICS_RADIUS: f64 = 12.0 * MICRO_METRE;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 =
    2.0 * MICRO_METRE * MICRO_METRE / (MINUTE * MINUTE);
pub const CELL_MECHANICS_DAMPING: f64 = 2.0 / MINUTE;

// Reaction parameters of the cell
pub const CELL_LIGAND_TURNOVER_RATE: f64 = 0.05 / SECOND;
pub const CELL_LIGAND_UPTAKE_RATE: f64 = 0.1 / SECOND;

// Parameters for domain
pub const DOMAIN_SIZE: f64 = 300.0 * MICRO_METRE;

// Parameters for Voxel Reaction+Diffusion
pub const VOXEL_LIGAND_DIFFUSION_CONSTANT: f64 = 100.0 * MICRO_METRE * MICRO_METRE / SECOND;

// Controller parameters
pub const TARGET_AVERAGE_CONC: f64 = 1.0 * MOLAR;

// Time parameters
pub const DT: f64 = 0.1 * SECOND;
pub const T_START: f64 = 0.0 * MINUTE;
pub const T_END: f64 = 60.0 * MINUTE;
pub const SAVE_INTERVAL: f64 = 0.5 * MINUTE;

// Meta Parameters to control solving
pub const N_THREADS: usize = 1;
pub const N_PLOTTING_THREADS: usize = 14;

mod bacteria_properties;
mod controller;
mod plotting;

use bacteria_properties::*;
use controller::*;
use plotting::*;

fn voxel_definition_strategy(voxel: &mut CartesianCuboidVoxel2<1>) {
    voxel.diffusion_constant = [VOXEL_LIGAND_DIFFUSION_CONSTANT].into();
    voxel.extracellular_concentrations = [0.0 * MOLAR].into();
    voxel.degradation_rate = [0.0].into();
    voxel.production_rate = [0.0].into();
}

fn create_domain() -> Result<CartesianCuboid2, CalcError> {
    CartesianCuboid2::from_boundaries_and_n_voxels([0.0; 2], [DOMAIN_SIZE, DOMAIN_SIZE], [12; 2])
}

fn run_main(
    strategy: &ControlStrategy,
    observer: &Observer,
    spatial_setup: &SpatialSetup,
) -> Result<(), SimulationError> {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // ###########################
    // ## DEFINE DOMAIN & CELLS ##
    // ###########################
    let (domain, cells) = spatial_setup.generate_domain_cells(&mut rng)?;

    // ##########################################
    // ## CREATE SUPERVISOR AND RUN SIMULATION ##
    // ##########################################
    let n_times = ((T_END - T_START) / DT).ceil() as usize + 1;
    let storage = StorageBuilder::new().location("out/sender_receiver").init();
    let save_path = storage.get_full_path();
    let setup = SimulationSetup::new(
        domain,
        cells,
        TimeSetup {
            t_start: 0.0,
            t_eval: (0..n_times)
                .map(|i| {
                    (
                        T_START + DT * i as f64,
                        (DT * i as f64) % SAVE_INTERVAL == 0.0,
                    )
                })
                .collect::<Vec<(f64, bool)>>(),
        },
        SimulationMetaParams {
            n_threads: N_THREADS,
            ..Default::default()
        },
        storage,
        SRController::new(TARGET_AVERAGE_CONC, 0.1 * MOLAR / SECOND, &save_path)
            .strategy(strategy.clone())
            .observer(observer.clone()),
    );

    let strategies = Strategies {
        voxel_definition_strategies: &voxel_definition_strategy,
    };

    let mut supervisor = SimulationSupervisor::initialize_with_strategies(setup, strategies);

    let mut simulation_result = supervisor.run_full_sim()?;

    // ######################
    // ## PLOT THE RESULTS ##
    // ######################
    simulation_result.plotting_config = PlottingConfig {
        n_threads: Some(N_PLOTTING_THREADS),
        image_size: 1500,
        image_type: ImageType::BitMap,
        ..Default::default()
    };

    simulation_result
        .plot_spatial_all_iterations_custom_cell_voxel_functions(plot_modular_cell, plot_voxel)?;
    Ok(())
}

fn main() {
    let strategies_observers = [
        (
            ControlStrategy::PID({
                let mut pid_default = PIDSettings::default();
                pid_default.t_d = 0.0;
                pid_default
            }),
            Observer::Predictor { weighting: 0.9 },
        ),
        (
            ControlStrategy::PID(PIDSettings::default()),
            Observer::Standard,
        ),
        (
            ControlStrategy::DelayODE(DelayODESettings::default()),
            Observer::Standard,
        ),
    ];

    let spatial_setups = [
        SpatialSetup::Default,
        SpatialSetup::Circular,
        SpatialSetup::CircularInverted,
        // SpatialSetup::Branch,
    ];

    for (strategy, observer) in strategies_observers.iter() {
        for spatial_setup in spatial_setups.iter() {
            run_main(strategy, observer, spatial_setup).unwrap();
            // Plot results
            std::process::Command::new("sh")
                .arg("-c")
                .arg("python plot.py")
                .spawn()
                .unwrap()
                .wait()
                .unwrap();
        }
    }
}
