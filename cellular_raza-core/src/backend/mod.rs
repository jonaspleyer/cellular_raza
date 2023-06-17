/// Use multiple os-threads and cpu-only resources
///
/// Parallelization is achieved by splitting the simualtion domain into as many chunks as threads are desired.
/// Communication between threads is handled by [crossbeam_channel] and synchronization by [hurdles::Barrier].
///
/// The user can manage the simulation flow by means of individual funtions or by creating a
/// [SimulationSupervisor](cpu_os_threads::supervisor::SimulationSupervisor).
pub mod cpu_os_threads;

/* pub trait Backend {
    type Setup;
    type SetupStrategies;
    type SnapShot;

    fn initialize(setup: Self::Setup) -> Self;
    fn initialize_with_strategies(setup: Self::Setup, strategies: Self::SetupStrategies) -> Self;
    fn initialize_from_full_snapshot(snapshot: Self::SnapShot) -> Self;

    fn run_full_simulation(&mut self) -> Result<(), SimulationError>;
}*/

pub mod chili;
