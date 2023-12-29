use super::particle_properties::*;

use cellular_raza::prelude::*;
use pyo3::prelude::*;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// One Dalton in SI units.
///
/// For a full description see [wikipedia](https://en.wikipedia.org/wiki/Dalton_(unit)).
pub const DALTON: f64 = 1.66053906660e-27;

/// One Angström in SI units.
///
/// For a description see [wikipedia](https://en.wikipedia.org/wiki/Angstrom).
pub const ANGSTROM: f64 = 1e-10;

/// One nanometre in SI units.
pub const NANOMETRE: f64 = 1e-9;

/// The Boltzmann-constant in SI units.
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

/// One Kelvin in SI units
pub const KELVIN: f64 = 1.0;

/// All settings which can be configured by the Python interface.
///
/// We aim to provide access to even the more lower-level settings
/// that can influence the results of our simulation.
/// Not all settings do make sense and some combinations can lead
/// to numerical integration problems.
///
/// For documentation of the individual settings, refer to the linked modules.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct SimulationSettings {
    /// Number of cargo particles in the simulation.
    pub n_cells_cargo: usize,

    /// Number of Atg11 particles in the simulation.
    pub n_cells_r11: usize,

    pub cell_radius_cargo: f64,
    pub cell_radius_r11: f64,

    pub mass_cargo: f64,
    pub mass_r11: f64,

    pub damping_cargo: f64,
    pub damping_r11: f64,

    pub kb_temperature_cargo: f64,
    pub kb_temperature_r11: f64,

    pub update_interval: usize,

    pub potential_strength_cargo_cargo: f64,
    pub potential_strength_r11_r11: f64,
    pub potential_strength_cargo_r11: f64,
    pub potential_strength_cargo_r11_avidity: f64,

    pub interaction_range_cargo_cargo: f64,
    pub interaction_range_r11_r11: f64,
    pub interaction_range_r11_cargo: f64,

    /// Integration step of the numerical simulation.
    pub dt: f64,
    /// Number of intgration steps done totally.
    pub n_times: usize,
    /// Specifies the frequency at which results are saved as json files.
    /// Lower the number for more saved results.
    pub save_interval: usize,

    /// Number of threads to use in the simulation.
    pub n_threads: usize,

    /// See [CartesianCuboid3]
    pub domain_size: f64,

    /// Upper bound of the radius in which the cargo particles will be spawned
    pub domain_cargo_radius_max: f64,

    /// Minimal radius outside of which r11 particles will be spawned
    /// Must be smaller than the domain_size
    pub domain_r11_radius_min: f64,

    /// See [CartesianCuboid3]
    pub domain_n_voxels: Option<usize>,

    /// Name of the folder to store the results in.
    pub storage_name: String,

    /// Determines if to add the current date at the end of the save path
    pub storage_name_add_date: bool,

    /// Do we want to show a progress bar
    pub show_progressbar: bool,

    /// The seed with which the simulation is initially configured
    pub random_seed: u64,
}

#[pymethods]
impl SimulationSettings {
    #[new]
    fn new() -> Self {
        // TODO for the future
        // let temperature = 300_f64 * KELVIN;
        // let thermodynamic_energy = BOLTZMANN_CONSTANT * temperature;
        // TODO for the future
        // let cell_radius_r11: f64 = 0.5*(22.938 + 16.259) * NANOMETRE;
        let cell_radius_r11: f64 = 1.0;
        // TODO for the future
        // let mass_r11 = 135002.0 * DALTON;
        let mass_r11 = 4.0 / 3.0 * std::f64::consts::PI * cell_radius_r11.powf(3.0);
        let mass_cargo = 3.0 * mass_r11;
        let cell_radius_cargo: f64 = 1.0 * cell_radius_r11;
        let dt = 1.0;

        SimulationSettings {
            n_cells_cargo: 200,
            n_cells_r11: 200,

            cell_radius_cargo,
            cell_radius_r11,

            mass_cargo,
            mass_r11,

            damping_cargo: 1.5,
            damping_r11: 1.5,

            kb_temperature_cargo: 0.0,
            kb_temperature_r11: 0.003,

            update_interval: 5,

            potential_strength_cargo_cargo: 0.03,
            potential_strength_r11_r11: 0.001,
            potential_strength_cargo_r11: 0.0,
            potential_strength_cargo_r11_avidity: 0.01,

            interaction_range_cargo_cargo: 0.4 * (cell_radius_cargo + cell_radius_r11),
            interaction_range_r11_r11: 0.4 * (cell_radius_cargo + cell_radius_r11),
            interaction_range_r11_cargo: 0.4 * (cell_radius_cargo + cell_radius_r11),
            dt,
            n_times: 40_001,
            save_interval: 100,

            n_threads: 1,

            domain_size: 20.0,
            domain_cargo_radius_max: 6.0,
            domain_r11_radius_min: 6.5,
            // TODO For the future
            // domain_size: 100_f64 * NANOMETRE,
            // domain_cargo_radius_max: 20_f64 * NANOMETRE,
            // domain_r11_radius_min: 40_f64 * NANOMETRE,
            domain_n_voxels: Some(4),

            storage_name: "out/autophagy".into(),
            storage_name_add_date: true,

            show_progressbar: true,

            random_seed: 1,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

fn save_simulation_settings(
    path: &std::path::PathBuf,
    simulation_settings: &SimulationSettings,
) -> PyResult<()> {
    // Also save the SimulationSettings into the same folder
    let mut save_path = path.clone();
    save_path.push("simulation_settings.json");
    let f = std::fs::File::create(save_path)?;
    let writer = std::io::BufWriter::new(f);
    serde_json::to_writer_pretty(writer, &simulation_settings).or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("serde_json error in writing simulation settings to file: {e}"),
        ))
    })?;
    Ok(())
}

fn generate_particle_pos(
    simulation_settings: &SimulationSettings,
    rng: &mut ChaCha8Rng,
    n: usize,
) -> PyResult<[f64; 3]> {
    let mut generate_position = |lower_radius: f64, upper_radius: f64| -> [f64; 3] {
        // Calculate middle of simulation domain
        let domain_half = simulation_settings.domain_size / 2.0;

        // Fix the first coordinate
        let x0_lower = lower_radius.max(0.0);
        let x0_upper = upper_radius.min(domain_half);
        let x0_sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        let x0 = x0_sign * rng.gen_range(x0_lower..x0_upper);

        // Determine upper and lower bounds for next coordinate
        let x1_lower_squared = lower_radius.powi(2) - x0.powi(2);
        let x1_lower = match x1_lower_squared > 0.0 {
            true => x1_lower_squared.sqrt(),
            false => 0.0,
        };
        let x1_upper_squared = upper_radius.powi(2) - x0.powi(2);
        let x1_upper = match x1_upper_squared < domain_half.powi(2) {
            true => x1_upper_squared.sqrt(),
            false => domain_half,
        };

        // Fix second coordinate
        let x1_sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        let x1 = x1_sign * rng.gen_range(x1_lower..x1_upper);

        // Determine upper and lower bounds for next coordinate
        let x2_lower_squared = x1_lower_squared - x1.powi(2);
        let x2_lower = match x2_lower_squared > 0.0 {
            true => x2_lower_squared.sqrt(),
            false => 0.0,
        };

        let x2_upper_squared = x1_upper_squared - x1.powi(2);
        let x2_upper = match x2_upper_squared < domain_half.powi(2) {
            true => x2_upper_squared.sqrt(),
            false => domain_half,
        };

        // Fix third coordinate
        let x2_sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        let x2 = x2_sign * rng.gen_range(x2_lower..x2_upper);

        [domain_half + x0, domain_half + x1, domain_half + x2]
    };

    let pos = if n < simulation_settings.n_cells_cargo {
        generate_position(0.0, simulation_settings.domain_cargo_radius_max)
    } else {
        generate_position(
            simulation_settings.domain_r11_radius_min,
            simulation_settings.domain_size,
        )
    };
    Ok(pos)
}

fn calculate_interaction_range_max(simulation_settings: &SimulationSettings) -> PyResult<f64> {
    // Calculate the maximal interaction range
    let i1 = simulation_settings.interaction_range_cargo_cargo;
    let i2 = simulation_settings.interaction_range_r11_cargo;
    let i3 = simulation_settings.interaction_range_r11_r11;
    let imax = i1.max(i2).max(i3);

    let r1 = simulation_settings.cell_radius_cargo;
    let r2 = simulation_settings.cell_radius_r11;
    let rmax = r1.max(r2);

    Ok(2.0 * rmax + imax)
}

fn create_particle_mechanics(
    simulation_settings: &SimulationSettings,
    rng: &mut ChaCha8Rng,
    n: usize,
) -> PyResult<Langevin3D> {
    let pos = generate_particle_pos(simulation_settings, rng, n)?;
    let mass = if n < simulation_settings.n_cells_cargo {
        simulation_settings.mass_cargo
    } else {
        simulation_settings.mass_r11
    };
    let damping = if n < simulation_settings.n_cells_cargo {
        simulation_settings.damping_cargo
    } else {
        simulation_settings.damping_r11
    };
    let kb_temperature = if n < simulation_settings.n_cells_cargo {
        simulation_settings.kb_temperature_cargo
    } else {
        simulation_settings.kb_temperature_r11
    };
    Ok(Langevin3D::new(
        pos,
        [0.0; 3].into(),
        mass,
        damping,
        kb_temperature,
        simulation_settings.update_interval,
    ))
}

fn create_particle_interaction(
    simulation_settings: &SimulationSettings,
    n: usize,
) -> PyResult<TypedInteraction> {
    Ok(TypedInteraction::new(
        if n < simulation_settings.n_cells_cargo {
            Species::Cargo
        } else {
            Species::R11
        },
        if n < simulation_settings.n_cells_cargo {
            simulation_settings.cell_radius_cargo
        } else {
            simulation_settings.cell_radius_r11
        },
        simulation_settings.potential_strength_cargo_cargo, // potential_strength_cargo_cargo
        simulation_settings.potential_strength_r11_r11,     // potential_strength_r11_r11
        simulation_settings.potential_strength_cargo_r11,   // potential_strength_cargo_r11
        simulation_settings.potential_strength_cargo_r11_avidity, // potential_strength_cargo_r11_avidity
        simulation_settings.interaction_range_cargo_cargo,        // interaction_range_cargo_cargo
        simulation_settings.interaction_range_r11_r11,            // interaction_range_r11_r11
        simulation_settings.interaction_range_r11_cargo,          // interaction_range_r11_cargo
    ))
}

/// Takes [SimulationSettings], runs the full simulation and returns the string of the output directory.
#[pyfunction]
pub fn run_simulation(
    simulation_settings: SimulationSettings,
) -> Result<std::path::PathBuf, pyo3::PyErr> {
    let mut rng = ChaCha8Rng::seed_from_u64(simulation_settings.random_seed);

    let particles = (0..simulation_settings.n_cells_cargo + simulation_settings.n_cells_r11)
        .map(|n| {
            let mechanics = create_particle_mechanics(&simulation_settings, &mut rng, n)?;
            let interaction = create_particle_interaction(&simulation_settings, n)?;
            Ok(Particle {
                mechanics,
                interaction,
            })
        })
        .collect::<Result<Vec<_>, pyo3::PyErr>>()?;

    let interaction_range_max = calculate_interaction_range_max(&simulation_settings)?;

    let domain = match simulation_settings.domain_n_voxels {
        Some(n_voxels) => CartesianCuboid3::from_boundaries_and_n_voxels(
            [0.0; 3],
            [simulation_settings.domain_size; 3],
            [n_voxels; 3],
        ),
        None => CartesianCuboid3::from_boundaries_and_interaction_ranges(
            [0.0; 3],
            [simulation_settings.domain_size; 3],
            [interaction_range_max; 3],
        ),
    }
    .or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Rust error in construction of simulation domain: {e}"),
        ))
    })?;

    let time = TimeSetup {
        t_start: 0.0,
        t_eval: (0..simulation_settings.n_times)
            .map(|n| {
                (
                    n as f64 * simulation_settings.dt,
                    n % simulation_settings.save_interval == 0,
                )
            })
            .collect(),
    };

    let meta_params = SimulationMetaParams {
        n_threads: simulation_settings.n_threads,
        ..Default::default()
    };

    let storage = StorageConfig::from_path(std::path::Path::new(&simulation_settings.storage_name))
        .add_date(simulation_settings.storage_name_add_date);

    let simulation_setup = create_simulation_setup!(
        Domain: domain,
        Cells: particles,
        Time: time,
        MetaParams: meta_params,
        Storage: storage
    );

    let mut supervisor = SimulationSupervisor::initialize_from_setup(simulation_setup);
    supervisor.config.show_progressbar = simulation_settings.show_progressbar;

    save_simulation_settings(&supervisor.storage.get_location(), &simulation_settings)?;

    let simulation_result = supervisor.run_full_sim().or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Rust error in simulation run: {e}"),
        ))
    })?;
    Ok(simulation_result.storage.get_location())
}
