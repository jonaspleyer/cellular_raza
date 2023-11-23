use super::particle_properties::*;

use cellular_raza::prelude::*;
use pyo3::prelude::*;

use nalgebra::Vector3;
use num::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

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
    pub n_cells_atg11_receptor: usize,

    /// Contains all paramters that all cargo particles will share in common.
    /// Only the position will be initially randomized. All other parameters will
    /// be used as is.
    pub particle_template_cargo: Py<ParticleTemplate>,

    /// Contains all paramters that all cargo particles will share in common.
    /// Only the position will be initially randomized. All other parameters will
    /// be used as is.
    pub particle_template_atg11_receptor: Py<ParticleTemplate>,

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

    /// Lower boundary of the cuboid in which the cargo cells will be placed initially
    pub domain_cargo_low: [f64; 3],

    /// Upper boundary of the cuboid in which the cargo cells will be placed initially
    pub domain_cargo_high: [f64; 3],

    /// Lower boundary of the cuboid in which the cargo cells will be placed initially
    pub domain_r11_low: [f64; 3],

    /// Upper boundary of the cuboid in which the cargo cells will be placed initially
    pub domain_r11_high: [f64; 3],

    /// Determines if the r11 particles will be placed outside of the domain of the
    /// cargo particles.
    pub domain_r11_avoid_cargo: bool,

    /// See [CartesianCuboid3]
    pub domain_n_voxels: Option<usize>,

    /// Name of the folder to store the results in.
    pub storage_name: String,

    /// Do we want to show a progress bar
    pub show_progressbar: bool,

    /// The seed with which the simulation is initially configured
    pub random_seed: u64,
}

fn create_particle_template(
    py: Python,
    species: Species,
    cell_radius: f64,
    damping: f64,
    kb_temperature: f64,
    update_interval: usize,
    potential_strength: f64,
    interaction_range: f64,
    clustering_strength: f64,
    avidity: f64,
) -> PyResult<ParticleTemplate> {
    Ok(ParticleTemplate {
        mechanics: Py::new(
            py,
            Langevin3D::new(
                [0.0; 3],                // pos
                cell_radius.powf(3_f64), // mass
                damping,                 // damping
                kb_temperature,          // kb_temperature
                update_interval,         // update_interval
            ),
        )?,
        interaction: Py::new(
            py,
            TypedInteraction::new(
                species,             // species
                cell_radius,         // cell_radius
                potential_strength,  // potential_strength
                interaction_range,   // interaction_range
                clustering_strength, // clustering_strength
                avidity,             // avidity
            ),
        )?,
    })
}

#[pymethods]
impl SimulationSettings {
    #[new]
    fn new(py: Python) -> PyResult<Self> {
        let cell_radius_atg11_receptor: f64 = 1.0;
        let cell_radius_cargo: f64 = 1.5 * cell_radius_atg11_receptor;
        let dt = 0.25;

        Ok(SimulationSettings {
            n_cells_cargo: 100,
            n_cells_atg11_receptor: 300,

            particle_template_cargo: Py::new(
                py,
                create_particle_template(
                    py,
                    Species::Cargo,          // species
                    cell_radius_cargo,       // cell_radius
                    1.5,                     // damping
                    0.0,                     // kb_temperature
                    5,                       // update_interval
                    0.03,                    // potential_strength
                    0.8 * cell_radius_cargo, // interaction_range
                    1.0,                     // clustering_strength
                    1.0,                     // avidity
                )?,
            )?,
            particle_template_atg11_receptor: Py::new(
                py,
                create_particle_template(
                    py,
                    Species::R11,                     // species
                    cell_radius_atg11_receptor,       // cell_radius
                    0.5,                              // damping
                    0.0025,                           // kb_temperature
                    5,                                // update_interval
                    0.02,                             // potential_strength
                    0.8 * cell_radius_atg11_receptor, // interaction_range
                    0.2,                              // clustering_strength
                    1.0,                              // avidity
                )?,
            )?,

            dt,
            n_times: 20_001,
            save_interval: 50,

            n_threads: 1,

            domain_size: 100.0,
            domain_cargo_low: [40.0; 3],
            domain_cargo_high: [60.0; 3],
            domain_r11_low: [0.0; 3],
            domain_r11_high: [100.0; 3],
            domain_r11_avoid_cargo: true,
            domain_n_voxels: Some(4),

            storage_name: "out/autophagy".into(),

            show_progressbar: true,

            random_seed: 1,
        })
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "{:#?}\nparticle_template_cargo:\n{:#?}\nparticle_template_atg11_receptor\n{:#?}",
            self,
            self.particle_template_cargo
                .extract::<ParticleTemplate>(py)?,
            self.particle_template_atg11_receptor
                .extract::<ParticleTemplate>(py)?
        ))
    }

    #[getter]
    fn get_particle_template_cargo(&mut self, py: Python) -> Py<ParticleTemplate> {
        self.particle_template_cargo.clone_ref(py)
    }
}

#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
pub struct Particle {
    #[Mechanics(Vector3<f64>, Vector3<f64>, Vector3<f64>)]
    pub mechanics: Langevin3D,

    #[Interaction(Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, usize, Species))]
    pub interaction: TypedInteraction,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct ParticleTemplate {
    pub mechanics: Py<Langevin3D>,
    pub interaction: Py<TypedInteraction>,
}

impl ParticleTemplate {
    fn into_particle(self, py: Python) -> PyResult<Particle> {
        let mechanics = self.mechanics.extract::<Langevin3D>(py)?;
        let interaction = self.interaction.extract::<TypedInteraction>(py)?;
        Ok(Particle {
            mechanics,
            interaction,
        })
    }
}

#[pymethods]
impl ParticleTemplate {
    #[new]
    fn new(mechanics: Langevin3D, interaction: TypedInteraction, py: Python) -> PyResult<Self> {
        Ok(Self {
            mechanics: Py::new(py, mechanics)?,
            interaction: Py::new(py, interaction)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

impl Cycle<Particle> for Particle {
    fn divide(
        _rng: &mut rand_chacha::ChaCha8Rng,
        _cell: &mut Particle,
    ) -> Result<Particle, DivisionError> {
        panic!()
    }

    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        _dt: &f64,
        _cell: &mut Particle,
    ) -> Option<CycleEvent> {
        None
    }
}

impl CellularReactions<Nothing, Nothing> for Particle {
    fn get_intracellular(&self) -> Nothing {
        Nothing::zero()
    }

    fn set_intracellular(&mut self, _concentration_vector: Nothing) {}

    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        _internal_concentration_vector: &Nothing,
        _external_concentration_vector: &Nothing,
    ) -> Result<(Nothing, Nothing), CalcError> {
        Ok((Nothing::zero(), Nothing::zero()))
    }
}

impl<Conc> InteractionExtracellularGradient<Particle, Conc> for Particle {
    fn sense_gradient(_cell: &mut Particle, _gradient: &Conc) -> Result<(), CalcError> {
        Ok(())
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
) -> PyResult<Vector3<f64>> {
    let pos = if n < simulation_settings.n_cells_cargo {
        Vector3::from([
            rng.gen_range(
                simulation_settings.domain_cargo_low[0].max(0.0)
                    ..simulation_settings.domain_cargo_high[0].min(simulation_settings.domain_size),
            ),
            rng.gen_range(
                simulation_settings.domain_cargo_low[1].max(0.0)
                    ..simulation_settings.domain_cargo_high[1].min(simulation_settings.domain_size),
            ),
            rng.gen_range(
                simulation_settings.domain_cargo_low[2].max(0.0)
                    ..simulation_settings.domain_cargo_high[2].min(simulation_settings.domain_size),
            ),
        ])
    } else {
        // We do not want to spawn the R11 particles in the middle where the cargo
        // will be placed. Thus we calculate where else we can spawn them.
        let mut loc = [0.0; 3];
        // Initially choose the location of the R11 particle
        for i in 0..3 {
            // Restrict it to at minimum 0.0 and at maximum the size of the domain
            let low = simulation_settings.domain_r11_low[i].max(0.0);
            let high = simulation_settings.domain_r11_high[i].min(simulation_settings.domain_size);
            loc[i] = rng.gen_range(low..high);
        }
        if simulation_settings.domain_r11_avoid_cargo {
            // Determine if the position chosen landed in the cargo region
            if loc.iter().enumerate().all(|(i, x)| {
                simulation_settings.domain_cargo_low[i] <= *x
                    && *x <= simulation_settings.domain_cargo_high[i]
            }) {
                // Choose one direction at random
                let j = rng.gen_range(0_usize..3_usize);
                // The cargo and r11 intervals must be intercepting
                // thus we divide the region into 2 subintervals both not containing the
                // cargo region.
                let r_low = simulation_settings.domain_r11_low[j].max(0.0);
                let r_high = simulation_settings.domain_r11_high[j].min(simulation_settings.domain_size);
                let c_low = simulation_settings.domain_cargo_low[j].max(0.0);
                let c_high = simulation_settings.domain_cargo_high[j].min(simulation_settings.domain_size);
                // First check that the r11 interval is not completely contained
                // inside the cargo interval
                if c_low <= r_low && r_high <= c_high {
                    return Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!(
                            "Error in definition of cargo and r11 domain while avoiding each other: Cargo_low: {:?} Cargo_high: {:?} R11_low: {:?} R11_high: {:?}",
                            simulation_settings.domain_cargo_low,
                            simulation_settings.domain_cargo_high,
                            simulation_settings.domain_r11_low,
                            simulation_settings.domain_r11_high,
                        ),
                    ));
                }
                // Check if the Cargo interval is fully contained in the R11 interval
                if r_low < c_low && c_high < r_high {
                    // We now know that we need to create 2 distinct intervals to sample
                    // for the R11 particle
                    let i1_low = r_low;
                    let i1_high = c_low;
                    let i1_dist = i1_high - i1_low;
                    let i2_low = c_high;
                    let i2_high = r_high;
                    let i2_dist = i2_high - i2_low;

                    // Calculate the probability to be in interval i1
                    let p = i1_dist / (i1_dist + i2_dist);
                    if rng.gen_bool(p) {
                        // We are in interval i1
                        loc[j] = rng.gen_range(i1_low..i1_high);
                    } else {
                        // We are in interval i2
                        loc[j] = rng.gen_range(i2_low..i2_high);
                    }
                }
                // Otherwise check if the cargo interval is lower than the R11 interval
                else if c_low <= r_low {
                    let i1_low = c_high;
                    let i1_high = r_high;
                    loc[j] = rng.gen_range(i1_low..i1_high);
                }
                // The only remaining option is r_high <= c_high
                else {
                    let i1_low = r_low;
                    let i1_high = c_low;
                    loc[j] = rng.gen_range(i1_low..i1_high);
                }
            }
        }
        Vector3::from(loc)
    };
    Ok(pos)
}

fn calculate_interaction_range_max(
    simulation_settings: &SimulationSettings,
    py: Python,
) -> PyResult<f64> {
    // Calculate the maximal interaction range
    let interaction_range_max = (simulation_settings
        .particle_template_cargo
        .borrow(py)
        .interaction
        .borrow(py)
        .interaction_range
        + 2.0
            * simulation_settings
                .particle_template_cargo
                .borrow(py)
                .interaction
                .borrow(py)
                .cell_radius)
        .max(
            simulation_settings
                .particle_template_atg11_receptor
                .borrow(py)
                .interaction
                .borrow(py)
                .interaction_range
                + 2.0
                    * simulation_settings
                        .particle_template_atg11_receptor
                        .borrow(py)
                        .interaction
                        .borrow(py)
                        .cell_radius,
        );
    Ok(interaction_range_max)
}

/// Takes [SimulationSettings], runs the full simulation and returns the string of the output directory.
#[pyfunction]
pub fn run_simulation(
    simulation_settings: SimulationSettings,
    py: Python,
) -> Result<std::path::PathBuf, pyo3::PyErr> {
    let mut rng = ChaCha8Rng::seed_from_u64(simulation_settings.random_seed);

    let particles = (0..simulation_settings.n_cells_cargo
        + simulation_settings.n_cells_atg11_receptor)
        .map(|n| {
            let pos = generate_particle_pos(&simulation_settings, &mut rng, n)?;
            let particle = if n < simulation_settings.n_cells_cargo {
                simulation_settings.particle_template_cargo.clone()
            } else {
                simulation_settings.particle_template_atg11_receptor.clone()
            };
            particle
                .borrow_mut(py)
                .mechanics
                .borrow_mut(py)
                .mechanics
                .set_pos(&pos);
            particle.extract::<ParticleTemplate>(py)?.into_particle(py)
        })
        .collect::<Result<Vec<_>, pyo3::PyErr>>()?;

    let interaction_range_max = calculate_interaction_range_max(&simulation_settings, py)?;

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
    };

    let storage =
        StorageConfig::from_path(simulation_settings.storage_name.clone().into());

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
