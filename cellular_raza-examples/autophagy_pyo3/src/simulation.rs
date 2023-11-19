use cellular_raza::prelude::*;
use pyo3::prelude::*;

use nalgebra::Vector3;
use num::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Particle Species of type Cargo or ATG11Receptor
///
/// We currently only distinguish between the cargo itself
/// and freely moving combinations of the receptor and ATG11.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[pyclass]
pub enum Species {
    Cargo,
    ATG11Receptor,
}

/// Interaction potential depending on the other cells species.
///
/// # Parameters
/// | Symbol | Parameter | Description |
/// | --- | --- | --- |
/// | $r_\text{interaction}$ | `interaction_range` | Maximal absolute interaction range. |
/// | $r_\text{cell}$ | `cell_radius` | Current radius of the cell. |
/// | $V_0$ | `potential_strength` | Strength of attraction and repelling. |
/// | $C_0$ | `clustering_strength` | Non-dimensional factor that describes how much stronger the attracting force is compared to the repelling force between two [ATG11Receptor](Species::ATG11Receptor) particles. |
/// | $\alpha$ | - | Relative interaction range. |
/// | $\sigma$ | - | Relative distance. |
/// | $\overrightarrow{d}$ | - | Direction in which the force will be acting. |
/// | $F(r)$ | - | Shape of the potential curve. |
/// | $\overrightarrow{F}(\overrightarrow{x_\text{ext}})$ | - | Resulting total force. |
///
/// # Spatial Cutoff
/// We impose a spatial cutoff which is calculated via
/// \\begin{equation}
///     c = \frac{1}{2}+\frac{1}{2}\text{sgn}\left(r_\text{interaction} + r_\text{cell,ext} + r_\text{cell,int} - r\right)
/// \\end{equation}
/// where $\text{sgn}(\dots)$ is the [signum](https://en.wikipedia.org/wiki/Sign_function) operator.
/// # Potential Shape
/// The potential is given by a repelling part at the center and an
/// attracting part when moving further away.
/// \\begin{align}
///     \alpha &= \frac{3}{2}\frac{r_\text{interaction}}{r_\text{cell,ext}+r_\text{cell,int}}\\\\
///     \sigma &= \frac{r}{r_\text{cell,int}+r_\text{cell,ext}}\\\\
///     \overrightarrow{d} &= \overrightarrow{x_\text{int}} - \overrightarrow{x_\text{ext}}\\\\
///     F(r) &= \frac{3}{\alpha^2}\left(3(\sigma - 1)^2 - 2\alpha(\sigma-1))\right)\\\\
///     \overrightarrow{F}(\overrightarrow{x_\text{ext}}) &= F(r) \overrightarrow{d}
/// \\end{align}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass]
pub struct TypedInteraction {
    #[pyo3(get, set)]
    pub species: Species,
    #[pyo3(get, set)]
    pub cell_radius: f64,
    #[pyo3(get, set)]
    pub epsilon: f64,
    #[pyo3(get, set)]
    pub bound: f64,
    #[pyo3(get, set)]
    pub cutoff: f64,
    #[pyo3(get, set)]
    pub clustering_strength: f64,
    neighbour_count: usize,
}

#[pymethods]
impl TypedInteraction {
    #[new]
    fn new(
        species: Species,
        cell_radius: f64,
        epsilon: f64,
        bound: f64,
        cutoff: f64,
        clustering_strength: f64,
    ) -> Self {
        Self {
            species,
            cell_radius,
            bound,
            epsilon,
            cutoff,
            clustering_strength,
            neighbour_count: 0,
        }
    }
}

impl Interaction<Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, Species)> for TypedInteraction {
    fn calculate_force_between(
        &self,
        own_pos: &Vector3<f64>,
        _own_vel: &Vector3<f64>,
        ext_pos: &Vector3<f64>,
        _ext_vel: &Vector3<f64>,
        ext_info: &(f64, Species),
    ) -> Option<Result<Vector3<f64>, CalcError>> {
        let (cell_radius_ext, species) = ext_info;
        let z = own_pos - ext_pos;
        let r = z.norm();
        let dir = z / r;
        // Calculate sigma from cell radius extern and intern
        let sigma = cell_radius_ext + self.cell_radius;

        let val =
            4.0 * self.epsilon / r * (12.0 * (sigma / r).powf(12.0) - 1.0 * (sigma / r).powf(1.0));
        let max = self.bound / r;
        let q = if self.cutoff >= r { 1.0 } else { 0.0 };
        let strength =
            q * max.min(val) * self.neighbour_count as f64 / (1.0 + self.neighbour_count as f64);

        // Calculate only attracting and repelling forces
        let attracting_force = dir * strength.max(0.0);
        let repelling_force = dir * strength.min(0.0);

        match (species, &self.species) {
            // R11 will bind to cargo
            (Species::Cargo, Species::ATG11Receptor) => {
                return Some(Ok(repelling_force + attracting_force))
            }

            // R11 forms clusters
            (Species::ATG11Receptor, Species::ATG11Receptor) => {
                return Some(Ok(
                    repelling_force + self.clustering_strength * attracting_force
                ))
            }

            // Cargo also attracts each other
            (Species::Cargo, Species::Cargo) => {
                return Some(Ok(
                    repelling_force + self.clustering_strength * attracting_force
                ))
            }

            (_, _) => return Some(Ok(repelling_force + attracting_force)),
        }
    }

    fn get_interaction_information(&self) -> (f64, Species) {
        (self.cell_radius, self.species.clone())
    }

    fn is_neighbour(
        &self,
        own_pos: &Vector3<f64>,
        ext_pos: &Vector3<f64>,
        inf: &(f64, Species),
    ) -> Result<bool, CalcError> {
        match (&self.species, &inf.1) {
            (Species::ATG11Receptor, Species::ATG11Receptor) => {
                Ok((own_pos - ext_pos).norm() <= self.cutoff)
            }
            (Species::Cargo, Species::Cargo) => Ok((own_pos - ext_pos).norm() <= self.cutoff),
            _ => Ok(false),
        }
    }

    fn react_to_neighbours(&mut self, neighbours: usize) -> Result<(), CalcError> {
        Ok(self.neighbour_count = neighbours)
    }
}

/// Random motion of particles
///
/// This is simply a wrapper class to be used with pyo3.
/// The documentation of the base struct can be found [here](cellular_raza::building_blocks::prelude::Langevin).
#[derive(CellAgent, Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct Langevin3D {
    #[Mechanics(Vector3<f64>, Vector3<f64>, Vector3<f64>)]
    pub mechanics: Langevin<3>,
}

#[pymethods]
impl Langevin3D {
    #[new]
    #[pyo3(signature = (pos, mass, damping, kb_temperature, update_interval))]
    ///
    /// Creates a new Langevin mechanics model with defined position, damping
    /// constant, mass and temperature.
    fn new(
        pos: [f64; 3],
        mass: f64,
        damping: f64,
        kb_temperature: f64,
        update_interval: usize,
    ) -> Self {
        Langevin3D {
            mechanics: Langevin::<3>::new(
                pos.into(),
                [0.0; 3].into(),
                mass,
                damping,
                kb_temperature,
                update_interval,
            ),
        }
    }

    #[getter(pos)]
    fn get_position(&self) -> [f64; 3] {
        self.mechanics.pos.into()
    }

    #[setter(pos)]
    fn set_position(&mut self, pos: [f64; 3]) {
        self.mechanics.pos = pos.into();
    }

    #[getter(damping)]
    fn get_damping(&self) -> f64 {
        self.mechanics.damping
    }

    #[setter(damping)]
    fn set_damping(&mut self, damping: f64) {
        self.mechanics.damping = damping;
    }

    #[getter(mass)]
    fn get_mass(&self) -> f64 {
        self.mechanics.mass
    }

    #[setter(mass)]
    fn set_mass(&mut self, mass: f64) {
        self.mechanics.mass = mass;
    }

    #[getter(kb_temperature)]
    fn get_kb_temperature(&self) -> f64 {
        self.mechanics.kb_temperature
    }

    #[setter(kb_temperature)]
    fn set_kb_temperature(&mut self, kb_temperature: f64) {
        self.mechanics.kb_temperature = kb_temperature;
    }
}

/// All settings which can be configured by the Python interface.
///
/// We aim to provide access to even the more lower-level settings
/// that can influence the results of our simulation.
/// Not all settings do make sense and some combinations can lead
/// to numerical integration problems.
///
/// For documentation of the individual settings, refer to the linked modules.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass]
pub struct SimulationSettings {
    /// Number of cargo particles in the simulation.
    #[pyo3(get, set)]
    pub n_cells_cargo: usize,

    /// Number of Atg11 particles in the simulation.
    #[pyo3(get, set)]
    pub n_cells_atg11_receptor: usize,

    /// Contains all paramters that all cargo particles will share in common.
    /// Only the position will be initially randomized. All other parameters will
    /// be used as is.
    #[pyo3(get, set)]
    pub particle_template_cargo: Py<ParticleTemplate>,

    /// Contains all paramters that all cargo particles will share in common.
    /// Only the position will be initially randomized. All other parameters will
    /// be used as is.
    #[pyo3(get, set)]
    pub particle_template_atg11_receptor: Py<ParticleTemplate>,

    /// Integration step of the numerical simulation.
    #[pyo3(get, set)]
    pub dt: f64,
    /// Number of intgration steps done totally.
    #[pyo3(get, set)]
    pub n_times: usize,
    /// Specifies the frequency at which results are saved as json files.
    /// Lower the number for more saved results.
    #[pyo3(get, set)]
    pub save_interval: usize,

    /// Number of threads to use in the simulation.
    #[pyo3(get, set)]
    pub n_threads: usize,

    /// See [CartesianCuboid3]
    #[pyo3(get, set)]
    pub domain_size: f64,

    /// See [CartesianCuboid3]
    #[pyo3(get, set)]
    pub domain_interaction_range: Option<f64>,

    /// Name of the folder to store the results in.
    #[pyo3(get, set)]
    pub storage_name: String,

    /// Do we want to show a progress bar
    #[pyo3(get, set)]
    pub show_progressbar: bool,
}

#[pymethods]
impl SimulationSettings {
    #[new]
    fn new(py: Python) -> PyResult<Self> {
        let cell_radius_atg11_receptor = 1.0;
        let dt = 0.25;

        Ok(SimulationSettings {
            n_cells_cargo: 100,
            n_cells_atg11_receptor: 300,

            particle_template_cargo: Py::new(
                py,
                ParticleTemplate {
                    mechanics: Py::new(
                        py,
                        Langevin3D {
                            mechanics: Langevin::<3>::new(Vector3::<f64>::zero(), Vector3::<f64>::zero(), 1.25 * cell_radius_atg11_receptor, 0.01, 0.02, 5),
                        },
                    )?,
                    interaction: Py::new(
                        py,
                        TypedInteraction::new(
                            Species::Cargo,
                            1.5,
                            2.0,
                            1.25 * cell_radius_atg11_receptor,
                            2.0,
                            0.5,
                        ),
                    )?,
                },
            )?,
            particle_template_atg11_receptor: Py::new(
                py,
                ParticleTemplate {
                    mechanics: Py::new(
                        py,
                        Langevin3D {
                            mechanics: Langevin::<3>::new(Vector3::<f64>::zero(), Vector3::<f64>::zero(), cell_radius_atg11_receptor, 0.5, 0.2, 5),
                        },
                    )?,
                    interaction: Py::new(
                        py,
                        TypedInteraction::new(
                            Species::ATG11Receptor,
                            cell_radius_atg11_receptor,
                            2.0,
                            1.0 * cell_radius_atg11_receptor,
                            2.0,
                            0.03,
                        ),
                    )?,
                },
            )?,

            dt,
            n_times: 20_001,
            save_interval: 50,

            n_threads: 1,

            domain_size: 100.0,
            domain_interaction_range: Some(25.0),

            storage_name: "out/autophagy".into(),

            show_progressbar: true,
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

    #[Interaction(Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, Species))]
    pub interaction: TypedInteraction,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass]
pub struct ParticleTemplate {
    #[pyo3(get, set)]
    pub mechanics: Py<Langevin3D>,

    #[pyo3(get, set)]
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

/// Takes [SimulationSettings], runs the full simulation and returns the string of the output directory.
#[pyfunction]
pub fn run_simulation(
    simulation_settings: SimulationSettings,
    py: Python,
) -> Result<std::path::PathBuf, pyo3::PyErr> {
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let particles = (0..simulation_settings.n_cells_cargo
        + simulation_settings.n_cells_atg11_receptor)
        .map(|n| {
            let low = 0.4 * simulation_settings.domain_size;
            let high = 0.6 * simulation_settings.domain_size;
            let pos = if n < simulation_settings.n_cells_cargo {
                Vector3::from([
                    rng.gen_range(low..high),
                    rng.gen_range(low..high),
                    rng.gen_range(low..high),
                ])
            } else {
                Vector3::from([
                    rng.gen_range(0.0..simulation_settings.domain_size),
                    rng.gen_range(0.0..simulation_settings.domain_size),
                    rng.gen_range(0.0..simulation_settings.domain_size),
                ])
            };
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

    // Calculate the maximal interaction range
    let interaction_range_max = simulation_settings
        .particle_template_cargo
        .borrow(py)
        .interaction
        .borrow(py)
        .cutoff
        .max(
            simulation_settings
                .particle_template_atg11_receptor
                .borrow(py)
                .interaction
                .borrow(py)
                .cutoff,
        );

    let interaction_range = match simulation_settings.domain_interaction_range {
        Some(range) => interaction_range_max.max(range),
        None => interaction_range_max,
    };
    let domain = CartesianCuboid3::from_boundaries_and_interaction_ranges(
        [0.0; 3],
        [simulation_settings.domain_size; 3],
        [interaction_range; 3],
    )
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

    let storage = StorageConfig::from_path("out/autophagy".into());

    let simulation_setup = create_simulation_setup!(
        Domain: domain,
        Cells: particles,
        Time: time,
        MetaParams: meta_params,
        Storage: storage.clone()
    );

    let mut supervisor = SimulationSupervisor::initialize_from_setup(simulation_setup);
    supervisor.config.show_progressbar = simulation_settings.show_progressbar;

    let simulation_result = supervisor.run_full_sim().or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Rust error in simulation run: {e}"),
        ))
    })?;
    Ok(simulation_result.storage.get_location())
}
