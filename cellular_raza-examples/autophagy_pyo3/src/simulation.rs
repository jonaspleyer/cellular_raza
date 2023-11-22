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
#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct TypedInteraction {
    pub species: Species,
    pub cell_radius: f64,
    pub potential_strength: f64,
    pub interaction_range: f64,
    pub clustering_strength: f64,
    neighbour_count: usize,
}

fn calcualte_avidity(own_neighbour_count: usize, ext_neighbour_count: usize) -> f64 {
    let n = 2.0;
    let alpha = 1.0;
    let nc = own_neighbour_count.min(ext_neighbour_count);
    let s = (nc as f64 / alpha).powf(n) / (1.0 + (nc as f64 / alpha).powf(n));
    //2.0 * self.neighbour_count as f64/(1.0 + self.neighbour_count as f64)
    let avidity = 1.3 * s; // * self.avidity
    avidity
}

impl Interaction<Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, usize, Species)>
    for TypedInteraction
{
    fn calculate_force_between(
        &self,
        own_pos: &Vector3<f64>,
        _own_vel: &Vector3<f64>,
        ext_pos: &Vector3<f64>,
        _ext_vel: &Vector3<f64>,
        ext_info: &(f64, usize, Species),
    ) -> Option<Result<Vector3<f64>, CalcError>> {
        // Calculate radius and direction
        let min_relative_distance_to_center = 0.3162277660168379;
        let (r, dir) =
            match (own_pos - ext_pos).norm() < self.cell_radius * min_relative_distance_to_center {
                false => {
                    let z = own_pos - ext_pos;
                    let r = z.norm();
                    (r, z.normalize())
                }
                true => {
                    let dir = match own_pos == ext_pos {
                        true => {
                            return None;
                        }
                        false => (own_pos - ext_pos).normalize(),
                    };
                    let r = self.cell_radius * min_relative_distance_to_center;
                    (r, dir)
                }
            };
        let (ext_radius, ext_neighbour_count, ext_species) = ext_info;
        // Introduce Non-dimensional length variable
        let sigma = r / (self.cell_radius + ext_radius);
        let bound = 4.0 + 1.0 / sigma;
        let spatial_cutoff =
            (1.0 + (self.interaction_range + self.cell_radius + ext_radius - r).signum()) * 0.5;

        // Calculate the strength of the interaction with correct bounds
        let strength = self.potential_strength
            * ((1.0 / sigma).powf(2.0) - (1.0 / sigma).powf(4.0))
                .min(bound)
                .max(-bound);

        // Calculate only attracting and repelling forces
        let attracting_force = dir * strength.max(0.0) * spatial_cutoff;
        let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

        let avidity = calcualte_avidity(self.neighbour_count, *ext_neighbour_count);

        match (ext_species, &self.species) {
            // R11 will bind to cargo
            (Species::Cargo, Species::ATG11Receptor) | (Species::ATG11Receptor, Species::Cargo) => {
                return Some(Ok(repelling_force + avidity * attracting_force))
            }

            // R11 forms clusters
            (Species::ATG11Receptor, Species::ATG11Receptor) => {
                return Some(Ok(
                    repelling_force + self.clustering_strength * attracting_force
                ))
            }

            (Species::Cargo, Species::Cargo) => {
                return Some(Ok(repelling_force + attracting_force))
            }
        }
    }

    fn get_interaction_information(&self) -> (f64, usize, Species) {
        (self.cell_radius, self.neighbour_count, self.species.clone())
    }

    fn is_neighbour(
        &self,
        own_pos: &Vector3<f64>,
        ext_pos: &Vector3<f64>,
        ext_inf: &(f64, usize, Species),
    ) -> Result<bool, CalcError> {
        match (&self.species, &ext_inf.2) {
            (Species::ATG11Receptor, Species::ATG11Receptor) => {
                Ok((own_pos - ext_pos).norm() <= 2.0 * (self.cell_radius + ext_inf.0))
            }
            (Species::Cargo, Species::Cargo) => {
                Ok((own_pos - ext_pos).norm() <= 2.0 * (self.cell_radius + ext_inf.0))
            }
            _ => Ok(false),
        }
    }

    fn react_to_neighbours(&mut self, neighbours: usize) -> Result<(), CalcError> {
        Ok(self.neighbour_count = neighbours)
    }
}

#[pymethods]
impl TypedInteraction {
    #[new]
    fn new(
        species: Species,
        cell_radius: f64,
        potential_strength: f64,
        interaction_range: f64,
        clustering_strength: f64,
    ) -> Self {
        Self {
            species,
            cell_radius,
            potential_strength,
            interaction_range,
            clustering_strength,
            neighbour_count: 0,
        }
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

    #[getter(update_interval)]
    fn get_update_interval(&self) -> usize {
        self.mechanics.update_interval
    }

    #[setter(update_interval)]
    fn set_update_interval(&mut self, update_interval: usize) {
        self.mechanics.update_interval = update_interval;
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

    /// Size of the centered cuboid in which the cargo cells will be placed initially
    pub domain_size_cargo: f64,

    /// See [CartesianCuboid3]
    pub domain_n_voxels: Option<usize>,

    /// Name of the folder to store the results in.
    pub storage_name: String,

    /// Do we want to show a progress bar
    pub show_progressbar: bool,
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
                    0.01,                    // potential_strength
                    0.8 * cell_radius_cargo, // interaction_range
                    1.3,                     // clustering_strength
                )?,
            )?,
            particle_template_atg11_receptor: Py::new(
                py,
                create_particle_template(
                    py,
                    Species::ATG11Receptor,           // species
                    cell_radius_atg11_receptor,       // cell_radius
                    0.5,                              // damping
                    0.02,                             // kb_temperature
                    5,                                // update_interval
                    0.02,                             // potential_strength
                    0.8 * cell_radius_atg11_receptor, // interaction_range
                    1.3,                              // clustering_strength
                )?,
            )?,

            dt,
            n_times: 20_001,
            save_interval: 50,

            n_threads: 1,

            domain_size: 100.0,
            domain_size_cargo: 20.0,
            domain_n_voxels: Some(4),

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

fn save_simulation_settings(path: &std::path::PathBuf, simulation_settings: &SimulationSettings) -> PyResult<()> {
    // Also save the SimulationSettings into the same folder
    let mut save_path = path.clone();
    save_path.push("simulation_settings.json");
    let f = std::fs::File::create(save_path)?;
    let writer = std::io::BufWriter::new(f);
    serde_json::to_writer_pretty(writer, &simulation_settings).or_else(|e| {
        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("serde_json error in writing simulation settings to file: {e}")
        ))
    })?;
    Ok(())
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
            let middle = simulation_settings.domain_size / 2.0;
            let low = middle - 0.5 * simulation_settings.domain_size_cargo;
            let high = middle + 0.5 * simulation_settings.domain_size_cargo;
            let pos = if n < simulation_settings.n_cells_cargo {
                Vector3::from([
                    rng.gen_range(low..high),
                    rng.gen_range(low..high),
                    rng.gen_range(low..high),
                ])
            } else {
                // We do not want to spawn the ATG11Receptor particles in the middle where the cargo
                // will be placed. Thus we calculate where else we can spawn them.
                let mut loc = [middle; 3];
                while loc.iter().all(|x| low <= *x && *x <= high) {
                    loc = [
                        rng.gen_range(0.0..simulation_settings.domain_size),
                        rng.gen_range(0.0..simulation_settings.domain_size),
                        rng.gen_range(0.0..simulation_settings.domain_size),
                    ];
                }
                Vector3::from(loc)
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

    let storage = StorageConfig::from_path("out/autophagy".into());

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
