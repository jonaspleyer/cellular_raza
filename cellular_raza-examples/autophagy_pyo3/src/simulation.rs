use cellular_raza::prelude::*;
use pyo3::prelude::{pyclass, pymethods};

use nalgebra::Vector3;
use num::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// All particle species
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
/// | $C_0$ | `clustering_strength` | Non-dimensional factor that describes how much stronger the attracting force is compared to the repelling force between two [R11](Species::R11) particles. |
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
    pub potential_strength: f64,
    #[pyo3(get, set)]
    pub interaction_range: f64,
    #[pyo3(get, set)]
    pub clustering_strength: f64,
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
        let (r, dir) = {
            let z = own_pos - ext_pos;
            let r = z.norm();
            (r, z.normalize())
        };

        let (cell_radius_ext, species) = ext_info;
        // Introduce Non-dimensional length variable
        let sigma = r / (self.cell_radius + cell_radius_ext);
        let spatial_cutoff = (1.0
            + (self.interaction_range + cell_radius_ext + self.cell_radius - r).signum())
            * 0.5;

        // Calculate the strength of the interaction with correct bounds
        let alpha = 3.0 / 2.0 * (self.interaction_range / (cell_radius_ext + self.cell_radius));
        let form =
            (3.0 * (sigma - 1.0).powf(2.0) - 2.0 * alpha * (sigma - 1.0)) * 3.0 / alpha.powf(2.0);
        let strength = -self.potential_strength * form.clamp(-1.0, 1.0);

        // Calculate only attracting and repelling forces
        let attracting_force = dir * strength.max(0.0) * spatial_cutoff;
        let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

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
                return Some(Ok(repelling_force + attracting_force))
            }

            (_, _) => return Some(Ok(repelling_force)),
        }
    }

    fn get_interaction_information(&self) -> (f64, Species) {
        (self.cell_radius, self.species.clone())
    }
}

/// Random motion of particles
///
/// # Parameters
/// | Symbol | Parameter | Description |
/// | --- | --- | --- |
/// | $\vec{x}$ | `pos` | Position of the particle. |
/// | $\vec{v}$ | `vel` | Velocity of the particle. |
/// | $\lambda$ | `dampening_constant` | Dampening constant of each particle. |
/// | $m$ | `mass` | Mass of the particle. |
/// | $v_r$ | `random_travel_velocity` | The absolute value of velocity which the particle is currently travelling at. |
/// | $\vec{d}$ | `random_direction_travel` | The direction in which the particle is currently tarvelling. |
/// | $t_r$ | `random_update_time` | The time until the next update of the random steps will be done. Set this to [f64::INFINITY] to never update the travel direction or velocity. |
///
/// # Position and Velocity Update
/// Positions and velocities are numerically integrated.
/// The differential equation which is solved corresponds to a euclidean equation of motion with dampening and a random part.
/// \\begin{align}
///     \frac{\partial}{\partial t}\vec{x} &= \vec{v}(t) + v_r(t)\vec{d}(t)\\\\
///     \frac{\partial}{\partial t}\vec{v} &= \frac{1}{m}\vec{F}(x, t) - \lambda\vec{v}(t)
/// \\end{align}
/// By choosing the `random_update_time` $t_r$ larger than the integration step, we can resolve smaller timesteps to more accurately solve the equations.
/// This procedure is recommended.
/// In this scheme, both $v_r$ and $\vec{d}$ depend on time in the sence that their values are changed at discrete time events.
/// The notation is slightly different to the usually used for stochastic processes.
#[derive(CellAgent, Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct Brownian3D {
    #[Mechanics(Vector3<f64>, Vector3<f64>, Vector3<f64>)]
    pub mechanics: Brownian<3>,
}

#[pymethods]
impl Brownian3D {
    #[new]
    #[pyo3(signature = (pos, diffusion_constant, kb_temperature, particle_random_update_interval))]
    ///
    /// Creates a new Brownian mechanics model with defined position, diffusion
    /// constant and temperature.
    fn new(
        pos: [f64; 3],
        diffusion_constant: f64,
        kb_temperature: f64,
        particle_random_update_interval: usize,
    ) -> Self {
        Brownian3D {
            mechanics: Brownian::<3>::new(
                pos.into(),
                diffusion_constant,
                kb_temperature,
                particle_random_update_interval,
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

    #[getter(diffusion_constant)]
    fn get_diffusion_constant(&self) -> f64 {
        self.mechanics.diffusion_constant
    }

    #[setter(diffusion_constant)]
    fn set_diffusion_constant(&mut self, diffusion_constant: f64) {
        self.mechanics.diffusion_constant = diffusion_constant;
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
    pub particle_template_cargo: Particle,

    /// Contains all paramters that all cargo particles will share in common.
    /// Only the position will be initially randomized. All other parameters will
    /// be used as is.
    #[pyo3(get, set)]
    pub particle_template_atg11_receptor: Particle,

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

    /// Name of the folder to store the results in.
    #[pyo3(get, set)]
    pub storage_name: String,

    /// Do we want to show a progress bar
    #[pyo3(get, set)]
    pub show_progressbar: bool,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        let cell_radius_atg11_receptor = 1.0;
        let dt = 0.25;

        SimulationSettings {
            n_cells_cargo: 100,
            n_cells_atg11_receptor: 300,

            particle_template_cargo: Particle {
                mechanics: Brownian3D {
                    mechanics: Brownian::<3>::new(Vector3::<f64>::zero(), 0.01, 0.02, 5),
                },
                interaction: TypedInteraction {
                    species: Species::Cargo,
                    cell_radius: 1.5,
                    potential_strength: 2.0,
                    interaction_range: 3.0 * cell_radius_atg11_receptor,
                    clustering_strength: 0.5,
                },
            },
            particle_template_atg11_receptor: Particle {
                mechanics: Brownian3D {
                    mechanics: Brownian::<3>::new(Vector3::<f64>::zero(), 0.5, 0.2, 5),
                },
                interaction: TypedInteraction {
                    species: Species::ATG11Receptor,
                    cell_radius: cell_radius_atg11_receptor,
                    potential_strength: 2.0,
                    interaction_range: 1.0 * cell_radius_atg11_receptor,
                    clustering_strength: 0.03,
                },
            },

            dt,
            n_times: 20_001,
            save_interval: 50,

            n_threads: 1,

            domain_size: 100.0,

            storage_name: "out/autophagy".into(),

            show_progressbar: true,
        }
    }
}

#[pymethods]
impl SimulationSettings {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
#[pyclass]
pub struct Particle {
    #[pyo3(get, set)]
    #[Mechanics(Vector3<f64>, Vector3<f64>, Vector3<f64>)]
    pub mechanics: Brownian3D,
    #[pyo3(get, set)]
    #[Interaction(Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, Species))]
    pub interaction: TypedInteraction,
}

#[pymethods]
impl Particle {
    #[new]
    fn new(mechanics: Brownian3D, interaction: TypedInteraction) -> Self {
        Self {
            mechanics,
            interaction,
        }
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
pub fn run_simulation_rs(
    simulation_settings: SimulationSettings,
) -> Result<std::path::PathBuf, SimulationError> {
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
            let mut particle = if n < simulation_settings.n_cells_cargo {
                simulation_settings.particle_template_cargo.clone()
            } else {
                simulation_settings.particle_template_atg11_receptor.clone()
            };
            particle.mechanics.mechanics.set_pos(&pos);
            particle
        })
        .collect::<Vec<_>>();

    // Calculate the maximal interaction range
    let interaction_range_max = simulation_settings
        .particle_template_cargo
        .interaction
        .interaction_range
        .max(
            simulation_settings
                .particle_template_atg11_receptor
                .interaction
                .interaction_range,
        );

    let domain = CartesianCuboid3::from_boundaries_and_interaction_ranges(
        [0.0; 3],
        [simulation_settings.domain_size; 3],
        [7.0 * interaction_range_max; 3],
    )?;

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

    let simulation_result = supervisor.run_full_sim()?;
    Ok(simulation_result.storage.get_location())
}
