use cellular_raza::prelude::*;
use pyo3::prelude::{pyclass, pymethods};

use nalgebra::Vector3;
use num::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[pyclass]
pub enum Species {
    Cargo,
    R11,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass]
pub struct CellSpecificInteraction {
    #[pyo3(get, set)]
    species: Species,
    #[pyo3(get, set)]
    cell_radius: f64,
    #[pyo3(get, set)]
    potential_strength: f64,
    #[pyo3(get, set)]
    interaction_range: f64,
    #[pyo3(get, set)]
    clustering_strength: f64,
}

impl Interaction<Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, Species)>
    for CellSpecificInteraction
{
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

        let (ext_radius, species) = ext_info;
        // Introduce Non-dimensional length variable
        let sigma = r / (self.cell_radius + ext_radius);
        let spatial_cutoff =
            (1.0 + (self.interaction_range + ext_radius + self.cell_radius - r).signum()) * 0.5;

        // Calculate the strength of the interaction with correct bounds
        let alpha = 3.0 / 2.0 * (self.interaction_range / (ext_radius + self.cell_radius));
        let form =
            (3.0 * (sigma - 1.0).powf(2.0) - 2.0 * alpha * (sigma - 1.0)) * 3.0 / alpha.powf(2.0);
        let strength = -self.potential_strength * form.clamp(-1.0, 1.0);

        // Calculate only attracting and repelling forces
        let attracting_force = dir * strength.max(0.0) * spatial_cutoff;
        let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

        match (species, &self.species) {
            // R11 will bind to cargo
            (Species::Cargo, Species::R11) => return Some(Ok(repelling_force + attracting_force)),
            (Species::R11, Species::Cargo) => return Some(Ok(repelling_force + attracting_force)),

            // R11 forms clusters
            (Species::R11, Species::R11) => {
                return Some(Ok(
                    repelling_force + self.clustering_strength * attracting_force
                ))
            }

            (_, _) => return Some(Ok(repelling_force)),
        }
    }

    fn get_interaction_information(&self) -> (f64, Species) {
        (self.cell_radius, self.species.clone())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct MyMechanics {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
    #[pyo3(get, set)]
    pub dampening_constant: f64,
    #[pyo3(get, set)]
    pub mass: f64,
    #[pyo3(get, set)]
    pub random_travel_velocity: f64,
    pub random_direction_travel: nalgebra::UnitVector3<f64>,
    #[pyo3(get, set)]
    pub random_update_time: f64,
}

#[pymethods]
impl MyMechanics {
    #[getter(pos)]
    fn get_pos(&self) -> [f64; 3] {
        self.pos.into()
    }

    #[setter(pos)]
    fn set_pos(&mut self, pos: [f64; 3]) {
        self.pos = pos.into();
    }

    #[getter(vel)]
    fn get_vel(&self) -> [f64; 3] {
        self.vel.into()
    }

    #[setter(vel)]
    fn set_vel(&mut self, vel: [f64; 3]) {
        self.vel = vel.into();
    }

    #[getter(random_direction_travel)]
    fn get_random_direction_travel(&self) -> [f64; 3] {
        self.random_direction_travel.clone_owned().into()
    }

    #[setter(random_direction_travel)]
    fn set_random_direction_travel(&mut self, random_direction_travel: [f64; 3]) {
        self.random_direction_travel =
            nalgebra::UnitVector3::new_normalize(random_direction_travel.into());
    }
}

impl Mechanics<Vector3<f64>, Vector3<f64>, Vector3<f64>> for MyMechanics {
    fn pos(&self) -> Vector3<f64> {
        self.pos
    }

    fn velocity(&self) -> Vector3<f64> {
        self.vel
    }

    fn set_pos(&mut self, p: &Vector3<f64>) {
        self.pos = *p;
    }

    fn set_velocity(&mut self, v: &Vector3<f64>) {
        self.vel = *v;
    }

    fn set_random_variable(&mut self, rng: &mut rand_chacha::ChaCha8Rng) -> Option<f64> {
        let phi = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let psi = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        self.random_direction_travel = nalgebra::UnitVector3::new_normalize(Vector3::from([
            phi.sin() * psi.cos(),
            phi.sin() * psi.sin(),
            phi.cos(),
        ]));
        Some(rng.gen_range(0.5..1.5) * self.random_update_time)
    }

    fn calculate_increment(
        &self,
        force: Vector3<f64>,
    ) -> Result<(Vector3<f64>, Vector3<f64>), CalcError> {
        let dx = self.vel + self.random_travel_velocity * self.random_direction_travel.into_inner();
        let dv = force / self.mass - self.dampening_constant * self.vel;
        Ok((dx, dv))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass]
pub struct SimulationSettings {
    #[pyo3(get, set)]
    pub n_cells_cargo: usize,
    #[pyo3(get, set)]
    pub n_cells_r11: usize,

    #[pyo3(get, set)]
    pub cell_dampening: f64,
    #[pyo3(get, set)]
    pub cell_radius_cargo: f64,
    #[pyo3(get, set)]
    pub cell_radius_r11: f64,

    #[pyo3(get, set)]
    pub cell_mechanics_interaction_range_cargo: f64,
    #[pyo3(get, set)]
    pub cell_mechanics_interaction_range_r11: f64,
    #[pyo3(get, set)]
    pub cell_mechanics_random_travel_velocity: f64,
    #[pyo3(get, set)]
    pub cell_mechanics_random_update_time: f64,

    #[pyo3(get, set)]
    pub cell_mechanics_potential_strength: f64,
    #[pyo3(get, set)]
    pub cell_mechanics_relative_clustering_strength: f64,

    #[pyo3(get, set)]
    pub dt: f64,
    #[pyo3(get, set)]
    pub n_times: usize,
    #[pyo3(get, set)]
    pub save_interval: usize,

    #[pyo3(get, set)]
    pub n_threads: usize,

    #[pyo3(get, set)]
    pub domain_size: f64,

    #[pyo3(get, set)]
    pub storage_name: String,

    #[pyo3(get, set)]
    pub show_progressbar: bool,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        let cell_radius_r11 = 1.0;
        let dt = 0.25;

        SimulationSettings {
            n_cells_cargo: 1,
            n_cells_r11: 500,

            cell_dampening: 1.0,
            cell_radius_cargo: 10.0,
            cell_radius_r11,

            cell_mechanics_interaction_range_cargo: 3.0 * cell_radius_r11,
            cell_mechanics_interaction_range_r11: 1.0 * cell_radius_r11,
            cell_mechanics_random_travel_velocity: 0.05,
            cell_mechanics_random_update_time: 200. * dt,

            cell_mechanics_potential_strength: 2.0,
            cell_mechanics_relative_clustering_strength: 0.03,

            dt,
            n_times: 2_001,
            save_interval: 2_001,

            n_threads: 3,

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

pub fn run_simulation_rs(
    simulation_settings: SimulationSettings,
) -> Result<std::path::PathBuf, SimulationError> {
    // Define the seed
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let cells = (0..simulation_settings.n_cells_cargo + simulation_settings.n_cells_r11)
        .map(|n| {
            let pos = if n == 0 {
                Vector3::from([simulation_settings.domain_size / 2.0; 3])
            } else {
                Vector3::from([
                    rng.gen_range(0.0..simulation_settings.domain_size),
                    rng.gen_range(0.0..simulation_settings.domain_size),
                    rng.gen_range(0.0..simulation_settings.domain_size),
                ])
            };
            let vel = Vector3::zero();
            let (cell_radius, species, interaction_range) = if n < simulation_settings.n_cells_cargo
            {
                (
                    simulation_settings.cell_radius_cargo,
                    Species::Cargo,
                    simulation_settings.cell_mechanics_interaction_range_cargo,
                )
            } else {
                (
                    simulation_settings.cell_radius_r11,
                    Species::R11,
                    simulation_settings.cell_mechanics_interaction_range_r11,
                )
            };
            ModularCell {
                mechanics: MyMechanics {
                    pos,
                    vel,
                    dampening_constant: simulation_settings.cell_dampening,
                    mass: 4. / 3. * std::f64::consts::PI * cell_radius.powf(3.0),
                    random_travel_velocity: if n < simulation_settings.n_cells_cargo {
                        0.0
                    } else {
                        simulation_settings.cell_mechanics_random_travel_velocity
                    },
                    random_direction_travel: Vector3::<f64>::y_axis(),
                    random_update_time: simulation_settings.cell_mechanics_random_update_time,
                },
                interaction: CellSpecificInteraction {
                    species,
                    potential_strength: simulation_settings.cell_mechanics_potential_strength,
                    interaction_range,
                    cell_radius,
                    clustering_strength: simulation_settings
                        .cell_mechanics_relative_clustering_strength,
                },
                cycle: NoCycle {},
                interaction_extracellular: NoExtracellularGradientSensing {},
                cellular_reactions: NoCellularreactions {},
            }
        })
        .collect::<Vec<_>>();

    let domain = CartesianCuboid3::from_boundaries_and_n_voxels(
        [0.0; 3],
        [simulation_settings.domain_size; 3],
        [3; 3],
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
    // storage.export_formats = vec![ExportOptions::Vtk];

    let simulation_setup = create_simulation_setup!(
        Domain: domain,
        Cells: cells,
        Time: time,
        MetaParams: meta_params,
        Storage: storage
    );

    let mut supervisor = SimulationSupervisor::initialize_from_setup(simulation_setup);
    supervisor.config.show_progressbar = simulation_settings.show_progressbar;
    // let simulation_result = run_full_simulation!(simulation_setup, [Mechanics, Interaction]);

    let simulation_result = supervisor.run_full_sim()?;
    Ok(simulation_result.storage.get_location())
}
