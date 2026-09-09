// #![deny(missing_docs)]
//! This package is an example of how to construct python bindings with documentation with
//! `cellular_raza <https://cellular-raza.com/>`_.

use std::any::Any;

use cellular_raza::prelude::*;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction, derive::*};

use serde::{Deserialize, Serialize};

use nalgebra::{Matrix2xX, Vector2};

use fungus::*;
use geometry::*;
use plant_cell::*;

mod fungus;
mod geometry;
mod plant_cell;

#[gen_stub_pyclass_enum]
#[pyclass(from_py_object)]
#[derive(Clone, Deserialize, Serialize)]
pub enum Agent {
    P(PlantCell),
    F(Fungus),
}

type V = Matrix2xX<f64>;

impl Agent {
    fn zero_force_default(&self) -> V {
        use Agent::*;
        match self {
            P(p) => nalgebra::Matrix2xX::zeros(p.position.ncols()),
            F(f) => nalgebra::Matrix2xX::zeros(f.mechanics.pos.nrows()),
        }
    }
}

#[pymethods]
impl Agent {
    fn is_fungus(&self) -> bool {
        match self {
            Agent::F(_) => true,
            _ => false,
        }
    }
}

impl Position<V> for Agent {
    fn pos(&self) -> V {
        use Agent::*;
        match self {
            P(p) => p.pos(),
            F(f) => f.mechanics.pos().transpose(),
        }
    }

    fn set_pos(&mut self, position: &V) {
        use Agent::*;
        match self {
            P(p) => p.set_pos(position),
            F(f) => {
                f.position_helper = position.clone();
                f.mechanics.set_pos(&position.transpose());
                if let Some((n, p)) = f.fixed_pos {
                    f.position_helper.column_mut(n)[0] = p[0];
                    f.position_helper.column_mut(n)[1] = p[1];
                    f.mechanics.pos.row_mut(n)[0] = p[0];
                    f.mechanics.pos.row_mut(n)[1] = p[1];
                }
            }
        }
    }
}

impl Velocity<V> for Agent {
    fn velocity(&self) -> V {
        use Agent::*;
        match self {
            P(p) => p.velocity(),
            F(f) => f.mechanics.velocity().transpose(),
        }
    }

    fn set_velocity(&mut self, velocity: &V) {
        use Agent::*;
        match self {
            P(p) => p.set_velocity(velocity),
            F(f) => f.mechanics.set_velocity(&velocity.transpose()),
        }
    }
}

impl Mechanics<V, V, V, f64> for Agent {
    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: f64,
    ) -> Result<(V, V), RngError> {
        use Agent::*;
        match self {
            P(p) => p.get_random_contribution(rng, dt),
            F(f) => f
                .mechanics
                .get_random_contribution(rng, dt)
                .map(|(p, v)| (p.transpose(), v.transpose())),
        }
    }

    fn calculate_increment(&self, force: V) -> Result<(V, V), CalcError> {
        use Agent::*;
        match self {
            P(p) => p.calculate_increment(force),
            F(f) => f
                .mechanics
                .calculate_increment(force.transpose())
                .map(|(p, v)| (p.transpose(), v.transpose())),
        }
    }
}

impl Cycle for Agent {
    fn update_cycle(
        _: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Self,
    ) -> Option<CycleEvent> {
        match cell {
            Agent::F(f) => f.mechanics.spring_length += f.growth_rate * dt,
            _ => (),
        }
        None
    }

    fn divide(_: &mut rand_chacha::ChaCha8Rng, _: &mut Self) -> Result<Self, DivisionError> {
        panic!("This function should never be called!")
    }
}

impl Agent {
    pub fn get_middle(&self) -> Vector2<f64> {
        use Agent::*;
        match self {
            P(p) => area_centroid(&p.position),
            F(f) => f.mechanics.pos.row_mean().transpose(),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub enum Inf {
    P,
    // Radius, Interaction Range, Strength, Potential_stiffness
    F(f64, f64, f64, f64),
}

fn force_plant_fungus(
    plant_pos: &V,
    fungus_pos: &V,
    radius: f64,
    cutoff: f64,
    strength: f64,
    potential_stiffness: f64,
) -> Result<(V, V), CalcError> {
    // Initialize Forces
    let mut force_plant = V::zeros(plant_pos.ncols());
    let mut force_fungus = V::zeros(fungus_pos.ncols());

    for (n, point) in fungus_pos.column_iter().enumerate() {
        let mut dist = f64::INFINITY;
        let mut h = 0.0;
        let mut k = 0;
        let mut force = Vector2::zeros();

        for (i, (v1, v2)) in plant_pos.column_iter().circular_tuple_windows().enumerate() {
            let (s, q) = closest_point_on_segment(&point, &v1, &v2);
            let z = q - point;
            let d = z.norm();

            if d < cutoff && d < dist {
                dist = d;
                h = s;
                k = i;
                // TODO this needs to be implemented!
                // force = z
                force = calculate_morse_interaction(
                    &q,
                    &Vector2::from([point[0], point[1]]),
                    0.0,
                    radius,
                    cutoff,
                    strength,
                    potential_stiffness,
                )?
                .1;
            }
        }
        if dist != f64::INFINITY {
            use core::ops::AddAssign;
            force_fungus.column_mut(n).add_assign(force);
            force_plant.column_mut(k).add_assign(-h * force);
            force_plant.column_mut(k).add_assign(-(1.0 - h) * force);
        }
    }

    Ok((force_plant, force_fungus))
}

impl Interaction<V, V, V, Inf> for Agent {
    fn calculate_force_between(
        &self,
        own_pos: &V,
        own_vel: &V,
        ext_pos: &V,
        ext_vel: &V,
        ext_info: &Inf,
    ) -> Result<(V, V), CalcError> {
        use Agent::*;
        match (self, ext_info) {
            (P(p), Inf::P) => p.calculate_force_between(own_pos, own_vel, ext_pos, ext_vel, &()),
            (P(_), Inf::F(r, ir, s, ps)) => {
                force_plant_fungus(&own_pos, &ext_pos, *r, *ir, *s, *ps)
            }
            (F(f), Inf::P) => {
                let (f1, f2) = force_plant_fungus(
                    &ext_pos,
                    &own_pos,
                    f.interaction.0.radius,
                    f.interaction.0.cutoff,
                    f.interaction.0.strength,
                    f.interaction.0.potential_stiffness,
                )?;
                Ok((f2, f1))
            }
            (F(f), Inf::F(r, _, _, _)) => f
                .interaction
                .calculate_force_between(
                    &own_pos.transpose(),
                    &own_vel.transpose(),
                    &ext_pos.transpose(),
                    &ext_vel.transpose(),
                    r,
                )
                .map(|(f1, f2)| (f1.transpose(), f2.transpose())),
        }
    }
}

impl InteractionInformation<Inf> for Agent {
    fn get_interaction_information(&self) -> Inf {
        use Agent::*;
        match self {
            P(_) => Inf::P,
            F(f) => Inf::F(
                f.interaction.0.radius,
                f.interaction.0.cutoff,
                f.interaction.0.strength,
                f.interaction.0.potential_stiffness,
            ),
        }
    }
}

/// Contains settings needed to specify the simulation
#[gen_stub_pyclass]
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone, Debug, approx::AbsDiffEq, PartialEq, Serialize, Deserialize)]
pub struct SimulationSettings {
    /// Overall domain size
    pub domain_size: f64,
    pub domain_force_dist: f64,
    pub domain_interaction_range: f64,
    /// Number of voxels to create subdivisions
    #[approx(equal)]
    pub n_voxels: usize,
    /// Time increment used to solve the simulation
    pub dt: f64,
    /// Maximum duration of the simulation
    pub t_max: f64,
    /// Frequency to store results
    pub save_interval: f64,
    /// Random initial seed
    #[approx(equal)]
    pub rng_seed: u64,
    // Cell-specific settings
    pub perimeter_mod: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl SimulationSettings {
    /// Creates a new :class:`SimulationSettings` class.
    #[new]
    fn new() -> Self {
        Self {
            domain_size: 30.0,
            domain_force_dist: 0.01,
            domain_interaction_range: 0.75,
            n_voxels: 3,
            dt: 0.05,
            t_max: 10.0,
            save_interval: 1.0,
            rng_seed: 0,
            perimeter_mod: 1.2,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

#[derive(Clone, Domain)]
pub struct MyDomain {
    #[DomainPartialDerive]
    #[DomainRngSeed]
    domain: CartesianCuboid<f64, 2>,
    force_dist: f64,
    interaction_range: f64,
}

impl DomainCreateSubDomains<MySubDomain> for MyDomain {
    type VoxelIndex = [usize; 2];
    type SubDomainIndex = usize;

    fn create_subdomains(
        &self,
        n_subdomains: core::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<Item = (Self::SubDomainIndex, MySubDomain, Vec<Self::VoxelIndex>)>,
        DecomposeError,
    > {
        Ok(self
            .domain
            .create_subdomains(n_subdomains)?
            .into_iter()
            .map(move |(n, subdomain, voxels)| {
                (
                    n,
                    MySubDomain {
                        subdomain,
                        force_dist: self.force_dist,
                        interaction_range: self.interaction_range,
                    },
                    voxels,
                )
            }))
    }
}

#[derive(Clone, SubDomain, Deserialize, Serialize)]
pub struct MySubDomain {
    #[Base]
    subdomain: CartesianSubDomain<f64, 2>,
    force_dist: f64,
    interaction_range: f64,
}

impl SortCells<Agent> for MyDomain {
    type VoxelIndex = [usize; 2];

    fn get_voxel_index_of(
        &self,
        cell: &Agent,
    ) -> Result<Self::VoxelIndex, cellular_raza::prelude::BoundaryError> {
        let pos = cell.get_middle();
        self.domain.get_voxel_index_of_raw(&pos)
    }
}

impl SubDomainMechanics<V, V> for MySubDomain {
    fn apply_boundary(&self, pos: &mut V, vel: &mut V) -> Result<(), BoundaryError> {
        for (mut p, mut v) in pos.column_iter_mut().zip(vel.column_iter_mut()) {
            let mut pi = [p[0], p[1]];
            let mut vi = [v[0], v[1]];
            self.subdomain.apply_boundary(&mut pi, &mut vi)?;
            p[0] = pi[0];
            p[1] = pi[1];
            v[0] = vi[0];
            v[1] = vi[1];
        }
        Ok(())
    }
}

impl SortCells<Agent> for MySubDomain {
    type VoxelIndex = [usize; 2];

    fn get_voxel_index_of(&self, cell: &Agent) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.get_middle();
        self.subdomain.get_index_of(pos)
    }
}

impl SubDomainForce<V, V, V, Inf> for MySubDomain {
    fn calculate_custom_force(
        &self,
        pos: &V,
        _: &V,
        inf: &Inf,
    ) -> Result<V, cellular_raza::concepts::CalcError> {
        match inf {
            Inf::F(_, _, _, _) => Ok(0.0 * pos),
            Inf::P => {
                let smin = self.subdomain.get_domain_min();
                let smax = self.subdomain.get_domain_max();

                let corners = nalgebra::matrix![
                    smin[0], smin[0], smax[0], smax[0];
                    smin[1], smax[1], smax[1], smin[1];
                ];

                let mut force = Matrix2xX::zeros(pos.ncols());

                for (p1, mut f) in pos.column_iter().zip(force.column_iter_mut()) {
                    for (v1, v2) in corners.column_iter().circular_tuple_windows() {
                        let (_, q) = closest_point_on_segment(&p1, &v1, &v2);
                        let x = q - p1;
                        let d = (x.norm() / self.interaction_range).clamp(0.0, 1.0);

                        if 0.0 < d && d < 1.0 {
                            use core::ops::AddAssign;
                            f.add_assign(-self.force_dist * (d - 1.0) / self.interaction_range * x);
                        }
                    }
                }
                Ok(force)
            }
        }
    }
}

#[allow(redundant_semicolons)]
fn custom_update_func<I, A, Com, Sy, const N: usize>(
    sbox: &mut SubDomainBox<I, MySubDomain, Agent, A, Com, Sy>,
    neighbor_sensing_func: impl Fn(&mut A, &Agent, &V, &V, &Inf) -> Result<(), CalcError>,
) -> Result<(), SimulationError>
where
    A: UpdateMechanics<V, V, V, N>,
    Com: Communicator<SubDomainPlainIndex, PosInformation<V, V, Inf>>,
    Com: Communicator<SubDomainPlainIndex, ForceInformation<V>>,
{
    let voxel_indices: Vec<_> = sbox.voxels.keys().map(|k| *k).collect();
    for voxel_index in voxel_indices {
        let n_cells = sbox.voxels[&voxel_index].cells.len();
        // Iterate over all cells in current voxel
        for n in 0..n_cells {
            // Needs to be borrowed here due to later usage in loop
            let vox = sbox.voxels.get_mut(&voxel_index).unwrap();

            // Intermediate helper variable
            let mut cells_mut = vox.cells.iter_mut();

            // Properties of cell 1
            if let Agent::P(c1) = &mut cells_mut.nth(n).unwrap().0.cell {
                let bbox1 = c1.bounding_box;

                // Initialize the position helper with the current position of the cell
                let pos1 = c1.position.clone();
                let mut pos_helper_1 = c1.position_helper.clone();

                macro_rules! apply_restrictions_c2 {
                    ($c2:expr) => {{
                        // Properties of cell 2
                        let pos2 = $c2.position.clone();
                        let bbox2 = $c2.bounding_box;
                        let pos_helper_2 = &mut $c2.position_helper;

                        // Only compute if bounding boxes are intersecting
                        if bounding_boxes_intersect(&bbox1, &bbox2) {
                            apply_restrictions(&pos1, &pos2, &mut pos_helper_1);
                            apply_restrictions(&pos2, &pos1, pos_helper_2);
                        }
                    }};
                };

                macro_rules! apply_restrictions_fungus {
                    ($f2:expr) => {{
                        let pos_fungus = $f2.mechanics.pos.transpose();
                        let pos_helper_fungus = &mut $f2.position_helper;

                        for (n, w) in pos1.column_iter().enumerate() {
                            let mut d = f64::INFINITY;
                            let mut m = 0;
                            let mut z = Vector2::zeros();
                            let mut h = 0.0;

                            for (k, (v1, v2)) in
                                pos_fungus.column_iter().tuple_windows().enumerate()
                            {
                                let (s, q) = closest_point_on_segment(&w, &v1, &v2);
                                let x = w - q; // Pointing from fungus to plant cell
                                let dist = x.norm();
                                if dist < $f2.interaction.0.radius && dist < d {
                                    d = dist;
                                    m = k;
                                    z = x;
                                    h = s;
                                }
                            }

                            if d != f64::INFINITY && d > 1e-6 {
                                use core::ops::AddAssign;
                                // z is pointing from fungus to plant cell
                                let y = z * ($f2.interaction.0.radius - d) / d;
                                pos_helper_1.column_mut(n).add_assign(0.5 * y);
                                pos_helper_fungus
                                    .column_mut(m)
                                    .add_assign(-0.5 * (1.0 - h) * y);
                                pos_helper_fungus.column_mut(m + 1).add_assign(-0.5 * h * y);
                            }
                        }
                    }};
                };

                // Iterate over all remaining cells in the voxel
                for _ in n + 1..n_cells {
                    match &mut cells_mut.nth(0).unwrap().0.cell {
                        Agent::P(c2) => apply_restrictions_c2!(c2),
                        Agent::F(f2) => apply_restrictions_fungus!(f2),
                    }
                }

                // Get neighbor cells and gather restrictions from them
                let neighbors = vox.neighbors.clone();
                for neighbor_index in &neighbors {
                    let neighbor = sbox.voxels.get_mut(&neighbor_index).unwrap();
                    let n_cells2 = neighbor.cells.len();
                    let mut cells_mut_2 = neighbor.cells.iter_mut();
                    for _ in 0..n_cells2 {
                        match &mut cells_mut_2.nth(0).unwrap().0.cell {
                            Agent::P(c2) => apply_restrictions_c2!(c2),
                            Agent::F(f2) => apply_restrictions_fungus!(f2),
                        }
                    }
                }

                // Update the position_helper of the cell
                if let Agent::P(c1) = &mut sbox
                    .voxels
                    .get_mut(&voxel_index)
                    .unwrap()
                    .cells
                    .get_mut(n)
                    .unwrap()
                    .0
                    .cell
                {
                    c1.position_helper = pos_helper_1;
                }
            }
        }
    }

    // Update all positions
    sbox.voxels.iter_mut().for_each(|(_, vox)| {
        vox.cells.iter_mut().for_each(|c| match &mut c.0.cell {
            Agent::P(c) => c.position.copy_from(&c.position_helper),
            Agent::F(c) => c.mechanics.pos.copy_from(&c.position_helper.transpose()),
        })
    });

    sbox.update_mechanics_interaction_step_1(neighbor_sensing_func)?;
    Ok(())
}

fn convert_agents(py: Python, agents: Vec<Py<PyAny>>) -> PyResult<Vec<Agent>> {
    agents
        .into_iter()
        .map(|a| {
            if let Ok(a) = a.extract::<Agent>(py) {
                return Ok(a);
            }
            let x: Result<PlantCell, _> = a.extract(py);
            let y: Result<Fungus, _> = a.extract(py);
            match (x, y) {
                (Ok(xi), _) => Ok(Agent::P(xi)),
                (Err(_), Ok(yi)) => Ok(Agent::F(yi)),
                (Err(_), Err(_)) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Could not extract Agent from type {:?}",
                    a.type_id()
                ))),
            }
        })
        .collect::<Result<Vec<_>, _>>()
}

fn storage_builder() -> StorageBuilder {
    cellular_raza::prelude::StorageBuilder::new()
        .priority([StorageOption::Ron, StorageOption::Memory])
}

fn load_agents<T>(
    cells: &StorageManager<CellIdentifier, (CellBox<Agent>, T)>,
) -> Result<std::collections::BTreeMap<u64, Vec<Agent>>, SimulationError>
where
    T: Clone + for<'a> serde::Deserialize<'a>,
{
    Ok(cells
        .load_all_elements()?
        .into_iter()
        .map(|(iteration, cells)| {
            let cells = cells
                .into_iter()
                .sorted_by_key(|x| x.0)
                .map(|(_, (c, _))| c.cell)
                .collect::<Vec<_>>();

            (iteration, cells)
        })
        .collect::<std::collections::BTreeMap<_, _>>())
}

/// Performs a complete numerical simulation of our system.
///
/// Args:
///     simulation_settings(SimulationSettings): The settings required to run the simulation
#[gen_stub_pyfunction]
#[pyfunction]
pub fn run_simulation<'py>(
    py: Python<'py>,
    settings: &SimulationSettings,
    agents: Vec<Py<PyAny>>,
) -> Result<std::path::PathBuf, SimulationError> {
    let agents = convert_agents(py, agents)?;
    Ok(run_simulation_rs(settings, agents)?)
}

pub fn run_simulation_rs(
    sim_settings: &SimulationSettings,
    agents: Vec<Agent>,
) -> Result<std::path::PathBuf, SimulationError> {
    // Domain Setup
    let domain_size = sim_settings.domain_size;
    let domain = MyDomain {
        domain: CartesianCuboid::from_boundaries_and_n_voxels(
            [0.0; 2],
            [domain_size; 2],
            [sim_settings.n_voxels; 2],
        )?,
        force_dist: sim_settings.domain_force_dist,
        interaction_range: sim_settings.domain_interaction_range,
    };

    // Storage Setup
    let storage_builder = storage_builder();

    // Time Setup
    let t0 = 0.0;
    let dt = sim_settings.dt;
    let time_stepper = cellular_raza::prelude::time::FixedStepsize::from_partial_save_interval(
        t0,
        dt,
        sim_settings.t_max,
        sim_settings.save_interval,
    )?;

    let settings = Settings {
        n_threads: 1.try_into().unwrap(),
        time: time_stepper,
        storage: storage_builder,
        progressbar: Some("Running Simulation".into()),
    };

    let storager = run_simulation!(
        domain: domain,
        agents: agents,
        settings: settings,
        aspects: [Mechanics, Interaction, DomainForce, Cycle],
        update_mechanics_interaction_step_1: custom_update_func,
        zero_force_default: |c: &Agent| c.zero_force_default(),
    )?;

    let opath = storager.get_path()?;
    let mut file = std::fs::File::create(opath.join("settings.ron"))?;
    let config = ron::ser::PrettyConfig::default();
    let options = ron::Options::default();
    options
        .to_io_writer_pretty(&mut file, &sim_settings, config)
        .map_err(|e| SimulationError::StorageError(StorageError::from(e)))?;

    Ok(opath)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn find_results(settings: &SimulationSettings) -> PyResult<Option<std::path::PathBuf>> {
    for path in
        glob::glob("out/*").map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    {
        let path = path
            .map_err(|e| std::io::Error::from(e))?
            .join("settings.ron");
        if let Ok(file) = std::fs::File::open(&path) {
            if ron::de::from_reader(&file)
                .is_ok_and(|s: SimulationSettings| approx::abs_diff_eq!(settings, &s))
            {
                // let s = ron::de::from_reader(&file).unwrap();
                // if approx::abs_diff_eq!(settings, &s, epsilon = 1e-8) {
                let mut opath = path;
                opath.pop();
                return Ok(Some(opath));
            }
        }
    }

    Ok(None)
}

fn _storage_builder_helper(path: &std::path::Path) -> StorageBuilder<true> {
    storage_builder()
        .location(path)
        .add_date(false)
        .suffix("cells")
        .init()
}

fn _cell_storager(
    path: &std::path::Path,
) -> Result<
    cellular_raza::prelude::StorageManager<CellIdentifier, (CellBox<Agent>, ron::Value)>,
    StorageError,
> {
    let storage_builder = _storage_builder_helper(&path);
    cellular_raza::prelude::StorageManager::open_or_create(storage_builder, 0)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn get_all_iterations(path: std::path::PathBuf) -> Result<Vec<u64>, SimulationError> {
    let storager = _cell_storager(&path)?;
    Ok(storager.get_all_iterations()?)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn load_all_results(
    path: std::path::PathBuf,
) -> Result<std::collections::BTreeMap<u64, Vec<Agent>>, SimulationError> {
    let cells = _cell_storager(&path)?;
    load_agents(&cells)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn load_results(
    iteration: u64,
    path: std::path::PathBuf,
) -> Result<Vec<Agent>, SimulationError> {
    let cells = _cell_storager(&path)?;
    Ok(cells
        .load_all_elements_at_iteration(iteration)?
        .into_iter()
        .sorted_by_key(|x| x.0)
        .map(|(_, x)| x.0.cell)
        .collect())
}

#[pymodule]
fn cr_rust_fungus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PlantCell>()?;
    m.add_class::<Fungus>()?;
    m.add_class::<Agent>()?;
    m.add_class::<SimulationSettings>()?;
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(find_results, m)?)?;
    m.add_function(wrap_pyfunction!(load_results, m)?)?;
    m.add_function(wrap_pyfunction!(load_all_results, m)?)?;
    m.add_function(wrap_pyfunction!(get_all_iterations, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
