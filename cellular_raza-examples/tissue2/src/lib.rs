// #![deny(missing_docs)]
//! This package is an example of how to construct python bindings with documentation with
//! `cellular_raza <https://cellular-raza.com/>`_.

use cellular_raza::prelude::*;
use numpy::PyArrayMethods;
use pyo3::{prelude::*, types::PyTuple, IntoPyObjectExt};
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction, derive::*};

use serde::{Deserialize, Serialize};

use nalgebra::{Matrix2xX, Vector2};

use geometry::*;

mod geometry;

/// Contains settings needed to specify the simulation
#[gen_stub_pyclass]
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct SimulationSettings {
    /// Overall domain size
    pub domain_size: f64,
    /// Number of voxels to create subdivisions
    pub n_voxels: usize,
    /// Time increment used to solve the simulation
    pub dt: f64,
    /// Maximum duration of the simulation
    pub t_max: f64,
    /// Frequency to store results
    pub save_interval: f64,
    /// Random initial seed
    pub rng_seed: u64,
}

#[gen_stub_pymethods]
#[pymethods]
impl SimulationSettings {
    /// Creates a new :class:`SimulationSettings` class.
    #[new]
    fn new() -> Self {
        Self {
            domain_size: 30.0,
            n_voxels: 3,
            dt: 0.05,
            t_max: 10.0,
            save_interval: 1.0,
            rng_seed: 0,
        }
    }
}

type Pos = Matrix2xX<f64>;
type Inf = ();

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Clone, Deserialize, Serialize)]
#[rustfmt::skip]
pub struct Agent {
    // Mechanical variables
    pub position: Matrix2xX<f64>,
    // This will be used to iteratively apply restrictions when calculating overlaps with other
    // polygons. We keep the original position fixed differences due to ordering of the applied
    // restrictions.
    position_helper: Matrix2xX<f64>,
    bounding_box: (Vector2<f64>, Vector2<f64>),
    // Interaction Parameters
    #[pyo3(get, set)] pub force_area: f64,
    #[pyo3(get, set)] pub force_perimeter: f64,
    #[pyo3(get, set)] pub force_dist: f64,
    #[pyo3(get, set)] pub force_angle: f64,
    #[pyo3(get, set)] pub min_dist: f64,
    #[pyo3(get, set)] pub target_area: f64,
    #[pyo3(get, set)] pub target_perimeter: f64,
    // Mechanical Parameters
    #[pyo3(get, set)] pub damping: f64,
    #[pyo3(get, set)] pub diffusion_constant: f64,
}

impl Agent {
    fn get_middle(&self) -> Vector2<f64> {
        area_centroid(&self.position)
    }
}

fn py_array_to_matrix(py_array: Bound<numpy::PyArray2<f64>>) -> Matrix2xX<f64> {
    let array = py_array.to_owned_array();
    nalgebra::Matrix2xX::from_fn(array.ncols(), |i, j| array[(i, j)])
}

fn matrix_to_py_array<'py>(
    py: Python<'py>,
    matrix: &Matrix2xX<f64>,
) -> Bound<'py, numpy::PyArray2<f64>> {
    let array =
        numpy::ndarray::Array2::<f64>::from_shape_fn(matrix.shape(), |(i, j)| matrix[(i, j)]);
    numpy::PyArray2::from_owned_array(py, array)
}

#[gen_stub_pymethods]
#[pymethods]
impl Agent {
    #[new]
    fn new(
        position: Bound<numpy::PyArray2<f64>>,
        force_area: f64,
        force_perimeter: f64,
        target_area: f64,
        force_angle: f64,
        force_dist: f64,
        min_dist: f64,
        target_perimeter: f64,
        damping: f64,
        diffusion_constant: f64,
    ) -> Self {
        let position = py_array_to_matrix(position);
        let position_helper = position.clone();
        let bounding_box = calculate_bbox(&position);

        Self {
            position,
            position_helper,
            bounding_box,
            force_area,
            force_perimeter,
            force_angle,
            force_dist,
            min_dist,
            target_area,
            target_perimeter,
            damping,
            diffusion_constant,
        }
    }

    fn get_perimeter(&self) -> f64 {
        get_polygon_perimeter(&self.position)
    }

    fn get_area(&self) -> f64 {
        get_polygon_area(&self.position)
    }

    #[getter]
    fn get_position<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        matrix_to_py_array(py, &self.position)
    }

    #[setter]
    fn set_position(&mut self, position: Bound<numpy::PyArray2<f64>>) {
        self.position = py_array_to_matrix(position);
    }
}

impl Position<Pos> for Agent {
    fn pos(&self) -> Pos {
        self.position.clone()
    }

    fn set_pos(&mut self, pos: &Pos) {
        // Clean up self intersections
        let pos = clean_self_intersections(pos.clone());
        self.position = pos.clone();
        self.position_helper = pos.clone();
        self.bounding_box = calculate_bbox(&self.position);
    }
}

impl Velocity<Pos> for Agent {
    fn velocity(&self) -> Pos {
        Matrix2xX::zeros(self.position.ncols())
    }

    fn set_velocity(&mut self, vel: &Pos) {}
}

impl Mechanics<Pos, Pos, Pos> for Agent {
    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: f64,
    ) -> Result<(Pos, Pos), RngError> {
        let mut dpos = Matrix2xX::zeros(self.position.ncols());
        for mut col in dpos.column_iter_mut() {
            col += (2.0 * self.diffusion_constant).sqrt() * wiener_process(rng, dt)?;
        }
        let dvel = Matrix2xX::zeros(self.position.ncols());
        Ok((dpos, dvel))
    }

    fn calculate_increment(&self, force: Pos) -> Result<(Pos, Pos), CalcError> {
        let ncols = self.position.ncols();
        let dv = Matrix2xX::zeros(ncols);

        let mut f = Matrix2xX::zeros(ncols);

        let area_diff = self.target_area - self.get_area();
        for i1 in 0..ncols {
            use core::ops::AddAssign;
            let i2 = (i1 + 1) % ncols;
            let i3 = (i1 + 2) % ncols;
            let p1 = self.position.column(i1);
            let p2 = self.position.column(i2);
            let p3 = self.position.column(i3);

            // Auxiliary variables
            let v1 = p2 - p1;
            let v2 = p3 - p2;

            // Calculate area force
            // Outward pointing direction
            let dir_p = v1.normalize();
            let dir_a = Vector2::from([dir_p[1], -dir_p[0]]);
            let force_a = self.force_area * area_diff;
            f.column_mut(i1).add_assign(0.5 * force_a * dir_a);
            f.column_mut(i2).add_assign(0.5 * force_a * dir_a);

            // Calculate perimeter force
            let force_p = self.force_perimeter * (self.target_perimeter / ncols as f64 - v1.norm());
            f.column_mut(i1).add_assign(-0.5 * force_p * dir_p);
            f.column_mut(i2).add_assign(0.5 * force_p * dir_p);

            // Calculate angle
            let angle = v1.angle(&v2);
            let force_dir = v1 - v2;
            let force_dir = if !approx::abs_diff_eq!(force_dir.norm(), 0.0, epsilon = EPSILON) {
                force_dir.normalize()
            } else {
                force_dir
            };
            let strength = 2.0 * self.force_angle * (ncols as f64) * (0.5 * angle).tan().min(1.0);
            let force = -force_dir * strength;
            f.column_mut(i1).add_assign(-0.5 * force);
            f.column_mut(i2).add_assign(force);
            f.column_mut(i3).add_assign(-0.5 * force);
        }

        Ok((force + f, dv))
    }
}

impl InteractionInformation<Inf> for Agent {
    fn get_interaction_information(&self) -> Inf {}
}

impl Interaction<Pos, Pos, Pos, Inf> for Agent {
    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        _: &Pos,
        ext_pos: &Pos,
        _: &Pos,
        ext_info: &Inf,
    ) -> Result<(Pos, Pos), CalcError> {
        let c1 = area_centroid(&own_pos);
        let c2 = area_centroid(&ext_pos);
        let dir = (c1 - c2).normalize();
        let d = (c1 - c2).norm() / (self.target_area / core::f64::consts::PI).sqrt();
        let force = self.force_dist * (1.0 - d) / (1.0 + d).max(0.0);
        let f1 = Matrix2xX::from_fn(own_pos.ncols(), |i, _| dir[i] * force);
        let f2 = Matrix2xX::from_fn(ext_pos.ncols(), |i, _| dir[i] * force);
        Ok((f1, f2))
    }
}

#[derive(Clone, Domain)]
pub struct MyDomain {
    #[DomainPartialDerive]
    #[DomainRngSeed]
    domain: CartesianCuboid<f64, 2>,
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
            .map(move |(n, subdomain, voxels)| (n, MySubDomain { subdomain }, voxels)))
    }
}

#[derive(Clone, SubDomain, Deserialize, Serialize)]
pub struct MySubDomain {
    #[Base]
    subdomain: CartesianSubDomain<f64, 2>,
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

impl SubDomainMechanics<Pos, Pos> for MySubDomain {
    fn apply_boundary(&self, pos: &mut Pos, vel: &mut Pos) -> Result<(), BoundaryError> {
        for (p, v) in pos.column_iter_mut().zip(vel.column_iter_mut()) {
            self.subdomain
                .apply_boundary(&mut [p[0], p[1]], &mut [v[0], v[1]])?;
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

fn custom_update_func<I, A, Com, Sy, const N: usize>(
    sbox: &mut SubDomainBox<I, MySubDomain, Agent, A, Com, Sy>,
    neighbor_sensing_func: impl Fn(&mut A, &Agent, &Pos, &Pos, &Inf) -> Result<(), CalcError>,
) -> Result<(), SimulationError>
where
    A: UpdateMechanics<Pos, Pos, Pos, N>,
    Com: Communicator<SubDomainPlainIndex, PosInformation<Pos, Pos, Inf>>,
    Com: Communicator<SubDomainPlainIndex, ForceInformation<Pos>>,
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
            let (c1, _) = cells_mut.nth(n).unwrap();
            let bbox1 = c1.cell.bounding_box;

            // Initialize the position helper with the current position of the cell
            let pos1 = c1.position.clone();
            let mut pos_helper_1 = c1.position_helper.clone();

            // Iterate over all remaining cells in the voxel
            for _ in n + 1..n_cells {
                // Properties of cell 2
                let (c2, _) = cells_mut.nth(0).unwrap();
                let pos2 = c2.position.clone();
                let bbox2 = c2.cell.bounding_box;

                // Only compute if bounding boxes are intersecting
                if bounding_boxes_intersect(&bbox1, &bbox2) {
                    apply_restrictions(&pos1, &mut pos_helper_1, &c2.position);
                    apply_restrictions(&pos2, &mut c2.position_helper, &pos1);
                }
            }

            // Get neighbor cells and gather restrictions from them
            let neighbors = vox.neighbors.clone();
            for neighbor_index in &neighbors {
                let neighbor = sbox.voxels.get_mut(&neighbor_index).unwrap();
                let n_cells2 = neighbor.cells.len();
                let mut cells_mut_2 = neighbor.cells.iter_mut();
                for _ in 0..n_cells2 {
                    let (c2, _) = cells_mut_2.nth(0).unwrap();
                    let pos2 = c2.position.clone();
                    apply_restrictions(&pos1, &mut pos_helper_1, &c2.position);
                    apply_restrictions(&pos2, &mut c2.position_helper, &pos1);
                }
            }

            // Update the position_helper of the cell
            let c1 = &mut sbox
                .voxels
                .get_mut(&voxel_index)
                .unwrap()
                .cells
                .get_mut(n)
                .unwrap()
                .0;
            c1.position_helper = pos_helper_1;
        }
    }

    // Update all positions
    sbox.voxels.iter_mut().for_each(|(_, vox)| {
        vox.cells
            .iter_mut()
            .for_each(|(c, _)| c.cell.position.copy_from(&c.cell.position_helper))
    });

    sbox.update_mechanics_interaction_step_1(neighbor_sensing_func)?;
    Ok(())
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
    agents: Vec<Agent>,
) -> Result<
    std::collections::BTreeMap<u64, Vec<(Bound<'py, numpy::PyArray2<f64>>, [f64; 4])>>,
    SimulationError,
> {
    // Domain Setup
    let domain_size = settings.domain_size;
    let domain = MyDomain {
        domain: CartesianCuboid::from_boundaries_and_n_voxels(
            [0.0; 2],
            [domain_size; 2],
            [settings.n_voxels; 2],
        )?,
    };

    // Storage Setup
    let storage_builder = cellular_raza::prelude::StorageBuilder::new()
        // .location("out")
        .priority([StorageOption::Memory]);

    // Time Setup
    let t0 = 0.0;
    let dt = settings.dt;
    let time_stepper = cellular_raza::prelude::time::FixedStepsize::from_partial_save_interval(
        t0,
        dt,
        settings.t_max,
        settings.save_interval,
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
        aspects: [Mechanics, Interaction],
        update_mechanics_interaction_step_1: custom_update_func,
        zero_force_default: |c: &Agent| nalgebra::Matrix2xX::zeros(c.position.ncols()),
    )?;

    let points = storager
        .cells
        .load_all_elements()?
        .into_iter()
        .map(|(iteration, cells)| {
            let cells = cells
                .into_iter()
                .map(|(_, (c, _))| {
                    let cell = c.cell;
                    let array = numpy::ndarray::Array2::<f64>::from_shape_fn(
                        cell.position.shape(),
                        |(i, j)| cell.position[(i, j)],
                    );
                    let array = numpy::PyArray2::from_owned_array(py, array);
                    (
                        array,
                        [
                            cell.get_area(),
                            cell.target_area,
                            cell.get_perimeter(),
                            cell.target_perimeter,
                        ],
                    )
                })
                .collect::<Vec<_>>();

            (iteration, cells)
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    Ok(points)
}

#[pymodule]
fn cr_tissue_2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Agent>()?;
    m.add_class::<SimulationSettings>()?;
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
