// #![deny(missing_docs)]
//! This package is an example of how to construct python bindings with documentation with
//! `cellular_raza <https://cellular-raza.com/>`_.

use cellular_raza::prelude::*;
use numpy::PyArrayMethods;
use pyo3::{prelude::*, types::PyTuple, IntoPyObjectExt};
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction, derive::*};

use serde::{Deserialize, Serialize};

use nalgebra::{Vector2, VectorView2};

// mod vertex_based;

// use vertex_based::*;

/// Contains settings needed to specify the simulation
#[gen_stub_pyclass]
#[pyclass(get_all, set_all)]
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
    /// Steps to take to approximate the size of the cell
    pub approximation_steps: usize,
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
            approximation_steps: 20,
        }
    }
}

type Pos = Vector2<f64>;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Inf {
    target_area: f64,
    current_area: f64,
    path: Vec<PathSegment>,
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone, Deserialize, Serialize)]
#[rustfmt::skip]
pub struct Agent {
    // Mechanical variables
    pub position: Vector2<f64>,
    pub velocity: Vector2<f64>,
    pub path: Vec<PathSegment>,
    #[pyo3(get, set)] pub current_area: f64,
    // Interaction Parameters
    #[pyo3(get, set)] pub force_area: f64,
    #[pyo3(get, set)] pub force_perimeter: f64,
    #[pyo3(get, set)] pub force_dist: f64,
    #[pyo3(get, set)] pub min_dist: f64,
    #[pyo3(get, set)] pub target_area: f64,
    #[pyo3(get, set)] pub target_perimeter: f64,
    // Mechanical Parameters
    #[pyo3(get, set)] pub damping: f64,
    #[pyo3(get, set)] pub diffusion_constant: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl Agent {
    #[new]
    fn new(
        position: Bound<numpy::PyArray1<f64>>,
        force_area: f64,
        force_perimeter: f64,
        target_area: f64,
        force_dist: f64,
        min_dist: f64,
        target_perimeter: f64,
        damping: f64,
        diffusion_constant: f64,
    ) -> Self {
        let position = position.to_owned_array();
        Self {
            position: [position[0], position[1]].into(),
            velocity: Vector2::zeros(),
            path: vec![],
            current_area: 0.0,
            force_area,
            force_perimeter,
            force_dist,
            min_dist,
            target_area,
            target_perimeter,
            damping,
            diffusion_constant,
        }
    }

    #[getter]
    fn get_position(&self) -> [f64; 2] {
        [self.position[0], self.position[1]]
    }

    #[setter]
    fn set_position(&mut self, position: [f64; 2]) {
        self.position[0] = position[0];
        self.position[1] = position[1];
    }
}

impl Position<Pos> for Agent {
    fn pos(&self) -> Pos {
        self.position
    }

    fn set_pos(&mut self, pos: &Pos) {
        self.position = *pos;
    }
}

impl Velocity<Pos> for Agent {
    fn velocity(&self) -> Pos {
        self.velocity
    }

    fn set_velocity(&mut self, vel: &Pos) {
        self.velocity = *vel;
    }
}

impl Mechanics<Pos, Pos, Pos> for Agent {
    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: f64,
    ) -> Result<(Pos, Pos), RngError> {
        let dpos = (2.0 * self.diffusion_constant).sqrt() * wiener_process(rng, dt)?;
        let dvel = Vector2::zeros();
        Ok((dpos, dvel))
    }

    fn calculate_increment(&self, force: Pos) -> Result<(Pos, Pos), CalcError> {
        let perimeter = self
            .path
            .iter()
            .map(|segm| match segm {
                PathSegment::Line { p1, p2 } => (p1 - p2).norm(),
                PathSegment::Arc {
                    angle1,
                    angle2,
                    radius,
                    ..
                } => (angle1 - angle2).abs() * radius,
            })
            .sum::<f64>();

        let mut force_perimeter = Pos::zeros();
        let mut force_area = Pos::zeros();
        for segm in self.path.iter() {
            let p1 = segm.pos1();

            let dir = self.position - p1;
            let perim_diff = perimeter - self.target_perimeter;
            force_perimeter -= self.force_perimeter * perim_diff * dir;

            let area = self.current_area;
            let area_diff = area - self.target_area;
            force_area -= self.force_area * area_diff * dir;
        }

        let dx = self.velocity;
        let dv = force_perimeter + force_area + force - self.damping * self.velocity;

        Ok((dx, dv))
    }
}

impl InteractionInformation<Inf> for Agent {
    fn get_interaction_information(&self) -> Inf {
        Inf {
            target_area: self.target_area,
            current_area: self.current_area,
            path: self.path.clone(),
        }
    }
}

fn get_intersecting_pathsegment(
    path1: &[PathSegment],
    path2: &[PathSegment],
) -> Option<PathSegment> {
    for s1 in path1.iter() {
        for s2 in path2.iter() {
            match (s1, s2) {
                (PathSegment::Line { p1, p2 }, PathSegment::Line { p1: q1, p2: q2 }) => {
                    let da = p2 - p1;
                    let db = q2 - q1;
                    let dab = p1 - q1;

                    // Cross product to check that they are parallel
                    let parallel = approx::abs_diff_eq!(da.perp(&db), 0.0, epsilon = 1e-3);
                    // Cross product to check that they are on the same line
                    let collinear = approx::abs_diff_eq!(dab.perp(&da), 0.0, epsilon = 1e-3);

                    if parallel && collinear {
                        let h0 = 0f64;
                        let h1 = 1f64;
                        let mut h2 = (q1 - p1).dot(&da) / da.norm_squared();
                        let mut h3 = (q2 - p1).dot(&da) / da.norm_squared();

                        if h2 > h3 {
                            std::mem::swap(&mut h2, &mut h3);
                        }

                        let u0 = h0.max(h3);
                        let u1 = h1.min(h3);
                        return Some(PathSegment::Line {
                            p1: p1 + u0 * da,
                            p2: p1 + u1 * da,
                        });
                    }
                }
                _ => (),
            }
        }
    }
    None
}

impl Interaction<Pos, Pos, Pos, Inf> for Agent {
    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        _: &Pos,
        ext_pos: &Pos,
        _: &Pos,
        _ext_info: &Inf,
    ) -> Result<(Pos, Pos), CalcError> {
        let dir = ext_pos - own_pos;
        let dist = dir.norm();
        if dist < self.min_dist {
            // Repulsive force due to close
            let f = -self.force_dist * dir.normalize();
            return Ok((f, -f));
        }

        /* if let Some(PathSegment::Line { p1, p2 }) =
            get_intersecting_pathsegment(&self.path, &ext_info.path)
        {
            /*
            let mut dir = own_pos - ext_pos;
            let d = dir.norm();
            dir /= d;

            let r1 = (self.target_area / core::f64::consts::PI).sqrt();
            let r2 = (ext_info.target_area / core::f64::consts::PI).sqrt();
            let range_scaling = (d / r1).max(1.0 / 3.0).powi(2);

            // Two area conditions
            let ac1 = self.target_area - self.current_area;
            let ac2 = ext_info.target_area - ext_info.current_area;

            // CASE 1: Distance is too small => Repulsive
            let f = if d < 0.8 * (r1 + r2) {
                self.force_strength / range_scaling * dir
            }
            // CASE 2: Any of the two areas is too small => Repulsive
            else if ac1 > 0.0 || ac2 > 0.0 {
                let a = 2.0 * (ac1.max(ac2)) / (self.target_area + ext_info.target_area);
                a * self.force_strength / range_scaling * dir
            }
            // CASE 3: Same Species => Attractive
            else if self.species == ext_info.species {
                -self.force_strength_species / range_scaling * dir
            }
            // CASE 4: Everything else => Attractive weak
            else {
                -self.force_strength_weak / range_scaling * dir
            };*/

            let area = self.current_area;
            let area_diff = area - self.target_area;
            let f = -self.force_area
                * area_diff
                * (own_pos - ext_pos).normalize()
                * 0.5
                * (p1 - own_pos).perp(&(p2 - own_pos)).abs()
                / area;

            // let f = f1 + f2;

            Ok((f, -f))
        } else {
            Ok((Pos::zeros(), Pos::zeros()))
        }*/
        Ok((Pos::zeros(), Pos::zeros()))
    }
}

/// ```
/// use tissue_pyo3::*;
/// use nalgebra::Vector2;
/// let o1 = Vector2::from([0.0, 0.0]);
/// let dir = Vector2::from([1.0, 0.0]);
/// let seg_start = Vector2::from([1.0, 1.0]);
/// let seg_end = Vector2::from([1.0, -1.0]);
///
/// let i1 = intersect_line_segment(o1, dir, seg_start, seg_end);
/// let i2 = intersect_line_segment(o1, dir, seg_end, seg_start);
///
/// assert!(i1 == i2);
/// assert!((i1.unwrap()[0] - 1.0).abs() < 1e-3);
/// assert!((i1.unwrap()[1] - 0.0).abs() < 1e-3);
/// ```
pub fn intersect_line_segment(
    line_origin: Pos,
    line_dir: Pos,
    seg_start: Pos,
    seg_end: Pos,
) -> Option<Pos> {
    // Vector components for the line direction
    let dx = line_dir[0];
    let dy = line_dir[1];

    // Vector components for the segment direction
    let sx = seg_end[0] - seg_start[0];
    let sy = seg_end[1] - seg_start[1];

    // The denominator is the 2D cross product of the two direction vectors.
    // If it's 0, the line and segment are parallel.
    let denom = dx * sy - dy * sx;

    if denom.abs() < 1e-10 {
        return None;
    }

    // Vector from line origin to segment start
    let ox = seg_start[0] - line_origin[0];
    let oy = seg_start[1] - line_origin[1];

    // Solve for u (the parameter for the segment)
    // u = (ox * dy - oy * dx) / (dx * sy - dy * sx)
    let u = (ox * dy - oy * dx) / denom;

    // Check if the intersection point lies within the segment bounds [0, 1]
    if (0.0..=1.0).contains(&u) {
        // We can solve for t to find the point, or just use u on the segment
        // Both yield the same point; using u is often more numerically stable
        // if line_dir is very small.
        Some(Pos::from([seg_start[0] + u * sx, seg_start[1] + u * sy]))
    } else {
        None
    }
}

#[derive(Clone, Domain)]
pub struct MyDomain {
    #[DomainPartialDerive]
    #[DomainRngSeed]
    domain: CartesianCuboid<f64, 2>,
    approximation_steps: usize,
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
        let approximation_steps = self.approximation_steps;
        Ok(self
            .domain
            .create_subdomains(n_subdomains)?
            .into_iter()
            .map(move |(n, subdomain, voxels)| {
                (
                    n,
                    MySubDomain {
                        subdomain,
                        approximation_steps,
                    },
                    voxels,
                )
            }))
    }
}

#[derive(Clone, SubDomain, Deserialize, Serialize)]
pub struct MySubDomain {
    #[Base]
    #[Mechanics]
    subdomain: CartesianSubDomain<f64, 2>,
    approximation_steps: usize,
}

impl SortCells<Agent> for MyDomain {
    type VoxelIndex = [usize; 2];

    fn get_voxel_index_of(
        &self,
        cell: &Agent,
    ) -> Result<Self::VoxelIndex, cellular_raza::prelude::BoundaryError> {
        let pos = cell.pos();
        self.domain.get_voxel_index_of_raw(&pos)
    }
}

impl SortCells<Agent> for MySubDomain {
    type VoxelIndex = [usize; 2];

    fn get_voxel_index_of(&self, cell: &Agent) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos();
        self.subdomain.get_index_of(pos)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PathSegment {
    Line {
        p1: Vector2<f64>,
        p2: Vector2<f64>,
    },
    Arc {
        center: Vector2<f64>,
        angle1: f64,
        angle2: f64,
        radius: f64,
    },
}

impl PathSegment {
    fn to_pytuple<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let elements = match self {
            PathSegment::Line { p1, p2 } => vec![
                "line".into_py_any(py)?,
                [p1[0], p1[1]].into_py_any(py)?,
                [p2[0], p2[1]].into_py_any(py)?,
            ],
            PathSegment::Arc {
                center,
                angle1,
                angle2,
                radius,
            } => vec![
                "arc".into_py_any(py)?,
                [center[0], center[1]].into_py_any(py)?,
                angle1.into_py_any(py)?,
                angle2.into_py_any(py)?,
                radius.into_py_any(py)?,
            ],
        };
        PyTuple::new(py, elements)
    }

    fn pos1(&self) -> Vector2<f64> {
        match self {
            PathSegment::Line { p1, p2: _ } => *p1,
            PathSegment::Arc {
                center,
                angle1,
                radius,
                ..
            } => center + Vector2::from([angle1.cos(), angle1.sin()]) * *radius,
        }
    }

    fn pos2(&self) -> Vector2<f64> {
        match self {
            PathSegment::Line { p1: _, p2 } => *p2,
            PathSegment::Arc {
                center,
                angle2,
                radius,
                ..
            } => center + Vector2::from([angle2.cos(), angle2.sin()]) * *radius,
        }
    }
}

fn get_polygon_area(middle: &Vector2<f64>, vertices: &nalgebra::Matrix2xX<f64>) -> f64 {
    let mut area = 0.0;
    let n_col = vertices.ncols();
    for i in 0..n_col {
        let v1 = vertices.column(i);
        let v2 = vertices.column((i + 1) % n_col);

        area += (v1 - middle).perp(&(v2 - middle)).abs();
    }
    area
}

fn get_polygon_perimeter(vertices: &nalgebra::Matrix2xX<f64>) -> f64 {
    let mut perimeter = 0.0;
    let n = vertices.ncols();
    for i in 0..n {
        perimeter += (vertices.column(i) - vertices.column((i + 1) % n)).norm();
    }
    perimeter
}

fn minimum_dist_to_line(point: &Vector2<f64>, l1: VectorView2<f64>, l2: VectorView2<f64>) -> f64 {
    let x1 = l1 - point;
    let x2 = l2 - point;
    let edge_len = (x1 - x2).norm();
    (x1[0] * x2[1] - x1[1] * x2[0]).abs() / edge_len
}

fn triangle_circle_intersection_area(radius: f64, v1: Vector2<f64>, v2: Vector2<f64>) -> f64 {
    let d1 = v1.norm();
    let d2 = v2.norm();

    let det = v1[0] * v2[1] - v1[1] * v2[0];
    let dv = v2 - v1;
    let dv_magn_sq = dv.norm_squared();
    if dv_magn_sq == 0.0 {
        return 0.0;
    }

    let t = (v1.dot(&dv) / dv_magn_sq).clamp(0.0, 1.0);
    let closest = v1 + t * dv;
    let dist_to_edge = closest.norm();

    if dist_to_edge >= radius {
        let angle = (v1.dot(&v2) / (d1 * d2)).clamp(0.0, 1.0).acos();
        return 0.5 * radius.powi(2) * angle;
    }

    if d1 <= radius && d2 <= radius {
        return 0.5 * det.abs();
    }

    let angle = (v1.dot(&v2) / (d1 * d2)).clamp(0.0, 1.0).acos();
    (0.5 * det.abs()).min(0.5 * radius.powi(2) * angle)
}

fn get_total_intersection_area(
    middle: &Vector2<f64>,
    vertices: &nalgebra::Matrix2xX<f64>,
    radius: f64,
) -> f64 {
    let mut area = 0.0;
    let n_cols = vertices.ncols();
    for i in 0..n_cols {
        let v1 = vertices.column(i) - middle;
        let v2 = vertices.column((i + 1) % n_cols) - middle;
        area += triangle_circle_intersection_area(radius, v1, v2);
    }
    area
}

/// Finds intersection points of a line segment p1-p2 and circle radius r.
fn get_circle_line_intersections(
    v1: Vector2<f64>,
    v2: Vector2<f64>,
    radius: f64,
) -> Vec<Vector2<f64>> {
    let d = v2 - v1;
    let dr2 = d.norm_squared();
    let v1_dot_d = v1.dot(&d);
    let v1_sq = v1.norm_squared();

    let a = dr2;
    let b = 2.0 * v1_dot_d;
    let c = v1_sq - radius.powi(2);

    let delta = b.powi(2) - 4.0 * a * c;
    if delta < 0.0 {
        return vec![];
    }

    let sqrt_delta = delta.sqrt();
    let t0 = (-b - sqrt_delta) / (2.0 * a);
    let t1 = (-b + sqrt_delta) / (2.0 * a);

    match (1.0 >= t0 && t0 >= 0.0, 1.0 >= t1 && t1 >= 0.0) {
        (true, true) => vec![v1 + t0 * d, v1 + t1 * d],
        (true, false) => vec![v1 + t0 * d],
        (false, true) => vec![v1 + t1 * d],
        (false, false) => vec![],
    }
}

fn construct_path(
    middle: &Vector2<f64>,
    vertices: &nalgebra::Matrix2xX<f64>,
    radius: f64,
) -> Vec<PathSegment> {
    let mut path = vec![];

    // Create line segments
    let n_cols = vertices.ncols();
    for i in 0..n_cols {
        let p1 = vertices.column(i);
        let p2 = vertices.column((i + 1) % n_cols);
        let v1 = p1 - middle;
        let v2 = p2 - middle;

        let intersections = get_circle_line_intersections(v1, v2, radius);

        // Case 1: Edge is inside
        let d1_in = v1.norm() <= radius;
        let d2_in = v2.norm() <= radius;

        match (d1_in, d2_in) {
            (true, true) => path.push((p1.into(), p2.into())),
            (false, true) => {
                if !intersections.is_empty() {
                    let entry_pt = intersections[0] + middle;
                    path.push((entry_pt, p2.into()));
                }
            }
            (true, false) => {
                if !intersections.is_empty() {
                    let exit_pt = intersections[0] + middle;
                    path.push((p1.into(), exit_pt));
                }
            }
            (false, false) => {
                if intersections.len() == 2 {
                    path.push((intersections[0] + middle, intersections[1] + middle))
                }
            }
        }
    }

    // Stich together with arcs
    let mut full_path = vec![];
    for i in 0..path.len() {
        full_path.push(PathSegment::Line {
            p1: path[i].0,
            p2: path[i].1,
        });
        let curr_end = path[i].1 - middle;
        let next_start = path[(i + 1) % path.len()].0 - middle;

        if (curr_end - next_start).norm() > 1e-9 {
            let start_angle = next_start[1].atan2(next_start[0]);
            let mut end_angle = curr_end[1].atan2(curr_end[0]);

            if end_angle <= start_angle {
                end_angle += 2.0 * core::f64::consts::PI;
            }
            full_path.push(PathSegment::Arc {
                center: *middle,
                angle1: start_angle,
                angle2: end_angle,
                radius,
            });
        }
    }
    full_path
}

fn calculate_area(middle: &Vector2<f64>, path: &[PathSegment]) -> f64 {
    let mut area = 0.0;
    for segm in path.iter() {
        match segm {
            PathSegment::Line { p1, p2 } => area += (p1 - middle).perp(&(p2 - middle)).abs(),
            PathSegment::Arc {
                angle1,
                angle2,
                radius,
                ..
            } => area += 0.5 * (angle1 - angle2).abs() * radius.powi(2),
        }
    }
    area
}

fn construct_constrained_path(
    middle: &Vector2<f64>,
    vertices: &nalgebra::Matrix2xX<f64>,
    target_area: f64,
    target_perimeter: f64,
    approximation_steps: usize,
) -> Vec<PathSegment> {
    let poly_area = get_polygon_area(middle, vertices);
    let poly_perim = get_polygon_perimeter(vertices);
    let n_cols = vertices.ncols();

    // CASE 1: FULL POLYGON
    // If the size of the bounding polygon is smaller than the target area, we return the polygon
    // itself
    if poly_area <= target_area || poly_perim <= target_perimeter {
        return (0..n_cols)
            .map(|n| PathSegment::Line {
                p1: vertices.column(n).into(),
                p2: vertices.column((n + 1) % n_cols).into(),
            })
            .collect();
    }

    // CASE 2: FULL CIRCLE
    // If a circle with correct radius fits inside the bounding polygon, then we return the circle
    let target_radius = (target_area / core::f64::consts::PI).sqrt();
    if (0..n_cols).all(|n| {
        minimum_dist_to_line(
            middle,
            vertices.column(n),
            vertices.column((n + 1) % n_cols),
        ) >= target_radius
    }) {
        return vec![PathSegment::Arc {
            center: *middle,
            angle1: 0.0,
            angle2: 2.0 * core::f64::consts::PI,
            radius: target_radius,
        }];
    }

    // CASE 3: SOMETHING IN BETWEEN
    let mut r_low = 0.0;
    let mut r_high = 0.0;
    for col in vertices.column_iter() {
        let d = (col - middle).norm();
        if d > r_high {
            r_high = d;
        }
    }

    // Binary search for the optimal radius
    for _ in 0..approximation_steps {
        let r_mid = 0.5 * (r_low + r_high);
        let new_area = get_total_intersection_area(middle, vertices, r_mid);
        if new_area < target_area {
            r_low = r_mid;
        } else {
            r_high = r_mid;
        }
    }

    construct_path(middle, vertices, 0.5 * (r_high + r_low))
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
    let info_all = sbox
        .voxels
        .iter_mut()
        .flat_map(|(_, voxel)| {
            voxel.cells.iter_mut().map(|(c, _)| {
                (
                    &mut c.cell.position,
                    &mut c.cell.path,
                    &mut c.cell.target_area,
                    &mut c.cell.target_perimeter,
                    &mut c.cell.current_area,
                )
            })
        })
        .collect::<Vec<_>>();

    use voronoice::*;
    let sites = info_all
        .iter()
        .map(|x| Point {
            x: x.0[0],
            y: x.0[1],
        })
        .collect::<Vec<_>>();

    let min = sbox.subdomain.subdomain.get_min();
    let max = sbox.subdomain.subdomain.get_max();
    let center = 0.5 * (min + max);
    let dx = max - min;
    let bounding_box = BoundingBox::new(
        Point {
            x: center[0],
            y: center[1],
        },
        dx[0],
        dx[1],
    );
    let voronoi = VoronoiBuilder::default()
        .set_sites(sites)
        .set_bounding_box(bounding_box)
        .set_lloyd_relaxation_iterations(0)
        .build();

    if let Some(voronoi) = voronoi {
        for ((site, path, target_area, target_perimeter, current_area), vcell) in
            info_all.into_iter().zip(voronoi.iter_cells())
        {
            let mut verts = nalgebra::Matrix2xX::zeros(1);
            for (n, v) in vcell.iter_vertices().enumerate() {
                if n >= verts.ncols() {
                    verts = verts.insert_column(n, 0.0);
                }
                verts[(0, n)] = v.x;
                verts[(1, n)] = v.y;
            }

            let new_path = construct_constrained_path(
                &site,
                &verts,
                *target_area,
                *target_perimeter,
                sbox.subdomain.approximation_steps,
            );
            *path = new_path;
            *current_area = calculate_area(&site, path);
        }
    }

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
    python: Python<'py>,
    settings: &SimulationSettings,
    agents: Vec<Agent>,
) -> Result<
    std::collections::BTreeMap<u64, Vec<([f64; 2], Vec<Bound<'py, PyTuple>>, f64)>>,
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
        approximation_steps: settings.approximation_steps,
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
    )?;

    let mut error = None;
    let points = storager
        .cells
        .load_all_elements()?
        .into_iter()
        .map(|(iteration, cells)| {
            let cells = cells
                .into_iter()
                .filter_map(|(_, (c, _))| {
                    let middle = [c.cell.position[0], c.cell.position[1]];
                    let p = c
                        .cell
                        .path
                        .into_iter()
                        .filter_map(|segm| match segm.to_pytuple(python) {
                            Ok(v) => Some(v),
                            Err(e) => {
                                if error.is_none() {
                                    error = Some(e);
                                }
                                None
                            }
                        })
                        .collect();

                    Some((middle, p, c.cell.current_area / c.cell.target_area))
                })
                .collect::<Vec<_>>();

            (iteration, cells)
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    if let Some(e) = error {
        Err(e)?;
    }

    Ok(points)
}

#[pymodule]
fn cr_tissue(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Agent>()?;
    m.add_class::<SimulationSettings>()?;
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
