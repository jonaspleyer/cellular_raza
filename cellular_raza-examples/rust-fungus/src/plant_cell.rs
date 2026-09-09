use cellular_raza::prelude::*;
use nalgebra::{Matrix2xX, Vector2};
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use serde::{Deserialize, Serialize};

use crate::geometry::*;

type Pos = Matrix2xX<f64>;
type Inf = ();

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Clone, Deserialize, Serialize)]
#[rustfmt::skip]
pub struct PlantCell {
    // Mechanical variables
    pub position: Matrix2xX<f64>,
    // This will be used to iteratively apply restrictions when calculating overlaps with other
    // polygons. We keep the original position fixed differences due to ordering of the applied
    // restrictions.
    pub position_helper: Matrix2xX<f64>,
    pub bounding_box: (Vector2<f64>, Vector2<f64>),
    // Interaction Parameters
    #[pyo3(get, set)] pub force_area: f64,
    #[pyo3(get, set)] pub force_perimeter: f64,
    #[pyo3(get, set)] pub force_dist: f64,
    #[pyo3(get, set)] pub force_angle: f64,
    #[pyo3(get, set)] pub interaction_range: f64,
    #[pyo3(get, set)] pub min_dist: f64,
    #[pyo3(get, set)] pub target_area: f64,
    #[pyo3(get, set)] pub target_perimeter: f64,
    // Mechanical Parameters
    #[pyo3(get, set)] pub damping: f64,
    #[pyo3(get, set)] pub diffusion_constant: f64,
}

impl PlantCell {
    pub fn get_middle(&self) -> Vector2<f64> {
        area_centroid(&self.position)
    }
}

pub(crate) fn py_array_to_matrix(py_array: Bound<numpy::PyArray2<f64>>) -> Matrix2xX<f64> {
    let array = py_array.to_owned_array();
    nalgebra::Matrix2xX::from_fn(array.ncols(), |i, j| array[(i, j)])
}

pub(crate) fn matrix_to_py_array<'py>(
    py: Python<'py>,
    matrix: &Matrix2xX<f64>,
) -> Bound<'py, numpy::PyArray2<f64>> {
    let array =
        numpy::ndarray::Array2::<f64>::from_shape_fn(matrix.shape(), |(i, j)| matrix[(i, j)]);
    numpy::PyArray2::from_owned_array(py, array)
}

#[gen_stub_pymethods]
#[pymethods]
impl PlantCell {
    #[new]
    pub fn new(
        position: Bound<numpy::PyArray2<f64>>,
        force_area: f64,
        force_perimeter: f64,
        target_area: f64,
        force_angle: f64,
        force_dist: f64,
        interaction_range: f64,
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
            interaction_range,
            min_dist,
            target_area,
            target_perimeter,
            damping,
            diffusion_constant,
        }
    }

    pub fn get_perimeter(&self) -> f64 {
        get_polygon_perimeter(&self.position)
    }

    pub fn get_area(&self) -> f64 {
        get_polygon_area(&self.position)
    }

    #[getter]
    pub fn get_position<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        matrix_to_py_array(py, &self.position)
    }

    #[setter]
    pub fn set_position(&mut self, position: Bound<numpy::PyArray2<f64>>) {
        self.position = py_array_to_matrix(position);
    }
}

impl Position<Pos> for PlantCell {
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

impl Velocity<Pos> for PlantCell {
    fn velocity(&self) -> Pos {
        Matrix2xX::zeros(self.position.ncols())
    }

    fn set_velocity(&mut self, _: &Pos) {}
}

impl Mechanics<Pos, Pos, Pos> for PlantCell {
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
            let force_dir =
                if !approx::abs_diff_eq!(force_dir.norm(), 0.0, epsilon = crate::geometry::EPSILON)
                {
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

impl InteractionInformation<Inf> for PlantCell {
    fn get_interaction_information(&self) -> Inf {}
}

impl Interaction<Pos, Pos, Pos, Inf> for PlantCell {
    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        _: &Pos,
        ext_pos: &Pos,
        _: &Pos,
        _: &Inf,
    ) -> Result<(Pos, Pos), CalcError> {
        use core::ops::AddAssign;
        let ncols1 = own_pos.ncols();
        let ncols2 = ext_pos.ncols();
        let mut own_force = Pos::zeros(ncols1);
        let mut ext_force = Pos::zeros(ncols2);
        for i in 0..ncols1 {
            let p1 = own_pos.column(i);
            // Calculate segment with closest distance
            let mut dist = f64::INFINITY;
            let mut f = Vector2::zeros();
            let mut t = 0.0;
            let (mut k1, mut k2) = (0, 0);
            for j1 in 0..ncols2 {
                let j2 = (j1 + 1) % ncols2;
                let v1 = ext_pos.column(j1);
                let v2 = ext_pos.column(j2);
                let (s, q) = closest_point_on_segment(&p1, &v1, &v2);

                // Calculate interaction betwen p1 and q
                let x = q - p1;
                let d = (x.norm() / self.interaction_range).clamp(0.0, 1.0);

                // Update calculated force if another distance was smaller
                if 0.0 < d && d < 1.0 && d < dist {
                    f = self.force_dist * (d - 1.0) / self.interaction_range * x;
                    t = s;
                    k1 = j1;
                    k2 = j2;
                    dist = d;
                }
            }
            // Apply forces
            own_force.column_mut(i).add_assign(-f);
            ext_force.column_mut(k1).add_assign(t * f);
            ext_force.column_mut(k2).add_assign((1.0 - t) * f);
        }
        Ok((own_force, ext_force))
    }
}
