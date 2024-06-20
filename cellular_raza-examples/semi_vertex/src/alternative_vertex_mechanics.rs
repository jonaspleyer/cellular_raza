use cellular_raza::building_blocks::generate_random_vector;
use cellular_raza::concepts::{CalcError, Mechanics, RngError};
use serde::{Deserialize, Serialize};

use itertools::Itertools;
use nalgebra::SVector;

/// Mechanics model which represents cells as vertices with edges between them.
///
/// The vertices are attached to each other with springs and a given length between each
/// vertex.
/// Furthermore, we define a central pressure that acts when the total cell area is greater
/// or smaller than the desired one.
/// Each vertex is damped individually by the same constant.
// TODO include more formulas for this model
///
/// See also [https://en.wikipedia.org/wiki/Regular_polygon].
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VertexMechanics2DAlternative<const D: usize> {
    points: nalgebra::SMatrix<f64, D, 2>,
    velocity: nalgebra::SMatrix<f64, D, 2>,
    random_vector: nalgebra::SMatrix<f64, D, 2>,
    /// TODO
    pub cell_boundary_length: f64,
    /// TODO
    pub cell_area: f64,
    /// TODO
    pub boundary_tension: f64,
    /// TODO
    pub central_pressure: f64,
    /// TODO
    pub damping_constant: f64,
    /// TODO
    pub diffusion_constant: f64,
}

impl<const D: usize> VertexMechanics2DAlternative<D> {
    /// Setter for the individual boundary lengths of the cell
    #[allow(unused)]
    pub fn set_boundary_length(&mut self, boundary_length: f64) {
        self.cell_boundary_length = boundary_length;
    }

    /// Creates a new vertex model in equilibrium conditions.
    ///
    /// The specified parameters are then used to carefully calculate relating properties of the
    /// model.
    /// We outline the formulas used.
    /// Given the number of vertices \\(N\\) in our model (specified by the const generic argument
    /// of the [VertexMechanics2DAlternative] struct),
    /// the resulting average angle when all nodes are equally distributed is
    /// \\[
    ///     \Delta\varphi = \frac{2\pi}{N}
    /// \\]
    /// Given the total area of the cell ([regular polygon](https://en.wikipedia.org/wiki/Regular_polygon)) \\(A\\), we can calculate the distance from the center point to the individual vertices by inverting
    /// \\[
    ///     A = r^2 \sin\left(\frac{\pi}{N}\right)\cos\left(\frac{\pi}{N}\right).
    /// \\]
    /// This formula can be derived by elementary geometric arguments.
    /// We can then generate the points \\(\vec{p}\_i\\) which make up the cell-boundary by using
    /// \\[
    ///     \vec{p}\_i = \vec{p}\_{mid} + r(\cos(i \Delta\varphi), \sin(i\Delta\varphi))^T.
    /// \\]
    /// From these points, their distance is calculated and passed as the individual boundary lengths.
    /// When randomization is turned on, these points will be slightly randomized in their radius and angle which might lead to non-equilibrium configurations.
    /// Pressure, damping and spring tensions are not impacted by randomization.
    #[allow(unused)]
    pub fn new(
        middle: SVector<f64, 2>,
        cell_area: f64,
        rotation_angle: f64,
        boundary_tension: f64,
        central_pressure: f64,
        damping_constant: f64,
        diffusion_constant: f64,
        randomize: Option<(f64, rand_chacha::ChaCha8Rng)>,
    ) -> Self {
        let n_rows = D;
        use rand::Rng;
        // Restrict the randomize variable between 0 and 1
        let r = match randomize {
            Some((rand, _)) => rand.clamp(0.0, 1.0),
            _ => 0.0,
        };
        let rng = || -> f64 {
            match randomize {
                Some((_, mut rng)) => rng.gen_range(0.0..1.0),
                None => 0.0,
            }
        };
        // Randomize the overall rotation angle
        let rotation_angle = (1.0 - r * rng.clone()()) * rotation_angle;
        // Calculate the angle fraction used to determine the points of the polygon
        let side_length = Self::calculate_single_side_length(cell_area);
        let radius = Self::calculate_outside_radius(cell_area);
        // Calculate the radius from cell area
        let points = nalgebra::SMatrix::<f64, D, 2>::from_row_iterator(
            (0..n_rows)
                .map(|n| {
                    let angle = rotation_angle
                        + 2.0 * Self::angle_fraction() * n as f64 * (1.0 - r * rng.clone()());
                    let radius_modified = radius * (1.0 + 0.5 * r * (1.0 - rng.clone()()));
                    [
                        middle.x + radius_modified * angle.cos(),
                        middle.y + radius_modified * angle.sin(),
                    ]
                    .into_iter()
                })
                .flatten(),
        );
        let cell_boundary_length = Self::calculate_boundary_length(cell_area);
        let calculated_cell_area =
            0.5 * n_rows as f64 * side_length * radius * Self::angle_fraction().cos();
        debug_assert!((cell_boundary_length - n_rows as f64 * side_length).abs() < 1e-3);
        debug_assert!(
            (0.25 * n_rows as f64 * side_length.powf(2.0) / Self::angle_fraction().tan()
                - cell_area)
                .abs()
                < 1e-3
        );
        debug_assert!((calculated_cell_area - cell_area).abs() < 1e-3);
        debug_assert!((Self::calculate_cell_area(cell_boundary_length) - cell_area).abs() < 1e-3);
        let res = VertexMechanics2DAlternative {
            points,
            velocity: nalgebra::SMatrix::<f64, D, 2>::zeros(),
            random_vector: nalgebra::SMatrix::<f64, D, 2>::zeros(),
            cell_boundary_length,
            boundary_tension,
            cell_area,
            central_pressure,
            damping_constant,
            diffusion_constant,
        };
        if r == 0.0 {
            debug_assert!(
                (res.calculate_current_boundary_length() - cell_boundary_length).abs() < 1e-3
            );
            debug_assert!((res.get_current_cell_area() - res.cell_area).abs() < 1e-3);
        }
        res
    }

    pub fn angle_fraction() -> f64 {
        std::f64::consts::PI / D as f64
    }

    pub fn calculate_outside_radius(cell_area: f64) -> f64 {
        0.5 * Self::calculate_single_side_length(cell_area) / Self::angle_fraction().sin()
    }

    pub fn calculate_single_side_length(cell_area: f64) -> f64 {
        let n_rows = D as f64;
        (4.0 * cell_area * Self::angle_fraction().tan() / n_rows).sqrt()
    }

    /// Calculates the boundary length of the regular polygon given the total area in equilibrium.
    ///
    /// The formula used is
    /// $$\\begin{align}
    ///     A &= \frac{L^2}{4n\tan\left(\frac{\pi}{n}\right)}\\\\
    ///     L &= \sqrt{4An\tan\left(\frac{\pi}{n}\right)}
    /// \\end{align}$$
    /// where $A$ is the total area, $n$ is the number of vertices and $L$ is the total boundary
    /// length.
    #[allow(unused)]
    pub fn calculate_boundary_length(cell_area: f64) -> f64 {
        let n_rows = D as f64;
        Self::calculate_single_side_length(cell_area) * n_rows
    }

    /// Calculates the cell area of the regular polygon in equilibrium.
    ///
    /// The formula used is identical the the one of [Self::calculate_boundary_length].
    #[allow(unused)]
    pub fn calculate_cell_area(boundary_length: f64) -> f64 {
        let n_rows = D as f64;
        boundary_length.powf(2.0) / (4.0 * Self::angle_fraction().tan()) / n_rows
    }

    /// Calculates the current area of the cell
    #[allow(unused)]
    pub fn get_current_cell_area(&self) -> f64 {
        // TODO this is only correct for convex polygons
        let middle = self.pos().row_mean();
        0.5 * self
            .points
            .row_iter()
            .circular_tuple_windows()
            .map(|(p1, p2)| {
                (p1 - middle)
                    .transpose()
                    .perp(&(p2 - middle).transpose())
                    .abs()
            })
            .sum::<f64>()
    }

    /// Calculate the current polygons boundary length
    pub fn calculate_current_boundary_length(&self) -> f64 {
        self.points
            .row_iter()
            .circular_tuple_windows::<(_, _)>()
            .map(|(p1, p2)| (p2 - p1).norm())
            .sum::<f64>()
    }

    /// Obtain current cell area
    #[allow(unused)]
    pub fn get_cell_area(&self) -> f64 {
        self.cell_area
    }

    /// Set the current cell area and adjust the length of edges such that the cell is still in
    /// equilibrium.
    #[allow(unused)]
    pub fn set_cell_area_and_boundary_length(&mut self, cell_area: f64) {
        // Calculate the relative difference to current area
        match self.cell_area {
            a if a == 0.0 => {
                let new_interaction_parameters = Self::new(
                    self.points
                        .row_iter()
                        .map(|v| v.transpose())
                        .sum::<nalgebra::Vector2<f64>>(),
                    cell_area,
                    0.0,
                    self.boundary_tension,
                    self.central_pressure,
                    self.damping_constant,
                    self.diffusion_constant,
                    None,
                );
                *self = new_interaction_parameters;
            }
            _ => {
                let relative_length_difference = (cell_area.abs() / self.cell_area.abs()).sqrt();
                // Calculate the new length of the cell boundary lengths
                self.cell_boundary_length *= relative_length_difference;
                self.cell_area = cell_area;
            }
        };
    }

    /// Change the internal cell area
    #[allow(unused)]
    pub fn set_cell_area(&mut self, cell_area: f64) {
        self.cell_area = cell_area;
    }
}

impl VertexMechanics2DAlternative<4> {
    /// Fill a specified rectangle with cells of 4 vertices
    #[allow(unused)]
    pub fn fill_rectangle(
        cell_area: f64,
        boundary_tension: f64,
        central_pressure: f64,
        damping_constant: f64,
        diffusion_constant: f64,
        rectangle: [SVector<f64, 2>; 2],
    ) -> Vec<Self> {
        let cell_side_length: f64 = cell_area.sqrt();
        let cell_side_length_padded: f64 = cell_side_length * 1.04;

        let number_of_cells_x: u64 =
            ((rectangle[1].x - rectangle[0].x) / cell_side_length_padded).floor() as u64;
        let number_of_cells_y: u64 =
            ((rectangle[1].y - rectangle[0].y) / cell_side_length_padded).floor() as u64;

        let start_x: f64 = rectangle[0].x
            + 0.5
                * ((rectangle[1].x - rectangle[0].x)
                    - number_of_cells_x as f64 * cell_side_length_padded);
        let start_y: f64 = rectangle[0].y
            + 0.5
                * ((rectangle[1].y - rectangle[0].y)
                    - number_of_cells_y as f64 * cell_side_length_padded);

        use itertools::iproduct;
        let filled_rectangle = iproduct!(0..number_of_cells_x, 0..number_of_cells_y)
            .map(|(i, j)| {
                let corner = (
                    start_x + (i as f64) * cell_side_length_padded,
                    start_y + (j as f64) * cell_side_length_padded,
                );

                let points = nalgebra::SMatrix::<f64, 4, 2>::from_row_iterator([
                    corner.0,
                    corner.1,
                    corner.0 + cell_side_length,
                    corner.1,
                    corner.0 + cell_side_length,
                    corner.1 + cell_side_length,
                    corner.0,
                    corner.1 + cell_side_length,
                ]);

                VertexMechanics2DAlternative {
                    points,
                    velocity: nalgebra::SMatrix::<f64, 4, 2>::zeros(),
                    random_vector: nalgebra::SMatrix::<f64, 4, 2>::zeros(),
                    cell_boundary_length: 4.0 * cell_side_length,
                    boundary_tension,
                    cell_area,
                    central_pressure,
                    damping_constant,
                    diffusion_constant,
                }
            })
            .collect::<Vec<_>>();

        filled_rectangle
    }
}

impl<const D: usize>
    Mechanics<
        nalgebra::SMatrix<f64, D, 2>,
        nalgebra::SMatrix<f64, D, 2>,
        nalgebra::SMatrix<f64, D, 2>,
    > for VertexMechanics2DAlternative<D>
{
    fn pos(&self) -> nalgebra::SMatrix<f64, D, 2> {
        self.points.clone()
    }

    fn velocity(&self) -> nalgebra::SMatrix<f64, D, 2> {
        self.velocity.clone()
    }

    fn set_pos(&mut self, pos: &nalgebra::SMatrix<f64, D, 2>) {
        // TODO
        // Make sure that the position is correctly ordered everytime
        let middle = pos.row_mean();
        let index_angle = (0..D)
            .map(|n_vertex| {
                let p1 = pos.row(n_vertex) - middle;
                let p2 = pos.row(0) - middle;
                let dot: f64 = p1.dot(&p2);
                let det: f64 = p1.transpose().perp(&p2.transpose());
                let res = (-det.atan2(dot)).rem_euclid(2.0 * std::f64::consts::PI);
                (n_vertex, res)
            })
            .sorted_by(|(_, a1), (_, a2)| a1.total_cmp(&a2));
        let mut new_pos = pos.clone();
        for (n, (n_vertex, _)) in index_angle.enumerate() {
            if n != n_vertex {
                new_pos.set_row(n, &pos.row(n_vertex));
            }
        }
        self.points = new_pos.clone();
    }

    fn set_velocity(&mut self, velocity: &nalgebra::SMatrix<f64, D, 2>) {
        self.velocity = velocity.clone();
    }

    fn calculate_increment(
        &self,
        force: nalgebra::SMatrix<f64, D, 2>,
    ) -> Result<(nalgebra::SMatrix<f64, D, 2>, nalgebra::SMatrix<f64, D, 2>), CalcError> {
        // Calculate the total internal force
        let area_diff = self.cell_area - self.get_current_cell_area();
        let boundary_diff = self.cell_boundary_length - self.calculate_current_boundary_length();

        // Calculate the difference of the current boundary to the specified one
        let mut center_force = nalgebra::RowSVector::<f64, 2>::zeros();
        let mut internal_force = self.points.clone() * 0.0;
        for (n_index, (point_1, point_2)) in self
            .points
            .row_iter()
            .circular_tuple_windows::<(_, _)>()
            .enumerate()
        {
            // Calculate forces arising from springs between points
            let p_21 = point_1 - point_2;
            let dir = p_21.normalize();
            let force1 = dir * boundary_diff * self.boundary_tension;
            let mut f1 = 0.5 * force1;
            let mut f2 = -0.5 * force1;

            // Calculate force arising from internal pressure
            {
                // Calculate the vector perpendicular to p_21 pointing outwards of the polygon
                let new_dir: nalgebra::RowSVector<f64, 2> = [-dir.y, dir.x].into();
                let force = new_dir * area_diff * self.central_pressure * p_21.norm();
                f1 += 0.5 * force;
                f2 += 0.5 * force;
                center_force -= force;
            }

            // Combine forces
            use core::ops::AddAssign;
            internal_force.row_mut(n_index).add_assign(&f1);
            internal_force.row_mut((n_index + 1) % D).add_assign(&f2);
        }
        internal_force.row_iter_mut().for_each(|mut r| r += center_force);
        debug_assert!(internal_force.row_sum().norm() < 1e-5);
        let dx = self.velocity.clone() + self.diffusion_constant * self.random_vector;
        let dv = force + internal_force - self.damping_constant * self.velocity.clone();
        if dx.iter().any(|xi| xi.is_nan()) {
            return Err(CalcError(format!("stopping sim due to nan value")));
        }
        Ok((dx, dv))
    }

    fn set_random_variable(
        &mut self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: f64,
    ) -> Result<(), RngError> {
        if dt != 0.0 {
            let random_vector: SVector<f64, 2> = generate_random_vector(rng, dt.sqrt())? / dt;
            self.random_vector.row_iter_mut().for_each(|mut r| {
                r *= 0.0;
                r += random_vector.transpose();
            });
        }
        Ok(())
    }
}

mod test {
    #[test]
    fn check_equilibrium_properties() {
        let middle = [0.0; 2].into();
        let cell_area = 150.0;
        let rotation_angle = 0.0;
        let boundary_tension = 0.1;
        let central_pressure = 0.1;
        let damping_constant = 0.1;
        let diffusion_constant = 0.01;
        let randomize = None;
        let mut cell = super::VertexMechanics2DAlternative::<4>::new(
            middle,
            cell_area,
            rotation_angle,
            boundary_tension,
            central_pressure,
            damping_constant,
            diffusion_constant,
            randomize,
        );

        let do_asserts = |cell: &super::VertexMechanics2DAlternative<4>| {
            assert_eq!(cell.get_cell_area(), cell_area);
            assert!((cell.get_current_cell_area() - cell_area).abs() < 1e-8);
            assert!(
                (cell.calculate_current_boundary_length() - 4.0 * cell_area.sqrt()).abs() < 1e-5
            );
        };
        do_asserts(&cell);

        use cellular_raza::concepts::Mechanics;
        /// Advance time a bit without external force
        /// This makes sure that we really are in the equilibrium
        use num::Zero;
        let dt = 0.01;
        for _ in 0..1_000 {
            let pos = cell.pos();
            let vel = cell.velocity();
            let (dx, dv) = cell
                .calculate_increment(nalgebra::SMatrix::<f64, 4, 2>::zero())
                .unwrap();
            cell.set_pos(&(pos + dt * dx));
            cell.set_velocity(&(vel + dt * dv));
            do_asserts(&cell);
        }
    }
}
