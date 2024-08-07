use std::marker::PhantomData;

use cellular_raza::building_blocks::{
    nearest_point_from_point_to_multiple_lines, ray_intersects_line_segment,
};
use cellular_raza::concepts::{
    CalcError, CellAgent, Interaction, Mechanics, Position, RngError, Velocity, Xapy,
};

use itertools::Itertools;
use nalgebra::{SVector, Vector2};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Vertices<F>(pub Vec<SVector<F, 2>>)
where
    F: Clone + std::fmt::Debug + PartialEq + 'static;

impl<F> Xapy<F> for Vertices<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
    SVector<F, 2>: core::ops::Mul<F, Output = SVector<F, 2>>,
    SVector<F, 2>: core::ops::Add<SVector<F, 2>, Output = SVector<F, 2>>,
    F: Copy,
{
    fn xapy(&self, a: F, y: &Self) -> Self {
        let mut new_values = self.clone();
        for i in 0..self.0.len().max(y.0.len()) {
            match (new_values.0.get_mut(i), y.0.get(i)) {
                (Some(s), Some(yi)) => *s = *s * a + *yi,
                (Some(s), None) => *s = *s * a,
                _ => (),
            }
        }
        new_values
    }
}

impl<F> Vertices<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
{
    pub fn from_value(n_vertices: usize, value: SVector<F, 2>) -> Self {
        Self((0..n_vertices).map(|_| value.clone()).collect())
    }
}

impl<F> Default for Vertices<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
{
    fn default() -> Self {
        Vertices(Vec::new())
    }
}

impl<F> num::Zero for Vertices<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
    F: nalgebra::RealField,
{
    fn zero() -> Self {
        Vertices(Vec::new())
    }

    fn is_zero(&self) -> bool {
        self.0.len() == 0 || self.0.iter().all(|xi| xi.is_zero())
    }
}

impl<F> Vertices<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
    F: core::ops::AddAssign + num::Zero + num::FromPrimitive + core::ops::Div<Output=F> + core::ops::DivAssign,
{
    pub fn mean(&self) -> nalgebra::SVector<F, 2> {
        self.0.iter().sum::<SVector<F, 2>>() / F::from_usize(self.0.len()).unwrap()
    }
}

impl<F> core::ops::Add for Vertices<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
    F: nalgebra::RealField,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut new_values = self;
        use core::ops::AddAssign;
        for i in 0..new_values.0.len() {
            match (new_values.0.get_mut(i), rhs.0.get(i)) {
                (Some(s), Some(x)) => s.add_assign(x),
                _ => (),
            }
        }
        new_values
    }
}

impl<F> core::ops::AddAssign for Vertices<F>
where
    SVector<F, 2>: core::ops::Add<Output = SVector<F, 2>> + Clone,
    F: Clone + std::fmt::Debug + PartialEq + 'static,
    F: nalgebra::RealField,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..self.0.len() {
            match (self.0.get_mut(i), rhs.0.get(i)) {
                (Some(s), Some(x)) => s.add_assign(x),
                _ => (),
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Triangulation {
    triangles: Vec<(usize, usize, usize)>,
    success: bool,
}

impl Triangulation {
    fn update_triangulation<F>(&mut self, _vertices: &Vec<SVector<F, 2>>) {
        // TODO
    }

    fn new<F>(vertices: &Vertices<F>) -> Result<Self, CalcError>
    where
        F: Clone + std::fmt::Debug + PartialEq + 'static,
        F: nalgebra::Scalar + std::cmp::PartialOrd,
        F: geo::CoordNum + geo::CoordFloat,
    {
        let polygon = geo::Polygon::new(
            geo::LineString::new(
                vertices
                    .0
                    .iter()
                    .map(|p| geo::Coord { x: p.x, y: p.y })
                    .collect::<Vec<_>>(),
            ),
            vec![],
        );
        use geo::TriangulateEarcut;
        let triangles: Vec<_> = polygon
            .earcut_triangles_raw()
            .triangle_indices
            .into_iter()
            .tuples::<(_, _, _)>()
            .collect();
        Ok(Triangulation {
            triangles,
            success: true,
        })
    }

    pub fn get_triangles(
        &self,
    ) -> Result<impl IntoIterator<Item = &(usize, usize, usize)>, CalcError> {
        if self.success {
            Ok(self.triangles.iter())
        } else {
            Err(CalcError(
                "building triangulation was not successful.\
                This may be due to the fact that the polygon in question is not a simple polygon."
                    .to_owned(),
            ))
        }
    }
}

#[cfg(test)]
mod test_triangulation {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn test_square() {
        let vertices = Vertices(vec![
            Vector2::from([0.0, 0.0]),
            Vector2::from([1.0, 0.0]),
            Vector2::from([1.0, 1.0]),
            Vector2::from([0.0, 1.0]),
        ]);
        let triangulation = Triangulation::new(&vertices).unwrap();
        assert_eq!(triangulation.triangles.len(), 2);
    }

    #[test]
    fn test_hexagon() {
        //     p2--p3
        //    /     \
        //   p1      p4
        //    \      /
        //     p6--p5
        let vertices = Vertices(vec![
            Vector2::from([-(2_f64.sqrt()), 0.0]),
            Vector2::from([-1.0, 1.0]),
            Vector2::from([1.0, 1.0]),
            Vector2::from([2_f64.sqrt(), 0.0]),
            Vector2::from([1.0, -1.0]),
            Vector2::from([-1.0, -1.0]),
        ]);
        let triangulation = Triangulation::new(&vertices).unwrap();
        assert_eq!(triangulation.triangles.len(), 4);
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Puzzle<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
{
    vertices: Vertices<F>,
    velocity: Vertices<F>,
    random_velocity: Vertices<F>,
    triangulation: Triangulation,
    pub angle_stiffness: F,
    pub surface_tension: F,
    pub boundary_length: F,
    pub cell_area: F,
    pub internal_pressure: F,
    pub diffusion_constant: F,
    pub damping: F,
}

impl<F> Puzzle<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
    F: nalgebra::RealField + num::Float + num::FromPrimitive,
{
    pub fn new_equilibrium(
        middle: impl Into<[F; 2]>,
        n_vertices: usize,
        angle_stiffness: F,
        surface_tension: F,
        boundary_length: F,
        cell_area: F,
        internal_pressure: F,
        diffusion_constant: F,
        damping: F,
        // TODO use this
        randomize: Option<(F, u64)>,
    ) -> Self
    where
        F: rand_distr::uniform::SampleUniform,
    {
        use rand::Rng;
        let randomizer = |n: u64| -> F {
            match &randomize {
                Some((r, seed)) => {
                    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed + n);
                    F::one()
                        + <F as num::Float>::clamp(r.clone(), F::zero(), F::one())
                            * rng.gen_range(-F::one()..F::one())
                }
                None => F::one(),
            }
        };
        // Restrict the randomize variable between 0 and 1
        let middle: [F; 2] = middle.into();
        if n_vertices % 2 == 0 {
            // TODO introduce 2 radii to obtain the correct boundary_length
            let n_vertices_float = F::from_usize(n_vertices).unwrap();
            let alpha_half = F::pi() / n_vertices_float;
            let radius = <F as num::Float>::sqrt(
                cell_area
                    / (n_vertices_float
                        * <F as num::Float>::sin(alpha_half)
                        * <F as num::Float>::cos(alpha_half)),
            );
            let vertices: Vec<_> = (0..n_vertices)
                .map(|n| {
                    let n_float = F::from_usize(n).unwrap();
                    let alpha = (F::one() + F::one()) * alpha_half;
                    let r = randomizer(n as u64);
                    SVector::<F, 2>::from([
                        middle[0] + r * radius * <F as num::Float>::cos(alpha * n_float),
                        middle[1] + r * radius * <F as num::Float>::sin(alpha * n_float),
                    ])
                })
                .collect();
            let vertices = Vertices(vertices);
            use num::Zero;
            Puzzle {
                vertices: vertices.clone(),
                velocity: vertices.xapy(F::zero(), &Vertices::zero()),
                random_velocity: vertices.xapy(F::zero(), &Vertices::zero()),
                triangulation: Triangulation::new(&vertices).unwrap(),
                angle_stiffness,
                surface_tension,
                boundary_length,
                cell_area,
                internal_pressure,
                diffusion_constant,
                damping,
            }
        } else {
            unimplemented!()
        }
    }

    fn ensure_vertices_are_simple(&self, new_vertices: Vertices<F>) -> Vertices<F> {
        // TODO implement this safekeeping mechanism
        new_vertices
    }
}

impl<F> Position<Vertices<F>> for Puzzle<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static + PartialOrd,
    F: nalgebra::RealField + num::Float,
{
    fn pos(&self) -> Vertices<F> {
        self.vertices.clone()
    }

    fn set_pos(&mut self, pos: &Vertices<F>) {
        self.vertices = self.ensure_vertices_are_simple(pos.clone());
        self.triangulation.update_triangulation(&self.vertices.0);
    }
}

impl<F> Velocity<Vertices<F>> for Puzzle<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static + PartialOrd,
{
    fn velocity(&self) -> Vertices<F> {
        self.velocity.clone()
    }

    fn set_velocity(&mut self, velocity: &Vertices<F>) {
        self.velocity = velocity.clone();
    }
}

impl<F> Mechanics<Vertices<F>, Vertices<F>, Vertices<F>, F> for Puzzle<F>
where
    F: core::ops::DivAssign
        + nalgebra::RealField
        + num::Float
        + core::ops::Mul<SVector<F, 2>, Output = SVector<F, 2>>
        + core::iter::Sum,
    rand_distr::StandardNormal: rand_distr::Distribution<F>,
{
    fn calculate_increment(
        &self,
        force: Vertices<F>,
    ) -> Result<(Vertices<F>, Vertices<F>), cellular_raza::prelude::CalcError> {
        use core::ops::AddAssign;
        let one_half = F::one() / (F::one() + F::one());
        let n_vertices = self.vertices.0.len();
        let mut internal_force = Vertices::from_value(n_vertices, SVector::<F, 2>::zeros());

        // Calculate the total boundary length of the cell
        let current_boundary_length = self
            .vertices
            .0
            .iter()
            .circular_tuple_windows::<(_, _)>()
            .map(|(v1, v2)| (v1 - v2).norm())
            .sum::<F>();

        // Iterate over all vertices and calculate forces arising from surface tension and
        // curvature
        for ((n1, v1), (n2, v2), (n3, v3)) in self
            .vertices
            .0
            .iter()
            .enumerate()
            .circular_tuple_windows::<(_, _, _)>()
        {
            let v21 = v1 - v2;
            let v23 = v3 - v2;

            // Calculate curvature contributions
            let angle = v21.angle(&v23);
            let length_fraction_and_dir = one_half * (v21 + v23) / current_boundary_length;
            let force_curvature = F::from_usize(n_vertices).unwrap()
                * self.angle_stiffness
                * (F::pi() - angle)
                * length_fraction_and_dir;

            internal_force
                .0
                .get_mut(n1)
                .unwrap()
                .add_assign(-one_half * force_curvature);
            // We made sure by using the modulus operator that these indices do not overflow
            internal_force
                .0
                .get_mut(n2)
                .unwrap()
                .add_assign(force_curvature);
            internal_force
                .0
                .get_mut(n3)
                .unwrap()
                .add_assign(-one_half * force_curvature);

            // Calculate surface tension contribution
            let force_surf_tension = self.surface_tension
                * (F::one()
                    - v21.norm() / self.boundary_length * F::from_usize(n_vertices).unwrap());
            internal_force
                .0
                .get_mut(n1)
                .unwrap()
                .add_assign(one_half * force_surf_tension * v21);
            internal_force
                .0
                .get_mut(n2)
                .unwrap()
                .add_assign(-one_half * force_surf_tension * v21);
        }

        // Calculate contributions by the internal pressure of the cell
        // This is where the triangulation comes into play.
        let current_cell_area = self
            .triangulation
            .get_triangles()?
            .into_iter()
            .map(|(nv1, nv2, nv3)| {
                let v1 = self.vertices.0.get(*nv1).unwrap();
                let v2 = self.vertices.0.get(*nv2).unwrap();
                let v3 = self.vertices.0.get(*nv3).unwrap();

                let v12 = v2 - v1;
                let v13 = v3 - v1;
                let area = <F as num::Float>::abs(v12.perp(&v13)) / (F::one() + F::one());
                area
            })
            .sum::<F>();
        let relative_area_diff = F::one() - current_cell_area / self.cell_area;

        let mut center_force = SVector::<F, 2>::zeros();
        for &(n1, n2, n3) in self.triangulation.get_triangles()?.into_iter() {
            let v1 = self.vertices.0[n1];
            let v2 = self.vertices.0[n2];
            let v3 = self.vertices.0[n3];
            let mut apply_force = |d: SVector<F, 2>, m1: usize, m2: usize| {
                if d.norm() != F::zero() {
                    let force_dir: SVector<F, 2> = [d.y, -d.x].into();
                    let pressure_force = self.internal_pressure * relative_area_diff * force_dir;
                    internal_force
                        .0
                        .get_mut(m1)
                        .unwrap()
                        .add_assign(&(one_half * pressure_force));
                    internal_force
                        .0
                        .get_mut(m2)
                        .unwrap()
                        .add_assign(&(one_half * pressure_force));
                    center_force.add_assign(-pressure_force);
                }
            };
            apply_force(v2 - v1, n1, n2);
            apply_force(v3 - v2, n2, n3);
            apply_force(v1 - v3, n3, n1);
        }
        internal_force.0.iter_mut().for_each(|f| *f += center_force);
        Ok((
            self.velocity.clone().xapy(F::one(), &self.random_velocity),
            internal_force.xapy(F::one(), &self.velocity.xapy(-self.damping, &force)),
        ))
    }

    fn set_random_variable(
        &mut self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: F,
    ) -> Result<(), cellular_raza::prelude::RngError> {
        use cellular_raza::building_blocks::wiener_process;
        let random_vel = wiener_process::<F, 2>(rng, dt)?;
        self.random_velocity
            .0
            .iter_mut()
            .for_each(|r| *r = self.diffusion_constant * random_vel);
        Ok(())
    }
}

#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
pub struct PuzzleInteraction<F, I, O, I1, I2>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
{
    #[MechanicsRaw]
    #[Velocity]
    pub puzzle: Puzzle<F>,
    pub bounding_min: Vector2<F>,
    pub bounding_max: Vector2<F>,
    pub inside_force: I,
    pub outside_force: O,
    pub phantom_inf_inside: PhantomData<I1>,
    pub phantom_inf_outside: PhantomData<I2>,
}

impl<F, I, O, I1, I2> PuzzleInteraction<F, I, O, I1, I2>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
{
    fn bounding_boxes_do_not_intersect(&self, min: &Vector2<F>, max: &Vector2<F>) -> bool
    where
        F: PartialOrd,
    {
        self.bounding_max.x < min.x
            || self.bounding_min.x > max.x
            || self.bounding_max.y < min.y
            || self.bounding_min.y > max.y
    }

    fn update_bounding_box(&mut self)
    where
        F: PartialOrd,
    {
        let mut min = self.puzzle.vertices.0[0].clone();
        let mut max = self.puzzle.vertices.0[0].clone();
        for vert in self.puzzle.vertices.0.iter() {
            if vert.x < min.x {
                min.x = vert.x.clone();
            }
            if vert.y < min.y {
                min.y = vert.y.clone();
            }
            if vert.x > max.x {
                max.x = vert.x.clone();
            }
            if vert.y > max.y {
                max.y = vert.y.clone();
            }
        }
        self.bounding_min = min;
        self.bounding_max = max;
    }
}

impl<F, I, O, I1, I2> Position<Vertices<F>> for PuzzleInteraction<F, I, O, I1, I2>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static + PartialOrd,
    F: nalgebra::RealField + num::Float,
{
    fn pos(&self) -> Vertices<F> {
        self.puzzle.pos()
    }

    fn set_pos(&mut self, pos: &Vertices<F>) {
        self.puzzle.set_pos(pos);
        self.update_bounding_box();
    }
}

impl<F, Ii, Io, I1, I2>
    Interaction<Vertices<F>, Vertices<F>, Vertices<F>, (Vector2<F>, Vector2<F>, I1, I2)>
    for PuzzleInteraction<F, Ii, Io, I1, I2>
where
    Ii: Interaction<Vector2<F>, Vector2<F>, Vector2<F>, I1>,
    Io: Interaction<Vector2<F>, Vector2<F>, Vector2<F>, I2>,
    F: Clone + std::fmt::Debug + PartialEq + 'static + num::Float,
    F: nalgebra::RealField + num::Float + num::FromPrimitive,
{
    fn calculate_force_between(
        &self,
        own_pos: &Vertices<F>,
        own_vel: &Vertices<F>,
        ext_pos: &Vertices<F>,
        ext_vel: &Vertices<F>,
        ext_info: &(Vector2<F>, Vector2<F>, I1, I2),
    ) -> Result<(Vertices<F>, Vertices<F>), CalcError> {
        use num::Zero;
        let min_ext = ext_info.0;
        let max_ext = ext_info.1;
        let mut total_force1 = own_pos.clone().xapy(F::zero(), &Vertices::zero());
        let mut total_force2 = ext_pos.clone().xapy(F::zero(), &Vertices::zero());
        // Check if the bounding boxes do intersect. If this is not the case, do only calculation
        // of outside interactions
        let boxes_do_not_intersect = self.bounding_boxes_do_not_intersect(&min_ext, &max_ext);
        let mut point_outside_polygon = self.bounding_min;
        // TODO think about using something else than one here
        point_outside_polygon.x -= F::one();

        for (m, point_ext) in ext_pos.0.iter().enumerate() {
            let ext_point_in_polygon = if !boxes_do_not_intersect {
                // Check if vertex is inside of other triangle.
                // If the bounding box was not successful,
                // we use the ray-casting algorithm to check.
                let n_intersections: usize = self
                    .puzzle
                    .vertices
                    .0
                    .iter()
                    .circular_tuple_windows::<(_, _)>()
                    .map(|line| {
                        ray_intersects_line_segment(
                            &(*point_ext, point_outside_polygon),
                            &(*line.0, *line.1),
                        ) as usize
                    })
                    .sum();
                // An even number means that the point was outside
                // while odd numbers mean that the point was inside.
                n_intersections % 2 == 1
            } else {
                false
            };

            // Find closest point on edge
            if let Some((n, (_, point, t))) = nearest_point_from_point_to_multiple_lines(
                &point_ext,
                &self
                    .puzzle
                    .vertices
                    .0
                    .iter()
                    .circular_tuple_windows::<(_, _)>()
                    .map(|(&x, &y)| (x, y))
                    .collect::<Vec<_>>(),
            ) {
                let n1 = n;
                let n2 = (n + 1) % self.puzzle.vertices.0.len();
                let average_vel = own_vel.0[n1] * (F::one() - t) + own_vel.0[n2] * t;
                let (f1, f2) = if ext_point_in_polygon {
                    // Inside Interaction
                    self.inside_force.calculate_force_between(
                        &point,
                        &average_vel,
                        &point_ext,
                        &ext_vel.0[n1],
                        &ext_info.2,
                    )?
                } else {
                    // Outside Interaction
                    self.outside_force.calculate_force_between(
                        &point,
                        &average_vel,
                        &point_ext,
                        &ext_vel.0[n1],
                        &ext_info.3,
                    )?
                };
                let n_vertices1 = F::from_usize(total_force1.0.len()).unwrap();
                let n_vertices2 = F::from_usize(total_force2.0.len()).unwrap();
                let x1 = total_force1.0.get_mut(n1).unwrap();
                *x1 += f1 * (F::one() - t) / n_vertices1;
                let x2 = total_force1.0.get_mut(n2).unwrap();
                let y1 = total_force2.0.get_mut(m).unwrap();
                *x2 += f1 * t / n_vertices1;
                *y1 += f2 / n_vertices2;
            }
        }
        Ok((total_force1, total_force2))
    }

    fn get_interaction_information(&self) -> (Vector2<F>, Vector2<F>, I1, I2) {
        (
            self.bounding_min,
            self.bounding_max,
            self.inside_force.get_interaction_information(),
            self.outside_force.get_interaction_information(),
        )
    }
}
