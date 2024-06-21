use cellular_raza::concepts::{CalcError, Mechanics};

use itertools::Itertools;
use nalgebra::SVector;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Vertices<F>(pub Vec<SVector<F, 2>>)
where
    F: Clone + std::fmt::Debug + PartialEq + 'static;

impl<F> core::ops::Add for Vertices<F>
where
    SVector<F, 2>: core::ops::Add<Output = SVector<F, 2>>,
    F: Clone + std::fmt::Debug + PartialEq + 'static,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // TODO consider changing this to a more complicated interpolation
        if self.0.len() == rhs.0.len() {
            let res: Vec<_> = self
                .0
                .into_iter()
                .zip(rhs.0.into_iter())
                .map(|(x, y)| x + y)
                .collect();
            Self(res)
        } else {
            self
        }
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

impl<F> core::ops::Sub<Vertices<F>> for Vertices<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
    SVector<F, 2>: core::ops::Sub<Output = SVector<F, 2>>,
{
    type Output = Vertices<F>;

    fn sub(self, rhs: Vertices<F>) -> Self::Output {
        // TODO consider changing this to a more complicated interpolation
        if self.0.len() == rhs.0.len() {
            let res: Vec<_> = self
                .0
                .into_iter()
                .zip(rhs.0.into_iter())
                .map(|(x, y)| x - y)
                .collect();
            Self(res)
        } else {
            self
        }
    }
}

impl<F> num::Zero for Vertices<F>
where
    F: Clone + std::fmt::Debug + PartialEq + 'static,
    nalgebra::SVector<F, 2>: num::Zero,
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
    F: core::ops::AddAssign + num::Zero + num::FromPrimitive + nalgebra::ClosedDiv,
{
    pub fn mean(&self) -> nalgebra::SVector<F, 2> {
        self.0.iter().sum::<SVector<F, 2>>() / F::from_usize(self.0.len()).unwrap()
    }
}

impl<F> core::ops::AddAssign for Vertices<F>
where
    SVector<F, 2>: core::ops::Add<Output = SVector<F, 2>> + Clone,
    F: Clone + std::fmt::Debug + PartialEq + 'static,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = (*self).clone() + rhs;
    }
}

impl<F> core::ops::Mul<F> for Vertices<F>
where
    nalgebra::SVector<F, 2>: core::ops::Mul<F, Output = nalgebra::SVector<F, 2>>,
    F: Clone + std::fmt::Debug + PartialEq + 'static,
{
    type Output = Vertices<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Vertices(
            self.0
                .into_iter()
                .map(|xi| xi * rhs.clone())
                .collect::<Vec<_>>(),
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Triangulation {
    triangles: Vec<(usize, usize, usize)>,
    success: bool,
}

impl Triangulation {
    fn update_triangulation<F>(&mut self, vertices: &Vec<SVector<F, 2>>) {}

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
            Vector2::from([-2_f64.sqrt(), 0.0]),
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
                        middle[0] + r * radius * <F as num::Float>::sin(alpha * n_float),
                    ])
                })
                .collect();
            let vertices = Vertices(vertices);
            Puzzle {
                vertices: vertices.clone(),
                velocity: vertices.clone() * F::zero(),
                random_velocity: vertices.clone() * F::zero(),
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

impl<F> Mechanics<Vertices<F>, Vertices<F>, Vertices<F>, F> for Puzzle<F>
where
    F: core::ops::DivAssign
        + nalgebra::RealField
        + num::Float
        + core::ops::Mul<SVector<F, 2>, Output = SVector<F, 2>>
        + core::iter::Sum,
    rand_distr::StandardNormal: rand_distr::Distribution<F>,
{
    fn pos(&self) -> Vertices<F> {
        self.vertices.clone()
    }

    fn set_pos(&mut self, pos: &Vertices<F>) {
        self.vertices = self.ensure_vertices_are_simple(pos.clone());
        self.triangulation.update_triangulation(&self.vertices.0);
    }

    fn velocity(&self) -> Vertices<F> {
        self.velocity.clone()
    }

    fn set_velocity(&mut self, velocity: &Vertices<F>) {
        self.velocity = velocity.clone();
    }

    fn calculate_increment(
        &self,
        force: Vertices<F>,
    ) -> Result<(Vertices<F>, Vertices<F>), cellular_raza::prelude::CalcError> {
        use core::ops::AddAssign;
        use itertools::Itertools;
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
            let force_curvature =
                self.angle_stiffness * (F::pi() - angle) * length_fraction_and_dir;

            internal_force
                .0
                .get_mut(n1)
                .unwrap()
                .add_assign(-one_half * force_curvature);
            // We made sure by using the modulus operator that these indices to not overflow
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
            let force_surf_tension =
                self.surface_tension * (F::one() - current_boundary_length / self.boundary_length);
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
                let area = <F as num::Float>::abs(v12.perp(&v13));
                area
            })
            .sum::<F>();
        let area_diff = self.cell_area - current_cell_area;

        let mut center_force = SVector::<F, 2>::zeros();
        for ((n1, v1), (n2, v2)) in self
            .vertices
            .0
            .iter()
            .enumerate()
            .circular_tuple_windows::<(_, _)>()
        {
            let d = v2 - v1;
            if d.norm() != F::zero() {
                let force_dir: SVector<F, 2> = [d.y, -d.x].into();
                let pressure_force = self.internal_pressure * area_diff * force_dir;
                internal_force
                    .0
                    .get_mut(n1)
                    .unwrap()
                    .add_assign(&(one_half * pressure_force));
                internal_force
                    .0
                    .get_mut(n2)
                    .unwrap()
                    .add_assign(&(one_half * pressure_force));
                center_force.add_assign(-pressure_force);
            }
        }
        internal_force.0.iter_mut().for_each(|f| *f += center_force);
        println!("{:8.2} {:8.2} {:8.2}",
            area_diff,
            self.boundary_length,
            current_boundary_length,
        );

        Ok((
            self.velocity.clone() + self.random_velocity.clone(),
            internal_force + force - self.velocity.clone() * self.damping,
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
