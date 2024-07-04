use cellular_raza::prelude::*;

use nalgebra::{Unit, Vector2};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DirectedSphericalMechanics {
    pub pos: Vector2<f64>,
    pub vel: Vector2<f64>,
    pub orientation: Unit<Vector2<f64>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OutsideInteraction {
    pub potential_strength: f64,
    pub interaction_range: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct InsideInteraction {
    pub potential_strength: f64,
    pub average_radius: f64,
}

impl Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>> for OutsideInteraction {
    fn calculate_force_between(
        &self,
        own_pos: &Vector2<f64>,
        _own_vel: &Vector2<f64>,
        ext_pos: &Vector2<f64>,
        _ext_vel: &Vector2<f64>,
        _ext_info: &(),
    ) -> Result<(Vector2<f64>, Vector2<f64>), CalcError> {
        // Calculate distance and direction between own and other point
        let z = ext_pos - own_pos;
        let r = z.norm();
        let dir = z.normalize();

        // Introduce Non-dimensional length variable
        let sigma = r / self.interaction_range;
        let spatial_cutoff = if r > self.interaction_range { 0.0 } else { 1.0 };

        // Calculate the strength of the interaction with correct bounds
        let strength = self.potential_strength * (1.0 - sigma);

        // Calculate only attracting and repelling forces
        let force = -dir * strength * spatial_cutoff;
        Ok((-force, force))
    }

    fn get_interaction_information(&self) -> () {}
}

impl Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>> for InsideInteraction {
    fn calculate_force_between(
        &self,
        own_pos: &Vector2<f64>,
        _own_vel: &Vector2<f64>,
        ext_pos: &Vector2<f64>,
        _ext_vel: &Vector2<f64>,
        _ext_info: &(),
    ) -> Result<(Vector2<f64>, Vector2<f64>), CalcError> {
        // Calculate direction between own and other point
        let z = own_pos - ext_pos;
        let r = z.norm();
        let dir = z.normalize();

        let force = self.potential_strength * dir / (0.5 + 0.5 * r / self.average_radius);
        Ok((-force, force))
    }

    fn get_interaction_information(&self) -> () {}
}

#[derive(Serialize, Deserialize, CellAgent, Clone, Debug)]
pub struct MyCell {
    #[Mechanics]
    pub mechanics: CustomVertexMechanics2D<f64>,
    pub interaction: VertexDerivedInteraction<OutsideInteraction, InsideInteraction>,
    pub growth_side: usize,
    pub growth_factor: f64,
    pub division_threshold_area: f64,
}

impl<F> Interaction<VertexPoint<F>, VertexPoint<F>, VertexPoint<F>, ((), ())> for MyCell
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
    VertexDerivedInteraction<OutsideInteraction, InsideInteraction>: Interaction<
        nalgebra::MatrixXx2<F>,
        nalgebra::MatrixXx2<F>,
        nalgebra::MatrixXx2<F>,
        ((), ()),
    >,
{
    fn get_interaction_information(&self) -> ((), ()) {
        ((), ())
    }

    fn calculate_force_between(
        &self,
        own_pos: &VertexPoint<F>,
        own_vel: &VertexPoint<F>,
        ext_pos: &VertexPoint<F>,
        ext_vel: &VertexPoint<F>,
        ext_info: &((), ()),
    ) -> Result<(VertexPoint<F>, VertexPoint<F>), CalcError> {
        let res =
            <VertexDerivedInteraction<OutsideInteraction, InsideInteraction> as Interaction<
                nalgebra::MatrixXx2<F>,
                nalgebra::MatrixXx2<F>,
                nalgebra::MatrixXx2<F>,
                ((), ()),
            >>::calculate_force_between(
                &self.interaction,
                &own_pos.0,
                &own_vel.0,
                &ext_pos.0,
                &ext_vel.0,
                ext_info,
            )?;
        Ok((VertexPoint(res.0), VertexPoint(res.1)))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VertexPoint<F>(pub nalgebra::MatrixXx2<F>)
where
    F: 'static + Clone + PartialEq + core::fmt::Debug;

impl<F> core::ops::Add<VertexPoint<F>> for VertexPoint<F>
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
    F: nalgebra::RealField,
{
    type Output = Self;

    fn add(self, rhs: VertexPoint<F>) -> Self::Output {
        if self.0.nrows() == rhs.0.nrows() {
            Self(self.0 + rhs.0)
        } else {
            let n_rows_min = self.0.nrows().min(rhs.0.nrows());
            let mut res = self.0;
            for n in 0..n_rows_min {
                use core::ops::AddAssign;
                res.row_mut(n).add_assign(rhs.0.row(n));
            }
            Self(res)
        }
    }
}

impl<F> core::ops::AddAssign for VertexPoint<F>
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
    F: nalgebra::RealField,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<F> core::ops::Sub<VertexPoint<F>> for VertexPoint<F>
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
    F: nalgebra::RealField,
{
    type Output = Self;

    fn sub(self, rhs: VertexPoint<F>) -> Self::Output {
        if self.0.nrows() == rhs.0.nrows() {
            Self(self.0 - rhs.0)
        } else {
            let n_rows_min = self.0.nrows().min(rhs.0.nrows());
            let mut res = self.0;
            for n in 0..n_rows_min {
                use core::ops::SubAssign;
                res.row_mut(n).sub_assign(rhs.0.row(n));
            }
            Self(res)
        }
    }
}

impl<F> num::Zero for VertexPoint<F>
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
    F: nalgebra::RealField,
{
    fn zero() -> Self {
        Self(nalgebra::MatrixXx2::zeros(6))
    }

    fn set_zero(&mut self) {
        self.0.iter_mut().for_each(|x| x.set_zero())
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

impl<F> core::ops::Neg for VertexPoint<F>
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
    F: nalgebra::RealField,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<F> core::ops::Mul<F> for VertexPoint<F>
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
    F: nalgebra::RealField,
{
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl<F> Default for VertexPoint<F>
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
    F: nalgebra::RealField,
    Self: Sized,
{
    fn default() -> Self {
        Self(nalgebra::MatrixXx2::zeros(6))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CustomVertexMechanics2D<F>
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
{
    pub pos: VertexPoint<F>,
    pub vel: VertexPoint<F>,
    pub spring_tensions: Vec<F>,
    pub n_kb_t: F,
    pub damping: F,
}

impl<F> CustomVertexMechanics2D<F>
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
    F: Copy + nalgebra::RealField,
{
    pub fn new(
        n_vertices: usize,
        middle: impl Into<[F; 2]>,
        area: F,
        spring_tension: F,
        n_kb_t: F,
        damping: F,
    ) -> Self {
        let middle: [F; 2] = middle.into();
        let mut pos = nalgebra::MatrixXx2::<F>::zeros(n_vertices);
        let four = F::one() + F::one() + F::one() + F::one();
        let n_vertices_float = F::from_usize(n_vertices).unwrap();
        let radius = (four * area * (F::pi() / n_vertices_float).tan() / n_vertices_float).sqrt();
        for n_row in 0..n_vertices {
            let angle =
                (F::one() + F::one()) * F::pi() * F::from_usize(n_row).unwrap() / n_vertices_float;
            let p = [
                middle[0] + radius * angle.cos(),
                middle[1] + radius * angle.sin(),
            ];
            pos.set_row(n_row, &p.into());
        }
        let pos = VertexPoint(pos);
        Self {
            pos,
            vel: VertexPoint(nalgebra::MatrixXx2::zeros(n_vertices)),
            spring_tensions: (0..n_vertices).map(|_| spring_tension).collect(),
            n_kb_t,
            damping,
        }
    }
}

impl<F> Mechanics<VertexPoint<F>, VertexPoint<F>, VertexPoint<F>> for CustomVertexMechanics2D<F>
where
    F: 'static + Clone + PartialEq + core::fmt::Debug,
    F: num::Float + nalgebra::RealField,
{
    fn pos(&self) -> VertexPoint<F> {
        self.pos.clone()
    }

    fn set_pos(&mut self, pos: &VertexPoint<F>) {
        // TODO can we enforce that this will always yield a simple polygon even if not starshaped?
        self.pos = pos.clone();
    }

    fn velocity(&self) -> VertexPoint<F> {
        self.vel.clone()
    }

    fn set_velocity(&mut self, velocity: &VertexPoint<F>) {
        self.vel = velocity.clone();
    }

    fn calculate_increment(
        &self,
        force: VertexPoint<F>,
    ) -> Result<(VertexPoint<F>, VertexPoint<F>), CalcError> {
        use core::ops::AddAssign;
        // IMPORTANT!
        // This algorithm assumes that we are dealing with a regular polygon where the vertices are
        // given in clockwise order.
        let mut internal_force = self.pos.clone() * F::zero();

        use itertools::Itertools;
        for (index, (p1, p2)) in self
            .pos
            .0
            .row_iter()
            .circular_tuple_windows::<(_, _)>()
            .enumerate()
        {
            let x = p1 - p2;
            let r = x.norm();
            if r > F::zero() {
                let dir = (p2 - p1) / r;
                let tension_force = self.spring_tensions[index] * r;
                internal_force
                    .0
                    .row_mut(index)
                    .add_assign(&(dir * tension_force));
                internal_force
                    .0
                    .row_mut((index + 1) % internal_force.0.nrows())
                    .add_assign(&(-dir * tension_force));
            }
        }

        // Perform a triangulation with the earcut method
        use geo::TriangulateEarcut;
        let polygon = geo::Polygon::new(
            geo::LineString::from(
                self.pos
                    .clone()
                    .0
                    .row_iter()
                    .map(|r| (r[0], r[1]))
                    .collect::<Vec<_>>(),
            ),
            vec![],
        );
        let triangles: Vec<_> = polygon
            .earcut_triangles_raw()
            .triangle_indices
            .into_iter()
            .tuples::<(_, _, _)>()
            .collect();
        let mut total_volume = F::zero();
        for &(n1, n2, n3) in triangles.iter() {
            let n1 = n1 % self.pos.0.nrows();
            let n2 = n2 % self.pos.0.nrows();
            let n3 = n3 % self.pos.0.nrows();
            let p1 = self.pos.0.row(n1);
            let p2 = self.pos.0.row(n2);
            let p3 = self.pos.0.row(n3);
            let increment =
                <F as num::Float>::abs((p1 - p2).transpose().dot(&(p3 - p2).transpose()))
                    / (F::one() + F::one());
            total_volume += increment;
        }
        let before_pressure = internal_force.clone();

        for &(n1, n2, n3) in triangles.iter() {
            let n1 = n1 % self.pos.0.nrows();
            let n2 = n2 % self.pos.0.nrows();
            let n3 = n3 % self.pos.0.nrows();
            let p1 = self.pos.0.row(n1);
            let p2 = self.pos.0.row(n2);
            let p3 = self.pos.0.row(n3);

            let mut apply_force = |ni: usize, nj: usize, pi, pj| {
                let d: nalgebra::RowVector2<F> = pi - pj;
                let dir = nalgebra::RowVector2::<F>::from([-d.y, d.x]);
                let force = self.n_kb_t / total_volume * d.norm();
                let one_half = F::one() / (F::one() + F::one());
                use core::ops::AddAssign;
                internal_force
                    .0
                    .row_mut(ni)
                    .add_assign(&(dir * force * one_half));
                internal_force
                    .0
                    .row_mut(nj)
                    .add_assign(&(dir * force * one_half));
            };
            apply_force(n1, n2, p1, p2);
            apply_force(n2, n3, p2, p3);
            apply_force(n3, n1, p3, p1);
        }
        let tension_contrib = before_pressure.0.norm();
        let pressure_contrib = (internal_force.clone() - before_pressure).0.norm();
        let total = tension_contrib + pressure_contrib;
        println!("{:6.4} {:6.4}", tension_contrib / total, pressure_contrib / total);

        let dx = self.vel.clone();
        let dv = internal_force + force - self.vel.clone() * self.damping;
        Ok((dx, dv))
    }
}

/* impl Cycle for MyCell {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Self,
    ) -> Option<CycleEvent> {
        let area = cell.mechanics.get_current_cell_area();
        cell.growth_side = match cell
            .mechanics
            .cell_boundary_lengths
            .iter()
            .enumerate()
            .min_by(|(_, t), (_, h)| t.partial_cmp(h).unwrap())
        {
            Some((i, _)) => i,
            None => 0,
        };
        for n in [cell.growth_side, (cell.growth_side + 3) % 6].into_iter() {
            *cell.mechanics.cell_boundary_lengths.get_mut(n).unwrap() += dt * cell.growth_factor;
        }
        let height = 2.0 * cell.mechanics.calculate_current_boundary_length() / 6.0
            * (std::f64::consts::PI / 3.0).sin();
        let new_area = area + dt * cell.growth_factor * height;
        cell.mechanics.cell_area = new_area;
        if new_area > cell.division_threshold_area {
            Some(CycleEvent::Division)
        } else {
            println!("{} {}", new_area, cell.division_threshold_area);
            None
        }
    }

    fn divide(_rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
        println!("Divide");
        let n1 = cell.growth_side;
        let n2 = (cell.growth_side + 3) % 6;

        // Get the growth edges and thus calculate the 3 new vertices
        let pos = cell.pos();
        let p11 = pos.row(n1);
        let p12 = pos.row((n1 + 1) % 6);
        let p21 = pos.row(n2);
        let p22 = pos.row((n2 + 1) % 6);
        let p1 = 0.5 * (p11 + p12);
        let p2 = 0.5 * (p21 + p22);
        let p3 = 0.5 * (p1 + p2);
        println!("{}", pos);
        println!("{} {} {}", p1, p3, p2);

        let mut new_pos1 = pos.clone();
        new_pos1.set_row((n1 + 1) % 6, &p1);
        new_pos1.set_row(n2, &p2);
        new_pos1.set_row((n1 + 2) % 6, &p3);
        println!("{new_pos1}");

        let mut new_pos2 = pos.clone();
        new_pos2.set_row(n1, &p1);
        new_pos2.set_row((n2 + 1) % 6, &p2);
        new_pos2.set_row((n2 + 2) % 6, &p3);
        println!("{new_pos2}");

        let area = cell.mechanics.get_current_cell_area();
        let new_area = area / 2.0;
        let new_boundary_length = VertexMechanics2D::<6>::calculate_boundary_length(new_area);
        println!(
            "{} {} {}",
            cell.division_threshold_area, new_area, new_boundary_length
        );
        let c1 = cell;
        c1.set_pos(&new_pos1);
        c1.mechanics.cell_area = new_area;
        c1.mechanics
            .cell_boundary_lengths
            .iter_mut()
            .for_each(|d| *d = new_boundary_length);
        let mut c2 = c1.clone();
        c2.set_pos(&new_pos2);
        Ok(c2)
    }
}*/
