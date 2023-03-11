use crate::concepts::errors::CalcError;
use crate::concepts::mechanics::Mechanics;

use itertools::Itertools;
use nalgebra::SVector;

use serde::{Serialize,Deserialize};


macro_rules! implement_mechanics_model_nd(
    ($model_name:ident, $dim:literal) => {
        #[derive(Clone,Debug,Serialize,Deserialize)]
        pub struct $model_name {
            pub pos: SVector<f64, $dim>,
            pub vel: SVector<f64, $dim>,
            pub dampening_constant: f64,
        }

        impl Mechanics<SVector<f64, $dim>, SVector<f64, $dim>, SVector<f64, $dim>> for $model_name {
            fn pos(&self) -> SVector<f64, $dim> {
                self.pos
            }

            fn velocity(&self) -> SVector<f64, $dim> {
                self.vel
            }

            fn set_pos(&mut self, p: &SVector<f64, $dim>) {
                self.pos = *p;
            }

            fn set_velocity(&mut self, v: &SVector<f64, $dim>) {
                self.vel = *v;
            }

            fn calculate_increment(&self, force: SVector<f64, $dim>) -> Result<(SVector<f64, $dim>, SVector<f64, $dim>), CalcError> {
                let dx = self.vel;
                let dv = force - self.dampening_constant * self.vel;
                Ok((dx, dv))
            }
        }
    }
);


implement_mechanics_model_nd!(MechanicsModel1D, 1);
implement_mechanics_model_nd!(MechanicsModel2D, 2);
implement_mechanics_model_nd!(MechanicsModel3D, 3);


#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct VertexMechanics2D<const D: usize> {
    points: nalgebra::SMatrix<f64, D, 2>,
    velocity: nalgebra::SMatrix<f64, D, 2>,
    cell_boundary_lengths: nalgebra::SVector<f64, D>,
    spring_tensions: nalgebra::SVector<f64, D>,
    cell_area: f64,
    central_pressure: f64,
    dampening_constant: f64,
}

pub type VertexVector2<const D: usize> = nalgebra::SMatrix<f64, D, 2>;
pub type VertexConnections2<const D: usize> = nalgebra::SVector<f64, D>;


impl<const D: usize> VertexMechanics2D<D> {
    /// Creates a new vertex model in equilibrium conditions.
    ///
    /// The specified parameters are then used to carefully calculate relating properties of the model.
    /// We outline the formulas used.
    /// Given the number of vertices \\(N\\) in our model (specified by the const generic argument of the [VertexMechanics2D] struct),
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
    /// Pressure, dampening and spring tensions are not impacted by randomization.
    pub fn new(
        middle: SVector<f64, 2>,
        cell_area: f64,
        rotation_angle: f64,
        spring_tensions: f64,
        central_pressure: f64,
        dampening_constant: f64,
        randomize: Option<(f64, rand_chacha::ChaCha8Rng)>,
    ) -> Self {
        use rand::Rng;
        // Restrict the randomize variable between 0 and 1
        let r = match randomize {
            Some((rand, _)) => rand.clamp(0.0, 1.0),
            _ => 0.0,
        };
        let rng = || -> f64 {match randomize {
            Some((_, mut rng)) => rng.gen_range(0.0..1.0),
            None => 0.0,
        }};
        // Randomize the overall rotation angle
        let rotation_angle = (1.0-r*rng.clone()())*rotation_angle;
        // Calculate the angle fraction used to determine the points of the polygon
        let angle_fraction = std::f64::consts::PI/D as f64;
        // Calculate the radius from cell area
        let radius = (cell_area/D as f64/angle_fraction.sin()/angle_fraction.cos()).sqrt();
        // TODO this needs to be calculated again
        let points = VertexVector2::<D>::from_row_iterator(
            (0..D).map(|n| {
                let angle = rotation_angle + 2.0 * angle_fraction * n as f64 * (1.0 - r*rng.clone()());
                let radius_modified = radius*(1.0 + 0.5*r*(1.0 - rng.clone()()));
                [
                    middle.x + radius_modified*angle.cos(),
                    middle.y + radius_modified*angle.sin()
                ].into_iter()
            }).flatten()
        );
        // Randomize the boundary lengths
        let cell_boundary_lengths = VertexConnections2::<D>::from_iterator(points.row_iter().circular_tuple_windows().map(|(p1, p2)| (p1-p2).norm()));
        VertexMechanics2D {
            points,
            velocity: VertexVector2::<D>::zeros(),
            cell_boundary_lengths,
            spring_tensions: VertexConnections2::<D>::from_element(spring_tensions),
            cell_area,
            central_pressure,
            dampening_constant,
        }
    }

    pub fn set_cell_area(&mut self, cell_area: f64) {
        // Calculate the relative difference to current area
        match self.cell_area {
            a if a==0.0 => {
                let new_interaction_parameters = Self::new(
                    self.points.row_iter().map(|v| v.transpose()).sum::<nalgebra::Vector2<f64>>(),
                    cell_area, 0.0,
                    self.spring_tensions.sum()/self.spring_tensions.len() as f64,
                    self.central_pressure,
                    self.dampening_constant,
                    None
                );
                *self = new_interaction_parameters;
            },
            _ => {
                let relative_difference = cell_area.abs()/self.cell_area.abs();
                // Calculate the new length of the cell boundary lengths
                self.cell_boundary_lengths.iter_mut().for_each(|length| *length *= relative_difference);
            }
        };
    }
}


impl<const D: usize> Mechanics<VertexVector2<D>, VertexVector2<D>, VertexVector2<D>> for VertexMechanics2D<D>
{
    fn pos(&self) -> VertexVector2<D> {
        self.points.clone()
    }

    fn velocity(&self) -> VertexVector2<D> {
        self.velocity.clone()
    }

    fn set_pos(&mut self, pos: &VertexVector2<D>) {
        self.points = pos.clone();
    }

    fn set_velocity(&mut self, velocity: &VertexVector2<D>) {
        self.velocity = velocity.clone();
    }

    fn calculate_increment(&self, force: VertexVector2<D>) -> Result<(VertexVector2<D>, VertexVector2<D>), CalcError> {
        // Calculate the total internal force
        let middle = self.points.row_sum()/self.points.shape().0 as f64;
        let current_ara: f64 = 0.5_f64*self.points.row_iter()
            .circular_tuple_windows()
            .map(|(p1, p2)| p1.transpose().perp(&p2.transpose()))
            .sum::<f64>();

        let mut internal_force = self.points.clone()*0.0;
        for (index, (point_1, point_2, point_3)) in self.points.row_iter().circular_tuple_windows::<(_,_,_)>().enumerate() {
            let tension_12 = self.spring_tensions[index];
            let tension_23 = self.spring_tensions[(index+1) % self.spring_tensions.len()];
            let mut force_2 = internal_force.row_mut((index+1) % self.points.shape().0);

            // Calculate forces arising from springs between points
            let p_21 = point_2-point_1;
            let p_23 = point_2-point_3;
            let force1 = p_21.normalize()*(self.cell_boundary_lengths[index] - p_21.norm()) * tension_12;
            let force2 = p_23.normalize()*(self.cell_boundary_lengths[(index+1) % self.cell_boundary_lengths.len()] - p_23.norm()) * tension_23;

            // Calculate force arising from internal pressure
            let middle_to_point_2 = point_2 - middle;
            let force3 = middle_to_point_2.normalize() * (self.cell_area - current_ara) * self.central_pressure;

            // Combine forces
            force_2 += force1 + force2 + force3;
        }
        let dx = self.velocity.clone();
        let dv = force + internal_force - self.dampening_constant * self.velocity.clone();
        Ok((dx, dv))
    }
}
