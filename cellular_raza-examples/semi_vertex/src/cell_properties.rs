use cellular_raza::prelude::*;

use nalgebra::{Unit, Vector2};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, core::fmt::Debug)]
pub struct DirectedSphericalMechanics {
    pub pos: Vector2<f64>,
    pub vel: Vector2<f64>,
    pub orientation: Unit<Vector2<f64>>,
}

#[derive(Serialize, Deserialize, Clone, core::fmt::Debug)]
pub struct OutsideInteraction {
    pub potential_strength: f64,
    pub interaction_range: f64,
}

#[derive(Serialize, Deserialize, Clone, core::fmt::Debug)]
pub struct InsideInteraction {
    pub potential_strength: f64,
    pub average_radius: f64,
}

#[derive(Serialize, Deserialize, CellAgent, Clone, core::fmt::Debug)]
pub struct MyCell<const D: usize> {
    #[Mechanics]
    pub mechanics: VertexMechanics2D<D>,
    #[Interaction]
    pub interaction: VertexDerivedInteraction<OutsideInteraction, InsideInteraction>,
    pub intracellular: Vector2<f64>,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub exchange: Vector2<f64>,
}

impl<const D: usize> ReactionsContact<Vector2<f64>, nalgebra::SMatrix<f64, D, 2>> for MyCell<D> {
    fn get_contact_information(&self) -> () {}
    fn calculate_contact_increment(
        &self,
        own_intracellular: &Vector2<f64>,
        ext_intracellular: &Vector2<f64>,
        own_pos: &nalgebra::SMatrix<f64, D, 2>,
        ext_pos: &nalgebra::SMatrix<f64, D, 2>,
        _rinf: &(),
    ) -> Result<(Vector2<f64>, Vector2<f64>), CalcError> {
        use itertools::Itertools;
        // Calculate overlap of borders
        let mut incr = Vector2::zeros();
        for (own_l1, own_l2) in own_pos.row_iter().circular_tuple_windows::<(_, _)>() {
            for (ext_l1, ext_l2) in ext_pos.row_iter().circular_tuple_windows() {
                let middle_own = 0.5 * (own_l1 + own_l2);
                let middle_ext = 0.5 * (ext_l1 + ext_l2);
                if (middle_own - middle_ext).norm()
                    < self.interaction.outside_interaction.interaction_range * 0.5
                {
                    incr.x += self.exchange.x * (ext_intracellular - own_intracellular).x;
                    incr.y += self.exchange.y * (ext_intracellular - own_intracellular).y;
                }
            }
        }
        Ok((incr, -incr))
    }
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
        let sigma = r / (self.interaction_range);
        let spatial_cutoff = if r > self.interaction_range { 0.0 } else { 1.0 };

        // Calculate the strength of the interaction with correct bounds
        let strength = self.potential_strength * (1.0 - sigma);

        // Calculate only attracting and repelling forces
        let force = -dir * strength * spatial_cutoff;
        Ok((-force, force))
    }

    fn get_interaction_information(&self) -> () {}
}

impl Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>>
    for InsideInteraction
{
    fn calculate_force_between(
        &self,
        own_pos: &Vector2<f64>,
        _own_vel: &Vector2<f64>,
        ext_pos: &Vector2<f64>,
        _ext_vel: &Vector2<f64>,
        _ext_info: &(),
    ) -> Result<(Vector2<f64>, Vector2<f64>), CalcError> {
        // Calculate direction between own and other point
        let z = ext_pos - own_pos;
        let r = z.norm();
        let dir = z.normalize();

        let force = self.potential_strength * dir / (0.5 + 0.5 * r / self.average_radius);
        Ok((-force, force))
    }

    fn get_interaction_information(&self) -> () {}
}

impl<const N: usize> Intracellular<Vector2<f64>> for MyCell<N> {
    fn get_intracellular(&self) -> Vector2<f64> {
        self.intracellular
    }

    fn set_intracellular(&mut self, intracellular: Vector2<f64>) {
        self.intracellular = intracellular;
    }
}

impl<const N: usize> Reactions<Vector2<f64>> for MyCell<N> {
    fn calculate_intracellular_increment(
        &self,
        intracellular: &Vector2<f64>,
    ) -> Result<Vector2<f64>, CalcError> {
        let a = intracellular.x;
        let s = intracellular.y;
        let da = self.k1 * s * a.powf(2.0) - self.k2 * a;
        let ds = -self.k1 * s * a.powf(2.0) + self.k3;
        println!("{:10.7} {:10.7} {:10.7} {:10.7}", a, s, da, ds);
        Ok([da, ds].into())
    }
}
