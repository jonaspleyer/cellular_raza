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
    pub intracellular: nalgebra::Vector3<f64>,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub k4: f64,
    pub k5: f64,
    pub contact_range: f64,
}

impl<const D: usize> ReactionsContact<nalgebra::Vector3<f64>, nalgebra::SMatrix<f64, D, 2>>
    for MyCell<D>
{
    fn get_contact_information(&self) -> () {}

    fn calculate_contact_increment(
        &self,
        own_intracellular: &nalgebra::Vector3<f64>,
        ext_intracellular: &nalgebra::Vector3<f64>,
        own_pos: &nalgebra::SMatrix<f64, D, 2>,
        ext_pos: &nalgebra::SMatrix<f64, D, 2>,
        _rinf: &(),
    ) -> Result<(nalgebra::Vector3<f64>, nalgebra::Vector3<f64>), CalcError> {
        let middle_own = own_pos.row_mean();
        let middle_ext = ext_pos.row_mean();
        let r = (middle_own - middle_ext).norm();
        if r < self.contact_range {
            let incr = [
                self.k3 * (ext_intracellular - own_intracellular)[0],
                0.0,
                0.0,
            ]
            .into();
            return Ok((incr, -incr));
        }
        let incr = [0.0; 3].into();
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

impl<const N: usize> Intracellular<nalgebra::Vector3<f64>> for MyCell<N> {
    fn get_intracellular(&self) -> nalgebra::Vector3<f64> {
        self.intracellular
    }

    fn set_intracellular(&mut self, intracellular: nalgebra::Vector3<f64>) {
        self.intracellular = intracellular;
    }
}

impl<const N: usize> Reactions<nalgebra::Vector3<f64>> for MyCell<N> {
    fn calculate_intracellular_increment(
        &self,
        intracellular: &nalgebra::Vector3<f64>,
    ) -> Result<nalgebra::Vector3<f64>, CalcError> {
        let ttgl = intracellular[0];
        let gl3 = intracellular[1];
        let ac = intracellular[2];
        let dttgl = self.k1 - self.k2 * ttgl - ttgl * gl3;
        let dgl3 = self.k4 * ac.powf(2.0) - self.k5 * gl3 - ttgl * gl3;
        let dac = ttgl * gl3 - ac;
        Ok([dttgl, dgl3, dac].into())
    }
}
