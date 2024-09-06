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
    pub growth_rate: f64,
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
        if r == 0.0 {
            return Ok(([0f64; 2].into(), [0f64; 2].into()));
        }
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

impl<const D: usize> Cycle for MyCell<D> {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Self,
    ) -> Option<CycleEvent> {
        let a = cell.mechanics.get_cell_area();
        cell.mechanics.set_cell_area(a + dt * cell.growth_rate);
        None
    }

    fn divide(_rng: &mut rand_chacha::ChaCha8Rng, _cell: &mut Self) -> Result<Self, DivisionError> {
        todo!()
    }
}
