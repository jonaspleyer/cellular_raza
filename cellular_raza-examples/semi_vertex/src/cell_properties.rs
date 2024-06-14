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
    pub growth_factor: f64,
    pub division_area_threshold: f64,
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
        let z = ext_pos - own_pos;
        let r = z.norm();
        let dir = z.normalize();

        let force = self.potential_strength * dir / (0.5 + 0.5 * r / self.average_radius);
        Ok((-force, force))
    }

    fn get_interaction_information(&self) -> () {}
}

impl<const D: usize> Cycle<MyCell<D>> for MyCell<D> {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut MyCell<D>,
    ) -> Option<CycleEvent> {
        let area = cell.mechanics.get_cell_area();
        let new_area = area + dt * cell.growth_factor;
        cell.mechanics.set_cell_area_and_boundary_length(new_area);
        if new_area > cell.division_area_threshold {
            Some(CycleEvent::Division)
        } else {
            None
        }
    }

    fn divide(
        rng: &mut rand_chacha::ChaCha8Rng,
        cell1: &mut MyCell<D>,
    ) -> Result<MyCell<D>, DivisionError> {
        // Pick one vertex at random
        use rand::Rng;
        let n_vertex_start: usize = rng.gen_range(0..D);
        let cell_pos = cell1.mechanics.pos();
        let pos_start = cell_pos.row(n_vertex_start);
        // Now we determine the new positions of the cells.
        let (new_pos1, new_pos2) = if D % 2 == 0 {
            // If we have an even number of vertices, the mechanics model will look something like
            // this:
            //   ___
            //  / | \
            // /  |  \
            // \  |  /
            //  \_|_/
            // Here, we introduce 1 new vertex on the cell-division axis and two new vertices on
            // the upper and lower horizontal edges.
            let n_opposite_pos = (n_vertex_start + D.div_euclid(2)) % D;
            let opposite_pos = cell_pos.row(n_opposite_pos);
            let mut new_pos1 = cell_pos.clone();
            let mut new_pos2 = cell_pos.clone();
            for i in 1..D.div_euclid(2) {
                let q = i as f64 / D as f64;
                let p_next = (1.0 - q) * pos_start + q * opposite_pos;
                new_pos1.set_row((n_vertex_start + i) % D, &p_next);
                new_pos2.set_row((n_vertex_start + D - i) % D, &p_next);
            }
            (new_pos1, new_pos2)
        } else {
            // In the case of uneven number of vertices, it is much harder to make a nice ascii
            // plot.
            // In this case, we have to introduce a new vertex on the old boundary.
            //    ___
            //   / | \
            //  /  |  \
            //  \  |  /
            //   \ | /
            //    \|/
            // The cell-division axis has one vertex right in the middle and another one which is
            // newly introduced on the edge (plot upper middle).
            let mut new_pos1 = cell_pos.clone();
            let mut new_pos2 = cell_pos.clone();
            (new_pos1, new_pos2)
        };
        cell1
            .mechanics
            .set_cell_area(0.5 * cell1.mechanics.get_cell_area());
        let mut cell2 = cell1.clone();
        cell1.mechanics.set_pos(&new_pos1);
        cell2.mechanics.set_pos(&new_pos2);
        Ok(cell2)
    }
}
