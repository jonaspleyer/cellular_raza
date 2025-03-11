use core::f32;

use cellular_raza::prelude::*;
use nalgebra::{SVector, Vector2};
use serde::{Deserialize, Serialize};

pub const N_REACTIONS: usize = 1;
pub type ReactionVector = nalgebra::SVector<f32, N_REACTIONS>;

#[derive(Serialize, Deserialize, Clone, core::fmt::Debug)]
pub struct MyInteraction {
    pub potential_strength: f32,
    pub exponent: f32,
    pub cell_radius: f32,
}

impl Interaction<Vector2<f32>, Vector2<f32>, Vector2<f32>, f32> for MyInteraction {
    fn calculate_force_between(
        &self,
        own_pos: &Vector2<f32>,
        _own_vel: &Vector2<f32>,
        ext_pos: &Vector2<f32>,
        _ext_vel: &Vector2<f32>,
        ext_radius: &f32,
    ) -> Result<(Vector2<f32>, Vector2<f32>), CalcError> {
        let z = own_pos - ext_pos;
        let r = z.norm();
        if r == 0.0 {
            return Ok((num::Zero::zero(), num::Zero::zero()));
        }

        let sigma = r / (self.cell_radius + ext_radius);
        let strength = if sigma <= 1.0 {
            self.potential_strength
        } else {
            0.0
        };
        let force = strength * z.normalize() / sigma.powf(self.exponent);
        Ok((force, -force))
    }

    fn get_interaction_information(&self) -> f32 {
        self.cell_radius
    }
}

impl Cycle<MyAgent, f32> for MyAgent {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        _dt: &f32,
        cell: &mut MyAgent,
    ) -> Option<CycleEvent> {
        // If the cell is not at the maximum size let it grow
        if cell.interaction.cell_radius > cell.division_radius {
            return Some(CycleEvent::Division);
        }
        None
    }

    fn divide(
        rng: &mut rand_chacha::ChaCha8Rng,
        c1: &mut MyAgent,
    ) -> Result<MyAgent, DivisionError> {
        // Clone existing cell
        let mut c2 = c1.clone();

        let r = c1.interaction.cell_radius;

        // Make both cells smaller
        // Also keep old cell larger
        c1.interaction.cell_radius /= std::f32::consts::SQRT_2;
        c2.interaction.cell_radius /= std::f32::consts::SQRT_2;

        // Generate cellular splitting direction randomly
        use rand::Rng;
        let angle_1 = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
        let dir_vec = nalgebra::Rotation2::new(angle_1) * nalgebra::Vector2::from([1.0, 0.0]);

        // Define new positions for cells
        // It is randomly chosen if the old cell is left or right
        let offset = dir_vec * r / std::f32::consts::SQRT_2;
        let old_pos = c1.pos();

        c1.set_pos(&(old_pos + offset));
        c2.set_pos(&(old_pos - offset));

        Ok(c2)
    }
}

// COMPONENT DESCRIPTION
// 0         CELL RADIUS
impl Intracellular<ReactionVector> for MyAgent {
    fn set_intracellular(&mut self, intracellular: ReactionVector) {
        self.interaction.cell_radius = intracellular[0];
    }

    fn get_intracellular(&self) -> ReactionVector {
        [self.interaction.cell_radius].into()
    }
}

impl ReactionsExtra<ReactionVector, ReactionVector> for MyAgent {
    fn calculate_combined_increment(
        &self,
        _intracellular: &ReactionVector,
        extracellular: &ReactionVector,
    ) -> Result<(ReactionVector, ReactionVector), CalcError> {
        let extra = extracellular;
        let u = self.uptake_rate;

        let uptake = u * extra;

        let incr_intra: ReactionVector = [self.growth_rate * uptake[0]].into();
        let incr_extra = -uptake;

        Ok((incr_intra, incr_extra))
    }
}

#[derive(Clone, Serialize, Deserialize, CellAgent)]
pub struct MyAgent {
    #[Mechanics]
    pub mechanics: NewtonDamped2DF32,
    #[Interaction]
    pub interaction: MyInteraction,
    pub uptake_rate: f32,
    pub division_radius: f32,
    pub growth_rate: f32,
}
