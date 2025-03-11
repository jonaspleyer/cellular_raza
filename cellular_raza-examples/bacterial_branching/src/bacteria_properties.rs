use core::f32::consts::{PI, SQRT_2};

use cellular_raza::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ReactionVector;

#[derive(Clone, Serialize, Deserialize, CellAgent)]
pub struct MyAgent {
    #[Mechanics]
    pub mechanics: NewtonDamped2DF32,
    #[Interaction]
    pub interaction: MorsePotentialF32,
    pub uptake_rate: f32,
    pub division_radius: f32,
    pub growth_rate: f32,
}

impl Cycle<MyAgent, f32> for MyAgent {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        _dt: &f32,
        cell: &mut MyAgent,
    ) -> Option<CycleEvent> {
        // If the cell is not at the maximum size let it grow
        if cell.interaction.radius > cell.division_radius {
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

        let r = c1.interaction.radius;

        // Make both cells smaller
        // Also keep old cell larger
        c1.interaction.radius /= SQRT_2;
        c2.interaction.radius /= SQRT_2;

        // Generate cellular splitting direction randomly
        use rand::Rng;
        let alpha = rng.gen_range(0.0..2.0 * PI);
        let dir_vec = nalgebra::Vector2::from([alpha.cos(), alpha.sin()]);

        // Define new positions for cells
        // It is randomly chosen if the old cell is left or right
        let offset = dir_vec * r / SQRT_2;
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
        self.interaction.radius = (intracellular[0] / PI).powf(0.5);
    }

    fn get_intracellular(&self) -> ReactionVector {
        [PI * self.interaction.radius.powf(2.0)].into()
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
