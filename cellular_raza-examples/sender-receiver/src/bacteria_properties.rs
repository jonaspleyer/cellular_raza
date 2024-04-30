use cellular_raza::prelude::*;

use serde::{Deserialize, Serialize};

pub const NUMBER_OF_REACTION_COMPONENTS: usize = 1;
pub type ReactionVector = nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>;
pub type MyCellType = ModularCell<
    NewtonDamped2D,
    MyInteraction,
    NoCycle,
    OwnReactions,
    NoExtracellularGradientSensing,
>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Species {
    Sender,
    Receiver,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OwnReactions {
    pub intracellular: ReactionVector,
    pub species: Species,
    pub sink_rate: ReactionVector,
    pub production_term: ReactionVector,
    pub uptake: ReactionVector,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MyInteraction {
    pub cell_radius: f64,
    pub potential_strength: f64,
    pub relative_interaction_range: f64,
}

impl Interaction<nalgebra::Vector2<f64>, nalgebra::Vector2<f64>, nalgebra::Vector2<f64>, f64>
    for MyInteraction
{
    fn calculate_force_between(
        &self,
        own_pos: &nalgebra::Vector2<f64>,
        _own_vel: &nalgebra::Vector2<f64>,
        ext_pos: &nalgebra::Vector2<f64>,
        _ext_vel: &nalgebra::Vector2<f64>,
        ext_radius: &f64,
    ) -> Result<nalgebra::Vector2<f64>, CalcError> {
        let min_relative_distance_to_center = 0.3162277660168379;
        let (r, dir) =
            match (own_pos - ext_pos).norm() < self.cell_radius * min_relative_distance_to_center {
                false => {
                    let z = own_pos - ext_pos;
                    let r = z.norm();
                    (r, z.normalize())
                }
                true => {
                    let dir = match own_pos == ext_pos {
                        true => {
                            return Ok([0.0; 2].into());
                        }
                        false => (own_pos - ext_pos).normalize(),
                    };
                    let r = self.cell_radius * min_relative_distance_to_center;
                    (r, dir)
                }
            };
        // Introduce Non-dimensional length variable
        let sigma = r / (self.cell_radius + ext_radius);
        let bound = 4.0 + 1.0 / sigma;
        let spatial_cutoff = (1.0
            + (self.relative_interaction_range * (self.cell_radius + ext_radius) - r).signum())
            * 0.5;

        // Calculate the strength of the interaction with correct bounds
        let strength = self.potential_strength
            * ((1.0 / sigma).powf(2.0) - (1.0 / sigma).powf(4.0))
                .min(bound)
                .max(-bound);

        // Calculate only attracting and repelling forces
        let attracting_force = dir * strength.max(0.0) * spatial_cutoff;
        let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

        Ok(repelling_force + attracting_force)
    }

    fn get_interaction_information(&self) -> f64 {
        self.cell_radius
    }
}

impl CellularReactions<ReactionVector> for OwnReactions {
    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        i: &ReactionVector,
        e: &ReactionVector,
    ) -> Result<(ReactionVector, ReactionVector), CalcError> {
        Ok(match self.species {
            Species::Sender => ([0.0].into(), self.production_term),
            Species::Receiver => (
                self.uptake * (e - i) - self.sink_rate * i,
                -self.uptake * (e - i),
            ),
        })
    }

    fn get_intracellular(&self) -> ReactionVector {
        self.intracellular
    }
    fn set_intracellular(&mut self, c: ReactionVector) {
        self.intracellular = c;
    }
}
