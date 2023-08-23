use cellular_raza::prelude::*;

use nalgebra::Vector2;
use num::Zero;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const NUMBER_OF_REACTION_COMPONENTS: usize = 1;
pub type ReactionVector = nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>;
pub type MyCellType =
    ModularCell<MechanicsModel2D, CellSpecificInteraction, OwnCycle, OwnReactions, GradientSensing>;

#[derive(Serialize, Deserialize, Clone, core::fmt::Debug)]
pub struct CellSpecificInteraction {
    pub potential_strength: f64,
    pub relative_interaction_range: f64,
    pub cell_radius: f64,
}

impl Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, f64> for CellSpecificInteraction {
    fn get_interaction_information(&self) -> f64 {
        self.cell_radius
    }

    fn calculate_force_between(
        &self,
        own_pos: &Vector2<f64>,
        _own_vel: &Vector2<f64>,
        ext_pos: &Vector2<f64>,
        _ext_vel: &Vector2<f64>,
        ext_radius: &f64,
    ) -> Option<Result<Vector2<f64>, CalcError>> {
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
                            return None;
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

        Some(Ok(repelling_force + attracting_force))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OwnCycle {
    age: f64,
    pub division_age: f64,
    pub maximum_cell_radius: f64,
    pub growth_rate: f64,
    pub food_threshold: f64,
    food_growth_rate_multiplier: f64,
    food_division_threshold: f64,
}

impl OwnCycle {
    pub fn new(
        division_age: f64,
        maximum_cell_radius: f64,
        growth_rate: f64,
        food_threshold: f64,
        food_growth_rate_multiplier: f64,
        food_division_threshold: f64,
    ) -> Self {
        OwnCycle {
            age: 0.0,
            division_age,
            maximum_cell_radius,
            growth_rate,
            food_threshold,
            food_growth_rate_multiplier,
            food_division_threshold,
        }
    }
}

impl Cycle<MyCellType> for OwnCycle {
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut MyCellType,
    ) -> Option<CycleEvent> {
        // If the cell is not at the maximum size let it grow
        if cell.interaction.cell_radius < cell.cycle.maximum_cell_radius {
            let growth_difference = (cell.cycle.maximum_cell_radius * cell.cycle.growth_rate * dt)
                .min(cell.cycle.maximum_cell_radius - cell.interaction.cell_radius);
            cell.cellular_reactions.intracellular_concentrations[0] -=
                cell.cycle.food_growth_rate_multiplier * growth_difference
                    / cell.cycle.maximum_cell_radius;
            cell.interaction.cell_radius += growth_difference;
        }

        // Increase the age of the cell and divide if possible
        cell.cycle.age += dt;

        // Calculate the modifier (between 0.0 and 1.0) based on food threshold
        let relative_division_food_level = ((cell.get_intracellular()[0]
            - cell.cycle.food_division_threshold)
            / (cell.cycle.food_threshold - cell.cycle.food_division_threshold))
            .clamp(0.0, 1.0);

        if
        // Check if the cell has aged enough
        cell.cycle.age > cell.cycle.division_age &&
            // Check if the cell has grown enough
            cell.interaction.cell_radius >= cell.cycle.maximum_cell_radius &&
            // Random selection but chance increased when significantly above the food threshold
            rng.gen_range(0.0..1.0) < relative_division_food_level
        {
            return Some(CycleEvent::Division);
        }
        None
    }

    fn divide(
        rng: &mut rand_chacha::ChaCha8Rng,
        c1: &mut MyCellType,
    ) -> Result<MyCellType, DivisionError> {
        // Clone existing cell
        let mut c2 = c1.clone();

        let r = c1.interaction.cell_radius;

        // Make both cells smaller
        // ALso keep old cell larger
        let relative_size_difference = 0.2;
        c1.interaction.cell_radius *= (1.0 + relative_size_difference) / std::f64::consts::SQRT_2;
        c2.interaction.cell_radius *= (1.0 - relative_size_difference) / std::f64::consts::SQRT_2;

        // Generate cellular splitting direction randomly
        let angle_1 = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let dir_vec = nalgebra::Rotation2::new(angle_1) * Vector2::from([1.0, 0.0]);

        // Define new positions for cells
        // It is randomly chosen if the old cell is left or right
        let offset = dir_vec * r / std::f64::consts::SQRT_2;
        let old_pos = c1.pos();

        c1.set_pos(&(old_pos + offset));
        c2.set_pos(&(old_pos - offset));

        // Decrease the amount of food in the cells
        c1.cellular_reactions.intracellular_concentrations *=
            (1.0 + relative_size_difference) * 0.5;
        c2.cellular_reactions.intracellular_concentrations *=
            (1.0 - relative_size_difference) * 0.5;

        // New cell is completely new so set age to 0
        c2.cycle.age = 0.0;

        Ok(c2)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OwnReactions {
    pub intracellular_concentrations: ReactionVector,
    pub turnover_rate: ReactionVector,
    pub production_term: ReactionVector,
    pub degradation_rate: ReactionVector,
    pub secretion_rate: ReactionVector,
    pub uptake_rate: ReactionVector,
}

impl CellularReactions<ReactionVector> for OwnReactions {
    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        internal_concentration_vector: &ReactionVector,
        external_concentration_vector: &ReactionVector,
    ) -> Result<(ReactionVector, ReactionVector), CalcError> {
        let mut increment_extracellular = ReactionVector::zero();
        let mut increment_intracellular = ReactionVector::zero();

        for i in 0..NUMBER_OF_REACTION_COMPONENTS {
            let uptake = self.uptake_rate[i] * external_concentration_vector[i];
            let secretion = self.secretion_rate[i] * internal_concentration_vector[i];
            increment_extracellular[i] = secretion - uptake;
            increment_intracellular[i] = self.production_term[i]
                - increment_extracellular[i]
                - self.turnover_rate[i] * internal_concentration_vector[i];
        }

        Ok((increment_intracellular, increment_extracellular))
    }

    fn get_intracellular(&self) -> ReactionVector {
        self.intracellular_concentrations
    }

    fn set_intracellular(&mut self, concentration_vector: ReactionVector) {
        self.intracellular_concentrations = concentration_vector;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientSensing {}

impl
    InteractionExtracellularGradient<
        MyCellType,
        nalgebra::SVector<Vector2<f64>, NUMBER_OF_REACTION_COMPONENTS>,
    > for GradientSensing
{
    fn sense_gradient(
        _cell: &mut MyCellType,
        _gradient: &nalgebra::SVector<Vector2<f64>, NUMBER_OF_REACTION_COMPONENTS>,
    ) -> Result<(), CalcError> {
        Ok(())
    }
}
