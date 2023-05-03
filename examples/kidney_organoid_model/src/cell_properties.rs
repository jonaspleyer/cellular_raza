use cellular_raza::backend::cpu_os_threads::prelude::*;
use cellular_raza::implementations::cell_properties::mechanics::VertexMechanics2D;

use nalgebra::{Unit, Vector2};
use serde::{Deserialize, Serialize};

use rand::Rng;

pub const NUMBER_OF_REACTION_COMPONENTS: usize = 4;
pub const NUMBER_OF_VERTICES: usize = 4;
pub type ReactionVector = nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>;
pub type InteractionInformation = ();
pub type MyCellType = ModularCell<
    VertexMechanics2D<NUMBER_OF_VERTICES>,
    VertexDerivedInteraction<OutsideInteraction, InsideInteraction>,
    OwnCycle,
    OwnReactions,
    GradientSensing,
>;

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
}

impl Interaction<Vector2<f64>, Vector2<f64>, InteractionInformation> for OutsideInteraction {
    fn calculate_force_on(
        &self,
        own_pos: &Vector2<f64>,
        ext_pos: &Vector2<f64>,
        _ext_info: &Option<InteractionInformation>,
    ) -> Option<Result<Vector2<f64>, CalcError>> {
        // Calculate distance and direction between own and other point
        let z = ext_pos - own_pos;
        let r = z.norm();
        let dir = z / r;

        // Introduce Non-dimensional length variable
        let sigma = r / (self.interaction_range);
        let spatial_cutoff = if r > self.interaction_range { 0.0 } else { 1.0 };

        // Calculate the strength of the interaction with correct bounds
        let strength = self.potential_strength * (1.0 - sigma);

        // Calculate only attracting and repelling forces
        let force = -dir * strength * spatial_cutoff;
        Some(Ok(force))
    }
}

impl Interaction<Vector2<f64>, Vector2<f64>, InteractionInformation> for InsideInteraction {
    fn calculate_force_on(
        &self,
        own_pos: &Vector2<f64>,
        ext_pos: &Vector2<f64>,
        _ext_info: &Option<InteractionInformation>,
    ) -> Option<Result<Vector2<f64>, CalcError>> {
        // Calculate direction between own and other point
        let z = ext_pos - own_pos;
        let dir = z.normalize();

        Some(Ok(self.potential_strength * dir))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OwnCycle {
    pub id: u64,
    age: f64,
    pub division_age: f64,
    divisions: u8,
    generation: u8,
    pub maximum_cell_area: f64,
    pub growth_rate: f64,
    food_growth_rate_multiplier: f64,
    food_death_threshold: f64,
    food_division_threshold: f64,
}

impl OwnCycle {
    pub fn new(
        id: u64,
        division_age: f64,
        maximum_cell_area: f64,
        growth_rate: f64,
        food_growth_rate_multiplier: f64,
        food_death_threshold: f64,
        food_division_threshold: f64,
    ) -> Self {
        OwnCycle {
            id,
            age: 0.0,
            division_age,
            divisions: 0,
            generation: 0,
            maximum_cell_area,
            growth_rate,
            food_growth_rate_multiplier,
            food_death_threshold,
            food_division_threshold,
        }
    }
}

impl Cycle<MyCellType> for OwnCycle {
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        c: &mut MyCellType,
    ) -> Option<CycleEvent> {
        // If the cell is not at the maximum size let it grow
        if c.mechanics.get_cell_area() < c.cycle.maximum_cell_area {
            let growth_difference = (c.cycle.maximum_cell_area * c.cycle.growth_rate * dt)
                .min(c.cycle.maximum_cell_area - c.mechanics.get_cell_area())
                .max(0.0);
            c.cellular_reactions.intracellular_concentrations[1] -=
                c.cycle.food_growth_rate_multiplier * growth_difference / c.cycle.maximum_cell_area;
            c.mechanics
                .set_cell_area(c.mechanics.get_cell_area() + growth_difference);
        }

        // Increase the age of the cell and divide if possible
        c.cycle.age += dt;

        // Calculate the modifier (between 0.0 and 1.0) based on food threshold
        /* let relative_division_food_level = (
            (c.get_intracellular()[1]-c.cycle.food_division_threshold)
            /(c.cellular_reactions.intracellular_concentrations_saturation_level[1]-c.cycle.food_division_threshold)
        ).clamp(0.0, 1.0);

        if
            // Check if the cell has aged enough
            c.cycle.age > c.cycle.division_age &&
            // Check if the cell has grown enough
            c.mechanics.get_cell_area() >= c.cycle.maximum_cell_area &&
            // Random selection but chance increased when significantly above the food threshold
            rng.gen_range(0.0..1.0) < relative_division_food_level
        {
            return Some(CycleEvent::Division);
        }*/

        // If the cell has not enough food let it die
        let relative_death_food_level = ((c.cycle.food_death_threshold - c.get_intracellular()[1])
            / c.cycle.food_death_threshold)
            .clamp(0.0, 1.0);
        if c.cellular_reactions.get_intracellular()[1] < 0.0
            && rng.gen_range(0.0..1.0) < relative_death_food_level
        {
            return Some(CycleEvent::Death);
        }
        None
    }

    fn divide(
        _rng: &mut rand_chacha::ChaCha8Rng,
        c1: &mut MyCellType,
    ) -> Result<Option<MyCellType>, DivisionError> {
        // Clone existing cell
        c1.cycle.generation += 1;
        let mut c2 = c1.clone();
        // let r = c1.interaction.base_interaction.cell_radius;

        // Make both cells smaller
        // ALso keep old cell larger
        let relative_size_difference = 0.2;
        c1.mechanics.set_cell_area(
            c1.mechanics.get_cell_area() * (1.0 + relative_size_difference)
                / std::f64::consts::SQRT_2,
        );
        c2.mechanics.set_cell_area(
            c2.mechanics.get_cell_area() * (1.0 - relative_size_difference)
                / std::f64::consts::SQRT_2,
        );

        // Generate cellular splitting direction randomly
        // let angle_1 = rng.gen_range(-std::f64::consts::FRAC_PI_8..std::f64::consts::FRAC_PI_8);
        // let dir_vec = nalgebra::Rotation2::new(angle_1) * c1.interaction.base_interaction.orientation;

        // Define new positions for cells
        // It is randomly chosen if the old cell is left or right
        // let sign = -1.0;//rng.gen_range(-1.0_f64..1.0_f64).signum();
        // let offset = sign*dir_vec.into_inner()*r/std::f64::consts::SQRT_2;
        // let old_pos = c1.pos();
        //
        // c1.set_pos(&(old_pos + offset));
        // c2.set_pos(&(old_pos - offset));
        //
        // Decrease the amount of food in the cells
        c1.cellular_reactions.intracellular_concentrations *=
            (1.0 + relative_size_difference) * 0.5;
        c2.cellular_reactions.intracellular_concentrations *=
            (1.0 - relative_size_difference) * 0.5;

        // Increase the amount of divisions that this cell has done
        c1.cycle.divisions += 1;

        // New cell is completely new so set age to 0
        c2.cycle.age = 0.0;

        Ok(Some(c2))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OwnReactions {
    pub intracellular_concentrations: ReactionVector,
    pub intracellular_concentrations_saturation_level: ReactionVector,
    pub production_term: ReactionVector,
    pub degradation_rate: ReactionVector,
    pub secretion_rate: ReactionVector,
    pub uptake_rate: ReactionVector,

    pub p1: f64,
    pub p2: f64,
    pub p3: f64,
    pub p4: f64,
}

impl CellularReactions<ReactionVector> for OwnReactions {
    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        internal_concentration_vector: &ReactionVector,
        external_concentration_vector: &ReactionVector,
    ) -> Result<(ReactionVector, ReactionVector), CalcError> {
        // Calculate a modifier which is clamps between 0.0 and 1.0 and will be 1.0 if
        // we are far away from the saturation level and 0.0 if we have reached it.
        use num::Zero;
        let mut increment_extracellular = ReactionVector::zero();
        let mut increment_intracellular = ReactionVector::zero();
        for i in 0..2 {
            let modifier_1 = (1.0
                - internal_concentration_vector[i]
                    / self.intracellular_concentrations_saturation_level[i])
                .clamp(0.0, 1.0);
            let modifier_2 = internal_concentration_vector[i]
                / self.intracellular_concentrations_saturation_level[i].max(0.0);
            let uptake = -self.uptake_rate[i] * external_concentration_vector[i] * modifier_1;
            let secretion = self.secretion_rate[i]
                * (internal_concentration_vector[i]
                    - self.intracellular_concentrations_saturation_level[i])
                * modifier_2;
            increment_extracellular[i] = uptake + secretion;
            increment_intracellular[i] = self.production_term[i] - increment_extracellular[i];
        }

        // Reduce the amount of food if a peak of the turing pattern is nearby
        increment_intracellular[1] -= self.production_term[1].abs()
            * (external_concentration_vector[2] / 4000.0).clamp(0.0, 1.0);

        // Equations for the Turing pattern
        let a = external_concentration_vector[2];
        let b = external_concentration_vector[3];
        increment_extracellular[2] = self.p1 - self.p2 * a + self.p3 * a.powf(2.0) * b;
        increment_extracellular[3] = self.p4 - self.p3 * a.powf(2.0) * b;
        // println!("{:8.4?} {:8.4?}", external_concentration_vector[2], external_concentration_vector[3]);
        // println!("Internal {:5.2?} External {:5.2?} Uptake {:5.2?} Secretion {:5.2?}", internal_concentration_vector, external_concentration_vector, uptake, secretion);
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
        // if gradient[0].norm()!=0.0 {
        //     cell.interaction.outside_interaction.orientation = nalgebra::Unit::new_normalize(-gradient[2]);
        // }
        Ok(())
    }
}
