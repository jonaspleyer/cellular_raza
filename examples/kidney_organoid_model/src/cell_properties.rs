use cellular_raza::backend::cpu_os_threads::prelude::*;
use cellular_raza::implementations::cell_properties::mechanics::VertexMechanics2D;

use nalgebra::{Unit, Vector2};
use serde::{Deserialize, Serialize};

use rand::Rng;

pub const NUMBER_OF_REACTION_COMPONENTS: usize = 2;
pub const NUMBER_OF_VERTICES: usize = 6;
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

impl Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, InteractionInformation>
    for OutsideInteraction
{
    fn calculate_force_on(
        &self,
        own_pos: &Vector2<f64>,
        _own_vel: &Vector2<f64>,
        ext_pos: &Vector2<f64>,
        _ext_vel: &Vector2<f64>,
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

impl Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, InteractionInformation>
    for InsideInteraction
{
    fn calculate_force_on(
        &self,
        own_pos: &Vector2<f64>,
        _own_vel: &Vector2<f64>,
        ext_pos: &Vector2<f64>,
        _ext_vel: &Vector2<f64>,
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
        cell: &mut MyCellType,
    ) -> Option<CycleEvent> {
        // If the cell is not at the maximum size let it grow
        if cell.mechanics.get_cell_area() < cell.cycle.maximum_cell_area {
            let growth_difference = (cell.cycle.maximum_cell_area * cell.cycle.growth_rate * dt)
                .min(cell.cycle.maximum_cell_area - cell.mechanics.get_cell_area())
                .max(0.0);
            cell.cellular_reactions.intracellular_concentrations[1] -=
                cell.cycle.food_growth_rate_multiplier * growth_difference
                    / cell.cycle.maximum_cell_area;
            cell.mechanics
                .set_cell_area(cell.mechanics.get_cell_area() + growth_difference);
        }

        // Increase the age of the cell and divide if possible
        cell.cycle.age += dt;

        // If the cell has not enough food let it die
        let relative_death_food_level = ((cell.cycle.food_death_threshold
            - cell.get_intracellular()[1])
            / cell.cycle.food_death_threshold)
            .clamp(0.0, 1.0);
        if cell.cellular_reactions.get_intracellular()[1] < 0.0
            && rng.gen_range(0.0..1.0) < relative_death_food_level
        {
            return Some(CycleEvent::Death);
        }
        None
    }

    fn divide(
        _rng: &mut rand_chacha::ChaCha8Rng,
        c1: &mut MyCellType,
    ) -> Result<MyCellType, DivisionError> {
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

        // Decrease the amount of food in the cells
        c1.cellular_reactions.intracellular_concentrations *=
            (1.0 + relative_size_difference) * 0.5;
        c2.cellular_reactions.intracellular_concentrations *=
            (1.0 - relative_size_difference) * 0.5;

        // Increase the amount of divisions that this cell has done
        c1.cycle.divisions += 1;

        // New cell is completely new so set age to 0
        c2.cycle.age = 0.0;

        Ok(c2)
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
