use cellular_raza::prelude::*;

use serde::{Serialize,Deserialize};
use nalgebra::{Unit,Vector2};

use rand::Rng;

pub const NUMBER_OF_REACTION_COMPONENTS: usize = 1;
pub type ReactionVector = nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>;
pub type MyCellType = ModularCell<Vector2<f64>, MechanicsModel2D, CellSpecificInteraction, OwnCycle, OwnReactions, GradientSensing>;


#[derive(Serialize,Deserialize,Clone,core::fmt::Debug)]
pub struct DirectedSphericalMechanics {
    pub pos: Vector2<f64>,
    pub vel: Vector2<f64>,
    pub orientation: Unit<Vector2<f64>>,
}



#[derive(Serialize,Deserialize,Clone,core::fmt::Debug)]
pub struct CellSpecificInteraction {
    pub potential_strength: f64,
    pub attraction_multiplier: f64,
    pub relative_interaction_range: f64,
    pub cell_radius: f64,
    pub orientation: Unit<Vector2<f64>>,
    pub polarity: i32,
}


impl Interaction<Vector2<f64>, Vector2<f64>, (f64, Unit<Vector2<f64>>)> for CellSpecificInteraction {
    fn get_interaction_information(&self) -> Option<(f64, Unit<Vector2<f64>>)> {
        Some((self.cell_radius, self.orientation.clone()))
    }

    fn calculate_force_on(&self, own_pos: &Vector2<f64>, ext_pos: &Vector2<f64>, ext_info: &Option<(f64, Unit<Vector2<f64>>)>) -> Option<Result<Vector2<f64>, CalcError>> {
        let min_relative_distance_to_center = 0.3162277660168379;
        let (r, dir) = match (own_pos-ext_pos).norm() < self.cell_radius*min_relative_distance_to_center {
            false => {
                let z = own_pos - ext_pos;
                let r = z.norm();
                (r, z.normalize())
            },
            true => {
                let dir = match own_pos==ext_pos {
                    true => self.orientation.into_inner(),
                    false => (own_pos - ext_pos).normalize(),
                };
                let r = self.cell_radius*min_relative_distance_to_center;
                (r, dir)
            }
        };
        match ext_info {
            Some((ext_radius, external_orientation)) => {
                // Introduce Non-dimensional length variable
                let sigma = r/(self.cell_radius + ext_radius);
                let bound = 4.0 + 1.0/sigma;
                let spatial_cutoff = (1.0+(self.relative_interaction_range*(self.cell_radius+ext_radius)-r).signum())*0.5;

                // Calculate the strength of the interaction with correct bounds
                let strength = self.potential_strength*((1.0/sigma).powf(2.0) - (1.0/sigma).powf(4.0)).min(bound).max(-bound);

                // Calculate the attraction modifier based on the different orientation value
                let attraction_orientation_modifier = dir.dot(external_orientation).abs();

                // Calculate only attracting and repelling forces
                let attracting_force = dir * (self.attraction_multiplier * attraction_orientation_modifier + 1.0) * strength.max(0.0) * spatial_cutoff;
                let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

                Some(Ok(repelling_force + attracting_force))
            },
            None => None,
        }
    }
}


#[derive(Serialize,Deserialize,Debug,Clone)]
pub struct OwnCycle {
    age: f64,
    pub division_age: f64,
    divisions: u8,
    generation: u8,
    pub maximum_cell_radius: f64,
    pub growth_rate: f64,
    cell_maximum_divisions: u8,
}


impl OwnCycle {
    pub fn new(division_age: f64, maximum_cell_radius: f64, growth_rate: f64, cell_maximum_divisions: u8) -> Self {
        OwnCycle {
            age: 0.0,
            division_age,
            divisions: 0,
            generation: 0,
            maximum_cell_radius,
            growth_rate,
            cell_maximum_divisions,
        }
    }
}


impl Cycle<MyCellType> for OwnCycle {
    fn update_cycle(_rng: &mut rand_chacha::ChaCha8Rng, dt: &f64, c: &mut MyCellType) -> Option<CycleEvent> {
        if c.interaction.cell_radius < c.cycle.maximum_cell_radius {
            c.interaction.cell_radius += (c.cycle.maximum_cell_radius * c.cycle.growth_rate * dt / c.cycle.division_age).min(c.cycle.maximum_cell_radius - c.interaction.cell_radius);
        }
        c.cycle.age += dt;
        if c.cycle.age >= c.cycle.division_age && c.cycle.divisions<c.cycle.cell_maximum_divisions {
            Some(CycleEvent::Division)
        } else {
            None
        }
    }

    fn divide(rng: &mut rand_chacha::ChaCha8Rng, c1: &mut MyCellType) -> Result<Option<MyCellType>, DivisionError> {
        // Clone existing cell
        c1.cycle.generation += 1;
        let mut c2 = c1.clone();
        let r = c1.interaction.cell_radius;

        // Make both cells smaller
        // ALso keep old cell larger
        let relative_size_difference = 0.2;
        c1.interaction.cell_radius *= (1.0+relative_size_difference)/std::f64::consts::SQRT_2;
        c2.interaction.cell_radius *= (1.0-relative_size_difference)/std::f64::consts::SQRT_2;

        // Generate cellular splitting direction randomly
        let angle_1 = std::f64::consts::FRAC_PI_2 + rng.gen_range(-std::f64::consts::FRAC_PI_8..std::f64::consts::FRAC_PI_8);
        let dir_vec = nalgebra::Rotation2::new(angle_1) * c1.interaction.orientation;

        // Define new positions for cells
        // It is randomly chosen if the old cell is left or right
        let sign = -1.0;//rng.gen_range(-1.0_f64..1.0_f64).signum();
        let offset = sign*dir_vec.into_inner()*r*0.5;
        let old_pos = c1.pos();

        c1.set_pos(&(old_pos + offset));
        c2.set_pos(&(old_pos - offset));

        // If we reach a certain cell-generation in the simulation, we change the angle of our

        // Increase the amount of divisions that this cell has done
        c1.cycle.divisions += 1;

        // New cell is completely new so set age to 0
        c2.cycle.age = 0.0;

        Ok(Some(c2))
    }
}

#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct OwnReactions {
    pub intracellular_concentrations: ReactionVector,
    pub production_term: ReactionVector,
    pub secretion_rate: ReactionVector,
    pub uptake_rate: ReactionVector,
}


impl CellularReactions<ReactionVector> for OwnReactions {
    fn calculate_intra_and_extracellular_reaction_increment(&self, internal_concentration_vector: &ReactionVector, external_concentration_vector: &ReactionVector) -> Result<(ReactionVector, ReactionVector), CalcError> {
        let increment_extracellular = - self.uptake_rate.component_mul(&external_concentration_vector) + self.secretion_rate.component_mul(&internal_concentration_vector);
        let increment_intracellular = self.production_term - increment_extracellular;
        // println!("{} {}", increment_intracellular, increment_extracellular);
        Ok((increment_intracellular, increment_extracellular))
    }

    fn get_intracellular(&self) -> ReactionVector {
        self.intracellular_concentrations
    }

    fn set_intracellular(&mut self, concentration_vector: ReactionVector) {
        self.intracellular_concentrations = concentration_vector;
    }
}
