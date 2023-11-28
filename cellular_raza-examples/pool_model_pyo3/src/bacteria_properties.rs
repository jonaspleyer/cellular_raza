use cellular_raza::prelude::*;

use nalgebra::Vector2;
use num::Zero;
use serde::{Deserialize, Serialize};

use pyo3::prelude::*;

pub const NUMBER_OF_REACTION_COMPONENTS: usize = 1;
pub type ReactionVector = nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct BacteriaMechanicsModel2D {
    pub pos: Vector2<f64>,
    pub vel: Vector2<f64>,
    #[pyo3(get, set)]
    pub dampening_constant: f64,
    #[pyo3(get, set)]
    pub mass: f64,
}

#[pymethods]
impl BacteriaMechanicsModel2D {
    #[getter(pos)]
    fn py_get_pos(&self) -> [f64; 2] {
        self.pos.into()
    }

    #[setter(pos)]
    fn py_set_pos(&mut self, pos: [f64; 2]) {
        self.pos = pos.into();
    }

    #[getter(vel)]
    fn py_get_vel(&self) -> [f64; 2] {
        self.vel.into()
    }

    #[setter(vel)]
    fn py_set_vel(&mut self, vel: [f64; 2]) {
        self.vel = vel.into();
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }
}

impl Mechanics<Vector2<f64>, Vector2<f64>, Vector2<f64>> for BacteriaMechanicsModel2D {
    fn pos(&self) -> Vector2<f64> {
        self.pos
    }

    fn velocity(&self) -> Vector2<f64> {
        self.vel
    }

    fn set_pos(&mut self, p: &Vector2<f64>) {
        self.pos = *p;
    }

    fn set_velocity(&mut self, v: &Vector2<f64>) {
        self.vel = *v;
    }

    fn calculate_increment(
        &self,
        force: Vector2<f64>,
    ) -> Result<(Vector2<f64>, Vector2<f64>), CalcError> {
        let dx = self.vel;
        let dv = force / self.mass - self.dampening_constant * self.vel;
        Ok((dx, dv))
    }
}

#[derive(Clone, Debug, CellAgent, Deserialize, Serialize)]
#[pyclass]
pub struct Bacteria {
    #[Mechanics(Vector2<f64>, Vector2<f64>, Vector2<f64>)]
    #[pyo3(get, set)]
    pub mechanics: BacteriaMechanicsModel2D,
    #[Interaction(Vector2<f64>, Vector2<f64>, Vector2<f64>, f64)]
    #[pyo3(get, set)]
    pub interaction: BacteriaInteraction,
    #[Cycle]
    #[pyo3(get, set)]
    pub cycle: BacteriaCycle,
    #[CellularReactions(nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>,)]
    #[pyo3(get, set)]
    pub cellular_reactions: BacteriaReactions,
    #[InteractionExtracellularGradient(nalgebra::SVector<Vector2<f64>, NUMBER_OF_REACTION_COMPONENTS>,)]
    pub interactionextracellulargradient: NoExtracellularGradientSensing,
}

#[derive(Serialize, Deserialize, Clone, core::fmt::Debug)]
#[pyclass]
pub struct BacteriaInteraction {
    #[pyo3(get, set)]
    pub potential_strength: f64,
    #[pyo3(get, set)]
    pub relative_interaction_range: f64,
    #[pyo3(get, set)]
    pub cell_radius: f64,
}

impl Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, f64> for BacteriaInteraction {
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
        // let attracting_force = dir * strength.max(0.0) * spatial_cutoff;
        // let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

        // Some(Ok(repelling_force + attracting_force))
        Some(Ok(dir * strength * spatial_cutoff))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct BacteriaCycle {
    #[pyo3(get, set)]
    pub age: f64,
    #[pyo3(get, set)]
    pub division_age: f64,
    #[pyo3(get, set)]
    pub maximum_cell_radius: f64,
    #[pyo3(get, set)]
    pub growth_rate: f64,
    #[pyo3(get, set)]
    pub food_threshold: f64,
    #[pyo3(get, set)]
    pub food_growth_rate_multiplier: f64,
    #[pyo3(get, set)]
    pub food_division_threshold: f64,
}

impl BacteriaCycle {
    pub fn new(
        division_age: f64,
        maximum_cell_radius: f64,
        growth_rate: f64,
        food_threshold: f64,
        food_growth_rate_multiplier: f64,
        food_division_threshold: f64,
    ) -> Self {
        BacteriaCycle {
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

impl Cycle<Bacteria> for BacteriaCycle {
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Bacteria,
    ) -> Option<CycleEvent> {
        use rand::Rng;
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
        c1: &mut Bacteria,
    ) -> Result<Bacteria, DivisionError> {
        use rand::Rng;
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
#[pyclass]
pub struct BacteriaReactions {
    pub intracellular_concentrations: ReactionVector,
    pub turnover_rate: ReactionVector,
    pub production_term: ReactionVector,
    pub degradation_rate: ReactionVector,
    pub secretion_rate: ReactionVector,
    pub uptake_rate: ReactionVector,
}

impl CellularReactions<ReactionVector> for BacteriaReactions {
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

impl Volume for Bacteria {
    fn get_volume(&self) -> f64 {
        4.0 / 3.0 * std::f64::consts::PI * self.interaction.cell_radius.powf(3.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientSensing {}

impl
    InteractionExtracellularGradient<
        Bacteria,
        nalgebra::SVector<Vector2<f64>, NUMBER_OF_REACTION_COMPONENTS>,
    > for GradientSensing
{
    fn sense_gradient(
        _cell: &mut Bacteria,
        _gradient: &nalgebra::SVector<Vector2<f64>, NUMBER_OF_REACTION_COMPONENTS>,
    ) -> Result<(), CalcError> {
        Ok(())
    }
}
