use cellular_raza::prelude::*;

use nalgebra::Vector2;
use num::Zero;
use serde::{Deserialize, Serialize};

use pyo3::prelude::*;

pub const NUMBER_OF_REACTION_COMPONENTS: usize = 2;
pub type ReactionVector = nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>;

#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct Bacteria {
    #[Mechanics(Vector2<f64>, Vector2<f64>, Vector2<f64>)]
    pub mechanics: NewtonDamped2D,
    #[Interaction(Vector2<f64>, Vector2<f64>, Vector2<f64>, f64)]
    pub interaction: BacteriaInteraction,
    #[Cycle]
    pub cycle: BacteriaCycle,
    #[CellularReactions(nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>,)]
    pub cellular_reactions: BacteriaReactions,
    #[InteractionExtracellularGradient(nalgebra::SVector<Vector2<f64>, NUMBER_OF_REACTION_COMPONENTS>,)]
    pub interactionextracellulargradient: GradientSensing,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct BacteriaInteraction {
    pub potential_strength: f64,
    pub cell_radius: f64,
}

#[pymethods]
impl BacteriaInteraction {
    fn __repr__(&self) -> String {
        format!("{self:#?}")
    }
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
        let z = ext_pos - own_pos;
        let r = z.norm();
        let sigma = r / (self.cell_radius + ext_radius);
        if sigma < 1.0 {
            let q = 0.2;
            let dir = z.normalize();
            let modifier = (1.0 + q) / (q + sigma);
            return Some(Ok(self.potential_strength * dir * modifier));
        } else {
            return Some(Ok(Vector2::zero()));
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass(get_all, set_all)]
pub struct BacteriaCycle {
    /// Conversion of the consumed food to cellular volume. In units $\frac{\text{volume}}{\text{food}}$.
    pub food_to_volume_conversion: f64,
    /// Threshold for the volume when the cell should divide
    pub volume_division_threshold: f64,
    pub lag_phase_transition_rate: f64,
}

#[pymethods]
impl BacteriaCycle {
    #[new]
    pub fn new(
        food_to_volume_conversion: f64,
        volume_division_threshold: f64,
        lag_phase_transition_rate: f64,
    ) -> Self {
        BacteriaCycle {
            food_to_volume_conversion,
            volume_division_threshold,
            lag_phase_transition_rate,
        }
    }
}

#[pymethods]
impl Bacteria {
    /// We can have a look at this paper https://doi.org/10.1128/jb.148.1.58-63.1981
    /// and see that the average density of E.Coli is between 1.080 and 1.100 g/ml
    /// This means we can safely set the density to 1.09
    pub fn volume_to_mass(&self, volume: f64) -> f64 {
        1.09 * volume
    }

    pub fn mass_to_volume(&self, mass: f64) -> f64 {
        mass / 1.09
    }
}

impl Cycle<Bacteria> for BacteriaCycle {
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Bacteria,
    ) -> Option<CycleEvent> {
        use rand::Rng;
        // Check if we are in lag phase and if so check if we want to convert to active state
        if cell.cellular_reactions.lag_phase_active {
            let p = rng.gen_bool(dt * cell.cycle.lag_phase_transition_rate);
            if p {
                cell.cellular_reactions.lag_phase_active = false;
            }
        }
        // Grow the cell if we are not in lag phase
        else {
            let cell_volume = cell.get_volume();
            // Calculate available food
            let food_available =
                cell.cellular_reactions.intracellular_concentrations[0] * cell_volume;

            // Calculate the total volume increment from the available food
            let volume_increment_available = food_available * cell.cycle.food_to_volume_conversion;

            // Calculate the actual increment we will do
            // It is either the total increment or by the difference to volume_division_threshold.
            // The last condition makes sure that we have not already exceeded the division threshold
            let volume_increment = volume_increment_available
                .min((cell.cycle.volume_division_threshold - cell_volume).max(0.0));

            // Grow the cell
            let radial_increment = (volume_increment / std::f64::consts::PI).sqrt();
            cell.interaction.cell_radius += radial_increment;

            // Reduce intracellular amount by the calculated consumed food.
            // Notice that we still divide by the old cell volume and not
            // the newer one which originates from the increased radius.
            let food_consumed = volume_increment / cell.cycle.food_to_volume_conversion;
            cell.cellular_reactions.intracellular_concentrations[0] -= food_consumed / cell_volume;

            // Set the cells new mass. Now we use the new volume
            cell.mechanics.mass = cell.volume_to_mass(cell.get_volume());
        }

        if cell.get_volume() >= cell.cycle.volume_division_threshold {
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
        c1.interaction.cell_radius /= std::f64::consts::SQRT_2;
        c2.interaction.cell_radius /= std::f64::consts::SQRT_2;

        // Generate cellular splitting direction randomly
        let angle_1 = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let dir_vec = nalgebra::Rotation2::new(angle_1) * Vector2::from([1.0, 0.0]);

        // Define new positions for cells
        // It is randomly chosen if the old cell is left or right
        let offset = dir_vec * r / std::f64::consts::SQRT_2;
        let old_pos = c1.pos();

        // Increase the remaining food in the bacteria by the fraction of volumes
        // from before and after the division. This exactly coincides with factor 2
        // since we half the volume of the cell
        c1.cellular_reactions.intracellular_concentrations[0] *= 2.0;
        c2.cellular_reactions.intracellular_concentrations[0] *= 2.0;

        c1.set_pos(&(old_pos + offset));
        c2.set_pos(&(old_pos - offset));

        Ok(c2)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass]
pub enum Species {
    S1,
    S2
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[pyclass]
pub struct BacteriaReactions {
    #[pyo3(get, set)]
    pub lag_phase_active: bool,
    #[pyo3(get, set)]
    pub species: Species,
    pub intracellular_concentrations: ReactionVector,
    pub uptake_rates: ReactionVector,
    pub production_rates: ReactionVector,
    pub inhibitions: ReactionVector,
}

#[pymethods]
impl BacteriaReactions {
    #[setter]
    pub fn set_intracellular_concentrations(
        &mut self,
        intracellular_concentrations: [f64; NUMBER_OF_REACTION_COMPONENTS],
    ) {
        self.intracellular_concentrations = intracellular_concentrations.into();
    }

    #[setter]
    pub fn set_uptake_rates(&mut self, uptake_rates: [f64; NUMBER_OF_REACTION_COMPONENTS]) {
        self.uptake_rates = uptake_rates.into();
    }

    #[setter]
    pub fn set_inhibitions(&mut self, inhibitions: [f64; NUMBER_OF_REACTION_COMPONENTS]) {
        self.inhibitions = inhibitions.into();
    }

    #[getter]
    pub fn get_intracellular_concentrations(&self) -> [f64; NUMBER_OF_REACTION_COMPONENTS] {
        self.intracellular_concentrations.into()
    }

    #[getter]
    pub fn get_uptake_rates(&self) -> [f64; NUMBER_OF_REACTION_COMPONENTS] {
        self.uptake_rates.into()
    }

    #[getter]
    pub fn get_inhibitions(&self) -> [f64; NUMBER_OF_REACTION_COMPONENTS] {
        self.inhibitions.into()
    }

    #[new]
    pub fn new(
        lag_phase_active: bool,
        species: Species,
        intracellular_concentrations: [f64; NUMBER_OF_REACTION_COMPONENTS],
        uptake_rates: [f64; NUMBER_OF_REACTION_COMPONENTS],
        production_rates: [f64; NUMBER_OF_REACTION_COMPONENTS],
        inhibitions: [f64; NUMBER_OF_REACTION_COMPONENTS],
    ) -> Self {
        Self {
            lag_phase_active,
            species,
            intracellular_concentrations: intracellular_concentrations.into(),
            uptake_rates: uptake_rates.into(),
            production_rates: production_rates.into(),
            inhibitions: inhibitions.into(),
        }
    }
}

impl CellularReactions<ReactionVector> for BacteriaReactions {
    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        _internal_concentration_vector: &ReactionVector,
        external_concentration_vector: &ReactionVector,
    ) -> Result<(ReactionVector, ReactionVector), CalcError> {
        // If we are in lag phase, we simply return a zero-vector
        if self.lag_phase_active {
            return Ok((ReactionVector::zero(), ReactionVector::zero()));
        }

        let inhib = self
            .inhibitions
            .component_mul(&external_concentration_vector)
            .add_scalar(1.0);

        let inc_int = self
            .uptake_rates
            .component_mul(&external_concentration_vector)
            .component_div(&inhib);

        let inc_ext = self.production_rates - inc_int;

        Ok((inc_int, inc_ext))
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
        std::f64::consts::PI * self.interaction.cell_radius.powf(2.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct GradientSensing;

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
