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
    #[Cycle]
    pub cycle: BacteriaCycle,
    #[Interaction(Vector2<f64>, Vector2<f64>, Vector2<f64>, f64)]
    #[CellularReactions(f64, ReactionVector)]
    pub cellular_reactions: BacteriaReactions,
    #[InteractionExtracellularGradient(nalgebra::SVector<Vector2<f64>, 2>,)]
    pub interactionextracellulargradient: GradientSensing,
}

#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct BacteriaTemplate {
    pub mechanics: Py<NewtonDamped2D>,
    pub cycle: Py<BacteriaCycle>,
    pub cellular_reactions: Py<BacteriaReactions>,
    pub interactionextracellulargradient: Py<GradientSensing>,
}

fn bacteria_default_volume() -> f64 {
    std::f64::consts::PI * 1.5_f64.powi(2)
}

fn bacteria_default_newton_damped() -> NewtonDamped2D {
    NewtonDamped2D::new(
        [0.0; 2],                         // pos
        [0.0; 2],                         // vel
        0.5,                              // damping
        1.09 * bacteria_default_volume(), // mass
    )
}

#[pymethods]
impl BacteriaTemplate {
    // TODO can we do this without using clone? Ie. without memory allocations?
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let bacteria = Bacteria::from(py, self.clone())?;
        Ok(format!("{:#?}", bacteria))
    }

    #[new]
    #[pyo3(signature = (
        mechanics=bacteria_default_newton_damped(),
        cycle=BacteriaCycle::default(),
        cellular_reactions=BacteriaReactions::default(),
    ))]
    fn new(
        py: Python,
        mechanics: NewtonDamped2D,
        cycle: BacteriaCycle,
        cellular_reactions: BacteriaReactions,
    ) -> PyResult<Self> {
        Ok(BacteriaTemplate {
            mechanics: Py::new(py, mechanics)?,
            cycle: Py::new(py, cycle)?,
            cellular_reactions: Py::new(py, cellular_reactions)?,
            interactionextracellulargradient: Py::new(py, GradientSensing)?,
        })
    }

    #[staticmethod]
    pub fn default(py: Python) -> PyResult<Self> {
        Ok(Self {
            mechanics: Py::new(py, bacteria_default_newton_damped())?,
            cycle: Py::new(py, BacteriaCycle::default())?,
            cellular_reactions: Py::new(py, BacteriaReactions::default())?,
            interactionextracellulargradient: Py::new(py, GradientSensing)?,
        })
    }
}

fn volume_to_radius(volume: f64) -> f64 {
    (volume / std::f64::consts::PI).sqrt()
}

#[pymethods]
impl Bacteria {
    #[staticmethod]
    pub fn from(py: Python, bacteria_template: BacteriaTemplate) -> PyResult<Self> {
        Ok(Self {
            mechanics: bacteria_template.mechanics.extract::<NewtonDamped2D>(py)?,
            cycle: bacteria_template.cycle.extract::<BacteriaCycle>(py)?,
            cellular_reactions: bacteria_template
                .cellular_reactions
                .extract::<BacteriaReactions>(py)?,
            interactionextracellulargradient: bacteria_template
                .interactionextracellulargradient
                .extract::<GradientSensing>(py)?,
        })
    }
    /// We can have a look at this paper https://doi.org/10.1128/jb.148.1.58-63.1981
    /// and see that the average density of E.Coli is between 1.080 and 1.100 g/ml
    /// This means we can safely set the density to 1.09
    pub fn volume_to_mass(&self, volume: f64) -> f64 {
        1.09 * volume
    }

    pub fn mass_to_volume(&self, mass: f64) -> f64 {
        mass / 1.09
    }

    pub fn increase_volume(&mut self, volume_increment: f64) {
        let current_volume = self.get_volume();
        let final_volume = current_volume + volume_increment;
        self.cellular_reactions.cell_volume = final_volume;
        self.mechanics.mass = self.volume_to_mass(final_volume);
    }

    pub fn cell_radius(&self) -> f64 {
        volume_to_radius(self.cellular_reactions.cell_volume)
    }
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

    #[new]
    #[pyo3(signature = (potential_strength=0.5, cell_radius=1.5))]
    pub fn new(potential_strength: f64, cell_radius: f64) -> Self {
        Self {
            potential_strength,
            cell_radius,
        }
    }

    #[staticmethod]
    pub fn default() -> Self {
        Self::new(0.5, 1.5)
    }
}

impl Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, f64> for BacteriaReactions {
    fn get_interaction_information(&self) -> f64 {
        volume_to_radius(self.cell_volume)
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
        let sigma = r / (self.cell_radius() + ext_radius);
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
    /// Threshold for the volume when the cell should divide
    pub volume_division_threshold: f64,
    // TODO think about replacing two variables for one
    pub lag_phase_transition_rate_1: f64,
    pub lag_phase_transition_rate_2: f64,
}

#[pymethods]
impl BacteriaCycle {
    #[new]
    #[pyo3(signature = (
        volume_division_threshold=2.0*bacteria_default_volume(),
        lag_phase_transition_rate_1=0.005,
        lag_phase_transition_rate_2=0.008,
    ))]
    pub fn new(
        volume_division_threshold: f64,
        lag_phase_transition_rate_1: f64,
        lag_phase_transition_rate_2: f64,
    ) -> Self {
        BacteriaCycle {
            volume_division_threshold,
            lag_phase_transition_rate_1,
            lag_phase_transition_rate_2,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        let bacteria_volume = bacteria_default_volume();
        Self {
            volume_division_threshold: 2.0 * bacteria_volume,
            lag_phase_transition_rate_1: 0.005,
            lag_phase_transition_rate_2: 0.008,
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
        // Check if we are in lag phase and if so check if we want to convert to active state
        if cell.cellular_reactions.lag_phase_active {
            let p = match cell.cellular_reactions.species {
                // TODO read out extracellular concentrations of nutrients here
                // change the probability to be
                // P = dt * lambda * N_external / N_external_initial
                Species::S1 => rng.gen_bool(dt * cell.cycle.lag_phase_transition_rate_1),
                Species::S2 => rng.gen_bool(dt * cell.cycle.lag_phase_transition_rate_2),
            };
            if p {
                cell.cellular_reactions.lag_phase_active = false;
            }
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

        let r = c1.cell_radius();

        // Make both cells smaller
        c1.cellular_reactions.cell_volume /= 2.0;
        c2.cellular_reactions.cell_volume /= 2.0;

        // Generate cellular splitting direction randomly
        let angle_1 = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let dir_vec = nalgebra::Rotation2::new(angle_1) * Vector2::from([1.0, 0.0]);

        // Define new positions for cells
        // It is randomly chosen if the old cell is left or right
        let offset = dir_vec * r / std::f64::consts::SQRT_2;
        let old_pos = c1.pos();

        c1.set_pos(&(old_pos + offset));
        c2.set_pos(&(old_pos - offset));

        Ok(c2)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass]
pub enum Species {
    S1,
    S2,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[pyclass(get_all, set_all)]
pub struct BacteriaReactions {
    pub potential_strength: f64,
    /// Conversion of the consumed food to cellular volume. In units $\frac{\text{volume}}{\text{food}}$.
    pub food_to_volume_conversion: f64,
    pub lag_phase_active: bool,
    pub species: Species,
    pub cell_volume: f64,
    pub uptake_rate: f64,
    pub inhibition_production_rate: f64,
    pub inhibition_coefficient: f64,
}

#[pymethods]
impl BacteriaReactions {
    #[new]
    #[pyo3(signature = (
        potential_strength=0.5,
        food_to_volume_conversion=1e-1,
        lag_phase_active=true,
        species=Species::S1,
        cell_volume=bacteria_default_volume(),
        uptake_rate=0.01,
        inhibition_production_rate=0.1,
        inhibition_coefficient=0.1,
    ))]
    pub fn new(
        potential_strength: f64,
        food_to_volume_conversion: f64,
        lag_phase_active: bool,
        species: Species,
        cell_volume: f64,
        uptake_rate: f64,
        inhibition_production_rate: f64,
        inhibition_coefficient: f64,
    ) -> Self {
        Self {
            potential_strength,
            food_to_volume_conversion,
            lag_phase_active,
            species,
            cell_volume: cell_volume,
            uptake_rate,
            inhibition_production_rate,
            inhibition_coefficient,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self::new(
            0.5,
            1e-1,
            true,
            Species::S1,
            bacteria_default_volume(),
            0.01,
            0.1,
            0.1,
        )
    }

    pub fn cell_radius(&self) -> f64 {
        volume_to_radius(self.cell_volume)
    }
}

impl CellularReactions<f64, ReactionVector> for BacteriaReactions {
    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        _internal_concentration_vector: &f64,
        external_concentration_vector: &ReactionVector,
    ) -> Result<(f64, ReactionVector), CalcError> {
        // If we are in lag phase, we simply return a zero-vector
        if self.lag_phase_active {
            return Ok((f64::zero(), ReactionVector::zero()));
        }

        let inc_ext = match self.species {
            // Species 1 does not feel the inhibition but produces it
            Species::S1 => [
                -self.uptake_rate * external_concentration_vector[0],
                self.inhibition_production_rate,
            ],
            // Species 2 feels the inhibition but does not produce it
            Species::S2 => {
                let inhib = 1.0 + self.inhibition_coefficient * external_concentration_vector[1];
                [
                    -self.uptake_rate * external_concentration_vector[0] / inhib,
                    0.0,
                ]
            }
        };

        // TODO
        let inc_int = -inc_ext[0]*self.food_to_volume_conversion*self.cell_volume;

        Ok((inc_int, inc_ext.into()))
    }

    fn get_intracellular(&self) -> f64 {
        self.cell_volume
    }

    fn set_intracellular(&mut self, new_volume: f64) {
        self.cell_volume = new_volume;
    }
}

impl Volume for Bacteria {
    fn get_volume(&self) -> f64 {
        // std::f64::consts::PI * self.interaction.cell_radius.powi(2)
        self.cellular_reactions.cell_volume
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
