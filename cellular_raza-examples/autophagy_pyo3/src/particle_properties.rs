use cellular_raza::prelude::*;
use nalgebra::Vector3;
use num::Zero;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Particle Species of type Cargo or R11
///
/// We currently only distinguish between the cargo itself
/// and freely moving combinations of the receptor and ATG11.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[pyclass]
pub enum Species {
    Cargo,
    R11,
}

/// Interaction potential depending on the other cells species.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct TypedInteraction {
    pub species: Species,
    pub cell_radius: f64,
    pub potential_strength_cargo_cargo: f64,
    pub potential_strength_r11_r11: f64,
    pub potential_strength_cargo_r11: f64,
    pub potential_strength_cargo_r11_avidity: f64,
    pub interaction_range_cargo_cargo: f64,
    pub interaction_range_r11_r11: f64,
    pub interaction_range_r11_cargo: f64,
    neighbour_count: usize,
}

fn calculate_avidity(own_neighbour_count: usize, ext_neighbour_count: usize) -> f64 {
    let n = 6.0;
    let alpha = 4.0;
    let nc = own_neighbour_count.min(ext_neighbour_count);
    let s = (nc as f64 / alpha).powf(n) / (1.0 + (nc as f64 / alpha).powf(n));
    s
}

impl Interaction<Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, usize, Species)>
    for TypedInteraction
{
    fn calculate_force_between(
        &self,
        own_pos: &Vector3<f64>,
        _own_vel: &Vector3<f64>,
        ext_pos: &Vector3<f64>,
        _ext_vel: &Vector3<f64>,
        ext_info: &(f64, usize, Species),
    ) -> Option<Result<Vector3<f64>, CalcError>> {
        // Calculate radius and direction
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
        let (ext_radius, ext_neighbour_count, ext_species) = ext_info;
        // Introduce Non-dimensional length variable
        let sigma = r / (self.cell_radius + ext_radius);
        let bound = 4.0 + 1.0 / sigma;
        let calculate_cutoff = |interaction_range| {
            if interaction_range + self.cell_radius + ext_radius >= r {
                1.0
            } else {
                0.0
            }
        };

        // Calculate the strength of the interaction with correct bounds
        let strength = ((1.0 / sigma).powf(2.0) - (1.0 / sigma).powf(4.0))
            .min(bound)
            .max(-bound);

        // Calculate only attracting and repelling forces
        let attracting_force = dir * strength.max(0.0);
        let repelling_force = dir * strength.min(0.0);

        match (ext_species, &self.species) {
            // R11 will bind to cargo
            (Species::Cargo, Species::R11) | (Species::R11, Species::Cargo) => {
                let avidity = self.potential_strength_cargo_r11_avidity
                    * calculate_avidity(self.neighbour_count, *ext_neighbour_count);
                let cutoff = calculate_cutoff(self.interaction_range_r11_cargo);
                let force = cutoff
                    * (self.potential_strength_cargo_cargo * repelling_force
                        + (self.potential_strength_cargo_r11 + avidity) * attracting_force);
                Some(Ok(force))
            }

            // R11 forms clusters
            (Species::Cargo, Species::Cargo) => {
                let cutoff = calculate_cutoff(self.interaction_range_cargo_cargo);
                Some(Ok(cutoff
                    * self.potential_strength_cargo_cargo
                    * (repelling_force + attracting_force)))
            }

            (Species::R11, Species::R11) => {
                let cutoff = calculate_cutoff(self.interaction_range_r11_r11);
                Some(Ok(cutoff
                    * self.potential_strength_r11_r11
                    * (repelling_force + attracting_force)))
            }
        }
    }

    fn get_interaction_information(&self) -> (f64, usize, Species) {
        (self.cell_radius, self.neighbour_count, self.species.clone())
    }

    fn is_neighbour(
        &self,
        own_pos: &Vector3<f64>,
        ext_pos: &Vector3<f64>,
        ext_inf: &(f64, usize, Species),
    ) -> Result<bool, CalcError> {
        match (&self.species, &ext_inf.2) {
            (Species::R11, Species::R11) | (Species::Cargo, Species::Cargo) => {
                Ok((own_pos - ext_pos).norm() <= 2.0 * (self.cell_radius + ext_inf.0))
            }
            _ => Ok(false),
        }
    }

    fn react_to_neighbours(&mut self, neighbours: usize) -> Result<(), CalcError> {
        Ok(self.neighbour_count = neighbours)
    }
}

impl Volume for Particle {
    fn get_volume(&self) -> f64 {
        1.0
    }
}

#[pymethods]
impl TypedInteraction {
    #[new]
    pub fn new(
        species: Species,
        cell_radius: f64,
        potential_strength_cargo_cargo: f64,
        potential_strength_r11_r11: f64,
        potential_strength_cargo_r11: f64,
        potential_strength_cargo_r11_avidity: f64,
        interaction_range_cargo_cargo: f64,
        interaction_range_r11_r11: f64,
        interaction_range_r11_cargo: f64,
    ) -> Self {
        Self {
            species,
            cell_radius,
            potential_strength_cargo_cargo,
            potential_strength_r11_r11,
            potential_strength_cargo_r11,
            potential_strength_cargo_r11_avidity,
            interaction_range_cargo_cargo,
            interaction_range_r11_r11,
            interaction_range_r11_cargo,
            neighbour_count: 0,
        }
    }
}

#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
pub struct Particle {
    #[Mechanics(Vector3<f64>, Vector3<f64>, Vector3<f64>)]
    pub mechanics: Langevin3D,

    #[Interaction(Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, usize, Species))]
    pub interaction: TypedInteraction,
}

impl Cycle<Particle> for Particle {
    fn divide(
        _rng: &mut rand_chacha::ChaCha8Rng,
        _cell: &mut Particle,
    ) -> Result<Particle, DivisionError> {
        panic!()
    }

    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        _dt: &f64,
        _cell: &mut Particle,
    ) -> Option<CycleEvent> {
        None
    }
}

impl CellularReactions<Nothing, Nothing> for Particle {
    fn get_intracellular(&self) -> Nothing {
        Nothing::zero()
    }

    fn set_intracellular(&mut self, _concentration_vector: Nothing) {}

    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        _internal_concentration_vector: &Nothing,
        _external_concentration_vector: &Nothing,
    ) -> Result<(Nothing, Nothing), CalcError> {
        Ok((Nothing::zero(), Nothing::zero()))
    }
}

impl<Conc> InteractionExtracellularGradient<Particle, Conc> for Particle {
    fn sense_gradient(_cell: &mut Particle, _gradient: &Conc) -> Result<(), CalcError> {
        Ok(())
    }
}
