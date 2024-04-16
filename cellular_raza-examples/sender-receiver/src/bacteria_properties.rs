use cellular_raza::prelude::*;

use num::Zero;
use serde::{Deserialize, Serialize};

use crate::DT;

pub const NUMBER_OF_REACTION_COMPONENTS: usize = 1;
pub type ReactionVector = nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>;
pub type MyCellType = ModularCell<
    NewtonDamped2D,
    MiePotential,
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
    pub species: Species,
    pub intracellular_concentrations: ReactionVector,
    pub turnover_rate: ReactionVector,
    pub production_term: ReactionVector,
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

#[derive(Clone, Serialize, Deserialize)]
pub struct ConcentrationController {
    pub target_average_conc: f64,
    pub k_p: f64,
    pub k_i: f64,
    pub k_d: f64,
    pub previous_values: Vec<f64>,
}

pub struct Observable(f64, usize);

impl Controller<MyCellType, Observable> for ConcentrationController {
    fn measure<'a, I>(&self, cells: I) -> Result<Observable, CalcError>
    where
        I: IntoIterator<Item = &'a CellAgentBox<MyCellType>> + Clone,
    {
        let mut n_cells = 0;
        let mut total_conc = 0.0;
        cells.into_iter().for_each(|cell| {
            total_conc += cell.cell.get_intracellular()[0];
            n_cells += 1;
        });
        Ok(Observable(total_conc / n_cells as f64, n_cells))
    }

    fn adjust<'a, 'b, I, J>(&mut self, measurements: I, cells: J) -> Result<(), ControllerError>
    where
        Observable: 'a,
        MyCellType: 'b,
        I: Iterator<Item = &'a Observable>,
        J: Iterator<Item = (&'b mut CellAgentBox<MyCellType>, &'b mut Vec<CycleEvent>)>,
    {
        // Combine the measurements
        let mut n_cells = 0;
        let mut total_conc = 0.0;
        measurements.into_iter().for_each(|measurement| {
            total_conc += measurement.0;
            n_cells += measurement.1;
        });

        // Calculate difference
        let average_conc = total_conc / n_cells as f64;
        let du = self.target_average_conc - average_conc;
        self.previous_values.push(du);

        // Make prediction

        // Calculate PID Controller
        let pn = self.previous_values.len();
        let derivative = if pn > 1 {
            (self.previous_values[pn-1] - self.previous_values[pn-2]) / DT
        } else {0.0};
        let integral = self.previous_values.iter().sum::<f64>() * DT;
        let controller_var = self.k_p * du + self.k_d * derivative + self.k_i * integral;

        // Adjust values
        cells.into_iter().for_each(|(c, _)| {
            match c.cell.cellular_reactions.species {
                Species::Sender => {
                    c.cell.cellular_reactions.production_term.add_scalar_mut(controller_var);
                },
                _ => (),
            }
        });
        Ok(())
    }
}
