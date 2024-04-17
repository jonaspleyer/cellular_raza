use cellular_raza::prelude::*;

use num::Zero;
use serde::{Deserialize, Serialize};

use crate::{
    CELL_LIGAND_TURNOVER_RATE, CELL_MECHANICS_RADIUS, DOMAIN_SIZE, DT, N_CELLS_INITIAL_RECEIVER,
    N_CELLS_INITIAL_SENDER,
};

pub const NUMBER_OF_REACTION_COMPONENTS: usize = 1;
pub type ReactionVector = nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>;
pub type MyCellType = ModularCell<
    NewtonDamped2D,
    MiePotential<3, 1, f64>,
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
    pub t_i: f64,
    pub t_d: f64,
    pub previous_dus: Vec<f64>,
    pub previous_production_values: Vec<f64>,
    pub with_mpc: bool,
    pub prediction_time: f64,
    pub sampling_prod_low: f64,
    pub sampling_prod_high: f64,
    pub sampling_steps: usize,
}

pub struct Observable(f64, usize);

impl Controller<MyCellType, Observable> for ConcentrationController {
    fn measure<'a, I>(&self, cells: I) -> Result<Observable, CalcError>
    where
        I: IntoIterator<Item = &'a CellAgentBox<MyCellType>> + Clone,
    {
        let mut n_cells = 0;
        let mut total_conc = 0.0;
        cells
            .into_iter()
            .for_each(|cell| match cell.cell.cellular_reactions.species {
                Species::Receiver => {
                    total_conc += cell.cell.get_intracellular()[0];
                    n_cells += 1;
                }
                _ => (),
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
        self.previous_dus.push(du);

        let new_production_term = if self.with_mpc {
            // Make prediction
            // Calculate how much ligand was produced already
            let produced_ligand = self.previous_production_values.iter().sum::<f64>()
                * DT
                * N_CELLS_INITIAL_SENDER as f64;

            // Calculate average intracellular concentration so far
            let degraded_ligand = self
                .previous_dus
                .iter()
                .map(|du| CELL_LIGAND_TURNOVER_RATE * (self.target_average_conc - du))
                .sum::<f64>()
                / self.previous_dus.len() as f64;

            // Set up cost function
            let cost_func = |production_term: f64| {
                // Past (known) values
                let previous = produced_ligand - degraded_ligand;

                // Make prediction
                let produced_ligand_predicted =
                    N_CELLS_INITIAL_SENDER as f64 * self.prediction_time * production_term;
                let degraded_ligand_predicted = N_CELLS_INITIAL_RECEIVER as f64
                    * self.prediction_time
                    * CELL_LIGAND_TURNOVER_RATE
                    * self
                        .previous_dus
                        .last()
                        .and_then(|x| Some(self.target_average_conc - x))
                        .or_else(|| Some(0.0))
                        .unwrap();

                // Combine values
                let predicted = produced_ligand_predicted - degraded_ligand_predicted;
                let vol_cells = (N_CELLS_INITIAL_SENDER + N_CELLS_INITIAL_RECEIVER) as f64
                    * CELL_MECHANICS_RADIUS.powf(2.0)
                    * std::f64::consts::PI;

                // Calculate average conc everywhere
                let vol_domain = DOMAIN_SIZE.powf(2.0);
                let approximated_conc = previous + predicted;
                let average_domain_conc = approximated_conc * vol_cells / (vol_domain + vol_cells);

                // Compare to target average_conc
                (self.target_average_conc - average_domain_conc).powf(2.0)
            };

            // Numerically minimize this function
            let production_term_sampling_increment =
                (self.sampling_prod_high - self.sampling_prod_low) / self.sampling_steps as f64;
            let res = (0..self.sampling_steps)
                .map(|i| self.sampling_prod_low + production_term_sampling_increment * i as f64)
                .map(|u| (u, cost_func(u)))
                .fold((0.0, f64::INFINITY), |(x1, x2), (c1, c2)| {
                    println!("{} {} {} {}", x1, x2, c1, c2);
                    if x2 < c2 {
                        (x1, x2)
                    } else {
                        (c1, c2)
                    }
                });

            // Write results to file
            use std::fs::File;
            use std::io::Write;
            let mut f = File::options()
                .append(true)
                .create(true)
                .open("controller_logs.csv")
                .unwrap();
            let line = format!("{},{},{},{}\n", self.target_average_conc - average_conc, average_conc, res.0, res.1);
            f.write(line.as_bytes()).unwrap();

            // If the approach was succesfull, we return the calculated value
            // otherwise retturn the standard one.
            if res.1 < f64::INFINITY {
                res.0.max(0.0)
            } else {
                0.0
            }
        } else {
            // Calculate PID Controller
            let pn = self.previous_dus.len();
            let derivative = if pn > 1 {
                (self.previous_dus[pn - 1] - self.previous_dus[pn - 2]) / DT
            } else {
                0.0
            };
            let integral = self.previous_dus.iter().sum::<f64>() * DT;

            let proportional = self.k_p * du;
            let differential = self.k_p * self.t_d * derivative;
            let integral = self.k_p * integral / self.t_i;
            // let controller_var = self.k_p * (du + self.t_d * derivative + integral / self.t_i);
            let controller_var = proportional + differential + integral;

            use std::fs::File;
            use std::io::Write;
            let mut f = File::options()
                .append(true)
                .create(true)
                .open("controller_logs.csv")
                .unwrap();
            let line = format!(
                "{},{},{},{},{},{}\n",
                average_conc, du, proportional, differential, integral, controller_var
            );
            f.write(line.as_bytes()).unwrap();

            (self
                .previous_production_values
                .last()
                .or_else(|| Some(&0.0))
                .unwrap()
                + controller_var)
                .max(0.0)
        };
        self.previous_production_values.push(new_production_term);

        // Adjust values
        cells
            .into_iter()
            .for_each(|(c, _)| match c.cell.cellular_reactions.species {
                Species::Sender => {
                    c.cell.cellular_reactions.production_term[0] = new_production_term;
                }
                _ => (),
            });
        Ok(())
    }
}
