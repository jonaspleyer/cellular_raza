use cellular_raza::prelude::*;

use nalgebra::DVector;
use num::Zero;
use ode_integrate::solvers::fixed_step::FixedStepSolvers;
use serde::{Deserialize, Serialize};

use crate::{CELL_LIGAND_TURNOVER_RATE, DT};

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

    pub save_path: std::path::PathBuf,
}

pub struct Observable(f64, usize);

#[derive(Clone)]
struct Parameters {
    delay: f64,
    sink: f64,
}

fn predict(
    production_history: &Vec<f64>,
    production_next: f64,
    parameters: &Parameters,
    n_compartments: usize,
    n_steps: usize,
    dt: f64,
) -> Result<Vec<DVector<f64>>, ControllerError> {
    // Define initial values
    let y0 = DVector::from_iterator(n_compartments + 1, (0..n_compartments + 1).map(|_| 2.0));
    let time_to_step = |t: f64| (t / dt) as usize;

    let ode = |y: &DVector<f64>,
               dy: &mut DVector<f64>,
               t: &f64,
               p: &Parameters|
     -> Result<(), CalcError> {
        let max_len = y.len();
        for i in 0..max_len - 1 {
            dy[i + 1] = p.delay * (y[i] - y[i + 1])
        }
        let step = time_to_step(*t);
        dy[0] = if production_history.len() == 0 {
            0.0
        } else if step < production_history.len() {
            production_history[step]
        } else {
            production_next
        };
        dy[max_len - 1] -= p.sink * y[max_len - 1];
        Ok(())
    };

    // Define time
    let t0 = 0.0;
    let steps = production_history.len() + n_steps;
    let t_series = (0..steps).map(|i| t0 + i as f64 * dt).collect::<Vec<_>>();

    // Solve the ode
    let res = ode_integrate::prelude::solve_ode_time_series_single_step_add(
        &y0,
        &t_series,
        &ode,
        &parameters,
        FixedStepSolvers::Rk4,
    )
    .or_else(|e| Err(cellular_raza::concepts::ControllerError(format!("{}", e))))?;
    Ok(res)
}

fn write_line_to_file(save_path: &std::path::Path, line: String) {
    use std::fs::File;
    use std::io::Write;
    let f = File::options()
        .append(true)
        .create(true)
        .open(save_path.join("controller_logs.csv"))
        .unwrap();
    let mut f = std::io::LineWriter::new(f);
    writeln!(f, "{}", line).unwrap();
}

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

        let du = if self.with_mpc {
            // Define parameters for prediction
            let parameters = Parameters {
                delay: 1.0,
                sink: CELL_LIGAND_TURNOVER_RATE,
            };
            let n_compartments = 10;

            // Make prediction
            let mut predicted_production_term = 0.0;
            let mut current_cost = f64::INFINITY;
            for i in 0..10 {
                let new_production_term = i as f64 * 0.1;
                let predicted_series = predict(
                    &self.previous_production_values,
                    new_production_term,
                    &parameters,
                    n_compartments,
                    20,
                    DT,
                )?;

                // Calculate difference to desired value
                let du = self.target_average_conc
                    - match predicted_series.last() {
                        Some(p) => p[n_compartments - 1],
                        None => average_conc,
                    };

                // Set up cost function
                let cost = du.powf(2.0);
                if cost < current_cost {
                    current_cost = cost;
                    predicted_production_term = new_production_term;
                }
            }

            // Write results to file
            let line = format!(
                "{},{},{},{}",
                average_conc,
                current_cost,
                predicted_production_term,
                self.previous_production_values
                    .last()
                    .or_else(|| Some(&0.0))
                    .unwrap()
            );
            write_line_to_file(&self.save_path, line);

            // Compare the predicted necessary production term with the last one which was active
            let res = if self.previous_production_values.len() > 0 {
                predicted_production_term
                    - self.previous_production_values[self.previous_production_values.len() - 1]
            } else {
                0.0
            };
            todo!("Something is missing here still");
            res
        } else {
            self.target_average_conc - average_conc
        };
        self.previous_dus.push(du);

        // Calculate PID Controller
        let pn = self.previous_dus.len();
        let derivative = if pn > 1 {
            (self.previous_dus[pn - 1] - self.previous_dus[pn - 2]) / DT
        } else {
            0.0

            // Write results to file
            let line = format!(
                "{},{},{},{},{},{}",
                average_conc, du, proportional, differential, integral, controller_var
            );
            write_line_to_file(&self.save_path, line);

        };
        let integral = self.previous_dus.iter().sum::<f64>() * DT;

        let proportional = self.k_p * du;
        let differential = self.k_p * self.t_d * derivative;
        let integral = self.k_p * integral / self.t_i;
        let controller_var = proportional + differential + integral;

        // Write results to file
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

        let new_production_term = (self
            .previous_production_values
            .last()
            .or_else(|| Some(&0.0))
            .unwrap()
            + controller_var)
            .max(0.0);

        // Push new production value
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
