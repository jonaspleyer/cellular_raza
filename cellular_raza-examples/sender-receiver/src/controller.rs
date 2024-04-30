use cellular_raza::concepts::{CellularReactions, Controller};
use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Clone, Deserialize, Serialize)]
pub struct SRController {
    pub target_concentration: f64,
    pub production_value_max: f64,
    previous_dus: Vec<f64>,
    previous_production_values: Vec<f64>,
    pub strategy: ControlStrategy,
}

#[derive(Clone, Deserialize, Serialize)]
pub enum ControlStrategy {
    PID(PIDSettings),
    DelayODE(DelayODESettings),
    Linear,
    Exponential,
    None,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct PIDSettings {
    /// Proportionality constant
    pub k_p: f64,
    /// Time scale of the differential part
    pub t_d: f64,
    /// Time scale of the integral part
    pub t_i: f64,
    /// Path where to save results
    pub save_path: std::path::PathBuf,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct DelayODESettings {
    pub sampling_prod_low: f64,
    pub sampling_prod_high: f64,
    pub sampling_steps: usize,
    pub prediction_time: f64,
    pub save_path: std::path::PathBuf,
}

pub fn write_line_to_file(save_path: &std::path::Path, line: String) {
    use std::fs::File;
    use std::io::Write;
    let f = File::options()
        .append(true)
        .create(true)
        .open(save_path)
        .unwrap();
    let mut f = std::io::LineWriter::new(f);
    writeln!(f, "{}", line).unwrap();
}

#[derive(Clone)]
pub struct ODEParameters {
    pub delay: f64,
    pub sink: f64,
}

impl SRController {
    pub fn new(target_concentration: f64, production_value_max: f64) -> Self {
        Self {
            target_concentration,
            production_value_max,
            previous_dus: Vec::new(),
            previous_production_values: Vec::new(),
            strategy: ControlStrategy::None,
        }
    }

    pub fn strategy(self, strategy: ControlStrategy) -> Self {
        Self { strategy, ..self }
    }

    fn pid_control(
        &mut self,
        average_concentration: f64,
        _n_cells: usize,
        pid_settings: PIDSettings,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // Calculate PID Controller
        let p_n = self.previous_dus.len();
        if p_n == 0 {
            return Ok(0.0);
        }
        let du = self.previous_dus[p_n - 1];

        // Calculate the derivative of the last two time points
        let derivative = if p_n > 1 {
            (self.previous_dus[p_n - 1] - self.previous_dus[p_n - 2]) / DT
        } else {
            0.0
        };
        let integral = self.previous_dus.iter().sum::<f64>() * DT;

        let proportional = pid_settings.k_p * du;
        let differential = pid_settings.k_p * pid_settings.t_d * derivative;
        let integral = pid_settings.k_p * integral / pid_settings.t_i;
        let controller_var = proportional + differential + integral;

        // Write results to file
        let line = format!(
            "{},{},{},{},{},{}",
            average_concentration, du, proportional, differential, integral, controller_var
        );
        write_line_to_file(&pid_settings.save_path, line);

        // Calculate new production term by incrementing old one
        let new_production_term = (self
            .previous_production_values
            .last()
            .or_else(|| Some(&0.0))
            .unwrap()
            + controller_var)
            .max(0.0);

        Ok(new_production_term)
    }

    fn delay_ode_control(
        &mut self,
        average_concentration: f64,
        _n_cells: usize,
        settings: DelayODESettings,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // Define parameters of the ODE we are solving
        let n_compartments = 6;
        let parameters = ODEParameters {
            delay: VOXEL_LIGAND_DIFFUSION_CONSTANT / DOMAIN_SIZE.powf(2.0)
                * ((n_compartments + 1) as f64).powf(2.0),
            sink: 15.0
                * CELL_LIGAND_TURNOVER_RATE
                * N_CELLS_INITIAL_RECEIVER as f64
                * CELL_MECHANICS_RADIUS.powf(2.0)
                / DOMAIN_SIZE.powf(2.0)
                * ((n_compartments + 1) as f64).powf(2.0),
        };

        // Make prediction
        use rayon::prelude::*;
        use std::f64;
        let (cost, predicted_production_term, predicted_conc) = (0..settings.sampling_steps)
            .into_par_iter()
            .map(|i| {
                let q = i as f64 / (settings.sampling_steps - 1) as f64;
                let tested_production_term = settings.sampling_prod_low
                    + q * (settings.sampling_prod_high - settings.sampling_prod_low);
                let predicted_series = predict(
                    &self.previous_production_values,
                    tested_production_term,
                    &parameters,
                    n_compartments,
                    (settings.prediction_time / DT).floor() as usize,
                    DT,
                )
                .unwrap();

                // Calculate difference to desired value
                let current_predicted_conc = predicted_series.last().unwrap()[n_compartments];
                let du = self.target_concentration - current_predicted_conc;
                let dv = self.target_concentration - average_concentration;

                // Set up cost function
                let current_cost = 0.75 * du.powf(2.0) + 0.25 * dv.powf(2.0);

                (current_cost, tested_production_term, current_predicted_conc)
            })
            .reduce(
                || (f64::INFINITY, 0.0, 0.0),
                |x, y| if x.0 <= y.0 { x } else { y },
            );

        // Write results to file
        let line = format!(
            "{},{},{},{}",
            average_concentration, cost, predicted_production_term, predicted_conc,
        );
        write_line_to_file(&settings.save_path, line);
        Ok(predicted_production_term)
    }

    fn linear_control(
        &mut self,
        _average_concentration: f64,
        _n_cells: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        unimplemented!();
    }

    fn exponential_control(
        &mut self,
        _average_concentration: f64,
        _n_cells: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        unimplemented!();
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SRObservable(f64, usize);

impl Controller<MyCellType, SRObservable> for SRController {
    fn measure<'a, I>(&self, cells: I) -> Result<SRObservable, cellular_raza::prelude::CalcError>
    where
        MyCellType: 'a + Serialize + for<'b> Deserialize<'b>,
        I: IntoIterator<Item = &'a cellular_raza::prelude::CellAgentBox<MyCellType>> + Clone,
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
        Ok(SRObservable(total_conc, n_cells))
    }

    fn adjust<'a, 'b, I, J>(
        &mut self,
        measurements: I,
        cells: J,
    ) -> Result<(), cellular_raza::prelude::ControllerError>
    where
        SRObservable: 'a,
        MyCellType: 'b + Serialize + for<'c> Deserialize<'c>,
        I: Iterator<Item = &'a SRObservable>,
        J: Iterator<
            Item = (
                &'b mut cellular_raza::prelude::CellAgentBox<MyCellType>,
                &'b mut Vec<cellular_raza::prelude::CycleEvent>,
            ),
        >,
    {
        // Calculate the current concentration
        let (total_concentration, n_cells) = measurements
            .into_iter()
            .fold((0.0, 0), |(total_conc, n_cells), SRObservable(c1, n1)| {
                (total_conc + c1, n_cells + n1)
            });
        let average_concentration = total_concentration / n_cells as f64;
        let du = self.target_concentration - average_concentration;
        self.previous_dus.push(du);

        // Apply chosen control strategy
        let new_production_value = match &self.strategy {
            ControlStrategy::PID(pid_settings) => {
                self.pid_control(average_concentration, n_cells, pid_settings.clone())
            }
            ControlStrategy::DelayODE(settings) => {
                self.delay_ode_control(average_concentration, n_cells, settings.clone())
            }
            ControlStrategy::Linear => self.linear_control(average_concentration, n_cells),
            ControlStrategy::Exponential => {
                self.exponential_control(average_concentration, n_cells)
            }
            ControlStrategy::None => Ok(0.0),
        }
        .unwrap();

        self.previous_production_values.push(new_production_value);

        // Apply new term to cells
        cells
            .into_iter()
            .for_each(|(cell, _)| match cell.cell.cellular_reactions.species {
                Species::Sender => {
                    cell.cell.cellular_reactions.production_term[0] =
                        new_production_value.min(self.production_value_max).max(0.0);
                }
                _ => (),
            });
        Ok(())
    }
}
