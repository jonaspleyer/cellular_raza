use cellular_raza::concepts::{CellularReactions, Controller};
use serde::{Deserialize, Serialize};

use crate::{MyCellType, Species, DT};

#[derive(Clone, Deserialize, Serialize)]
pub struct SRController {
    pub target_concentration: f64,
    previous_dus: Vec<f64>,
    previous_production_values: Vec<f64>,
    pub strategy: ControlStrategy,
}

#[derive(Clone, Deserialize, Serialize)]
pub enum ControlStrategy {
    PID(PIDSettings),
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

impl SRController {
    pub fn new(target_concentration: f64) -> Self {
        Self {
            target_concentration,
            previous_dus: Vec::new(),
            previous_production_values: Vec::new(),
            strategy: ControlStrategy::None,
        }
    }

    pub fn strategy(self, strategy: ControlStrategy) -> Self {
        Self { strategy, ..self }
    }

    fn pid_control<'a, J>(
        &mut self,
        average_concentration: f64,
        _n_cells: usize,
        pid_settings: PIDSettings,
        cells: J,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        J: Iterator<
            Item = (
                &'a mut cellular_raza::prelude::CellAgentBox<MyCellType>,
                &'a mut Vec<cellular_raza::prelude::CycleEvent>,
            ),
        >,
    {
        // Calculate PID Controller
        let pn = self.previous_dus.len();
        if pn == 0 {
            return Ok(());
        }
        let du = self.previous_dus[pn - 1];

        // Calculate the derivative of the last two time points
        let derivative = if pn > 1 {
            (self.previous_dus[pn - 1] - self.previous_dus[pn - 2]) / DT
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
        self.previous_production_values.push(new_production_term);

        // Apply new term to cells
        cells
            .into_iter()
            .for_each(|(cell, _)| match cell.cell.cellular_reactions.species {
                Species::Sender => {
                    cell.cell.cellular_reactions.production_term[0] = new_production_term
                }
                _ => (),
            });
        Ok(())
    }

    fn linear_control(
        &mut self,
        average_concentration: f64,
        n_cells: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn exponential_control(
        &mut self,
        average_concentration: f64,
        n_cells: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
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
        match &self.strategy {
            ControlStrategy::PID(pid_settings) => {
                self.pid_control(average_concentration, n_cells, pid_settings.clone(), cells)
            }
            ControlStrategy::Linear => self.linear_control(average_concentration, n_cells),
            ControlStrategy::Exponential => {
                self.exponential_control(average_concentration, n_cells)
            }
            ControlStrategy::None => Ok(()),
        }
        .unwrap();

        // Write information to file
        Ok(())
    }
}
