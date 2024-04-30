use cellular_raza::prelude::*;

use nalgebra::DVector;
use ode_integrate::solvers::fixed_step::FixedStepSolvers;
use serde::{Deserialize, Serialize};

use crate::{
    ODEParameters,
    write_line_to_file,
    CELL_LIGAND_TURNOVER_RATE, CELL_MECHANICS_RADIUS, DOMAIN_SIZE, DT, N_CELLS_INITIAL_RECEIVER,
    VOXEL_LIGAND_DIFFUSION_CONSTANT,
};

pub const NUMBER_OF_REACTION_COMPONENTS: usize = 1;
pub type ReactionVector = nalgebra::SVector<f64, NUMBER_OF_REACTION_COMPONENTS>;
pub type MyCellType = ModularCell<
    NewtonDamped2D,
    MyInteraction,
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
    pub intracellular: ReactionVector,
    pub species: Species,
    pub sink_rate: ReactionVector,
    pub production_term: ReactionVector,
    pub uptake: ReactionVector,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MyInteraction {
    pub cell_radius: f64,
    pub potential_strength: f64,
    pub relative_interaction_range: f64,
}

impl Interaction<
    nalgebra::Vector2<f64>,
    nalgebra::Vector2<f64>,
    nalgebra::Vector2<f64>,
    f64
>
    for MyInteraction
{
    fn calculate_force_between(
        &self,
        own_pos: &nalgebra::Vector2<f64>,
        _own_vel: &nalgebra::Vector2<f64>,
        ext_pos: &nalgebra::Vector2<f64>,
        _ext_vel: &nalgebra::Vector2<f64>,
        ext_radius: &f64,
    ) -> Result<nalgebra::Vector2<f64>, CalcError> {
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
                            return Ok([0.0; 2].into());
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
        let attracting_force = dir * strength.max(0.0) * spatial_cutoff;
        let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

        Ok(repelling_force + attracting_force)
    }

    fn get_interaction_information(&self) -> f64 {
        self.cell_radius
    }
}

impl CellularReactions<ReactionVector> for OwnReactions {
    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        i: &ReactionVector,
        e: &ReactionVector,
    ) -> Result<(ReactionVector, ReactionVector), CalcError> {
        Ok(match self.species {
            Species::Sender => ([0.0].into(), self.production_term),
            Species::Receiver => (
                self.uptake * (e - i) - self.sink_rate * i,
                -self.uptake * (e - i),
            ),
        })
    }

    fn get_intracellular(&self) -> ReactionVector {
        self.intracellular
    }
    fn set_intracellular(&mut self, c: ReactionVector) {
        self.intracellular = c;
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ConcentrationController {
    pub target_average_conc: f64,
    pub k_p: f64,
    pub t_i: f64,
    pub t_d: f64,
    pub previous_dus: Vec<f64>,
    pub previous_dvs: Vec<f64>,

    pub previous_production_values: Vec<f64>,

    pub control_method: ControlScheme,
    pub prediction_time: f64,
    pub sampling_prod_low: f64,
    pub sampling_prod_high: f64,
    pub sampling_steps: usize,

    pub save_path: std::path::PathBuf,
}

pub struct Observable(f64, usize);

pub fn predict(
    production_history: &Vec<f64>,
    production_next: f64,
    parameters: &ODEParameters,
    n_compartments: usize,
    n_steps: usize,
    dt: f64,
) -> Result<Vec<DVector<f64>>, ControllerError> {
    // Define initial values: interpolate between current production_rate_next * dt and the known
    // current concentration at the position of the cells.
    let y0 = DVector::from_iterator(n_compartments + 1, (0..n_compartments + 1).map(|_| 0.0));
    let time_to_step = |t: f64| (t / dt) as usize;

    let ode = |y: &DVector<f64>,
               dy: &mut DVector<f64>,
               t: &f64,
               p: &ODEParameters|
     -> Result<(), CalcError> {
        let max_len = y.len();
        for i in 0..max_len - 2 {
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
        dy[max_len - 1] = p.delay * y[max_len - 2] - p.sink * y[max_len - 1];
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
    // for r in res.iter() {
    //     for ri in r.iter() {
    //         print!("{:9.4} ", ri);
    //     }
    //     println!("");
    // }
    Ok(res)
}

#[derive(Clone, Deserialize, Serialize)]
pub enum ControlScheme {
    PID,
    DelayODE,
    LinearFit,
    ExponentialFit,
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
        assert!(average_conc.is_nan() == false);

        use ControlScheme::*;
        let new_production_term = match self.control_method {
            DelayODE => {
                // Define parameters for prediction
                let n_compartments = 6;
                let parameters = ODEParameters {
                    delay: VOXEL_LIGAND_DIFFUSION_CONSTANT / DOMAIN_SIZE.powf(2.0)
                        * ((n_compartments + 1) as f64).powf(2.0),
                    sink: CELL_LIGAND_TURNOVER_RATE
                        * N_CELLS_INITIAL_RECEIVER as f64
                        * CELL_MECHANICS_RADIUS.powf(2.0)
                        / DOMAIN_SIZE.powf(2.0)
                        * ((n_compartments + 1) as f64).powf(2.0),
                };

                // Make prediction
                use rayon::prelude::*;
                use std::f64;
                let (cost, predicted_production_term, predicted_conc) = (0..self.sampling_steps)
                    .into_par_iter()
                    .map(|i| {
                        let q = i as f64 / (self.sampling_steps - 1) as f64;
                        let tested_production_term = self.sampling_prod_low
                            + q * (self.sampling_prod_high - self.sampling_prod_low);
                        let predicted_series = predict(
                            &self.previous_production_values,
                            tested_production_term,
                            &parameters,
                            n_compartments,
                            (self.prediction_time / DT).floor() as usize,
                            DT,
                        )
                        .unwrap();

                        // Calculate difference to desired value
                        let current_predicted_conc =
                            predicted_series.last().unwrap()[n_compartments];
                        let du = self.target_average_conc - current_predicted_conc;
                        let dv = self.target_average_conc - average_conc;

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
                    average_conc, cost, predicted_production_term, predicted_conc,
                );
                write_line_to_file(&self.save_path, line);

                // From that calculate the new production term
                predicted_production_term
            }
            LinearFit => {
                // Interpolation strategy
                let n_values = self.previous_dus.len();
                debug_assert_eq!(n_values, self.previous_production_values.len());

                if n_values > 0 {
                    let n_prediction_steps =
                        ((self.prediction_time / DT).floor() as usize).min(n_values);

                    // Make linear prediction
                    let predictor = |tested_production_term: f64, output: &mut String| -> f64 {
                        // Concentration value at previous time point
                        let y0 = self.target_average_conc
                            - self.previous_dus[n_values - n_prediction_steps];
                        // Production term at previous time point
                        let a0 = self.previous_production_values[n_values - n_prediction_steps];
                        // Concentration currently
                        let y1 = average_conc;
                        // Tested production_term
                        let a1 = tested_production_term;

                        // Calculate factor in linear formula
                        let integrated_production_values = (n_values - n_prediction_steps
                            ..n_values)
                            .map(|i| self.previous_production_values[i] * DT)
                            .sum::<f64>();
                        let factor = (y1 - y0) / integrated_production_values;

                        // Make a prediction with a linear model
                        let y2 = y1 + a1 * factor * self.prediction_time;
                        output.extend(format!("prod: {:7.3} y2: {:7.3} y1: {:7.3} y0: {:7.3} a0: {:7.3} a1: {:7.3}", tested_production_term, y2, y1, y0, a0, a1).chars());

                        // Return the prediction
                        y2
                    };

                    let partition_cost = 0.5;
                    // Set up cost function
                    let cost_func = |predicted_conc: f64| {
                        let du = self.target_average_conc - average_conc;
                        let dv = self.target_average_conc - predicted_conc;
                        partition_cost * du.powf(2.0) + (1.0 - partition_cost) * dv.powf(2.0)
                    };

                    // Optimize this cost function
                    use rayon::prelude::*;
                    let (cost, predicted_production_term, predicted_conc) = (0..self
                        .sampling_steps)
                        .into_iter()
                        .into_par_iter()
                        .map(|i| {
                            let mut output = String::new();
                            let q = i as f64 / (self.sampling_steps - 1) as f64;
                            let tested_production_term = self.sampling_prod_low
                                + q * (self.sampling_prod_high - self.sampling_prod_low);
                            let predicted_conc = predictor(tested_production_term, &mut output);
                            let cost = cost_func(predicted_conc);
                            println!("cost: {:7.3} {}", cost, output);
                            (cost, tested_production_term, predicted_conc)
                        })
                        .reduce(
                            || (f64::INFINITY, 0.0, 0.0),
                            |x, y| if x.0 < y.0 { x } else { y },
                        );
                    // println!("Picked: {:7.3} {:7.3} {:7.3}", cost, predicted_production_term, predicted_conc);

                    let line = format!(
                        "{},{},{},{}",
                        average_conc, cost, predicted_production_term, predicted_conc,
                    );
                    write_line_to_file(&self.save_path, line);
                    predicted_production_term

                    /*
                    // Use PID Controller to handle setting production term
                    // Calculate PID Controller for du
                    let du_pn = self.previous_dus.len();
                    let du = self.target_average_conc - average_conc;
                    let du_derivative = if du_pn > 1 {
                        (self.previous_dus[du_pn - 1] - self.previous_dus[du_pn - 2]) / DT
                    } else {
                        0.0
                    };
                    let du_proportional = self.k_p * du;
                    let du_differential = self.k_p * self.t_d * du_derivative;
                    let du_integral =
                        self.k_p / self.t_i * self.previous_dus.iter().sum::<f64>() * DT;
                    let du_controller_var = du_proportional + du_differential + du_integral;

                    // Calculate PID Controller for dv
                    let dv_pn = self.previous_dvs.len();
                    let dv = self.target_average_conc - predicted_conc;
                    self.previous_dvs.push(dv);
                    let dv_derivative = if dv_pn > 1 {
                        (self.previous_dvs[dv_pn - 1] - self.previous_dvs[dv_pn - 2]) / DT
                    } else {
                        0.0
                    };
                    let dv_proportional = self.k_p * dv;
                    let dv_differential = self.k_p * self.t_d * dv_derivative;
                    let dv_integral =
                        self.k_p / self.t_i * self.previous_dvs.iter().sum::<f64>() * DT;
                    let dv_controller_var = dv_proportional + dv_differential + dv_integral;

                    // Combine both results
                    let controller_var = partition_cost * du_controller_var
                        + (1.0 - partition_cost) * dv_controller_var;
                    println!("{} {}", du_controller_var, dv_controller_var);
                    (self
                        .previous_production_values
                        .last()
                        .or_else(|| Some(&0.0))
                        .unwrap()
                        + controller_var)
                        .max(0.0)*/
                } else {
                    0.0
                }
            }
            ExponentialFit => {
                let n_values = self.previous_production_values.len();
                let n_prediction_steps =
                    ((self.prediction_time / DT).floor() as usize).min(n_values);
                if n_values < n_prediction_steps {
                    0.0
                } else {
                    let predictor = |production_term: f64| -> f64 {
                        // Calculate the predicted concentration depending on the sink and
                        // source term with an exponential approximation
                        let a = production_term;
                        let b = CELL_LIGAND_TURNOVER_RATE;

                        let y0 = average_conc * b / a;
                        let time_interval = self.prediction_time;
                        let predicted_conc_future =
                            a / b * (1.0 + (y0 - 1.0) * (-b * time_interval).exp());

                        println!(
                            "{:7.3} {:7.3} {:7.3} {:7.3}",
                            a, b, average_conc, predicted_conc_future
                        );

                        predicted_conc_future
                    };

                    let cost_func = |predicted_conc: f64| {
                        let du = self.target_average_conc - average_conc;
                        let dv = self.target_average_conc - predicted_conc;
                        du.abs() * dv.abs()
                    };

                    use rayon::prelude::*;
                    let (cost, predicted_production_term, predicted_conc) = (0..self
                        .sampling_steps)
                        .into_par_iter()
                        .map(|i| {
                            // Calculate variable to test
                            let tested_production_term = self.sampling_prod_low
                                + i as f64 / (self.sampling_steps - 1) as f64
                                    * (self.sampling_prod_high - self.sampling_prod_low);

                            // Calculate prediction for concentration in future
                            let predicted_final_conc = predictor(tested_production_term);

                            // Calculate cost associated with that
                            let cost = cost_func(predicted_final_conc);

                            (cost, tested_production_term, predicted_final_conc)
                        })
                        .reduce(
                            || (f64::INFINITY, 0.0, 0.0),
                            |x, y| if x.0 < y.0 { x } else { y },
                        );
                    println!(
                        "Chose {:7.3} {:7.3} {:7.3}",
                        predicted_production_term, cost, predicted_conc
                    );

                    let line = format!(
                        "{},{},{},{}",
                        average_conc, cost, predicted_production_term, predicted_conc,
                    );
                    write_line_to_file(&self.save_path, line);
                    predicted_production_term
                }
            }
            PID => {
                let du = self.target_average_conc - average_conc;

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
                let controller_var = proportional + differential + integral;

                // Write results to file
                let line = format!(
                    "{},{},{},{},{},{}",
                    average_conc, du, proportional, differential, integral, controller_var
                );
                write_line_to_file(&self.save_path, line);

                (self
                    .previous_production_values
                    .last()
                    .or_else(|| Some(&0.0))
                    .unwrap()
                    + controller_var)
                    .max(0.0)
            }
        };

        // Push new values
        let du = self.target_average_conc - average_conc;
        self.previous_dus.push(du);
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
