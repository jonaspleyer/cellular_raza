use cellular_raza::concepts::domain_old::*;
use cellular_raza::concepts::reactions_old::*;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Clone, Deserialize, Serialize)]
pub enum SpatialSetup {
    Default,
    Circular,
    Branch,
}

fn create_cell(pos: [f64; 2], species: Species) -> MyCellType {
    let pos = Vector2::from(pos);
    ModularCell {
        mechanics: NewtonDamped2D {
            pos,
            vel: Vector2::zero(),
            damping_constant: CELL_MECHANICS_DAMPING,
            mass: 1.0,
        },
        interaction: MyInteraction {
            cell_radius: CELL_MECHANICS_RADIUS,
            potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
            relative_interaction_range: 1.5,
        },
        interaction_extracellular: NoExtracellularGradientSensing,
        cycle: NoCycle,
        cellular_reactions: OwnReactions {
            species,
            intracellular: [0.0].into(),
            production_term: [0.0].into(),
            sink_rate: [CELL_LIGAND_TURNOVER_RATE].into(),
            uptake: [CELL_LIGAND_UPTAKE_RATE].into(),
        },
        volume: std::f64::consts::PI * CELL_MECHANICS_RADIUS.powf(2.0),
    }
}

fn default_setup(
    rng: &mut ChaCha8Rng,
) -> Result<(CartesianCuboid2, Vec<MyCellType>), SimulationError> {
    let domain = create_domain(DOMAIN_SIZE)?;
    let cells = (0..N_CELLS_INITIAL_SENDER + N_CELLS_INITIAL_RECEIVER)
        .map(|n_cell| {
            let y = DOMAIN_SIZE * rng.gen_range(0.3..0.7);
            let (species, x) = if n_cell < N_CELLS_INITIAL_SENDER {
                let x = DOMAIN_SIZE * rng.gen_range(0.0..0.2);
                (Species::Sender, x)
            } else {
                let x = DOMAIN_SIZE * rng.gen_range(0.8..1.0);
                (Species::Receiver, x)
            };
            create_cell([x, y], species)
        })
        .collect();
    Ok((domain, cells))
}

fn circular_setup(
    rng: &mut ChaCha8Rng,
) -> Result<(CartesianCuboid2, Vec<MyCellType>), SimulationError> {
    const NEW_DOMAIN_SIZE: f64 = 1.5 * DOMAIN_SIZE;
    let domain = create_domain(NEW_DOMAIN_SIZE)?;
    let cells = (0..N_CELLS_INITIAL_SENDER + N_CELLS_INITIAL_RECEIVER)
        .map(|n_cell| {
            let phi = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
            let (species, r) = if n_cell < N_CELLS_INITIAL_SENDER {
                let r = 0.5 * NEW_DOMAIN_SIZE * rng.gen_range(0.7..1.0);
                (Species::Sender, r)
            } else {
                let r = 0.5 * NEW_DOMAIN_SIZE * rng.gen_range(0.0..0.3);
                (Species::Receiver, r)
            };
            create_cell(
                [
                    0.5 * NEW_DOMAIN_SIZE + r * phi.cos(),
                    0.5 * NEW_DOMAIN_SIZE + r * phi.sin(),
                ],
                species,
            )
        })
        .collect();
    Ok((domain, cells))
}

fn branch_setup(
    rng: &mut ChaCha8Rng,
) -> Result<(CartesianCuboid2, Vec<MyCellType>), SimulationError> {
    let domain = create_domain(DOMAIN_SIZE)?;
    let cells = (0..N_CELLS_INITIAL_SENDER + N_CELLS_INITIAL_RECEIVER)
        .map(|n_cell| {
            let (species, x, y) = if n_cell < N_CELLS_INITIAL_SENDER {
                let y = DOMAIN_SIZE * rng.gen_range(0.3..0.7);
                let x = DOMAIN_SIZE * rng.gen_range(0.0..0.2);
                (Species::Sender, x, y)
            } else {
                let x = rng.gen_range(0.0..0.4 * DOMAIN_SIZE);
                let y = 0.5 * DOMAIN_SIZE + rng.gen_range(-1_f64..1_f64).signum() * x;
                (Species::Receiver, DOMAIN_SIZE - x, y)
            };
            create_cell([x, y], species)
        })
        .collect();
    Ok((domain, cells))
}

impl SpatialSetup {
    pub fn generate_domain_cells(
        &self,
        rng: &mut ChaCha8Rng,
    ) -> Result<(CartesianCuboid2, impl IntoIterator<Item = MyCellType>), SimulationError> {
        match &self {
            SpatialSetup::Default => default_setup(rng),
            SpatialSetup::Circular => circular_setup(rng),
            SpatialSetup::Branch => branch_setup(rng),
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            SpatialSetup::Default => "spatial_default",
            SpatialSetup::Circular => "spatial_circular",
            SpatialSetup::Branch => "spatial_branch",
        }
        .to_owned()
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SRController {
    pub target_concentration: f64,
    pub production_value_max: f64,
    previous_dus: Vec<f64>,
    previous_production_values: Vec<f64>,
    pub strategy: ControlStrategy,
    pub observer: Observer,
    pub save_path: std::path::PathBuf,
}

#[derive(Clone, Deserialize, Serialize)]
pub enum ControlStrategy {
    PID(PIDSettings),
    DelayODE(DelayODESettings),
}

impl ControlStrategy {
    pub fn to_string(&self) -> String {
        match self {
            ControlStrategy::DelayODE(_) => "delay_ode",
            ControlStrategy::PID(_) => "pid",
        }
        .to_owned()
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub enum Observer {
    Standard,
    Predictor { weighting: f64 },
}

impl Observer {
    pub fn to_string(&self) -> String {
        match self {
            Observer::Standard => "obs_standard",
            #[allow(unused)]
            Observer::Predictor { weighting } => "obs_predictor",
        }
        .to_owned()
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct PIDSettings {
    /// Proportionality constant
    pub k_p: f64,
    /// Time scale of the differential part
    pub t_d: f64,
    /// Time scale of the integral part
    pub t_i: f64,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct DelayODESettings {
    pub sampling_prod_low: f64,
    pub sampling_prod_high: f64,
    pub sampling_steps: usize,
    pub prediction_time: f64,
}

impl Default for PIDSettings {
    fn default() -> Self {
        PIDSettings {
            k_p: 0.075 * MOLAR / MINUTE,
            t_d: 1.0 * MINUTE,
            t_i: 10.0 * MINUTE,
        }
    }
}

impl Default for DelayODESettings {
    fn default() -> Self {
        Self {
            sampling_prod_low: 0.04 * MOLAR / SECOND,
            sampling_prod_high: 0.06 * MOLAR / SECOND,
            sampling_steps: 3,
            prediction_time: 4.0 * MINUTE,
        }
    }
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
    pub sink: f64,
    pub diffusion: f64,
    pub dx: f64,
}

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
    let time_to_step = |t: f64| (t / dt).round() as usize;

    let ode = |y: &DVector<f64>,
               dy: &mut DVector<f64>,
               t: &f64,
               p: &ODEParameters|
     -> Result<(), CalcError> {
        let max_len = y.len();
        let current_step = time_to_step(*t);
        if current_step == 0 {
            debug_assert_eq!(*t, 0.0);
        }
        let alpha = if production_history.len() == 0 {
            0.0
        } else if current_step < production_history.len().min(n_steps) {
            let start_index = production_history.len() - production_history.len().min(n_steps);
            debug_assert!(start_index + current_step < production_history.len());
            production_history[start_index + current_step]
        } else {
            production_next
        };
        dy[0] = alpha + p.diffusion / p.dx.powf(2.0) * (y[1] - y[0]);
        dy[max_len - 1] = -p.sink * y[max_len - 1]
            + p.diffusion / p.dx.powf(2.0) * (y[max_len - 2] - y[max_len - 1]);
        for i in 1..max_len - 1 {
            dy[i] = p.diffusion / p.dx.powf(2.0) * (y[i + 1] - 2.0 * y[i] + y[i - 1]);
        }
        Ok(())
    };

    // Define time
    let t0 = 0.0;
    let total_steps = production_history.len().min(n_steps) + n_steps;
    let t_series = (0..total_steps)
        .map(|i| t0 + i as f64 * dt)
        .collect::<Vec<_>>();

    // Solve the ode
    let res = ode_integrate::prelude::solve_ode_time_series_single_step_add(
        &y0,
        &t_series,
        &ode,
        &parameters,
        ode_integrate::prelude::Rk4,
    )
    .or_else(|e| Err(cellular_raza::concepts::ControllerError(format!("{}", e))))?;
    Ok(res)
}

impl SRController {
    pub fn new(
        target_concentration: f64,
        production_value_max: f64,
        save_path: &std::path::Path,
    ) -> Self {
        Self {
            target_concentration,
            production_value_max,
            previous_dus: Vec::new(),
            previous_production_values: Vec::new(),
            strategy: ControlStrategy::PID(PIDSettings::default()),
            observer: Observer::Standard,
            save_path: save_path.into(),
        }
    }

    pub fn strategy(self, strategy: ControlStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn observer(self, observer: Observer) -> Self {
        Self { observer, ..self }
    }

    fn observer_standard(&mut self, average_concentration: f64) -> f64 {
        self.target_concentration - average_concentration
    }

    fn observer_predictor(&mut self, average_concentration: f64, weighting: f64) -> f64 {
        let alpha = self
            .previous_production_values
            .last()
            .or_else(|| Some(&0.0))
            .unwrap()
            .min(self.production_value_max);
        let beta = CELL_LIGAND_TURNOVER_RATE;
        let predicted_conc = alpha / beta;
        let dv = self.target_concentration - predicted_conc;
        let du = self.target_concentration - average_concentration;
        let q = weighting.min(1.0).max(0.0);
        q * du + (1.0 - q) * dv
    }

    fn pid_control(
        &mut self,
        average_concentration: f64,
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
        write_line_to_file(&self.save_path.join("pid_controller.csv"), line);

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
        let n_compartments = 2;
        let parameters = ODEParameters {
            diffusion: VOXEL_LIGAND_DIFFUSION_CONSTANT,
            dx: DOMAIN_SIZE / 12.0,
            sink: CELL_LIGAND_TURNOVER_RATE,
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
                // Set up cost function
                let current_predicted_conc = predicted_series.last().unwrap()[n_compartments];
                let du = self.target_concentration - current_predicted_conc;
                let current_cost = du.powf(2.0);

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
        write_line_to_file(&self.save_path.join("delay_ode_mpc.csv"), line);
        Ok(predicted_production_term)
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
        let du = match self.observer {
            Observer::Standard => self.observer_standard(average_concentration),
            Observer::Predictor { weighting } => {
                self.observer_predictor(average_concentration, weighting)
            }
        };
        // let du = self.target_concentration - average_concentration;
        self.previous_dus.push(du);

        // Apply chosen control strategy
        let new_production_value = match &self.strategy {
            ControlStrategy::PID(pid_settings) => {
                self.pid_control(average_concentration, pid_settings.clone())
            }
            ControlStrategy::DelayODE(settings) => {
                self.delay_ode_control(average_concentration, n_cells, settings.clone())
            }
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
