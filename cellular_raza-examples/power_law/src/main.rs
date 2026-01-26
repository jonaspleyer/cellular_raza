use cellular_raza::prelude::*;

use clap::Parser;
use nalgebra::Vector3;
use num::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

type P = nalgebra::Vector3<f64>;
type Inf = (f64, Healthstate);

#[derive(Clone, Deserialize, Serialize)]
enum Healthstate {
    Healthy,
    Selfinfected,
    Infected,
}

impl Healthstate {
    fn is_infected(&self) -> bool {
        match self {
            Healthstate::Healthy => false,
            Healthstate::Selfinfected | Healthstate::Infected => true,
        }
    }
}

#[derive(CellAgent, Clone, Deserialize, Serialize)]
struct Cell {
    interaction: MorsePotential,
    #[Mechanics]
    mechanics: Langevin3D,
    health_state: Healthstate,
    death_range: f64,
    infection_rate_others: f64,
    infection_rate_self: f64,
    neighbor_infected: usize,
    growth_rate: f64,
    death_rate: f64,
    division_size: f64,
}

impl NeighborSensing<P, usize, Inf> for Cell {
    fn accumulate_information(
        &self,
        own_pos: &P,
        ext_pos: &P,
        ext_inf: &Inf,
        accumulator: &mut usize,
    ) -> Result<(), CalcError> {
        if (own_pos - ext_pos).norm() < self.death_range && ext_inf.1.is_infected() {
            *accumulator += 1;
        }
        Ok(())
    }

    fn react_to_neighbors(&mut self, accumulator: &usize) -> Result<(), CalcError> {
        self.neighbor_infected = *accumulator;
        Ok(())
    }

    fn clear_accumulator(accumulator: &mut usize) {
        *accumulator = 0;
    }
}

impl InteractionInformation<Inf> for Cell {
    fn get_interaction_information(&self) -> Inf {
        (self.interaction.radius, self.health_state.clone())
    }
}

impl Interaction<P, P, P, Inf> for Cell {
    fn calculate_force_between(
        &self,
        own_pos: &P,
        own_vel: &P,
        ext_pos: &P,
        ext_vel: &P,
        ext_info: &Inf,
    ) -> Result<(P, P), CalcError> {
        self.interaction
            .calculate_force_between(own_pos, own_vel, ext_pos, ext_vel, &ext_info.0)
    }
}

impl Cycle<Cell> for Cell {
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Cell,
    ) -> Option<CycleEvent> {
        let a = cell.neighbor_infected as f64;
        // Probability to infect from others
        let cond1 = rng.random_bool(cell.infection_rate_others * dt * a / (1.0 + a));
        // Probability to infect by itself
        let cond2 = rng.random_bool(cell.infection_rate_self * dt);

        if cond1 {
            cell.health_state = Healthstate::Infected;
            Some(CycleEvent::PhasedDeath)
        } else if cond2 {
            cell.health_state = Healthstate::Selfinfected;
            Some(CycleEvent::PhasedDeath)
        } else if cell.interaction.radius >= cell.division_size {
            Some(CycleEvent::Division)
        } else {
            cell.interaction.radius += dt * cell.growth_rate;
            cell.interaction.cutoff = 2.0 * cell.interaction.radius;
            None
        }
    }

    fn divide(rng: &mut rand_chacha::ChaCha8Rng, c1: &mut Cell) -> Result<Cell, DivisionError> {
        use rand::Rng;
        // Clone existing cell
        let mut c2 = c1.clone();

        let r = c1.interaction.radius;

        // Make both cells smaller
        c1.interaction.radius /= core::f64::consts::SQRT_2;
        c2.interaction.radius /= core::f64::consts::SQRT_2;
        c1.interaction.cutoff = 2.0 * c1.interaction.radius;
        c2.interaction.cutoff = 2.0 * c1.interaction.radius;

        // Generate cellular splitting direction randomly
        let theta = rng.random_range(0.0..2.0 * core::f64::consts::PI);
        let phi = rng.random_range(0.0..2.0 * core::f64::consts::PI);
        let dir_vec = nalgebra::Vector3::from([
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ]);

        // Define new positions for cells
        // It is randomly chosen if the old cell is left or right
        let offset = dir_vec * r / std::f64::consts::SQRT_2;
        let old_pos = c1.pos();

        c1.set_pos(&(old_pos + offset));
        c2.set_pos(&(old_pos - offset));

        Ok(c2)
    }

    fn update_conditional_phased_death(
        _: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Cell,
    ) -> Result<bool, DeathError> {
        cell.interaction.radius -= cell.death_rate * dt;
        cell.interaction.cutoff = 2.0 * cell.interaction.radius;
        Ok(cell.interaction.radius < cell.division_size / 5.0)
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 100)]
    n_cells: usize,
    #[arg(long, default_value_t = 400.0)]
    domain_size: f64,
    #[arg(long, default_value_t = 24)]
    n_threads: usize,
    #[arg(long, default_value_t = 0.2)]
    time_dt: f64,
    #[arg(long, default_value_t = 30_000)]
    time_n: u64,
    #[arg(long, default_value_t = 20)]
    time_save_interval: u64,
    #[arg(long, default_value_t = 0.5)]
    cell_damping: f64,
    #[arg(long, default_value_t = 0.3)]
    cell_kb_temperature: f64,
    #[arg(long, default_value_t = 6.0)]
    cell_radius: f64,
    #[arg(long, default_value_t = 0.25)]
    cell_potential_stiffness: f64,
    #[arg(long, default_value_t = 0.01)]
    cell_strength: f64,
    #[arg(long, default_value_t = 24.0)]
    cell_death_range: f64,
    #[arg(long, default_value_t = 0.2)]
    cell_infection_rate_others: f64,
    #[arg(long, default_value_t = 0.00001)]
    cell_infection_rate_self: f64,
    #[arg(long, default_value_t = 9.0)]
    cell_division_size: f64,
    #[arg(long, default_value_t = 0.1)]
    cell_growth_rate: f64,
    #[arg(long, default_value_t = 0.5)]
    cell_death_rate: f64,
}

fn main() -> Result<(), SimulationError> {
    // Define the seed
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let Args {
        n_cells,
        domain_size,
        n_threads,
        time_dt: dt,
        time_n: n_times,
        time_save_interval: save_interval,
        cell_damping,
        cell_kb_temperature,
        cell_radius,
        cell_potential_stiffness,
        cell_strength,
        cell_death_range,
        cell_infection_rate_others,
        cell_infection_rate_self,
        cell_division_size,
        cell_growth_rate,
        cell_death_rate,
    } = Args::parse();

    let cells = (0..n_cells)
        .map(|_| {
            let pos = Vector3::from([
                rng.random_range(0.0..domain_size),
                rng.random_range(0.0..domain_size),
                rng.random_range(0.0..domain_size),
            ]);
            let radius = rng.random_range(0.5 * cell_radius..cell_division_size);
            Cell {
                mechanics: Langevin3D {
                    pos,
                    vel: Vector3::zero(),
                    mass: 1.0,
                    damping: cell_damping,
                    kb_temperature: cell_kb_temperature,
                },
                interaction: MorsePotential {
                    radius,
                    potential_stiffness: cell_potential_stiffness,
                    cutoff: 2.0 * radius,
                    strength: cell_strength,
                },
                health_state: Healthstate::Healthy,
                death_range: cell_death_range,
                infection_rate_others: cell_infection_rate_others,
                infection_rate_self: cell_infection_rate_self,
                neighbor_infected: 0,
                division_size: cell_division_size,
                growth_rate: cell_growth_rate,
                death_rate: cell_death_rate,
            }
        })
        .collect::<Vec<_>>();

    let domain = CartesianCuboid::from_boundaries_and_interaction_range(
        [0.0; 3],
        [domain_size; 3],
        2.0 * cell_division_size,
    )?;

    let time = cellular_raza::core::time::FixedStepsize::from_partial_save_steps(
        0.0,
        dt,
        n_times,
        save_interval,
    )?;
    let storage_builder = StorageBuilder::new().location("out/cell_sorting");

    let settings = cellular_raza::core::backend::chili::Settings {
        n_threads: n_threads.try_into().unwrap(),
        time,
        storage: storage_builder,
        progressbar: Some("".into()),
    };

    run_simulation!(
        domain,
        agents: cells,
        settings,
        aspects: [Mechanics, Interaction, Cycle, NeighborSensing]
    )?;
    Ok(())
}
