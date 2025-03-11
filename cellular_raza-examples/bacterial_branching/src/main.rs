use cellular_raza::prelude::*;

use clap::{Args, Parser};
use nalgebra::Vector2;
use num::Zero;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use serde::{Deserialize, Serialize};

mod bacteria_properties;
mod subdomain;

use bacteria_properties::*;
use subdomain::*;

#[derive(Clone, Args, Debug)]
#[group()]
#[clap(next_help_heading = Some("Bacteria"))]
struct BacterialParameters {
    /// Number of cells to put into simulation in the Beginning
    #[arg(short, long, default_value_t = 3)]
    n_bacteria_initial: u32,

    /// Mechanical parameters
    #[arg(short, long, default_value_t = 6.0)]
    radius: f32,
    #[arg(long, default_value_t = 0.2)]
    exponent: f32,
    #[arg(long, default_value_t = 4.0)]
    potential_strength: f32,
    #[arg(long, default_value_t = 1.0)]
    damping_constant: f32,

    /// Parameters for cell cycle
    #[arg(short, long, default_value_t = 0.8)]
    uptake_rate: f32,
    #[arg(short, long, default_value_t = 3.0)]
    growth_rate: f32,
}

#[derive(Clone, Args, Debug)]
#[group()]
#[clap(next_help_heading = Some("Domain"))]
struct DomainParameters {
    /// Parameters for domain
    #[arg(short, long, default_value_t = 300.0)]
    domain_size: f32,
    /// Discretization used to model the diffusion process
    #[arg(long, default_value_t = 20.0)]
    reactions_dx: f32,
    /// Parameters for Voxel Reaction+Diffusion
    #[arg(long, default_value_t = 25.0)]
    diffusion_constant: f32,
    /// Initial concentration of food in domain
    #[arg(long, default_value_t = 10.0)]
    initial_concentration: f32,
}

#[derive(Clone, Args, Debug)]
#[group()]
#[clap(next_help_heading = Some("Time"))]
struct TimeParameters {
    /// Time parameters
    #[arg(long, default_value_t = 0.05)]
    dt: f32,
    #[arg(long, default_value_t = 100.0)]
    tmax: f32,
    #[arg(long, default_value_t = 100)]
    save_interval: usize,
}

#[derive(Clone, Parser, Debug)]
#[command(version, about, long_about = None)]
struct Parameters {
    #[command(flatten)]
    bacteria: BacterialParameters,

    #[command(flatten)]
    domain: DomainParameters,

    #[command(flatten)]
    time: TimeParameters,

    #[clap(help_heading = Some("Other"))]
    /// Meta Parameters to control solving
    #[arg(long, default_value_t = 2)]
    threads: usize,
}

fn main() -> Result<(), SimulationError> {
    let parameters = Parameters::parse();
    run_sim(parameters)
}

fn run_sim(parameters: Parameters) -> Result<(), SimulationError> {
    let Parameters {
        bacteria:
            BacterialParameters {
                n_bacteria_initial,
                radius: cell_radius,
                exponent,
                potential_strength,
                damping_constant,
                uptake_rate,
                growth_rate,
            },
        domain:
            DomainParameters {
                domain_size,
                reactions_dx,
                diffusion_constant,
                initial_concentration,
            },
        time:
            TimeParameters {
                dt,
                tmax: t_max,
                save_interval,
            },
        threads: n_threads,
    } = parameters;

    let starting_domain_x_low = domain_size / 2.0 - 50.0;
    let starting_domain_x_high = domain_size / 2.0 + 50.0;
    let starting_domain_y_low = domain_size / 2.0 - 50.0;
    let starting_domain_y_high = domain_size / 2.0 + 50.0;

    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    let cells = (0..n_bacteria_initial)
        .map(|_| {
            let x = rng.gen_range(starting_domain_x_low..starting_domain_x_high);
            let y = rng.gen_range(starting_domain_y_low..starting_domain_y_high);

            let pos = Vector2::from([x, y]);
            MyAgent {
                mechanics: NewtonDamped2DF32 {
                    pos,
                    vel: Vector2::zero(),
                    damping_constant,
                    mass: 1.0,
                },
                interaction: MyInteraction {
                    cell_radius,
                    exponent,
                    potential_strength,
                },
                uptake_rate,
                division_radius: cell_radius * 2.0,
                growth_rate,
            }
        })
        .collect::<Vec<_>>();

    let cond = dt - 0.5 * reactions_dx / diffusion_constant;
    if cond >= 0.0 {
        println!("Warning: The stability condition dt <= 0.5 dx^2/D for the integration method is not satisfied. Results can be inaccurate.");
    }

    let domain = CartesianDiffusion2D {
        domain: CartesianCuboid::from_boundaries_and_interaction_range(
            [0.0; 2],
            [domain_size, domain_size],
            domain_voxel_size,
        )?,
        reactions_dx: [reactions_dx; 2].into(),
        diffusion_constant,
        initial_value: ReactionVector::from([initial_concentration]),
    };

    let storage = StorageBuilder::new().priority([StorageOption::SerdeJson]);
    let time = FixedStepsize::from_partial_save_freq(0.0, dt, t_max, save_interval)?;
    let settings = Settings {
        n_threads: n_threads.try_into().unwrap(),
        time,
        storage,
        show_progressbar: true,
    };

    let _storager = run_simulation!(
        agents: cells,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction, ReactionsExtra, Cycle],
        parallelizer: Rayon,
    )?;
    Ok(())
}
