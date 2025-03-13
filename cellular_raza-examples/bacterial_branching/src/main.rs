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
    #[arg(short, long, default_value_t = 5)]
    n_bacteria_initial: u32,
    #[arg(short, long, default_value_t = 6.0)]
    radius: f32,
    /// Multiple of the radius at which the cell will divide
    #[arg(long, default_value_t = 2.0)]
    division_threshold: f32,
    #[arg(long, default_value_t = 0.15)]
    potential_stiffness: f32,
    #[arg(long, default_value_t = 2.0)]
    potential_strength: f32,
    #[arg(long, default_value_t = 1.0)]
    damping_constant: f32,
    #[arg(short, long, default_value_t = 1.0)]
    uptake_rate: f32,
    #[arg(short, long, default_value_t = 13.0)]
    growth_rate: f32,
}

#[derive(Clone, Args, Debug)]
#[group()]
#[clap(next_help_heading = Some("Domain"))]
struct DomainParameters {
    /// Overall size of the domain
    #[arg(short, long, default_value_t = 3000.0)]
    domain_size: f32,
    #[arg(
        long,
        default_value_t = 30.0,
        help = "\
        Size of one voxel containing individual cells.\n\
        This value should be chosen `>=3*RADIUS`.\
    "
    )]
    voxel_size: f32,
    /// Size of the square for initlal placement of bacteria
    #[arg(long, default_value_t = 100.0)]
    domain_starting_size: f32,
    /// Discretization of the diffusion process
    #[arg(long, default_value_t = 20.0)]
    reactions_dx: f32,
    #[arg(long, default_value_t = 80.0)]
    diffusion_constant: f32,
    #[arg(long, default_value_t = 10.0)]
    initial_concentration: f32,
}

#[derive(Clone, Args, Debug)]
#[group()]
#[clap(next_help_heading = Some("Time"))]
struct TimeParameters {
    #[arg(long, default_value_t = 0.1)]
    dt: f32,
    #[arg(long, default_value_t = 2000.0)]
    tmax: f32,
    #[arg(long, default_value_t = 200)]
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
                division_threshold,
                potential_stiffness,
                potential_strength,
                damping_constant,
                uptake_rate,
                growth_rate,
            },
        domain:
            DomainParameters {
                domain_size,
                voxel_size: domain_voxel_size,
                domain_starting_size,
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

    let ds = domain_size / 2.0;
    let dx = domain_starting_size / 2.0;

    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    let cells = (0..n_bacteria_initial)
        .map(|_| {
            let x = rng.gen_range(ds - dx..ds + dx);
            let y = rng.gen_range(ds - dx..ds + dx);

            let pos = Vector2::from([x, y]);
            MyAgent {
                mechanics: NewtonDamped2DF32 {
                    pos,
                    vel: Vector2::zero(),
                    damping_constant,
                    mass: 1.0,
                },
                interaction: MorsePotentialF32 {
                    radius: cell_radius,
                    potential_stiffness,
                    cutoff: 2.0 * division_threshold * cell_radius,
                    strength: potential_strength,
                },
                uptake_rate,
                division_radius: division_threshold * cell_radius,
                growth_rate,
            }
        })
        .collect::<Vec<_>>();

    let cond = dt - 0.5 * reactions_dx / diffusion_constant;
    if cond >= 0.0 {
        println!(
            "❗❗❗WARNING❗❗❗\n\
            The stability condition \
            dt <= 0.5 dx^2/D for the integration \
            method is not satisfied. This can \
            lead to solving errors and inaccurate \
            results."
        );
    }

    if domain_voxel_size < division_threshold * cell_radius {
        println!(
            "❗❗❗WARNING❗❗❗\n\
            The domain_voxel_size {domain_voxel_size} has been chosen \
            smaller than the length of the interaction {}. This \
            will probably yield incorrect results.",
            division_threshold * cell_radius,
        );
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
