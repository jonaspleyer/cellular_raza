use std::usize;

use cellular_raza::core::backend::chili;
use cellular_raza::{core::time::FixedStepsize, prelude::*};

use clap::Parser;
use kdam::BarExt;
use nalgebra::Vector3;
use num::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
enum Species {
    RedCell,
    BlueCell,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CellSpecificInteraction {
    species: Species,
    cell_radius: f64,
    potential_strength: f64,
    relative_interaction_range: f64,
}

impl Interaction<Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, Species)>
    for CellSpecificInteraction
{
    fn calculate_force_between(
        &self,
        own_pos: &Vector3<f64>,
        _own_vel: &Vector3<f64>,
        ext_pos: &Vector3<f64>,
        _ext_vel: &Vector3<f64>,
        ext_info: &(f64, Species),
    ) -> Result<(Vector3<f64>, Vector3<f64>), CalcError> {
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
                            return Ok((Vector3::zeros(), Vector3::zeros()));
                        }
                        false => (own_pos - ext_pos).normalize(),
                    };
                    let r = self.cell_radius * min_relative_distance_to_center;
                    (r, dir)
                }
            };
        let (ext_radius, species) = ext_info;
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

        if *species == self.species {
            Ok((
                repelling_force + attracting_force,
                -repelling_force - attracting_force,
            ))
        } else {
            Ok((repelling_force, -repelling_force))
        }
    }

    fn get_interaction_information(&self) -> (f64, Species) {
        (self.cell_radius, self.species.clone())
    }
}

#[derive(CellAgent, Clone, Serialize, Deserialize)]
struct Cell {
    #[Mechanics]
    mechanics: NewtonDamped3D,
    #[Interaction]
    interaction: CellSpecificInteraction,
}

fn run_simulation(
    n_cells_1: usize,
    n_cells_2: usize,
    n_threads: std::num::NonZeroUsize,
    domain_size: f64,
    n_steps: usize,
    dt: f64,
) -> Result<(), chili::SimulationError> {
    // Define the seed
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let cells = (0..n_cells_1 + n_cells_2).map(|n| {
        let pos = Vector3::from([
            rng.gen_range(0.0..domain_size),
            rng.gen_range(0.0..domain_size),
            rng.gen_range(0.0..domain_size),
        ]);
        Cell {
            mechanics: NewtonDamped3D {
                pos,
                vel: Vector3::zero(),
                damping_constant: 2.0,
                mass: 1.0,
            },
            interaction: CellSpecificInteraction {
                species: match n <= n_cells_1 {
                    true => Species::BlueCell,
                    false => Species::RedCell,
                },
                potential_strength: 2.0,
                relative_interaction_range: 1.5,
                cell_radius: 6.0,
            },
        }
    });

    let domain = CartesianCuboid3New::from_boundaries_and_interaction_ranges(
        [0.0; 3],
        [domain_size; 3],
        [1.5 * 6.0 * 2.0; 3],
    )?;

    let time = FixedStepsize::from_partial_save_steps(0.0, dt, n_steps as u64, n_steps as u64 + 1)?;

    let storage = StorageBuilder::new();

    let settings = chili::Settings {
        n_threads,
        time,
        storage,
        show_progressbar: false,
    };

    chili::run_simulation!(
        agents: cells,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction],
    )?;
    Ok(())
}

#[derive(Deserialize, Serialize)]
struct DomainSample {
    // Configuration settings
    name: String,
    id: i32,
    // Results
    n_domain_size: usize,
    times: Vec<u128>,
}

impl Args {
    fn create_kdam_bar(
        &self,
        init_fmt_string: impl Into<String>,
        total: usize,
    ) -> Option<kdam::Bar> {
        if self.no_output {
            None
        } else {
            Some(kdam::tqdm!(
                desc = init_fmt_string,
                total = total,
                position = 0
            ))
        }
    }

    fn set_description(progress_bar: &mut Option<kdam::Bar>, desc: impl Into<String>) {
        match progress_bar.as_mut() {
            Some(bar) => {
                bar.set_description(desc);
                match bar.update(1) {
                    Ok(_) => (),
                    Err(e) => println!("Progressbar could not be updated with error: {e}"),
                }
            }
            None => (),
        }
    }
}

fn cell_scaling(args: &Args) -> Vec<DomainSample> {
    let mut samples = vec![];
    let mut progress_bar = args.create_kdam_bar("", args.domain_sizes.len() * args.sample_size);
    for &n_domain_size in args.domain_sizes.iter() {
        // Reset the progress bar
        let n_cells = 10 * 4_usize.pow(n_domain_size as u32);
        // The domain is sliced into voxels of size [18.0; 3]
        // Thus we want to have domains with size that is a multiplicative of 18.0
        let domain_size = 36_f64 * 4_f64.powf(1.0 / 3.0 * n_domain_size as f64);
        let mut times = vec![];
        for n_sample in 0..args.sample_size {
            let now = std::time::Instant::now();
            criterion::black_box(|| {
                run_simulation(
                    n_cells,
                    n_cells,
                    1.try_into().unwrap(),
                    domain_size,
                    10,
                    0.25,
                )
                .unwrap();
            })();
            let t = now.elapsed().as_nanos();
            times.push(t);
            Args::set_description(
                &mut progress_bar,
                format!("Domain Size {} Sample {}", n_domain_size, n_sample),
            );
        }
        samples.push(DomainSample {
            name: args.name.clone(),
            id: args.id,
            n_domain_size,
            times,
        });
    }
    samples
}

fn thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");
    group.sample_size(10);

    for n_threads in 1..48 {
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::new("n_threads", n_threads),
            &n_threads,
            |b, &n_threads| {
                b.iter(|| {
                    run_simulation(
                        10_000,
                        10_000,
                        n_threads.try_into().unwrap(),
                        210.0,
                        5,
                        0.25,
                    )
                })
            },
        );
    }
    group.finish();
}

/// Create new cell_sorting benchmark for thread or domain_size scaling
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the current runs such as name of the device to be benchmarked
    #[arg(required = true)]
    name: String,

    /// Identifier for benchmark results. Negative values generate a new id.
    #[arg(short, long, default_value_t = -1)]
    id: i32,

    /// Output directory of benchmark results
    #[arg(short, long, default_value_t = format!("benchmark_results"))]
    output_directory: String,

    /// List of number of threads to benchmark
    #[arg(short, long, default_values_t = Vec::<usize>::new(), num_args=0..)]
    threads: Vec<usize>,

    /// List of domain sizes to benchmark
    #[arg(short, long, default_values_t = Vec::<usize>::new(), num_args=0..)]
    domain_sizes: Vec<usize>,

    /// Number of samples to be generated for each measurement
    #[arg(short, long, default_value_t = 5)]
    sample_size: usize,

    /// Do not save results. This takes priority against the overwrite settings.
    #[arg(long, default_value_t = false)]
    no_save: bool,

    /// Overwrite existing results
    #[arg(long, default_value_t = true)]
    overwrite: bool,

    /// Disables output
    #[arg(long, default_value_t = false)]
    no_output: bool,
}

fn main() {
    let args = Args::parse();

    if !args.no_output {
        println!("Generating Results for device {}", args.name);
    }
    let domain_samples = cell_scaling(&args);
    for sample in domain_samples {
        println!("{:#?}", sample.times);
    }
}
