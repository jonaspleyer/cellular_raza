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

impl CLIArgs {
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

fn problem_size_scaling(args: &CLIArgs) -> Vec<DomainSample> {
    let mut samples = vec![];
    let mut progress_bar = args.create_kdam_bar("", args.problem_sizes.len() * args.sample_size);
    for &n_domain_size in args.problem_sizes.iter() {
        // Reset the progress bar
        let n_cells = 10 * 4_usize.pow(n_domain_size as u32);
        // The domain is sliced into voxels of size [18.0; 3]
        // Thus we want to have domains with size that is a multiplicative of 18.0
        let domain_size = 36_f64 * 4_f64.powf(1.0 / 3.0 * n_domain_size as f64);
        let mut times = vec![];
        // Try to load from a file
        let ds = match DomainSample::try_read_from_file(&args) {
            None => {
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
                    CLIArgs::set_description(
                        &mut progress_bar,
                        format!("Domain Size {} Sample {}", n_domain_size, n_sample),
                    );
                }
                let ds = DomainSample {
                    name: args.name.clone(),
                    id: args.id,
                    n_domain_size,
                    times,
                };
                match ds.store_to_file(&args) {
                    Ok(_) => (),
                    Err(_) => println!("Could not save to file"),
                }
                ds
            }
            Some(ds) => ds,
        };
        samples.push(ds);
    }
    samples
}

#[derive(Deserialize, Serialize)]
struct ThreadSample {
    // Configuration
    name: String,
    id: i32,
    // Results
    n_threads: usize,
    times: Vec<u128>,
}

fn thread_scaling(args: &CLIArgs) -> Vec<ThreadSample> {
    let mut samples = vec![];
    let mut progress_bar = args.create_kdam_bar("", args.threads.len() * args.sample_size);
    for &n_threads in args.threads.iter() {
        // Do warm-up run
        let mut times = vec![];
        for n_sample in 0..args.sample_size {
            let now = std::time::Instant::now();
            criterion::black_box(|| {
                run_simulation(
                    10_000,
                    10_000,
                    n_threads.try_into().unwrap(),
                    210.0,
                    5,
                    0.25,
                )
                .unwrap();
            })();
            let t = now.elapsed().as_nanos();
            times.push(t);
            CLIArgs::set_description(
                &mut progress_bar,
                format!("Threads: {} Sample: {}", n_threads, n_sample),
            );
        }
        samples.push(ThreadSample {
            name: args.name.clone(),
            id: args.id,
            n_threads,
            times,
        });
    }
    samples
}

trait Storage
where
    Self: Sized,
{
    fn store_to_file(&self, args: &CLIArgs) -> std::io::Result<()>;
    fn try_read_from_file(args: &CLIArgs) -> Option<Self>;
}

impl<T> Storage for T
where
    T: Serialize + for<'a> Deserialize<'a>,
{
    fn store_to_file(&self, args: &CLIArgs) -> std::io::Result<()> {
        let storage_path = args.get_storage_path();
        std::fs::create_dir_all(&storage_path.parent().unwrap())?;
        let buffer = std::fs::File::create(storage_path)?;
        serde_json::to_writer(buffer, self)?;
        Ok(())
    }

    fn try_read_from_file(args: &CLIArgs) -> Option<Self> {
        let path = args.get_storage_path();
        match std::fs::File::open(path) {
            Ok(file) => {
                let reader = std::io::BufReader::new(file);
                match serde_json::from_reader(reader) {
                    Ok(u) => {
                        Some(u)
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// Create new cell_sorting benchmark for thread or domain_size scaling
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Name of the current runs such as name of the device to be benchmarked
    // TODO use this
    #[arg(required = true)]
    name: String,

    /// Identifier for benchmark results. Negative values generate a new id.
    // TODO use this
    #[arg(short, long, default_value_t = -1)]
    id: i32,

    /// Output directory of benchmark results
    // TODO use this
    #[arg(short, long, default_value_t = format!("benchmark_results"))]
    output_directory: String,

    /// List of number of threads to benchmark
    #[arg(short, long, default_values_t = Vec::<usize>::new(), num_args=0..)]
    threads: Vec<usize>,

    /// List of domain sizes to benchmark
    #[arg(short, long, default_values_t = Vec::<usize>::new(), num_args=0..)]
    problem_sizes: Vec<usize>,

    /// Number of samples to be generated for each measurement
    #[arg(short, long, default_value_t = 5)]
    sample_size: usize,

    /// Do not save results. This takes priority against the overwrite settings.
    // TODO use this
    #[arg(long, default_value_t = false)]
    no_save: bool,

    /// Overwrite existing results
    // TODO use this
    #[arg(long, default_value_t = true)]
    overwrite: bool,

    /// Disables output
    #[arg(long, default_value_t = false)]
    no_output: bool,
}

impl CLIArgs {
    fn get_storage_path(&self) -> std::path::PathBuf {
        // TODO check if id is negative and then create new id if so
        std::path::PathBuf::from(&self.output_directory)
            .join(&self.name)
            .join(format!("{:010}.json", self.id))
    }
}

fn main() {
    let args = CLIArgs::parse();

    if !args.no_output {
        println!("Generating Results for device {}", args.name);
    }
    let thread_samples = thread_scaling(&args);
    for sample in thread_samples {
        println!("{:#?}", sample.times);
    }
    let domain_samples = problem_size_scaling(&args);
    for sample in domain_samples {
        println!("{:#?}", sample.times);
    }
}
