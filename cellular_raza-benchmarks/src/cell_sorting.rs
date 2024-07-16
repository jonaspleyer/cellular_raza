use std::usize;

use cellular_raza::core::backend::chili;
use cellular_raza::{core::time::FixedStepsize, prelude::*};

use clap::{Parser, Subcommand};
use kdam::BarExt;
use nalgebra::Vector3;
use num::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use tempdir::TempDir;

// SIMULATION SPECIFIC CODE

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

#[derive(Clone, Debug, Eq, Deserialize, PartialEq, Serialize)]
struct SimSettings {
    n_cells_1: usize,
    n_cells_2: usize,
    n_threads: std::num::NonZeroUsize,
    /// Multiple of 0.1
    domain_size: usize,
    n_steps: usize,
    /// Multiple of 0.01
    dt: usize,
}

fn run_simulation(sim_settings: &SimSettings) -> Result<(), chili::SimulationError> {
    let n_cells_1 = sim_settings.n_cells_1;
    let n_cells_2 = sim_settings.n_cells_2;
    let n_threads = sim_settings.n_threads;
    let domain_size = sim_settings.domain_size as f64 * 0.1;
    let n_steps = sim_settings.n_steps;
    let dt = sim_settings.dt as f64 * 0.01;

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

    let temp_dir = TempDir::new("out_tmp").unwrap();
    let storage = StorageBuilder::new().location(temp_dir.path());

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

    fn set_description_and_update(
        progress_bar: &mut Option<kdam::Bar>,
        desc: impl Into<String>,
        update: Option<usize>,
    ) {
        match progress_bar.as_mut() {
            Some(bar) => {
                bar.set_description(desc);
                match update {
                    Some(steps) => match bar.update(steps) {
                        Ok(_) => (),
                        Err(e) => println!("Progressbar could not be updated with error: {e}"),
                    },
                    None => (),
                }
            }
            None => (),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BenchmarkResult {
    simulation_settings: SimSettings,
    times: Vec<u128>,
}

fn run_sim(
    args: &CLIArgs,
    settings: Vec<SimSettings>,
    formatter: impl Fn(&SimSettings, usize) -> String,
    main: impl Fn(&SimSettings),
    save_prefix: &str,
) -> Vec<BenchmarkResult> {
    let mut samples = vec![];
    let mut progress_bar = args.create_kdam_bar("", settings.len() * args.sample_size);
    for setting in settings.into_iter() {
        let res = match BenchmarkResult::try_read_from_file(args, &setting, save_prefix) {
            Ok(Some(r)) => {
                // Loading previous runs
                CLIArgs::set_description_and_update(
                    &mut progress_bar,
                    formatter(&setting, args.sample_size),
                    Some(args.sample_size),
                );
                r
            }
            Ok(None) | Err(_) => {
                // Do warm-up run before main
                match &mut progress_bar {
                    Some(bar) => bar.set_description("Doing Warmup"),
                    None => (),
                }
                criterion::black_box(|| main(&setting))();

                // Do main benchmark run
                let mut times = vec![];
                for n_sample in 0..args.sample_size {
                    let now = std::time::Instant::now();
                    criterion::black_box(|| main(&setting))();
                    let t = now.elapsed().as_nanos();
                    times.push(t);
                    CLIArgs::set_description_and_update(
                        &mut progress_bar,
                        formatter(&setting, n_sample),
                        Some(1),
                    );
                }
                let br = BenchmarkResult {
                    simulation_settings: setting,
                    times,
                };
                match br.store_to_file(args, save_prefix) {
                    Ok(_) => (),
                    Err(e) => println!("Storing to file failed with error: {e}"),
                }
                br
            }
        };
        samples.push(res);
    }
    samples
}

fn n_domain_size_to_domain_size_and_n_cells(n_domain_size: usize) -> (usize, usize) {
    let n_cells = 10 * 4_usize.pow(n_domain_size as u32);
    let domain_size = (360_f64 * 4_f64.powf(1.0 / 3.0 * n_domain_size as f64)).round() as usize;
    println!("{} {}", domain_size, n_cells);
    (domain_size, n_cells)
}

fn problem_size_scaling(
    args: &CLIArgs,
    domain_sizes: Vec<usize>,
    n_threads: usize,
) -> Vec<BenchmarkResult> {
    let simulation_settings: Vec<_> = domain_sizes
        .into_iter()
        .map(|n_domain_size| {
            let (domain_size, n_cells) = n_domain_size_to_domain_size_and_n_cells(n_domain_size);
            // The domain is sliced into voxels of size [18.0; 3]
            // Thus we want to have domains with size that is a multiplicative of 18.0
            SimSettings {
                n_cells_1: n_cells,
                n_cells_2: n_cells,
                n_threads: n_threads.try_into().unwrap(),
                domain_size,
                n_steps: 10,
                dt: 10,
            }
        })
        .collect();
    run_sim(
        args,
        simulation_settings,
        |setting: &SimSettings, n_sample: usize| {
            format!(
                "Domain Size: {} Sample: {}",
                setting.domain_size,
                n_sample + 1
            )
        },
        |settings: &SimSettings| {
            run_simulation(settings).unwrap();
        },
        "sim-size",
    )
}

fn thread_scaling(args: &CLIArgs, threads: Vec<usize>) -> Vec<BenchmarkResult> {
    let simulation_settings: Vec<_> = threads
        .into_iter()
        .map(|n_threads| SimSettings {
            n_cells_1: 10_000,
            n_cells_2: 10_000,
            n_threads: n_threads.try_into().unwrap(),
            domain_size: 2100,
            n_steps: 5,
            dt: 10,
        })
        .collect();
    run_sim(
        args,
        simulation_settings,
        |setting: &SimSettings, n_sample: usize| {
            format!("Threads: {} Sample: {}", setting.n_threads, n_sample + 1)
        },
        |settings: &SimSettings| {
            run_simulation(settings).unwrap();
        },
        "thread-scaling",
    )
}

impl BenchmarkResult {
    fn get_next_index_value(
        storage_path: &std::path::Path,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let mut index = 0;
        for globresult in glob::glob(&format!("{}/*.json", storage_path.to_string_lossy()))? {
            let res = globresult?;
            if let Some(file_name) = res.file_name() {
                let new_index: u32 = file_name
                    .to_string_lossy()
                    .split(".json")
                    .next()
                    .unwrap()
                    .parse()?;
                index = new_index.max(index);
            }
        }
        Ok(index + 1)
    }

    fn get_storage_path(
        args: &CLIArgs,
        save_prefix: impl Into<std::path::PathBuf>,
    ) -> std::path::PathBuf {
        args.get_storage_base_path().join(save_prefix.into())
    }

    fn get_file_path(
        args: &CLIArgs,
        save_prefix: impl Into<std::path::PathBuf>,
    ) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
        let storage_path = Self::get_storage_path(args, save_prefix);
        std::fs::create_dir_all(&storage_path)?;
        let index = Self::get_next_index_value(&storage_path)?;
        Ok(storage_path.join(format!("{index:010}.json")))
    }

    fn store_to_file(
        &self,
        args: &CLIArgs,
        save_prefix: impl Into<std::path::PathBuf>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = Self::get_file_path(args, save_prefix)?;
        let buffer = std::fs::File::create(file_path)?;
        serde_json::to_writer(buffer, self)?;
        Ok(())
    }

    fn try_read_from_file(
        args: &CLIArgs,
        sim_settings: &SimSettings,
        save_prefix: impl Into<std::path::PathBuf>,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let storage_path = Self::get_storage_path(args, save_prefix);
        // Get names of all files in this directory which end on json
        for globresult in glob::glob(&format!("{}/*.json", storage_path.to_string_lossy()))? {
            if let Ok(file_path) = globresult {
                let file = std::fs::File::open(&file_path)?;
                let reader = std::io::BufReader::new(file);
                match serde_json::from_reader::<_, BenchmarkResult>(reader) {
                    Ok(u) => {
                        if &u.simulation_settings == sim_settings {
                            return Ok(Some(u));
                        }
                    }
                    Err(e) => println!(
                        "\
                        File {} might not be matching storage format.\
                        Encountered error {e}",
                        file_path.to_string_lossy()
                    ),
                }
            }
        }
        Ok(None)
    }
}

#[derive(Subcommand, Debug)]
enum SubCommand {
    /// Thread scaling benchmark
    Threads {
        /// List of thread configurations to benchmark
        threads: Vec<usize>,
    },
    /// Simulation Size scaling benchmark
    SimSize {
        /// List of problem sizes to benchmark
        problem_sizes: Vec<usize>,
        #[arg(short, default_value_t = 1)]
        n_threads: usize,
    },
}

/// Create new cell_sorting benchmark for thread or domain_size scaling
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Name of the current runs such as name of the device to be benchmarked
    // TODO use this
    #[arg(required = true)]
    name: String,

    /// Output directory of benchmark results
    // TODO use this
    #[arg(short, long, default_value_t = format!("benchmark_results"))]
    output_directory: String,

    #[command(subcommand)]
    commands: Option<SubCommand>,

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
    fn get_storage_base_path(&self) -> std::path::PathBuf {
        std::path::PathBuf::from(&self.output_directory).join(&self.name)
    }
}

fn main() {
    let args = CLIArgs::parse();

    if let Some(command) = &args.commands {
        if !args.no_output {
            println!("Generating Results for device {}", args.name);
        }
        match command {
            SubCommand::Threads { threads } => {
                thread_scaling(&args, threads.clone());
            }
            SubCommand::SimSize { problem_sizes, n_threads } => {
                problem_size_scaling(&args, problem_sizes.clone(), *n_threads);
            }
        }
    }
}
