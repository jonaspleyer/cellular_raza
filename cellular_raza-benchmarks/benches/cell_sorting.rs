use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use cellular_raza::prelude::*;

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
        ext_info: &Option<(f64, Species)>,
    ) -> Option<Result<Vector3<f64>, CalcError>> {
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
                            return None;
                        }
                        false => (own_pos - ext_pos).normalize(),
                    };
                    let r = self.cell_radius * min_relative_distance_to_center;
                    (r, dir)
                }
            };
        match ext_info {
            Some((ext_radius, species)) => {
                // Introduce Non-dimensional length variable
                let sigma = r / (self.cell_radius + ext_radius);
                let bound = 4.0 + 1.0 / sigma;
                let spatial_cutoff = (1.0
                    + (self.relative_interaction_range * (self.cell_radius + ext_radius) - r)
                        .signum())
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
                    return Some(Ok(repelling_force + attracting_force));
                } else {
                    return Some(Ok(repelling_force));
                }
            }
            None => None,
        }
    }

    fn get_interaction_information(&self) -> Option<(f64, Species)> {
        Some((self.cell_radius, self.species.clone()))
    }
}

fn run_simulation(
    n_cells_1: usize,
    n_cells_2: usize,
    n_threads: usize,
    domain_size: f64,
    n_times: usize,
    dt: f64,
) -> Result<(), SimulationError> {
    // Define the seed
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let cells = (0..n_cells_1 + n_cells_2)
        .map(|n| {
            let pos = Vector3::from([
                rng.gen_range(0.0..domain_size),
                rng.gen_range(0.0..domain_size),
                rng.gen_range(0.0..domain_size),
            ]);
            ModularCell {
                mechanics: MechanicsModel3D {
                    pos,
                    vel: Vector3::zero(),
                    dampening_constant: 2.0,
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
                cycle: NoCycle {},
                interaction_extracellular: NoExtracellularGradientSensing {},
                cellular_reactions: NoCellularreactions {},
            }
        })
        .collect::<Vec<_>>();

    let domain = CartesianCuboid3::from_boundaries_and_interaction_ranges(
        [0.0; 3],
        [domain_size; 3],
        [1.5 * 6.0 * 2.0; 3],
    )?;

    let time = TimeSetup {
        t_start: 0.0,
        t_eval: (0..n_times).map(|n| (n as f64 * dt, false)).collect(),
    };

    let meta_params = SimulationMetaParams { n_threads };

    let storage = StorageConfig::from_path("out/cell_sorting".into());

    let simulation_setup = SimulationSetup::new(domain, cells, time, meta_params, storage, ());

    let mut supervisor = SimulationSupervisor::initialize_from_setup(simulation_setup);
    supervisor.config.show_progressbar = false;

    supervisor.run_full_sim()?;
    Ok(())
}

fn cell_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("cell_scaling");

    for n_cells in [10, 50, 100, 500, 1000, 5000].into_iter() {
        group.bench_with_input(
            BenchmarkId::new("n_cells-domain_size", n_cells),
            &n_cells,
            |b, &n_cells| {
                b.iter(|| {
                    run_simulation(
                        n_cells,
                        n_cells,
                        1,
                        10.0 + (n_cells as f64 / 0.002).powf(1.0 / 3.0),
                        10,
                        0.25,
                    )
                })
            },
        );
    }
    group.finish();
}

fn thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");

    for n_threads in 1..9 {
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::new("n_threads", n_threads),
            &n_threads,
            |b, &n_threads| {
                b.iter(|| run_simulation(1_000, 1_000, n_threads as usize, 100.0, 5, 0.25))
            },
        );
    }
    group.finish();
}

criterion_group!(benches, cell_scaling, thread_scaling);
criterion_main!(benches);
