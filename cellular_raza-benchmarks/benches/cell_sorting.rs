use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use cellular_raza::core::backend::chili;
use cellular_raza::{core::time::FixedStepsize, prelude::*};

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
            return Some(Ok(repelling_force + attracting_force));
        } else {
            return Some(Ok(repelling_force));
        }
    }

    fn get_interaction_information(&self) -> (f64, Species) {
        (self.cell_radius, self.species.clone())
    }
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
        ModularCell {
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
            cycle: NoCycle {},
            interaction_extracellular: NoExtracellularGradientSensing,
            cellular_reactions: NoCellularReactions,
            volume: 0.0,
        }
    });

    let domain = CartesianCuboid3New::from_boundaries_and_interaction_ranges(
        [0.0; 3],
        [domain_size; 3],
        [1.5 * 6.0 * 2.0; 3],
    )?;

    let time = FixedStepsize::from_partial_save_steps(0.0, dt, n_steps as u64, n_steps as u64 + 1)?;

    let storage = StorageBuilder::new().location("out/cell_sorting");

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

fn cell_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("cell_scaling");
    group.sample_size(10);

    for i in 2..8 {
        let n_cells = 10 * 4_usize.pow(i);
        // The domain is sliced into voxels of size [18.0; 3]
        // Thus we want to have domains with size that is a multiplicative of 18.0
        let domain_size = 36_f64 * 4_f64.powf(1.0 / 3.0 * i as f64);
        group.bench_with_input(
            BenchmarkId::new("n_cells-domain_size", n_cells),
            &n_cells,
            |b, &n_cells| {
                b.iter(|| {
                    run_simulation(
                        n_cells,
                        n_cells,
                        1.try_into().unwrap(),
                        domain_size,
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

criterion_group!(benches, cell_scaling, thread_scaling);
criterion_main!(benches);
