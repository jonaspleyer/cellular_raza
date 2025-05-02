use criterion::{Criterion, criterion_group, criterion_main};
use serde::{Deserialize, Serialize};

use cellular_raza::prelude::*;

fn run_sim(n_lattice: usize) -> Result<(), Box<dyn std::error::Error>> {
    #[derive(Clone, Debug, Serialize, CellAgent)]
    struct Agent {
        #[Mechanics]
        mechanics: NewtonDamped3D,
        #[Interaction]
        interaction: MorsePotential,
    }
    let domain_size = 100.0;
    let dx = domain_size / n_lattice as f64;
    let agents = (0..n_lattice)
        .zip(0..n_lattice)
        .zip(0..n_lattice)
        .zip(0..2)
        .map(|(((i, j), k), l)| Agent {
            mechanics: NewtonDamped3D {
                pos: [
                    (0.3 + i as f64 + 0.3 * l as f64) * dx,
                    (0.3 + j as f64 + 0.3 * l as f64) * dx,
                    (0.3 + k as f64 + 0.3 * l as f64) * dx,
                ]
                .into(),
                vel: num::zero(),
                mass: 10.0,
                damping_constant: 0.1,
            },
            interaction: MorsePotential {
                potential_stiffness: 0.5,
                strength: 0.01,
                cutoff: 1.1 * dx,
                radius: 0.5 * dx,
            },
        });
    let time = FixedStepsize::from_partial_save_steps(0.0, 0.01, 10, 1)?;
    let storage = StorageBuilder::new().priority([]);
    let domain =
        CartesianCuboid::from_boundaries_and_n_voxels([0.0; 3], [domain_size; 3], [n_lattice; 3])?;
    let settings = Settings {
        n_threads: 1.try_into().unwrap(),
        time,
        storage,
        show_progressbar: false,
    };
    run_simulation!(
        agents,
        domain,
        settings,
        aspects: [Mechanics, Interaction],
    )?;
    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("Mechanics(NewtonDamped3D)-Interaction(MorsePotential)-Lattice");
    group.sample_size(40);
    group.measurement_time(std::time::Duration::from_secs_f64(15.));

    for n_lattice in [4, 8, 16, 32] {
        group.bench_function(format!("{n_lattice}^3 Voxels"), |b| {
            b.iter(|| run_sim(n_lattice).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
