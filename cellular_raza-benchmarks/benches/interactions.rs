use criterion::{Criterion, criterion_group, criterion_main};
use serde::{Deserialize, Serialize};

use cellular_raza::prelude::*;

#[derive(Clone, Debug, Serialize, CellAgent)]
struct Agent {
    #[Mechanics]
    mechanics: NewtonDamped3D,
    #[Interaction]
    interaction: MorsePotential,
}

type Prepared = (
    Vec<Agent>,
    CartesianCuboid<f64, 3>,
    Settings<FixedStepsize<f64>, false>,
);

fn prepare_sim(
    n_lattice: usize,
    n_subspacing: usize,
) -> Result<Prepared, Box<dyn std::error::Error>> {
    let domain_size = 100.0;
    let dx = domain_size / n_lattice as f64;
    let ds = 1.0 / (n_subspacing + 1) as f64;
    let agents = itertools::iproduct!(0..n_lattice, 0..n_lattice, 0..n_lattice, 0..n_subspacing)
        .map(|(i, j, k, l)| Agent {
            mechanics: NewtonDamped3D {
                pos: [
                    (0.3 + i as f64 + ds * l as f64) * dx,
                    (0.3 + j as f64 + ds * l as f64) * dx,
                    (0.3 + k as f64 + ds * l as f64) * dx,
                ]
                .into(),
                vel: num::zero(),
                mass: 10.0,
                damping_constant: 0.1,
            },
            interaction: MorsePotential {
                potential_stiffness: 0.5,
                strength: 0.001,
                cutoff: 1.1 * dx,
                radius: 0.25 * dx,
            },
        })
        .collect();
    let time = FixedStepsize::from_partial_save_steps(0.0, 0.01, 1_000, 1000)?;
    let storage = StorageBuilder::new().priority([]);
    let domain =
        CartesianCuboid::from_boundaries_and_n_voxels([0.0; 3], [domain_size; 3], [n_lattice; 3])?;
    let settings = Settings {
        n_threads: 1.try_into().unwrap(),
        time,
        storage,
        show_progressbar: false,
    };
    Ok((agents, domain, settings))
}

fn run_sim(prepared: Prepared) -> Result<(), SimulationError> {
    let (agents, domain, settings) = prepared;
    run_simulation!(
        agents,
        domain,
        settings,
        aspects: [Mechanics, Interaction],
    )?;
    Ok(())
}

fn subdomains_communicate(c: &mut Criterion) {
    let mut group = c.benchmark_group("SubDomains-Communicate");
    group.sample_size(40);
    group.measurement_time(std::time::Duration::from_secs_f64(15.));

    for n_lattice in [4, 8, 16, 32] {
        group.bench_function(format!("{n_lattice}^3 Voxels"), |b| {
            let prepared = prepare_sim(n_lattice, 3).unwrap();
            b.iter(|| run_sim(prepared.clone()).unwrap())
        });
    }
    group.finish();
}

fn one_subdomain_many_cells(c: &mut Criterion) {
    let mut group = c.benchmark_group("SubDomain-Single");
    group.bench_function("1000cells", |b| {
        let (agents, domain, settings) = prepare_sim(10).unwrap();
        let domain = CartesianCuboid::from_boundaries_and_n_voxels(
            [0.0; 3],
            [domain.get_max()[0]; 3],
            [1; 3],
        )
        .unwrap();
        b.iter(|| run_sim((agents.clone(), domain.clone(), settings.clone())).unwrap())
    });
}

criterion_group!(benches, subdomains_communicate, one_subdomain_many_cells);
criterion_main!(benches);
