use cellular_raza_building_blocks::{Brownian3D, CartesianCuboid3New};
use cellular_raza_core::storage::StorageInterfaceLoad;
use serde::{Deserialize, Serialize};

struct Parameters {
    n_particles: usize,
    domain_size: f64,

    dt: f64,
    n_steps: u64,
    save_interval: u64,

    diffusion_constant: f64,

    storage_name: std::path::PathBuf,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            #[cfg(debug_assertions)]
            n_particles: 160,
            #[cfg(not(debug_assertions))]
            n_particles: 800,
            domain_size: 200.0,

            dt: 1e-3,
            #[cfg(debug_assertions)]
            n_steps: 100,
            #[cfg(not(debug_assertions))]
            n_steps: 5_000,
            save_interval: 50,

            diffusion_constant: 1.0,
            storage_name: "out/brownian".into(),
        }
    }
}

#[allow(unused)]
fn brownian(parameters: &Parameters) -> Result<(), Box<dyn std::error::Error>> {
    let domain_size = parameters.domain_size;
    assert!(domain_size > 0.0);
    let domain =
        CartesianCuboid3New::from_boundaries_and_n_voxels([0.0; 3], [domain_size; 3], [3; 3])?;

    let time = cellular_raza_core::time::FixedStepsize::from_partial_save_steps(
        0.0,
        parameters.dt,
        parameters.n_steps,
        parameters.save_interval,
    )?;

    // Use a temporary directory to automatically clean up all files in it
    use tempdir::TempDir;
    let dir = TempDir::new("tempdir").unwrap();
    let location = dir.path().join(&parameters.storage_name);
    let storage = cellular_raza_core::storage::StorageBuilder::new()
        .priority([cellular_raza_core::storage::StorageOption::Sled])
        .location(location)
        .init();

    let settings = cellular_raza_core::backend::chili::Settings {
        n_threads: 1.try_into().unwrap(),
        time,
        storage,
        show_progressbar: false,
    };

    let initial_position = nalgebra::Vector3::from([domain_size / 2.0; 3]);
    let particles = (0..parameters.n_particles)
        .map(|_| Brownian3D::new(initial_position.into(), parameters.diffusion_constant, 1.0));

    let storage_access = cellular_raza_core::backend::chili::run_simulation!(
        agents: particles,
        domain: domain,
        settings: settings,
        aspects: [Mechanics]
    )?;

    let positions = storage_access
        .cells
        .load_all_elements()?
        .into_iter()
        .map(|(iteration, particles)| {
            (
                iteration,
                particles
                    .into_iter()
                    .map(|(_, (p, _))| (p.identifier, p.cell.pos))
                    .collect::<std::collections::HashMap<_, _>>(),
            )
        })
        .collect::<std::collections::HashMap<u64, _>>();

    let square_displacements = positions
        .into_iter()
        .map(|(iteration, positions)| {
            (
                iteration,
                positions
                    .into_iter()
                    .map(|(id, pos)| (pos - initial_position).norm_squared())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<std::collections::HashMap<u64, Vec<f64>>>();

    let means = square_displacements
        .iter()
        .map(|(iteration, displs)| (iteration, displs.iter().sum::<f64>() / displs.len() as f64))
        .collect::<std::collections::HashMap<_, _>>();

    let std_dev_err = square_displacements
        .iter()
        .map(|(iteration, displs)| {
            let sigma = (displs
                .iter()
                .map(|displ| (displ - means[&iteration]).powf(2.0))
                .sum::<f64>()
                / displs.len() as f64)
                .sqrt();
            (iteration, (sigma, sigma / (displs.len() as f64).sqrt()))
        })
        .collect::<std::collections::HashMap<_, _>>();

    for &iteration in means.keys() {
        let expected = 6.0
            * parameters.diffusion_constant
            * (*iteration - min_iteration) as f64
            * parameters.dt;
        assert!((means[iteration] - expected).abs() <= std_dev[iteration]);
    }
    Ok(())
}

// TODO activate in the future #[test]
fn brownian_1() {
    let parameters = Parameters::default();
    brownian(&parameters).unwrap();
}

// TODO activate in the future #[test]
fn brownian_2() {
    let mut parameters = Parameters::default();
    parameters.diffusion_constant = 0.5;
    parameters.storage_name = "out/brownian_2".into();
    brownian(&parameters).unwrap();
}

// TODO activate in the future #[test]
fn brownian_3() {
    let mut parameters = Parameters::default();
    parameters.diffusion_constant = 0.25;
    parameters.storage_name = "out/brownian_3".into();
    brownian(&parameters).unwrap();
}
