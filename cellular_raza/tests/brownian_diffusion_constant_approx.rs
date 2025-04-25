use core::f64;

use cellular_raza_building_blocks::*;
use cellular_raza_core::{storage::StorageInterfaceLoad, time::FixedStepsize};
use nalgebra::SVector;
use serde::{Deserialize, Serialize};

struct Parameters {
    n_particles: usize,
    domain_size: f64,

    dt: f64,
    n_steps: u64,
    save_interval: u64,

    diffusion_constant: f64,

    storage_name: std::path::PathBuf,
    random_seed: u64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            n_particles: 160,
            domain_size: 200.0,

            dt: 1e-3,
            n_steps: 200,
            save_interval: 1,

            diffusion_constant: 1.0,
            storage_name: "out/brownian".into(),
            random_seed: 0,
        }
    }
}

fn define_settings(
    parameters: &Parameters,
) -> Result<
    cellular_raza_core::backend::chili::Settings<FixedStepsize<f64>, true>,
    Box<dyn std::error::Error>,
> {
    let time = cellular_raza_core::time::FixedStepsize::from_partial_save_steps(
        0.0,
        parameters.dt,
        parameters.n_steps,
        parameters.save_interval,
    )?;

    // Only save results when debug_assertions are not active.
    let location = parameters.storage_name.to_owned();
    let storage = cellular_raza_core::storage::StorageBuilder::new()
        .priority([
            cellular_raza_core::storage::StorageOption::Memory,
            #[cfg(not(debug_assertions))]
            cellular_raza_core::storage::StorageOption::SerdeJson,
        ])
        .location(location)
        .init();

    Ok(cellular_raza_core::backend::chili::Settings {
        n_threads: 1.try_into().unwrap(),
        time,
        storage,
        show_progressbar: false,
    })
}

fn calculate_mean_std_dev<const D: usize>(
    parameters: &Parameters,
    positions: impl IntoIterator<Item = (u64, Vec<SVector<f64, D>>)>,
) -> Result<std::collections::HashMap<u64, (f64, f64)>, Box<dyn std::error::Error>> {
    let initial_position = SVector::from([parameters.domain_size / 2.0; D]);
    let square_displacements = positions
        .into_iter()
        .map(|(iteration, positions)| {
            (
                iteration,
                positions
                    .into_iter()
                    .map(|pos| (pos - initial_position).norm_squared())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<std::collections::HashMap<u64, Vec<f64>>>();

    let means = square_displacements
        .iter()
        .map(|(iteration, displs)| (*iteration, displs.iter().sum::<f64>() / displs.len() as f64))
        .collect::<std::collections::HashMap<_, _>>();

    let means_std_dev = square_displacements
        .iter()
        .map(|(iteration, displs)| {
            let sigma = (displs
                .iter()
                .map(|displ| (displ - means[&iteration]).powf(2.0))
                .sum::<f64>()
                / displs.len() as f64)
                .sqrt();
            (*iteration, (means[&iteration], sigma))
        })
        .collect::<std::collections::HashMap<_, _>>();
    Ok(means_std_dev)
}

fn analyze_positions_brownian<const D: usize>(
    parameters: &Parameters,
    positions: impl IntoIterator<Item = (u64, Vec<SVector<f64, D>>)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let means_std_dev = calculate_mean_std_dev(&parameters, positions)?;

    // Calculate probability for values to be identical
    for &iteration in means_std_dev.keys() {
        let expected =
            2.0 * D as f64 * parameters.diffusion_constant * iteration as f64 * parameters.dt;
        let (mean, std_dev) = means_std_dev[&iteration];
        assert!((expected - mean).abs() <= 3.5 * std_dev);
    }
    Ok(())
}

macro_rules! test_brownian {
    ($parameters:ident, $particle_name:ident, $d:literal) => {{
        let domain_size = $parameters.domain_size;
        assert!(domain_size > 0.0);
        let mut domain =
            CartesianCuboid::from_boundaries_and_n_voxels([0.0; $d], [domain_size; $d], [3; $d])?;
        domain.rng_seed = $parameters.random_seed;

        let initial_position = nalgebra::SVector::from([domain_size / 2.0; $d]);
        let particles = (0..$parameters.n_particles)
            .map(|_| $particle_name::new(
                initial_position.into(),
                $parameters.diffusion_constant,
                1.0
            ));

        let settings = define_settings(&$parameters)?;

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
                        .map(|(_, (p, _))| cellular_raza::concepts::Position::pos(&p.cell))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<std::collections::HashMap<u64, _>>();

        analyze_positions_brownian(&$parameters, positions)?;
        Result::<(), Box<dyn std::error::Error>>::Ok(())
    }}
}

#[test]
fn diffusion_constant_brownian_3d_1() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/brownian_3d_1".into();
    parameters.diffusion_constant = 1.0;
    parameters.random_seed = 1;
    test_brownian!(parameters, Brownian3D, 3)?;
    Ok(())
}

#[test]
fn diffusion_constant_brownian_3d_2() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.diffusion_constant = 0.5;
    parameters.storage_name = "out/brownian_3d_2".into();
    parameters.random_seed = 2;
    test_brownian!(parameters, Brownian3D, 3)?;
    Ok(())
}

#[test]
fn diffusion_constant_brownian_3d_3() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.diffusion_constant = 0.25;
    parameters.storage_name = "out/brownian_3d_3".into();
    parameters.random_seed = 3;
    test_brownian!(parameters, Brownian3D, 3)?;
    Ok(())
}

#[test]
fn diffusion_constant_brownian_2d_1() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/brownian_2d_1".into();
    parameters.diffusion_constant = 1.0;
    parameters.random_seed = 4;
    test_brownian!(parameters, Brownian2D, 2)?;
    Ok(())
}

#[test]
fn diffusion_constant_brownian_2d_2() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.diffusion_constant = 0.5;
    parameters.storage_name = "out/brownian_2d_2".into();
    parameters.random_seed = 5;
    test_brownian!(parameters, Brownian2D, 2)?;
    Ok(())
}

#[test]
fn diffusion_constant_brownian_2d_3() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.diffusion_constant = 0.25;
    parameters.storage_name = "out/brownian_2d_3".into();
    parameters.random_seed = 6;
    test_brownian!(parameters, Brownian2D, 2)?;
    Ok(())
}

#[test]
fn diffusion_constant_brownian_1d_1() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/brownian_1d_1".into();
    parameters.diffusion_constant = 1.0;
    parameters.random_seed = 7;
    test_brownian!(parameters, Brownian1D, 1)?;
    Ok(())
}

#[test]
fn diffusion_constant_brownian_1d_2() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.diffusion_constant = 0.5;
    parameters.storage_name = "out/brownian_1d_2".into();
    parameters.random_seed = 8;
    test_brownian!(parameters, Brownian1D, 1)?;
    Ok(())
}

#[test]
fn diffusion_constant_brownian_1d_3() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.diffusion_constant = 0.25;
    parameters.storage_name = "out/brownian_1d_3".into();
    parameters.random_seed = 9;
    test_brownian!(parameters, Brownian1D, 1)?;
    Ok(())
}

fn analyze_positions_langevin<const D: usize>(
    parameters: &Parameters,
    positions: impl IntoIterator<Item = (u64, Vec<SVector<f64, D>>)>,
    damping: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let means_std_dev = calculate_mean_std_dev(&parameters, positions)?;

    // Calculate probability for values to be identical
    let mass = 1.0;
    let kb_temperature = kb_temp(mass, parameters.diffusion_constant, damping);
    for &iteration in means_std_dev.keys() {
        // Here we assume that the initial velocity v(0) = 0
        // https://en.wikipedia.org/wiki/Langevin_equation#Trajectories_of_free_Brownian_particles
        if iteration > 1 {
            let (mean, std_dev) = means_std_dev[&iteration];
            let time = iteration as f64 * parameters.dt;
            let expected = -(D as f64) * kb_temperature / mass / damping.powf(2.0)
                * (1.0 - (-damping * time).exp())
                * (3.0 - (-damping * time).exp())
                + 2.0 * D as f64 * kb_temperature / mass * time / damping;
            assert!((expected - mean).abs() <= 3.5 * std_dev);
        }
    }
    Ok(())
}

fn kb_temp(mass: f64, diffusion_constant: f64, damping: f64) -> f64 {
    diffusion_constant * damping * mass
}

macro_rules! test_langevin {
    (
        $parameters:ident,
        $particle_name:ident,
        $d:literal,
        $damping:literal
    ) => {{
        let domain_size = $parameters.domain_size;
        assert!(domain_size > 0.0);
        let mut domain =
            CartesianCuboid::from_boundaries_and_n_voxels([0.0; $d], [domain_size; $d], [3; $d])?;
        domain.rng_seed = $parameters.random_seed;
        let initial_position = nalgebra::SVector::from([domain_size / 2.0; $d]);
        let mass = 1.0;
        let kb_temperature = $parameters.diffusion_constant * $damping * mass;
        let particles = (0..$parameters.n_particles)
            .map(|_| $particle_name {
                pos: initial_position.into(),
                vel: [0.0; $d].into(),
                mass,
                damping: $damping,
                kb_temperature,
            });
        let settings = define_settings(&$parameters)?;

        let storage_access = cellular_raza_core::backend::chili::run_simulation!(
            agents: particles,
            domain: domain,
            settings: settings,
            aspects: [Mechanics],
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
                        .map(|(_, (p, _))| cellular_raza::concepts::Position::pos(&p.cell))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<std::collections::HashMap<u64, _>>();

        analyze_positions_langevin(&$parameters, positions, $damping)?;
        Result::<(), Box<dyn std::error::Error>>::Ok(())
    }}
}

#[test]
fn diffusion_constant_langevin_3d_1() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_3d_1".into();
    parameters.diffusion_constant = 80.0;
    parameters.random_seed = 1;
    test_langevin!(parameters, Langevin3D, 3, 10.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_3d_2() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_3d_2".into();
    parameters.diffusion_constant = 40.0;
    parameters.random_seed = 2;
    test_langevin!(parameters, Langevin3D, 3, 10.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_3d_3() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_3d_3".into();
    parameters.diffusion_constant = 20.0;
    parameters.random_seed = 3;
    test_langevin!(parameters, Langevin3D, 3, 10.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_3d_4() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_3d_4".into();
    parameters.diffusion_constant = 40.0;
    parameters.random_seed = 4;
    test_langevin!(parameters, Langevin3D, 3, 1.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_3d_5() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_3d_5".into();
    parameters.diffusion_constant = 40.0;
    parameters.random_seed = 5;
    test_langevin!(parameters, Langevin3D, 3, 0.1)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_2d_1() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_2d_1".into();
    parameters.diffusion_constant = 80.0;
    parameters.random_seed = 6;
    test_langevin!(parameters, Langevin2D, 2, 10.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_2d_2() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_2d_2".into();
    parameters.diffusion_constant = 40.0;
    parameters.random_seed = 7;
    test_langevin!(parameters, Langevin2D, 2, 10.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_2d_3() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_2d_3".into();
    parameters.diffusion_constant = 20.0;
    parameters.random_seed = 8;
    test_langevin!(parameters, Langevin2D, 2, 10.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_2d_4() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_2d_4".into();
    parameters.diffusion_constant = 20.0;
    parameters.random_seed = 9;
    test_langevin!(parameters, Langevin2D, 2, 1.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_2d_5() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_2d_5".into();
    parameters.diffusion_constant = 20.0;
    parameters.random_seed = 10;
    test_langevin!(parameters, Langevin2D, 2, 0.1)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_1d_1() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_1d_1".into();
    parameters.diffusion_constant = 80.0;
    parameters.random_seed = 11;
    test_langevin!(parameters, Langevin1D, 1, 10.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_1d_2() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_1d_2".into();
    parameters.diffusion_constant = 40.0;
    parameters.random_seed = 12;
    test_langevin!(parameters, Langevin1D, 1, 10.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_1d_3() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_1d_3".into();
    parameters.diffusion_constant = 20.0;
    parameters.random_seed = 13;
    test_langevin!(parameters, Langevin1D, 1, 10.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_1d_4() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_1d_4".into();
    parameters.diffusion_constant = 20.0;
    parameters.random_seed = 14;
    test_langevin!(parameters, Langevin1D, 1, 1.0)?;
    Ok(())
}

#[test]
fn diffusion_constant_langevin_1d_5() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = Parameters::default();
    parameters.storage_name = "out/langevin_1d_5".into();
    parameters.diffusion_constant = 20.0;
    parameters.random_seed = 15;
    test_langevin!(parameters, Langevin1D, 1, 0.1)?;
    Ok(())
}
