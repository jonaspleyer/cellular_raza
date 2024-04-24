use cellular_raza_building_blocks::{Brownian3D, CartesianCuboid3New};
use cellular_raza_concepts::{CalcError, CellAgent, Interaction, Mechanics, RngError};
use cellular_raza_core::storage::StorageInterfaceLoad;
use rand_chacha::rand_core::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
struct MyInteraction;

impl<Pos, Vel, For> Interaction<Pos, Vel, For> for MyInteraction
where
    For: num::Zero,
{
    fn calculate_force_between(
        &self,
        _own_pos: &Pos,
        _own_vel: &Vel,
        _ext_pos: &Pos,
        _ext_vel: &Vel,
        _ext_info: &(),
    ) -> Result<For, cellular_raza::prelude::CalcError> {
        Ok(For::zero())
    }
    fn get_interaction_information(&self) -> () {}
}

#[derive(CellAgent, Clone, Deserialize, Serialize)]
struct Particle {
    #[Mechanics]
    mechanics: Brownian3D,
    #[Interaction]
    interaction: MyInteraction,
}

struct Parameters {
    n_particles: usize,
    domain_size: f64,

    dt: f64,
    t_max: f64,
    save_interval: f64,

    diffusion_constant: f64,

    storage_name: std::path::PathBuf,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            n_particles: 800,
            domain_size: 200.0,

            dt: 0.1,
            t_max: 100.0,
            save_interval: 1.0,

            diffusion_constant: 1.0,
            storage_name: "out/brownian".into(),
        }
    }
}

fn brownian(parameters: &Parameters) -> Result<(), Box<dyn std::error::Error>> {
    let domain_size = parameters.domain_size;
    assert!(domain_size > 0.0);
    let domain =
        CartesianCuboid3New::from_boundaries_and_n_voxels([0.0; 3], [domain_size; 3], [3; 3])?;

    let time = cellular_raza_core::time::FixedStepsize::from_partial_save_interval(
        0.0,
        parameters.dt,
        parameters.t_max,
        parameters.save_interval,
    )?;

    let storage = cellular_raza_core::storage::StorageBuilder::new()
        .priority([cellular_raza_core::storage::StorageOption::Sled])
        .location(parameters.storage_name.clone())
        .init();

    let settings = cellular_raza_core::backend::chili::Settings {
        n_threads: 1.try_into().unwrap(),
        time,
        storage,
        show_progressbar: false,
    };

    let particles = (0..parameters.n_particles).map(|_| {
        let pos = [domain_size / 2.0; 3];
        Particle {
            mechanics: Brownian3D::new(pos, parameters.diffusion_constant, 1.0),
            interaction: MyInteraction,
        }
    });

    let storage_access = cellular_raza_core::backend::chili::run_simulation!(
        agents: particles,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction]
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
                    .map(|(_, (p, _))| (p.identifier, p.cell.mechanics.pos))
                    .collect::<std::collections::HashMap<_, _>>(),
            )
        })
        .collect::<std::collections::HashMap<u64, _>>();

    let min_iteration = *positions.keys().into_iter().min().unwrap();
    let initial_positions = positions[&min_iteration].clone();
    let square_displacements = positions
        .into_iter()
        .map(|(iteration, positions)| {
            (
                iteration,
                positions
                    .into_iter()
                    .map(|(id, pos)| (pos - initial_positions[&id]).norm_squared())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<std::collections::HashMap<u64, Vec<f64>>>();

    let means = square_displacements
        .iter()
        .map(|(iteration, displs)| (iteration, displs.iter().sum::<f64>() / displs.len() as f64))
        .collect::<std::collections::HashMap<_, _>>();

    let std_dev = square_displacements
        .iter()
        .map(|(iteration, displs)| {
            (
                iteration,
                (displs
                    .iter()
                    .map(|displ| (displ - means[&iteration]).powf(2.0))
                    .sum::<f64>()
                    / displs.len() as f64)
                    .sqrt(),
            )
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
