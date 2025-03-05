use cellular_raza::building_blocks::*;
use cellular_raza::core::backend::chili::*;
use cellular_raza::core::storage::*;
use StorageOption::*;

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

fn main_sim(
    storage_options: impl IntoIterator<Item = StorageOption>,
) -> Result<BTreeMap<u64, BTreeMap<CellIdentifier, Langevin3D>>, SimulationError> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let agents = (0..10).map(|_| Langevin3D {
        pos: [
            rng.gen_range(-1e3..-5e2),
            rng.gen_range(-1e3..-5e2),
            rng.gen_range(-1e3..-5e2),
        ]
        .into(),
        vel: [0.0; 3].into(),
        mass: 1.0,
        damping: 0.001,
        kb_temperature: 1e-10,
    });
    let domain = CartesianCuboid::from_boundaries_and_n_voxels([-1e3; 3], [-5e2; 3], [3; 3])?;
    let time = cellular_raza::core::time::FixedStepsize::from_partial_save_steps(0.0, 0.1, 100, 1)?;
    // Create temporary directory which will delete itself after usage
    let tmp_dir = tempfile::TempDir::new()?;
    let storage = StorageBuilder::new()
        .location(tmp_dir.path())
        .priority(storage_options);
    let settings = Settings {
        n_threads: 1.try_into().unwrap(),
        time,
        storage,
        show_progressbar: false,
    };
    let storager = run_simulation!(
        agents: agents,
        domain: domain,
        settings: settings,
        aspects: [Mechanics],
        determinism: true,
    )?;
    Ok(storager
        .cells
        .load_all_elements()?
        .into_iter()
        .map(|(iter, cells)| {
            (
                iter,
                cells
                    .into_iter()
                    .map(|(id, (cbox, _))| (id, cbox.cell))
                    .collect(),
            )
        })
        .collect())
}

#[test]
fn storage_serde_json() -> Result<(), SimulationError> {
    let r1 = main_sim([SerdeJson])?;
    let r2 = main_sim([SerdeJson])?;
    let r3 = main_sim([SerdeJson])?;
    assert_eq!(r1, r2);
    assert_eq!(r2, r3);
    Ok(())
}

#[test]
fn storage_ron() -> Result<(), SimulationError> {
    let r1 = main_sim([Ron])?;
    let r2 = main_sim([Ron])?;
    let r3 = main_sim([Ron])?;
    assert_eq!(r1, r2);
    assert_eq!(r2, r3);
    Ok(())
}

#[test]
fn storage_sled() -> Result<(), SimulationError> {
    let r1 = main_sim([Sled])?;
    let r2 = main_sim([Sled])?;
    let r3 = main_sim([Sled])?;
    assert_eq!(r1, r2);
    assert_eq!(r2, r3);
    Ok(())
}

#[test]
fn storage_sled_temp() -> Result<(), SimulationError> {
    let r1 = main_sim([SledTemp])?;
    let r2 = main_sim([SledTemp])?;
    let r3 = main_sim([SledTemp])?;
    assert_eq!(r1, r2);
    assert_eq!(r2, r3);
    Ok(())
}

#[test]
fn storage_memory() -> Result<(), SimulationError> {
    let r1 = main_sim([Memory])?;
    let r2 = main_sim([Memory])?;
    let r3 = main_sim([Memory])?;
    assert_eq!(r1, r2);
    assert_eq!(r2, r3);
    Ok(())
}
