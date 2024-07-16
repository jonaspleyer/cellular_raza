use cellular_raza::building_blocks::*;
use cellular_raza::concepts::*;
use cellular_raza::core::storage::*;
use cellular_raza_core::backend::chili::CellIdentifier;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(CellAgent, Clone, Debug, Deserialize, Serialize, PartialEq)]
struct MyAgent {
    #[Mechanics]
    mechanics: NewtonDamped3D,
    #[Interaction]
    interaction: MorsePotential,
}

macro_rules! set_up_and_return (
    ($domain:ident, $agents:ident, $settings:ident, [$($aspect:ident),*]) => {{
        let storager = cellular_raza::core::backend::chili::run_simulation!(
            domain: $domain,
            agents: $agents,
            settings: $settings,
            aspects: [$($aspect),*],
            determinism: false,
        )?;
        Result::<_, cellular_raza::core::backend::chili::SimulationError>::Ok(storager
            .cells
            .load_all_elements()?
            .into_iter()
            .map(|(iteration, agents)| {
                (
                    iteration,
                    agents
                        .into_iter()
                        .map(|(identifier, (agent, _))| (identifier, agent.cell))
                        .collect(),
                )
            })
            .collect())
    }}
);

macro_rules! ensure_results_identical(
    ($test_name:ident, $func_name:ident) => {
        #[test]
        fn $test_name () -> Result<(), Box<dyn std::error::Error>> {
            let mut results = (0..10)
                .map(|_| $func_name())
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            let r1 = results.pop().unwrap();
            for res in results {
                for (iteration, agents) in res.into_iter() {
                    let t1 = r1.get(&iteration).unwrap();
                    let t2 = agents;
                    assert_eq!(*t1, t2);
                }
            }
            Ok(())
        }
    }
);

fn test_newton_damped(
) -> Result<HashMap<u64, HashMap<CellIdentifier, MyAgent>>, Box<dyn std::error::Error>> {
    let domain =
        cellular_raza::building_blocks::CartesianCuboid::from_boundaries_and_interaction_range(
            [0f64; 3], [100.0; 3], 20.0,
        )?;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    #[cfg(debug_assertions)]
    let n_agents = 10;
    #[cfg(not(debug_assertions))]
    let n_agents = 50;
    let agents = (0..n_agents).map(|_| MyAgent {
        mechanics: NewtonDamped3D {
            pos: [
                rng.gen_range(0.0..100.0),
                rng.gen_range(0.0..100.0),
                rng.gen_range(0.0..100.0),
            ]
            .into(),
            vel: [0.0; 3].into(),
            damping_constant: 0.1,
            mass: 1.0,
        },
        interaction: MorsePotential {
            strength_attracting: 0.01,
            strength_repelling: 0.1,
            length_attracting: 3.0,
            length_repelling: 2.0,
            cutoff: 4.0,
        },
    });
    let storage = StorageBuilder::new().priority([StorageOption::Memory]);
    let time = cellular_raza::core::time::FixedStepsize::from_partial_save_steps(
        0.0,
        0.01,
        #[cfg(debug_assertions)]
        100,
        #[cfg(not(debug_assertions))]
        1_000,
        10,
    )?;
    let settings = cellular_raza::core::backend::chili::Settings {
        n_threads: 1.try_into().unwrap(),
        show_progressbar: false,
        storage,
        time,
    };

    let res = set_up_and_return!(domain, agents, settings, [Mechanics, Interaction]);
    Ok(res?)
}

ensure_results_identical!(
    determinism_newton_damped_mechanics_morse_interaction,
    test_newton_damped
);

fn test_pure_brownian(
) -> Result<HashMap<u64, HashMap<CellIdentifier, Brownian3D>>, Box<dyn std::error::Error>> {
    let domain =
        cellular_raza::building_blocks::CartesianCuboid::from_boundaries_and_interaction_range(
            [-30f64; 3],
            [100.0; 3],
            20.0,
        )?;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let agents = (0..10).map(|_| {
        Brownian3D::new(
            [
                rng.gen_range(-30.0..100.0),
                rng.gen_range(-30.0..100.0),
                rng.gen_range(-30.0..100.0),
            ]
            .into(),
            1e-12,
            1e-10,
        )
    });
    let storage = StorageBuilder::new().priority([StorageOption::Memory]);
    let time =
        cellular_raza::core::time::FixedStepsize::from_partial_save_steps(0.0, 0.01, 100, 10)?;
    let settings = cellular_raza::core::backend::chili::Settings {
        n_threads: 1.try_into().unwrap(),
        show_progressbar: false,
        storage,
        time,
    };

    let res = set_up_and_return!(domain, agents, settings, [Mechanics]);
    Ok(res?)
}

ensure_results_identical!(determinism_brownian_mechanics, test_pure_brownian);
