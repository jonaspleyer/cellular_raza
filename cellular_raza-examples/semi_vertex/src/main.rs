use backend::chili;
use cellular_raza::prelude::*;

use plotters::drawing::IntoDrawingArea;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const MICRON: f64 = 1e-0;

pub const SECOND: f64 = 1e0;
pub const MINUTE: f64 = 60e0;

// Number of cells
pub const N_CELLS: usize = 1;

// Mechanical parameters
pub const CELL_MECHANICS_AREA: f64 = 500.0 * MICRON * MICRON;
pub const CELL_MECHANICS_SPRING_TENSION: f64 = 0.0001;
pub const CELL_MECHANICS_CENTRAL_PRESSURE: f64 = 0.0001;
pub const CELL_MECHANICS_INTERACTION_RANGE: f64 = 5.0 * MICRON;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 1.0;
pub const CELL_MECHANICS_DAMPING_CONSTANT: f64 = 0.01 / SECOND;
pub const CELL_MECHANICS_DIFFUSION_CONSTANT: f64 = 0.0 * MICRON * MICRON / SECOND;

// Cycle
pub const CELL_CYCLE_GROWTH_FACTOR: f64 = 0.0 * MICRON * MICRON / SECOND;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 600.0 * MICRON;
pub const DOMAIN_SIZE_Y: f64 = 600.0 * MICRON;

// Time parameters
pub const DT: f64 = 0.01 * SECOND;
pub const T_END: f64 = 10.0 * MINUTE;
pub const SAVE_INTERVAL: f64 = 4.0 * SECOND;

// Meta Parameters to control solving
pub const N_THREADS: usize = 1;

mod alternative_vertex_mechanics;
mod cell_properties;
mod custom_domain;
mod plotting;

use alternative_vertex_mechanics::VertexMechanics2DAlternative;
use cell_properties::*;
use custom_domain::*;
use plotting::*;
use time::FixedStepsize;

fn main() -> Result<(), chili::SimulationError> {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // Define the simulation domain
    let domain = MyDomain {
        cuboid: CartesianCuboid::from_boundaries_and_interaction_range(
            [0.0; 2],
            [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
            2.0 * CELL_MECHANICS_INTERACTION_RANGE
                .max((CELL_MECHANICS_AREA / std::f64::consts::PI).sqrt()),
        )?,
    };

    // Define cell agents
    let cells = (0..N_CELLS)
        .map(|_| MyCell {
            mechanics: VertexMechanics2DAlternative::<6>::new(
                [
                    rng.gen_range(0.4 * DOMAIN_SIZE_X..0.6 * DOMAIN_SIZE_X),
                    rng.gen_range(0.4 * DOMAIN_SIZE_Y..0.6 * DOMAIN_SIZE_Y),
                ]
                .into(),
                CELL_MECHANICS_AREA,
                rng.gen_range(0.0..2.0 * std::f64::consts::PI),
                CELL_MECHANICS_SPRING_TENSION,
                CELL_MECHANICS_CENTRAL_PRESSURE,
                CELL_MECHANICS_DAMPING_CONSTANT,
                CELL_MECHANICS_DIFFUSION_CONSTANT,
                Some((0.3, rng.clone())),
            ),
            interaction: VertexDerivedInteraction::from_two_forces(
                OutsideInteraction {
                    potential_strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
                    interaction_range: CELL_MECHANICS_INTERACTION_RANGE,
                },
                InsideInteraction {
                    potential_strength: 1.5 * CELL_MECHANICS_POTENTIAL_STRENGTH,
                    average_radius: CELL_MECHANICS_AREA.sqrt(),
                },
            ),
            growth_factor: CELL_CYCLE_GROWTH_FACTOR,
            division_area_threshold: 2.0 * CELL_MECHANICS_AREA,
        })
        .collect::<Vec<_>>();

    // Define settings for storage and time solving
    let settings = chili::Settings {
        time: FixedStepsize::from_partial_save_interval(0.0, DT, T_END, SAVE_INTERVAL)?,
        n_threads: N_THREADS.try_into().unwrap(),
        show_progressbar: true,
        storage: StorageBuilder::new().location("out/semi_vertex"),
    };

    // Run the simulation
    let storager = chili::run_simulation!(
        agents: cells,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction],
    )?;

    // Plot the results
    let save_path = storager.get_path()?.join("images");
    std::fs::create_dir(&save_path)?;
    let all_iterations = storager.cells.get_all_iterations()?;

    println!("");
    use rayon::prelude::*;
    kdam::par_tqdm!(all_iterations.into_par_iter())
        .map(move |iteration| -> Result<(), chili::SimulationError> {
            let cells = storager.cells.load_all_elements_at_iteration(iteration)?;
            let img_path = save_path.join(format!("snapshot_{:08}.png", iteration));
            let domain_size_x = DOMAIN_SIZE_X.round() as u32;
            let domain_size_y = DOMAIN_SIZE_Y.round() as u32;
            let root =
                plotters::prelude::BitMapBackend::new(&img_path, (domain_size_x, domain_size_y))
                    .into_drawing_area();
            root.fill(&plotters::prelude::WHITE)?;
            let mut root = root.apply_coord_spec(plotters::prelude::Cartesian2d::<
                plotters::coord::types::RangedCoordf64,
                plotters::coord::types::RangedCoordf64,
            >::new(
                0.0..DOMAIN_SIZE_X,
                0.0..DOMAIN_SIZE_X,
                (0..domain_size_x as i32, 0..domain_size_y as i32),
            ));
            for (_, (cell, _)) in cells {
                plot_cell(&cell.cell, &mut root)?;
            }
            root.present()?;
            Ok(())
        })
        .collect::<Result<Vec<_>, chili::SimulationError>>()?;
    Ok(())
}
