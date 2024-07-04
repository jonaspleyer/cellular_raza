use backend::chili;
use cellular_raza::prelude::*;

use plotters::drawing::IntoDrawingArea;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

// Number of cells
pub const N_CELLS: usize = 30;

pub const MICROMETRE: f64 = 1e-6;
pub const KILOGRAM: f64 = 1e0;
pub const METRE: f64 = 1e0;
pub const SECOND: f64 = 1e0;
pub const PASCAL: f64 = KILOGRAM / METRE / SECOND / SECOND;
pub const BAR: f64 = 100_000.0 * PASCAL;
pub const NEWTON: f64 = KILOGRAM * METRE / SECOND / SECOND;

// Mechanical parameters
pub const CELL_MECHANICS_AREA: f64 = 500.0 * MICROMETRE * MICROMETRE;
pub const CELL_MECHANICS_SPRING_TENSION: f64 = 1e1 * NEWTON / METRE;
pub const CELL_MECHANICS_N_KB_T: f64 = 4e0 * BAR * CELL_MECHANICS_AREA;

// TODO find units!
pub const CELL_MECHANICS_INTERACTION_RANGE: f64 = 10.0 * MICROMETRE;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 1e-4 * NEWTON;
pub const CELL_MECHANICS_DAMPING_CONSTANT: f64 = 2.0 / SECOND;
pub const CELL_MECHANICS_DIFFUSION_CONSTANT: f64 = 0.2;

// Parameters for domain
pub const DOMAIN_SIZE_X: f64 = 150.0 * MICROMETRE;
pub const DOMAIN_SIZE_Y: f64 = 150.0 * MICROMETRE;

// Time parameters
pub const N_TIMES: u64 = 1_002;
pub const DT: f64 = 0.01 * SECOND;
pub const T_START: f64 = 0.0 * SECOND;
pub const SAVE_INTERVAL: u64 = 2;

// Meta Parameters to control solving
pub const N_THREADS: usize = 1;

mod cell_properties;
mod custom_domain;
mod plotting;

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
    let dx = 1.05 * CELL_MECHANICS_AREA.sqrt();
    let n_x_max = (0.8 * DOMAIN_SIZE_X / dx).floor();
    let n_y_max = (0.8 * DOMAIN_SIZE_Y / dx).floor();
    let cells = (0..N_CELLS)
        .map(|n_cell| {
            let n_x = n_cell as f64 % n_x_max;
            let n_y = (n_cell as f64 / n_y_max).floor();
            MyCell {
                mechanics: CustomVertexMechanics2D::new(
                    6,
                    [
                        0.1 * DOMAIN_SIZE_X + n_x * dx + 0.5 * (n_y % 2.0) * dx,
                        0.1 * DOMAIN_SIZE_Y + n_y * dx,
                    ],
                    CELL_MECHANICS_AREA,
                    // rng.gen_range(0.0..2.0 * std::f64::consts::PI),
                    CELL_MECHANICS_SPRING_TENSION,
                    CELL_MECHANICS_N_KB_T,
                    CELL_MECHANICS_DAMPING_CONSTANT,
                    // CELL_MECHANICS_DIFFUSION_CONSTANT,
                    // None,
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
                growth_side: 0,
                growth_factor: 0.1,
                division_threshold_area: CELL_MECHANICS_AREA * 1.1,
            }
        })
        .collect::<Vec<_>>();

    // Define settings for storage and time solving
    let settings = chili::Settings {
        time: FixedStepsize::from_partial_save_steps(0.0, DT, N_TIMES, SAVE_INTERVAL)?,
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
            let domain_size_x = (DOMAIN_SIZE_X / MICROMETRE).round() as u32 * 10;
            let domain_size_y = (DOMAIN_SIZE_Y / MICROMETRE).round() as u32 * 10;
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
