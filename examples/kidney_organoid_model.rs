use cellular_raza::prelude::*;
use cellular_raza::impls_cell_models::modular_cell::ModularCell;

use nalgebra::{Unit,Vector2};

use serde::{Serialize,Deserialize};

use rand_chacha::ChaCha8Rng;
use rand::{SeedableRng,Rng};

use plotters::{
    prelude::{DrawingArea,Cartesian2d,Circle,ShapeStyle},
    coord::types::RangedCoordf64,
    backend::BitMapBackend,
};


// Constants of the simulation
pub const N_CELLS_1: u32 = 30;
pub const N_CELLS_2: u32 = 0;

// Mechanical parameters
pub const CELL_RADIUS: f64 = 20.0;
pub const CELL_POTENTIAL_STRENGTH: f64 = 0.5;
pub const CELL_VELOCITY_REDUCTION: f64 = 2.0;
pub const CELL_VELOCITY_REDUCTION_MAX: f64 = 20.0;
pub const CELL_VELOCITY_REDUCTION_RATE: f64 = 5e-4;
pub const ATTRACTION_MULTIPLIER: f64 = 2.0;

// Parameters for cell cycle
pub const DIVISION_AGE: f64 = 250.0;

// Parameters for domain
pub const N_VOXEL_X: usize = 15;
pub const N_VOXEL_Y: usize = 15;
pub const DOMAIN_SIZE_X: f64 = 3000.0;
pub const DOMAIN_SIZE_Y: f64 = 3000.0;
pub const DX_VOXEL_X: f64 = DOMAIN_SIZE_X / N_VOXEL_X as f64;
pub const DX_VOXEL_Y: f64 = DOMAIN_SIZE_Y / N_VOXEL_Y as f64;

// Time parameters
pub const N_TIMES: usize = 50_001;
pub const DT: f64 = 1.0;
pub const T_START: f64 = 0.0;
pub const SAVE_INTERVAL: usize = 50;

// Meta Parameters to control solving
pub const N_THREADS: usize = 14;


#[derive(Serialize,Deserialize,Clone,core::fmt::Debug,std::cmp::PartialEq)]
enum CellType {
    One,
    Two,
}


#[derive(Serialize,Deserialize,Clone,core::fmt::Debug)]
struct DirectedSphericalMechanics {
    pub pos: Vector2<f64>,
    pub vel: Vector2<f64>,
    pub orientation: Unit<Vector2<f64>>,
}



#[derive(Serialize,Deserialize,Clone,core::fmt::Debug)]
struct CellSpecificInteraction {
    pub potential_strength: f64,
    pub attraction_multiplier: f64,
    pub cell_radius: f64,
    pub celltype: CellType,
    pub orientation: Unit<Vector2<f64>>,
}


impl Interaction<Vector2<f64>, Vector2<f64>, (f64, CellType, Unit<Vector2<f64>>)> for CellSpecificInteraction {
    fn get_interaction_information(&self) -> Option<(f64, CellType, Unit<Vector2<f64>>)> {
        Some((self.cell_radius, self.celltype.clone(), self.orientation.clone()))
    }

    fn calculate_force_on(&self, own_pos: &Vector2<f64>, ext_pos: &Vector2<f64>, ext_info: &Option<(f64, CellType, Unit<Vector2<f64>>)>) -> Option<Result<Vector2<f64>, CalcError>> {
        let (r, dir) = match (own_pos-ext_pos).norm() < self.cell_radius/10.0 {
            false => {
                let z = own_pos - ext_pos;
                let r = z.norm();
                (r, z.normalize())
            },
            true => {
                let dir = match own_pos==ext_pos {
                    true => self.orientation.into_inner(),
                    false => (own_pos - ext_pos).normalize(),
                };
                let r = self.cell_radius/10.0;
                (r, dir)
            }
        };
        match ext_info {
            Some((ext_radius, celltype, external_orientation)) => {
                // Introduce Non-dimensional length variable
                let sigma = r/(self.cell_radius + ext_radius);
                let bound = 4.0 + 1.0/sigma;
                let spatial_cutoff = (1.0+(1.5*(self.cell_radius+ext_radius)-r).signum())*0.5;
                
                // Calculate the strength of the interaction with correct bounds
                let strength = self.potential_strength*((1.0/sigma).powf(2.0) - (1.0/sigma).powf(4.0)).min(bound).max(-bound);
                
                // Calculate the attraction modifier based on the different orientation value
                let attraction_orientation_modifier = dir.dot(external_orientation).abs();

                // Calculate only attracting and repelling forces
                let attracting_force = dir * self.attraction_multiplier * attraction_orientation_modifier * strength.max(0.0) * spatial_cutoff;
                let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

                if attracting_force.iter().any(|e| e.is_nan()) {
                    println!("{:8.4?} {:8.4?} {:?}", own_pos, ext_pos, ext_info);
                    println!("{}", own_pos==ext_pos);
                }
                if repelling_force.iter().any(|e| e.is_nan()) {
                    println!("{:8.4?} {:8.4?} {:?}", own_pos, ext_pos, ext_info);
                    println!("{}", own_pos==ext_pos);
                }
                
                if *celltype == self.celltype {
                    Some(Ok(repelling_force + attracting_force))
                } else {
                    Some(Ok(repelling_force))
                }
            },
            None => None,
        }
    }
}


#[derive(Serialize,Deserialize,Debug,Clone)]
struct OwnCycle {
    age: f64,
    pub division_age: f64,
    has_divided: bool,
}


impl OwnCycle {
    fn new(division_age: f64) -> Self {
        OwnCycle {
            age: 0.0,
            division_age,
            has_divided: false
        }
    }
}


impl cellular_raza::concepts::cycle::Cycle<ModularCell<Vector2<f64>, MechanicsModel2D, CellSpecificInteraction, OwnCycle>> for OwnCycle {
    fn update_cycle(_rng: &mut rand_chacha::ChaCha8Rng, dt: &f64, c: &mut ModularCell<Vector2<f64>, MechanicsModel2D, CellSpecificInteraction, OwnCycle>) -> Option<CycleEvent> {
        if c.interaction.cell_radius < CELL_RADIUS {
            c.interaction.cell_radius += 2.0 * CELL_RADIUS * dt / DIVISION_AGE;
        }
        c.cycle.age += dt;
        if c.cycle.age >= c.cycle.division_age && c.cycle.has_divided == false {
            Some(CycleEvent::Division)
        } else {
            None
        }
    }

    fn divide(rng: &mut rand_chacha::ChaCha8Rng, c1: &mut ModularCell<Vector2<f64>, MechanicsModel2D, CellSpecificInteraction, OwnCycle>) -> Result<Option<ModularCell<Vector2<f64>, MechanicsModel2D, CellSpecificInteraction, OwnCycle>>, DivisionError> {
        // Clone existing cell
        let mut c2 = c1.clone();
        let r = c1.interaction.cell_radius;

        // Make both cells smaller
        c1.interaction.cell_radius /= std::f64::consts::SQRT_2;
        c2.interaction.cell_radius /= std::f64::consts::SQRT_2;

        // Generate cellular splitting direction randomly
        let angle_1 = std::f64::consts::FRAC_PI_2 + rng.gen_range(-std::f64::consts::FRAC_PI_8..std::f64::consts::FRAC_PI_8);
        let dir_vec = nalgebra::Rotation2::new(angle_1) * c1.interaction.orientation;

        // Define new positions for cells
        // It is randomly chosen if the old cell is left or right
        let sign = rng.gen_range(-1.0_f64..1.0_f64).signum();
        let offset = sign*dir_vec.into_inner()*r*0.5;
        let old_pos = c1.pos();

        c1.set_pos(&(old_pos + offset));
        c2.set_pos(&(old_pos - offset));

        // Set flag true that first cell has already divided
        c1.cycle.has_divided = true;

        // New cell is completely new so set age to 0
        c2.cycle.age = 0.0;

        // println!("Division succeeded! new positions: {:5.1?} {:5.1?}", c1.pos(), c2.pos());
        Ok(Some(c2))
    }
}


fn plot_modular_cell
    (modular_cell: &ModularCell<Vector2<f64>, MechanicsModel2D, CellSpecificInteraction, OwnCycle>, root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), SimulationError>
{
    let cell_border_color = plotters::prelude::BLACK;
    let cell_inside_color = match modular_cell.interaction.celltype {
        CellType::One => plotters::prelude::full_palette::GREEN,
        CellType::Two => plotters::prelude::full_palette::ORANGE,
    };
    let cell_orientation_color = plotters::prelude::full_palette::BLACK;

    let relative_border_thickness = 0.25;

    // Plot the cell border
    let dx = root.get_x_range().end - root.get_x_range().start;
    let dx_pix = root.get_x_axis_pixel_range().end - root.get_x_axis_pixel_range().start;

    let s = modular_cell.interaction.cell_radius / dx * dx_pix as f64;
    let cell_border = Circle::new(
        (modular_cell.pos().x, modular_cell.pos().y),
        s,
        Into::<ShapeStyle>::into(&cell_border_color).filled(),
    );
    root.draw(&cell_border).unwrap();

    // Plot the inside of the cell
    let cell_inside = Circle::new(
        (modular_cell.pos().x, modular_cell.pos().y),
        s*(1.0 - relative_border_thickness),
        Into::<ShapeStyle>::into(&cell_inside_color).filled(),
    );
    root.draw(&cell_inside).unwrap();

    // Plot the orientation as a line in the cell
    let rotation = nalgebra::Rotation2::new(std::f64::consts::FRAC_PI_2);
    let directed_offset = (1.0 - 0.5*relative_border_thickness) * modular_cell.interaction.cell_radius * (rotation * modular_cell.interaction.orientation).into_inner();
    let start = modular_cell.pos() - directed_offset;
    let end = modular_cell.pos() + directed_offset;
    let orientation_pointer = plotters::element::PathElement::new(
        [(start.x, start.y),
        (end.x, end.y)],
        Into::<ShapeStyle>::into(&cell_orientation_color).filled().stroke_width((s/5.0).ceil() as u32),
    );
    root.draw(&orientation_pointer).unwrap();
    Ok(())
}


fn main() {
    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // ###################################### DEFINE SIMULATION DOMAIN ######################################
    // Define the simulation domain
    let domain = CartesianCuboid2 {
        min: [0.0; 2],
        max: [DOMAIN_SIZE_X, DOMAIN_SIZE_Y],
        n_vox: [N_VOXEL_X, N_VOXEL_Y],
        voxel_sizes: [DX_VOXEL_X, DX_VOXEL_Y],
    };

    // ###################################### DEFINE CELLS IN SIMULATION ######################################
    // Cells of Type 1
    let relative_outer_border = 0.3;
    let mut cells = (0..N_CELLS_1).map(|_| {
        let pos = Vector2::<f64>::from([
            rng.gen_range(relative_outer_border*DOMAIN_SIZE_X..(1.0-relative_outer_border)*DOMAIN_SIZE_X),
            rng.gen_range(relative_outer_border*DOMAIN_SIZE_Y..(1.0-relative_outer_border)*DOMAIN_SIZE_Y)]);
        ModularCell {
        mechanics: cellular_raza::impls_cell_models::modular_cell::MechanicsOptions::Mechanics(MechanicsModel2D {
            pos,
            vel: Vector2::from([0.0, 0.0]),
            dampening_constant: CELL_VELOCITY_REDUCTION,
        }),
        interaction: CellSpecificInteraction {
            potential_strength: CELL_POTENTIAL_STRENGTH,
            attraction_multiplier: ATTRACTION_MULTIPLIER,
            cell_radius: CELL_RADIUS,
            celltype: CellType::One,
            orientation: Unit::<Vector2<f64>>::new_normalize(Vector2::<f64>::from([1.0, 0.0])),//pos.y/DOMAIN_SIZE_Y - 1.0])),
        },
        cycle: OwnCycle::new(rng.gen_range(0.8*DIVISION_AGE..1.2*DIVISION_AGE)),
    }}).collect::<Vec<_>>();

    // Cells of Type 2
    let cells_type2 = (0..N_CELLS_2).map(|_| ModularCell {
        mechanics: cellular_raza::impls_cell_models::modular_cell::MechanicsOptions::Mechanics(MechanicsModel2D {
            pos: Vector2::<f64>::from([
                rng.gen_range(0.0..DOMAIN_SIZE_X),
                rng.gen_range(0.0..DOMAIN_SIZE_Y)]),
            vel: Vector2::from([0.0, 0.0]),
            dampening_constant: CELL_VELOCITY_REDUCTION,
        }),
        interaction: CellSpecificInteraction {
            potential_strength: CELL_POTENTIAL_STRENGTH,
            attraction_multiplier: ATTRACTION_MULTIPLIER,
            cell_radius: CELL_RADIUS,
            celltype: CellType::Two,
            orientation: Unit::<Vector2<f64>>::new_normalize(Vector2::<f64>::from([rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)])),
        },
        cycle: OwnCycle::new(DIVISION_AGE),
    });
    cells.extend(cells_type2);

    // ###################################### CREATE SUPERVISOR AND RUN SIMULATION ######################################
    let setup = SimulationSetup {
        domain,
        cells,
        time: TimeSetup {
            t_start: 0.0,
            t_eval: (0..N_TIMES).map(|i| (T_START + DT * i as f64, i % SAVE_INTERVAL == 0, i % SAVE_INTERVAL == 0)).collect::<Vec<(f64, bool, bool)>>(),
        },
        meta_params: SimulationMetaParams {
            n_threads: N_THREADS
        },
        database: SledDataBaseConfig {
            name: "out/simulation_custom_cells".to_owned().into(),
        }
    };

    let mut supervisor = SimulationSupervisor::from(setup);

    supervisor.run_full_sim().unwrap();

    supervisor.end_simulation().unwrap();

    supervisor.plotting_config = PlottingConfig {
        n_threads: Some(16),
        image_size: 2000,
    };

    // ###################################### PLOT THE RESULTS ######################################
    supervisor.plot_cells_at_every_iter_bitmap_with_cell_plotting_func(&plot_modular_cell).unwrap();
}
