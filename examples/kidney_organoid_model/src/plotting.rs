use cellular_raza::prelude::*;
use cellular_raza::impls_cell_models::modular_cell::ModularCell;

use plotters::{
    prelude::{DrawingArea,Cartesian2d,Circle,ShapeStyle},
    coord::types::RangedCoordf64,
    backend::BitMapBackend,
};

use nalgebra::Vector2;

use crate::cell_properties::*;


pub fn plot_voxel
    (voxel: &CartesianCuboidVoxel2Reactions1, root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), SimulationError>
{
    // Define lower and upper bounds for our values
    let lower_bound = 0.0;
    let upper_bound = 1000.0;
    let concentration = voxel.get_total_extracellular()[0];
    // println!("{}", concentration);
    let h = ((concentration-lower_bound)/(upper_bound-lower_bound)).min(1.0).max(0.0);

    // This should give a greyscale color palette
    let voxel_color = create_viridis_color(h);
    let circle = plotters::prelude::Rectangle::new(
        [(voxel.get_min()[0], voxel.get_min()[1]), (voxel.get_max()[0], voxel.get_max()[1])],
        Into::<ShapeStyle>::into(&voxel_color).filled()
    );

    root.draw(&circle)?;
    Ok(())
}


pub fn plot_modular_cell
    (modular_cell: &ModularCell<Vector2<f64>, MechanicsModel2D, CellSpecificInteraction, OwnCycle, OwnReactions>, root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), SimulationError>
{
    let cell_border_color = plotters::prelude::BLACK;
    let cell_inside_color = plotters::prelude::full_palette::GREEN;
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
    root.draw(&cell_border)?;

    // Plot the inside of the cell
    let cell_inside = Circle::new(
        (modular_cell.pos().x, modular_cell.pos().y),
        s*(1.0 - relative_border_thickness),
        Into::<ShapeStyle>::into(&cell_inside_color).filled(),
    );
    root.draw(&cell_inside)?;

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
    root.draw(&orientation_pointer)?;
    Ok(())
}
