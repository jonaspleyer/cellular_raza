use cellular_raza::prelude::*;

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

    // This should give a nice colormap
    let voxel_color = ViridisRGB::get_color_normalized(concentration, lower_bound, upper_bound);
    let rectangle = plotters::prelude::Rectangle::new(
        [(voxel.get_min()[0], voxel.get_min()[1]), (voxel.get_max()[0], voxel.get_max()[1])],
        Into::<ShapeStyle>::into(&voxel_color).filled()
    );
    root.draw(&rectangle)?;

    // Also plot the direction in which the current concentration is pointing
    let gradient = voxel.extracellular_gradient[0];
    let strength = gradient.norm();
    let gradient_upper_bound = 100.0;
    let start = Vector2::from(voxel.get_middle());
    let end = if gradient!=Vector2::from([0.0; 2]) {start + (gradient/strength.max(gradient_upper_bound)).component_mul(&Vector2::from(voxel.get_dx()))/2.0} else {start};
    let pointer = plotters::element::PathElement::new(
        [(start.x, start.y),
        (end.x, end.y)],
        Into::<ShapeStyle>::into(&plotters::prelude::BLACK).filled().stroke_width(2),
    );
    root.draw(&pointer)?;
    Ok(())
}


pub fn plot_modular_cell
    (modular_cell: &MyCellType, root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), SimulationError>
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
    let directed_offset = (1.0 - 0.5*relative_border_thickness) * modular_cell.interaction.cell_radius * modular_cell.interaction.orientation.into_inner();
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
