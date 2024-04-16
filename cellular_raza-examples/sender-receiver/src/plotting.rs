use cellular_raza::prelude::*;

use plotters::{
    backend::BitMapBackend,
    coord::types::RangedCoordf64,
    prelude::{Cartesian2d, Circle, DrawingArea, ShapeStyle},
    style::colors::colormaps::{ColorMap, DerivedColorMap, ViridisRGB},
    style::RGBColor,
};

use crate::bacteria_properties::*;

pub fn plot_voxel(
    voxel: &CartesianCuboidVoxel2<NUMBER_OF_REACTION_COMPONENTS>,
    root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) -> Result<(), DrawingError> {
    // Define lower and upper bounds for our values
    let lower_bound = 0.0;
    let upper_bound = 12.0;
    let concentration = voxel.get_total_extracellular()[0];

    // This should give a nice colormap
    let voxel_color = ViridisRGB::get_color_normalized(concentration, lower_bound, upper_bound);
    let rectangle = plotters::prelude::Rectangle::new(
        [
            (voxel.get_min()[0], voxel.get_min()[1]),
            (voxel.get_max()[0], voxel.get_max()[1]),
        ],
        Into::<ShapeStyle>::into(&voxel_color).filled(),
    );
    root.draw(&rectangle)?;
    Ok(())
}

pub fn plot_modular_cell(
    modular_cell: &MyCellType,
    root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) -> Result<(), DrawingError> {
    let cell_border_color = plotters::prelude::BLACK;

    let relative_border_thickness = 0.25;

    // Plot the cell border
    let dx = root.get_x_range().end - root.get_x_range().start;
    let dx_pix = root.get_x_axis_pixel_range().end - root.get_x_axis_pixel_range().start;

    let s = modular_cell.interaction.get_radius() / dx * dx_pix as f64;
    let cell_border = Circle::new(
        (modular_cell.pos().x, modular_cell.pos().y),
        s,
        Into::<ShapeStyle>::into(&cell_border_color).filled(),
    );
    root.draw(&cell_border)?;

    /* let lower_bound = 0.0;
    let upper_bound = modular_cell.cycle.food_threshold / 10.0;

    // Define colormap
    let derived_colormap = DerivedColorMap::new(&[RGBColor(102, 52, 83), RGBColor(247, 126, 201)]);

    // Plot the inside of the cell
    let cell_inside_color = derived_colormap.get_color_normalized(
        modular_cell.get_intracellular()[0],
        lower_bound,
        upper_bound,
    );

    // let cell_inside_color = Life::get_color_normalized(modular_cell.get_intracellular()[1], 0.0, modular_cell.cellular_reactions.intracellular_concentrations_saturation_level[1]);
    let cell_inside = Circle::new(
        (modular_cell.pos().x, modular_cell.pos().y),
        s * (1.0 - relative_border_thickness),
        Into::<ShapeStyle>::into(&cell_inside_color).filled(),
    );
    root.draw(&cell_inside)?;*/
    Ok(())
}
