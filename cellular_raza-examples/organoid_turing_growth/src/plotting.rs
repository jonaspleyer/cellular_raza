use cellular_raza::building_blocks::cartesian_cuboid_n_old::CartesianCuboidVoxel2;
use cellular_raza::concepts::domain_old::ExtracellularMechanics;
use cellular_raza::concepts::reactions_old::*;
use cellular_raza::prelude::*;

use plotters::{
    backend::BitMapBackend,
    coord::types::RangedCoordf64,
    prelude::{Cartesian2d, Circle, DrawingArea, ShapeStyle},
    style::colors::colormaps::{ColorMap, DerivedColorMap, ViridisRGB},
    style::RGBColor,
};

use nalgebra::Vector2;

use crate::cell_properties::*;

pub fn plot_voxel(
    voxel: &CartesianCuboidVoxel2<NUMBER_OF_REACTION_COMPONENTS>,
    root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) -> Result<(), DrawingError> {
    // Define lower and upper bounds for our values
    let lower_bound = 0.0;
    let upper_bound = 1000.0;
    let concentration = voxel.get_total_extracellular()[2];

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

    // Also plot the direction in which the current concentration is pointing
    let gradient = voxel.extracellular_gradient[2];
    let strength = gradient.norm();
    let gradient_upper_bound =
        (upper_bound - lower_bound) / voxel.get_dx().iter().sum::<f64>() * 2.0;
    let start = Vector2::from(voxel.get_middle());
    let end = if gradient != Vector2::from([0.0; 2]) {
        start
            + (gradient / strength.max(gradient_upper_bound))
                .component_mul(&Vector2::from(voxel.get_dx()))
                / 2.0
    } else {
        start
    };
    let pointer = plotters::element::PathElement::new(
        [(start.x, start.y), (end.x, end.y)],
        Into::<ShapeStyle>::into(&plotters::prelude::BLACK)
            .filled()
            .stroke_width(2),
    );
    root.draw(&pointer)?;
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

    let s = modular_cell.interaction.cell_radius / dx * dx_pix as f64;
    let cell_border = Circle::new(
        (modular_cell.pos().x, modular_cell.pos().y),
        s,
        Into::<ShapeStyle>::into(&cell_border_color).filled(),
    );
    root.draw(&cell_border)?;

    // Create colormap
    let life_colormap = DerivedColorMap::new(&[
        RGBColor(153, 0, 0),
        RGBColor(255, 153, 51),
        RGBColor(120, 255, 120),
        RGBColor(51, 255, 51),
    ]);
    let poison_colormap = DerivedColorMap::new(&[
        RGBColor(11, 54, 38),
        RGBColor(10, 110, 73),
        RGBColor(63, 171, 104),
        RGBColor(141, 235, 113),
    ]);

    // Plot the inside of the cell
    let cell_inside_color = match modular_cell.cycle.is_ureter {
        true => life_colormap.get_color_normalized(
            modular_cell.get_intracellular()[1],
            0.0,
            modular_cell
                .cellular_reactions
                .intracellular_concentrations_saturation_level[1],
        ),
        false => poison_colormap.get_color_normalized(
            modular_cell.get_intracellular()[1],
            0.0,
            modular_cell
                .cellular_reactions
                .intracellular_concentrations_saturation_level[1],
        ),
    };
    // let cell_inside_color = Life::get_color_normalized(modular_cell.get_intracellular()[1], 0.0, modular_cell.cellular_reactions.intracellular_concentrations_saturation_level[1]);
    let cell_inside = Circle::new(
        (modular_cell.pos().x, modular_cell.pos().y),
        s * (1.0 - relative_border_thickness),
        Into::<ShapeStyle>::into(&cell_inside_color).filled(),
    );
    root.draw(&cell_inside)?;
    Ok(())
}
