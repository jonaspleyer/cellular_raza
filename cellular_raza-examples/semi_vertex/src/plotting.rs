use cellular_raza::prelude::*;

use plotters::{
    backend::BitMapBackend,
    coord::types::RangedCoordf64,
    prelude::{Cartesian2d, DrawingArea, ShapeStyle},
    style::colors::colormaps::ViridisRGB,
    style::RGBColor,
};

use nalgebra::Vector2;

use crate::cell_properties::*;

pub fn plot_voxel(
    voxel: &CartesianCuboidVoxel2Vertex<NUMBER_OF_VERTICES, NUMBER_OF_REACTION_COMPONENTS>,
    root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) -> Result<(), DrawingError> {
    // Define lower and upper bounds for our values
    let lower_bound = 0.0;
    let upper_bound = 1000.0;
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

    // Also plot the direction in which the current concentration is pointing
    let gradient = voxel.extracellular_gradient[0];
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
    // let cell_orientation_color = plotters::prelude::full_palette::BLACK;

    let relative_border_thickness = 0.1;

    // Plot the cell border
    let pos = modular_cell.pos();
    let middle = pos.row_sum() / pos.shape().0 as f64;
    // Calculate the paths
    let path_points = pos
        .row_iter()
        .map(|row| {
            (
                middle[0] + row[0] - middle[0],
                middle[1] + row[1] - middle[1],
            )
        })
        .collect::<Vec<_>>();
    // Define the style
    let cell_border = plotters::element::Polygon::new(
        path_points.clone(),
        Into::<ShapeStyle>::into(&cell_border_color),
    );
    root.draw(&cell_border)?;

    // Define color inside of cell
    let cell_inside_color = RGBColor(28, 173, 28);
    let cell_inside = plotters::element::Polygon::new(
        path_points
            .clone()
            .iter()
            .map(|point| {
                (
                    middle[0] + (1.0 - relative_border_thickness) * (point.0 - middle[0]),
                    middle[1] + (1.0 - relative_border_thickness) * (point.1 - middle[1]),
                )
            })
            .collect::<Vec<_>>(),
        Into::<ShapeStyle>::into(&cell_inside_color).filled(),
    );
    root.draw(&cell_inside)?;
    Ok(())
}
