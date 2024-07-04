use cellular_raza::prelude::*;

use plotters::{
    backend::BitMapBackend,
    coord::types::RangedCoordf64,
    prelude::{Cartesian2d, DrawingArea, ShapeStyle},
    style::RGBColor,
};

use crate::cell_properties::*;

pub fn plot_cell(
    cell: &MyCell,
    root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) -> Result<(), DrawingError> {
    let cell_border_color = plotters::prelude::BLACK;
    // let cell_orientation_color = plotters::prelude::full_palette::BLACK;

    let relative_border_thickness = 0.1;

    // Plot the cell border
    let pos = cell.pos();
    let middle = pos.0.row_sum() / pos.0.nrows() as f64;
    // Calculate the paths
    let path_points = pos
        .0
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
