use cellular_raza::prelude::*;

use plotters::{
    backend::BitMapBackend,
    coord::types::RangedCoordf64,
    prelude::{Cartesian2d, DrawingArea, ShapeStyle},
    style::{colors::colormaps::ColorMap, RGBColor},
};

use crate::cell_properties::*;

pub fn plot_cell<const D: usize>(
    cell: &MyCell<D>,
    root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) -> Result<(), DrawingError> {
    let cell_border_color = plotters::prelude::BLACK;
    // let cell_orientation_color = plotters::prelude::full_palette::BLACK;

    let relative_border_thickness = 0.1;

    // Plot the cell border
    let pos = cell.pos();
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
    let colormap = plotters::style::colors::colormaps::DerivedColorMap::new(&[
        plotters::style::colors::BLACK,
        RGBColor(100, 255, 100),
    ]);
    let cell_inside_color = colormap.get_color_normalized(cell.intracellular.x, 0.0, 0.1);
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
