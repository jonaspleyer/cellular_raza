use cellular_raza::building_blocks::cartesian_cuboid_n_old::CartesianCuboidVoxel2;
use cellular_raza::concepts::domain_old::ExtracellularMechanics;
use cellular_raza::prelude::*;

use plotters::{
    backend::BitMapBackend,
    coord::types::RangedCoordf64,
    prelude::{Cartesian2d, Circle, DrawingArea, ShapeStyle},
    style::{
        colors::colormaps::{ColorMap, DerivedColorMap},
        RGBColor,
    },
};

use crate::{bacteria_properties::*, TARGET_AVERAGE_CONC};

pub fn plot_voxel(
    voxel: &CartesianCuboidVoxel2<NUMBER_OF_REACTION_COMPONENTS>,
    root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) -> Result<(), DrawingError> {
    use plotters::prelude::*;

    // Define lower and upper bounds for our values
    let lower_bound = 0.0;
    let upper_bound = 5.0 * TARGET_AVERAGE_CONC;
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

    // let text = plotters::element::Text::new(
    //     format!("{:6.3}", voxel.get_total_extracellular()[0]),
    //     (
    //         0.5 * (voxel.get_min()[0] + voxel.get_max()[0]),
    //         0.5 * (voxel.get_min()[1] + voxel.get_max()[1]),
    //     ),
    //     TextStyle::from(("normal", 20).into_font()).color(&BLACK),
    // );
    // root.draw(&text)?;
    Ok(())
}

pub fn plot_modular_cell(
    modular_cell: &MyCellType,
    root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) -> Result<(), DrawingError> {
    // Define colormap
    let lower_bound = 0.0;
    let upper_bound = 2.0 * TARGET_AVERAGE_CONC;
    let derived_colormap = DerivedColorMap::new(&[
        RGBColor(168, 39, 60),
        RGBColor(43, 168, 39),
        RGBColor(168, 86, 39),
    ]);

    // Get inside color of the cell
    let cell_inside_color = match modular_cell.cellular_reactions.species {
        Species::Sender => plotters::style::colors::full_palette::GREY,
        Species::Receiver => derived_colormap.get_color_normalized(
            modular_cell.get_intracellular()[0],
            lower_bound,
            upper_bound,
        ),
    };

    // Plot the cell border
    let dx = root.get_x_range().end - root.get_x_range().start;
    let dx_pix = root.get_x_axis_pixel_range().end - root.get_x_axis_pixel_range().start;

    let s = modular_cell.interaction.cell_radius / dx * dx_pix as f64;
    let cell_border = Circle::new(
        (modular_cell.pos().x, modular_cell.pos().y),
        s,
        Into::<ShapeStyle>::into(&plotters::prelude::BLACK).filled(),
    );

    // Plot the inside of the cell
    let q = 0.8 * s;
    let cell_inside = Circle::new(
        (modular_cell.pos().x, modular_cell.pos().y),
        q,
        Into::<ShapeStyle>::into(&cell_inside_color).filled(),
    );

    root.draw(&cell_border)?;
    root.draw(&cell_inside)?;
    Ok(())
}
