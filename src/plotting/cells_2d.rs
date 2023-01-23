use crate::plotting::spatial::PlotSelf;


use plotters::{
    backend::DrawingBackend,
    coord::Shift,
    prelude::{
        DrawingArea,
        Circle},
    style::ShapeStyle,
};


macro_rules! implement_draw_cell_2d (
    ($cell:ty, $domain:ty, $index:ty, $voxel:ty, $($ni:tt),+) => {
        impl<Db, E> PlotSelf<Db, E, $domain, $cell, $index, $voxel> for $cell
        where
            Db: DrawingBackend<ErrorType=E>,
            E: std::error::Error + std::marker::Sync + std::marker::Send,
        {
            fn plot_self(&self, domain: &$domain, root: &mut DrawingArea<Db, Shift>) -> Result<(), E> {
                let cell_border_color = plotters::prelude::BLACK;
                let cell_inside_color = plotters::prelude::full_palette::PURPLE;
                // Get size of backend
                let size = root.dim_in_pixel();
                let offset = root.get_base_pixel();
                
                // Get position of cell
                let domain_size = [
                    $(domain.max[$ni] - domain.min[$ni],)+
                ];
                let relative_pos = [
                    $((self.pos[$ni] - domain.min[$ni])/domain_size[$ni],)+
                ];
                let draw_middle = (
                    $((relative_pos[$ni] * size.0 as f64).round() as i32 + offset.$ni,)+
                );
                
                let cell_border = Circle::new(
                    draw_middle,
                    (self.cell_radius as f64).round() as i32,
                    Into::<ShapeStyle>::into(&cell_border_color).filled(),
                );
                root.draw(&cell_border).unwrap();

                let cell_inside = Circle::new(
                    draw_middle,
                    ((self.cell_radius as f64) * 0.6).round() as i32,
                    Into::<ShapeStyle>::into(&cell_inside_color).filled(),
                );
                root.draw(&cell_inside).unwrap();
                Ok(())
            }
        }
    }
);


implement_draw_cell_2d!(crate::impls_cell_models::custom_cell_nd::CustomCell2D, crate::impls_domain::cartesian_cuboid_n::CartesianCuboid2, [usize; 2], crate::impls_domain::cartesian_cuboid_n::CartesianCuboidVoxel2, 0, 1);
implement_draw_cell_2d!(crate::impls_cell_models::standard_cell_2d::StandardCell2D, crate::impls_domain::cartesian_cuboid_n::CartesianCuboid2, [usize; 2], crate::impls_domain::cartesian_cuboid_n::CartesianCuboidVoxel2, 0, 1);


/*
pub fn plot_current_cells_2d(
    time: f64,
    cells: Vec<CellModel2D>,
    domain_size: [f64; 2],
    n_voxel: [usize; 2],
    cell_radius: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let image_size: u32 = 1000;
    let top_border = (image_size as f64 * 0.1).round() as u32;

    let data: Vec<(i32, i32)> = cells
        .iter()
        .map(|cell|
            ((image_size as f64 * (cell.mechanics.pos()[0]+domain_size[0]) / 2.0 / domain_size[0]).round() as i32,
             (top_border as f64 + image_size as f64 * (cell.mechanics.pos()[1]+domain_size[1]) / 2.0 / domain_size[1]).round() as i32)
        ).collect();

    let filename = format!("out/snapshot_{:08.2}.png", time);
    let root = BitMapBackend::new(&filename, (image_size, top_border + image_size)).into_drawing_area();
    root.fill(&WHITE)?;

    let (box_top, _box_bot) = root.split_vertically(top_border);

    // Draw the mesh lines
    for n in 0..n_voxel[0] {
        let x = (n as f64 / n_voxel[0] as f64 * image_size as f64).round() as i32;
        let pe_x = PathElement::new(vec!((x, top_border as i32),(x, (image_size + top_border) as i32)), Into::<ShapeStyle>::into(&BLACK).filled());
        root.draw(&pe_x)?;
    }

    for n in 0..n_voxel[1] {
        let y = (n as f64 / n_voxel[1] as f64 * image_size as f64).round() as i32;
        let pe_y = PathElement::new(vec!((0, y + top_border as i32),(image_size as i32, y + top_border as i32)), Into::<ShapeStyle>::into(&BLACK).filled());
        root.draw(&pe_y)?;
    }

    let fonttype = "sans-mono";
    let fontsize = (top_border as f64 * 0.4).round() as i32;

    box_top.draw_text(
        &format!("Cells: {:10.0}", cells.len()),
        &TextStyle::from((fonttype, fontsize)
            .into_font())
            .color(&BLACK),
        (0, (0.1 * top_border as f64).round() as i32)
    )?;
    box_top.draw_text(
        &format!("Time: {:12.1}", time),
        &TextStyle::from((fonttype, fontsize)
            .into_font())
            .color(&BLACK),
        (0, (top_border as f64 * 0.6).round() as i32)
    )?;

    // Draw a circle per cell
    for pos in data {
        // Color the border of the cell
        let border_color = BLACK;
        let scale = 
        if domain_size[0] < domain_size[1] {
            domain_size[0]
        } else {
            domain_size[1]
        };
        root.draw(&Circle::new(
            pos,
            (cell_radius * image_size as f64 / scale / 5.0).round() as i32,
            Into::<ShapeStyle>::into(&border_color).filled(),
        ))?;

        // Color the inner part of the cell
        let inner_color = RGBAColor(128, 128, 128, 0.9);
        root.draw(&Circle::new(
            pos,
            (cell_radius * 0.75 * image_size as f64 / scale / 5.0).round() as i32,
            Into::<ShapeStyle>::into(&inner_color).filled(),
        ))?;
    }

    root.present()?;
    Ok(())
}
*/
