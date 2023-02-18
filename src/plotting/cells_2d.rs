use crate::concepts::errors::DrawingError;
use crate::plotting::spatial::PlotSelf;


use plotters::{
    backend::DrawingBackend,
    coord::cartesian::Cartesian2d,
    coord::types::RangedCoordf64,
    prelude::{
        DrawingArea,
        Circle},
    style::ShapeStyle,
};


macro_rules! implement_draw_cell_2d (
    ($cell:ty, $($ni:tt),+) => {
        impl PlotSelf for $cell
        {
            fn plot_self<Db>
            (&self, root: &mut DrawingArea<Db, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), DrawingError>
            where
                Db: DrawingBackend,
            {
                let cell_border_color = plotters::prelude::BLACK;
                let cell_inside_color = plotters::prelude::full_palette::PURPLE;

                // Plot the cell border
                let dx = root.get_x_range().end - root.get_x_range().start;
                // let dy = root.get_y_range().end - root.get_y_range().start;
                let dx_pix = root.get_x_axis_pixel_range().end - root.get_x_axis_pixel_range().start;
                // let dy_pix = root.get_y_axis_pixel_range().end - root.get_y_axis_pixel_range().start;

                let s = self.cell_radius / dx * dx_pix as f64;
                // println!("{:?} {:?} {}", root.get_x_range(), root.get_x_axis_pixel_range(), s);
                let cell_border = Circle::new(
                    (self.pos.x, self.pos.y),
                    s,
                    Into::<ShapeStyle>::into(&cell_border_color).filled(),
                );
                root.draw(&cell_border).unwrap();

                // Plot the inside of the cell
                let cell_inside = Circle::new(
                    (self.pos.x, self.pos.y),
                    s*0.6,
                    Into::<ShapeStyle>::into(&cell_inside_color).filled(),
                );
                root.draw(&cell_inside).unwrap();
                Ok(())
            }
        }
    }
);


implement_draw_cell_2d!(crate::impls_cell_models::custom_cell_nd::CustomCell2D, 0, 1);
implement_draw_cell_2d!(crate::impls_cell_models::standard_cell_2d::StandardCell2D, 0, 1);


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
