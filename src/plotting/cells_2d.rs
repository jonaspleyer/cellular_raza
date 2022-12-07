use crate::impls_cell_properties::cell_model::*;
use crate::concepts::mechanics::*;

use plotters::prelude::*;


pub fn plot_current_cells_2d(
    time: f64,
    cells: Vec<CellModel>,
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
