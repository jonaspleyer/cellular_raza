use cellular_raza_concepts::errors::*;
use cellular_raza_concepts::plotting::*;

use plotters::{
    backend::DrawingBackend,
    coord::cartesian::Cartesian2d,
    coord::types::RangedCoordf64,
    prelude::{Circle, DrawingArea},
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

implement_draw_cell_2d!(crate::cell_models::custom_cell_nd::CustomCell2D, 0, 1);
implement_draw_cell_2d!(crate::cell_models::standard_cell_2d::StandardCell2D, 0, 1);
