use crate::concepts::errors::DrawingError;

use plotters::backend::DrawingBackend;
use plotters::prelude::BitMapBackend;
use plotters::prelude::DrawingArea;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::types::RangedCoordf64;


pub trait CreatePlottingRoot//, E>
    // E: std::error::Error + std::marker::Sync + std::marker::Send,
{
    fn create_bitmap_root<'a>(&self, image_size: u32, filename: &'a String) -> Result<DrawingArea<BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>, DrawingError>;
    // TODO implement this as well
    // fn create_svg_root<'a>(&self, image_size: u32, filename: &'a String) -> DrawingArea<SVGBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>;
}


pub trait PlotSelf {
    fn plot_self<Db>
    (&self, root: &mut DrawingArea<Db, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), DrawingError>
    where
        Db: DrawingBackend;
}


use crate::concepts::cell::CellAgentBox;
use serde::{Serialize,Deserialize};


impl<C> PlotSelf for CellAgentBox<C>
where
    C: PlotSelf + Serialize + for<'a> Deserialize<'a>,
{
    fn plot_self<Db>
    (&self, root: &mut DrawingArea<Db, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), DrawingError>
    where
        Db: DrawingBackend,
    {
        self.cell.plot_self(root)
    }
}
