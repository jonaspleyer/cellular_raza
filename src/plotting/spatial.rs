use crate::concepts::errors::DrawingError;

use plotters::backend::DrawingBackend;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::BitMapBackend;
use plotters::prelude::DrawingArea;
use plotters::prelude::SVGBackend;

pub trait CreatePlottingRoot //, E>
// E: std::error::Error + std::marker::Sync + std::marker::Send,
{
    fn create_bitmap_root<'a, T>(
        &self,
        image_size: u32,
        filename: &'a T,
    ) -> Result<
        DrawingArea<BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
        DrawingError,
    >
    where
        T: AsRef<std::path::Path> + ?Sized;
    // TODO implement this as well
    /* fn create_svg_root<'a>(
        &self,
        image_size: u32,
        filename: &'a String,
    ) -> Result<
        DrawingArea<SVGBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
        DrawingError,
    >;*/
}

pub trait PlotSelf {
    fn plot_self<Db>(
        &self,
        root: &mut DrawingArea<Db, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    ) -> Result<(), DrawingError>
    where
        Db: DrawingBackend;

    fn plot_self_bitmap(
        &self,
        root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>
    ) -> Result<(), DrawingError> {
        self.plot_self(root)
    }

    fn plot_self_svg(
        &self,
        root: &mut DrawingArea<SVGBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>
    ) -> Result<(), DrawingError> {
        self.plot_self(root)
    }
}

use crate::concepts::cell::CellAgentBox;
use serde::{Deserialize, Serialize};

impl<C> PlotSelf for CellAgentBox<C>
where
    C: PlotSelf + Serialize + for<'a> Deserialize<'a>,
{
    fn plot_self<Db>(
        &self,
        root: &mut DrawingArea<Db, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    ) -> Result<(), DrawingError>
    where
        Db: DrawingBackend,
    {
        self.cell.plot_self(root)
    }
}
