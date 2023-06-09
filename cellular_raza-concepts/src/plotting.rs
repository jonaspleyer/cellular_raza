use plotters::backend::DrawingBackend;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::BitMapBackend;
use plotters::prelude::DrawingArea;
use plotters::prelude::SVGBackend;

use crate::errors::DrawingError;

/// Creates a new plotting root which can then be drawn upon.
pub trait CreatePlottingRoot //, E>
// E: std::error::Error + std::marker::Sync + std::marker::Send,
{
    /// Creates a bitmap plotting root.
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

/// Allows elements of the simulation such as cells and voxels to draw themselves onto a plotting root.
/// Typically, voxels will draw first and cells afterwards.
pub trait PlotSelf {
    /// Define which elements to draw when plotting the element itself.
    fn plot_self<Db>(
        &self,
        root: &mut DrawingArea<Db, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    ) -> Result<(), DrawingError>
    where
        Db: DrawingBackend;

    /// Overload for backend to have a purely bitmap function.
    /// User are not expected to change this function.
    fn plot_self_bitmap(
        &self,
        root: &mut DrawingArea<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    ) -> Result<(), DrawingError> {
        self.plot_self(root)
    }

    /// Overload for backend to have a purely bitmap function.
    /// User are not expected to change this function.
    fn plot_self_svg(
        &self,
        root: &mut DrawingArea<SVGBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    ) -> Result<(), DrawingError> {
        self.plot_self(root)
    }
}

use crate::cell::CellAgentBox;
use serde::{Deserialize, Serialize};

impl<Cel> PlotSelf for CellAgentBox<Cel>
where
    Cel: PlotSelf + Serialize + for<'a> Deserialize<'a>,
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
