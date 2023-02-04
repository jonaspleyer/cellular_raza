use plotters::backend::DrawingBackend;
use plotters::prelude::DrawingArea;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::types::RangedCoordf64;


pub trait CreatePlottingRoot<'a, Db>//, E>
where
    Db: DrawingBackend,
    // E: std::error::Error + std::marker::Sync + std::marker::Send,
{
    fn create_plotting_root(&self, image_size: u32, filename: &'a String) -> DrawingArea<Db, Cartesian2d<RangedCoordf64, RangedCoordf64>>;
}


pub trait PlotSelf<Db, E>
where
    Db: DrawingBackend<ErrorType=E>,
    E: std::error::Error + std::marker::Sync + std::marker::Send,
{
    fn plot_self(&self, root: &mut DrawingArea<Db, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), E>;
}


use crate::concepts::cell::CellAgentBox;
use crate::concepts::cell::CellAgent;
use crate::concepts::mechanics::{Position,Force,Velocity};


impl<Db, E, Pos, For, Vel, C> PlotSelf<Db, E> for CellAgentBox<Pos, For, Vel, C>
where
    Db: DrawingBackend<ErrorType=E>,
    E: std::error::Error + std::marker::Sync + std::marker::Send,
    Pos: Position,
    For: Force,
    Vel: Velocity,
    C: CellAgent<Pos, For, Vel> + PlotSelf<Db, E>,
{
    fn plot_self(&self, root: &mut DrawingArea<Db, Cartesian2d<RangedCoordf64, RangedCoordf64>>) -> Result<(), E> {
        self.cell.plot_self(root)
    }
}
