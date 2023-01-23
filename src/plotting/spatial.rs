use plotters::backend::DrawingBackend;
use plotters::prelude::DrawingArea;
use plotters::coord::Shift;


use crate::concepts::domain::Domain;


pub trait CreatePlottingRoot<Db, E>
where
    Db: DrawingBackend<ErrorType=E>,
    E: std::error::Error + std::marker::Sync + std::marker::Send,
{
    fn create_plotting_root(&self) -> Result<DrawingArea<Db, Shift>, E>;
}


pub trait PlotSelf<Db, E, Dom, C, I, V>
where
    Db: DrawingBackend<ErrorType=E>,
    E: std::error::Error + std::marker::Sync + std::marker::Send,
    Dom: Domain<C, I, V>,
{
    fn plot_self(&self, domain: &Dom, root: &mut DrawingArea<Db, Shift>) -> Result<(), E>;
}


use crate::concepts::cell::CellAgentBox;
use crate::concepts::cell::CellAgent;
use crate::concepts::mechanics::{Position,Force,Velocity};
use crate::concepts::domain::{Index,Voxel};
use crate::prelude::DomainBox;


impl<Db, E, Dom, Pos, For, Vel, C, I, V> PlotSelf<Db, E, DomainBox<C, I, V, Dom>, CellAgentBox<Pos, For, Vel, C>, I, V> for CellAgentBox<Pos, For, Vel, C>
where
    Db: DrawingBackend<ErrorType=E>,
    E: std::error::Error + std::marker::Sync + std::marker::Send,
    Dom: Domain<C, I, V>,
    I: Index,
    V: Voxel<I, Pos, For>,
    Pos: Position,
    For: Force,
    Vel: Velocity,
    C: CellAgent<Pos, For, Vel> + PlotSelf<Db, E, Dom, C, I, V>,
{
    fn plot_self(&self, domain: &DomainBox<C, I, V, Dom>, root: &mut DrawingArea<Db, Shift>) -> Result<(), E> {
        self.cell.plot_self(&domain.domain_raw, root)
    }
}
