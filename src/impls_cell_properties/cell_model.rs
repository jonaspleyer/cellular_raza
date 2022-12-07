use crate::impls_cell_properties::cycle::*;
use crate::impls_cell_properties::death::*;
use crate::impls_cell_properties::interaction::*;
use crate::impls_cell_properties::mechanics::*;
use crate::impls_cell_properties::flags::*;


// Implement Sender/Sync traits
unsafe impl Send for CellModel {}
unsafe impl Sync for CellModel {}


#[derive(Clone)]
pub struct CellModel {
    pub mechanics: MechanicsModel,
    pub cell_cycle: CycleModel,
    pub death_model: DeathModel,
    pub interaction: LennardJones,
    pub flags: Flags,
    pub id: u32,
}
