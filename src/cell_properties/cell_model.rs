use crate::cell_properties::cycle::*;
use crate::cell_properties::death::*;
use crate::cell_properties::interaction::*;
use crate::cell_properties::mechanics::*;
use crate::cell_properties::flags::*;


#[derive(Clone)]
pub struct CellModel {
    pub mechanics: MechanicsModel,
    pub cell_cycle: CycleModel,
    pub death_model: DeathModel,
    pub interaction: LennardJones,
    pub flags: Flags,
    pub id: u32,
}
