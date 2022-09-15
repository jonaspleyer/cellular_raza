use crate::cell_properties::cycle::*;
use crate::cell_properties::death::*;
use crate::cell_properties::interaction::*;
use crate::cell_properties::spatial::*;


use rand::distributions::Standard;

#[derive(Clone)]
pub struct CellModel<'a, P> {
    pub spatial: SpatialModel,
    pub cell_cycle: CycleModel,
    pub death_model: DeathModel,
    pub interaction: InteractionModel<'a, P>,
    pub rng_model: Standard,
}
