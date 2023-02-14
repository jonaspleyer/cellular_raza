use crate::concepts::cycle::*;
// use crate::impls_cell_properties::cell_model::CellModel;

use serde::{Serialize, Deserialize};


#[derive(Clone,Debug,Serialize, Deserialize)]
pub struct NoCycle {}

impl<Cel> Cycle<Cel> for NoCycle {
    fn update_cycle(_: &f64, _: &mut Cel) {}
}
