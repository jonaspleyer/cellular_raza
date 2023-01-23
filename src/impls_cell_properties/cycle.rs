use crate::concepts::cycle::*;
// use crate::impls_cell_properties::cell_model::CellModel;

use serde::{Serialize, Deserialize};


#[derive(Clone,Debug,Serialize, Deserialize)]
pub struct NoCycle {}

impl<Cel> Cycle<Cel> for NoCycle {
    fn update_cycle(_: &f64, _: &mut Cel) {}
}


#[derive(Clone)]
pub struct CellCycle {
    pub lifetime: f64,
}


#[derive(Clone)]
pub struct CycleModel {
    pub cycles: Vec<CellCycle>,
    pub current_age: f64,
    pub current_cycle: usize,
}


impl From<&Vec<CellCycle>> for CycleModel {
    fn from(v: &Vec<CellCycle>) -> Self {
        CycleModel {
            cycles: v.clone(),
            current_age: 0.0,
            current_cycle: 0,
        }
    }
}
