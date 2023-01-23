use crate::concepts::cycle::*;
// use crate::impls_cell_properties::cell_model::CellModel;


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
