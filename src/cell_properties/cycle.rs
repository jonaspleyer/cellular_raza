use serde::{Serialize,Deserialize};


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct CellCycle {
    pub lifetime: f64,
}


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct CycleModel {
    pub cycle1: CellCycle,
    pub cycle2: CellCycle,
    pub cycle3: CellCycle,
    pub cycle4: CellCycle,
}
