use crate::concepts::cycle::*;
use crate::cell_properties::cell_model::CellModel;


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


impl Cycle<CellModel> for CycleModel {
    fn update_cycle(dt: &f64, cell: &mut CellModel) {
        cell.cell_cycle.current_age += dt;
        if cell.cell_cycle.current_age >= cell.cell_cycle.cycles[cell.cell_cycle.current_cycle].lifetime {
            cell.cell_cycle.current_cycle += 1;
            cell.cell_cycle.current_age = 0.0;
        }
        if cell.cell_cycle.current_cycle == cell.cell_cycle.cycles.len() {
            cell.flags.removal = true;
        }
    }
}
