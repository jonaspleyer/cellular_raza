use crate::cell_properties::cell_model::CellModel;


pub trait Cycle {
    fn update(dt: &f64, cell: &mut CellModel);
}
