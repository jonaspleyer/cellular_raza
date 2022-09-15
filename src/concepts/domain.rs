use crate::cell_properties::cell_model::*;


pub trait Domain  {
    fn apply_boundary(&self, cell: &mut CellModel);
}
