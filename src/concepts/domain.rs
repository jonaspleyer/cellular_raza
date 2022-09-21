use crate::cell_properties::cell_model::*;
use crate::concepts::errors::*;


pub trait Domain  {
    fn apply_boundary(&self, cell: &mut CellModel) -> Result<(),BoundaryError>;
}
