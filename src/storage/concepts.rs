use std::collections::HashMap;

use crate::concepts::errors::SimulationError;

pub enum StorageIdent {
    Cell,
    Voxel,
    MultiVoxel,
}

impl StorageIdent {
    pub const fn value(self) -> u16 {
        match self {
            StorageIdent::Cell => 1,
            StorageIdent::Voxel => 2,
            StorageIdent::MultiVoxel => 3,
        }
    }
}

pub trait StorageInterface<Id, Element> {
    fn store_single_element(
        &self,
        iteration: u64,
        identifier: Id,
        element: Element,
    ) -> Result<(), SimulationError>;

    fn store_batch_elements(
        &self,
        iteration: u64,
        identifiers_elements: Vec<(Id, Element)>,
    ) -> Result<(), SimulationError> {
        for (id, element) in identifiers_elements.into_iter() {
            self.store_single_element(iteration, id, element)?;
        }
        Ok(())
    }

    // TODO decide if these functions should be &mut self instead of &self
    // This could be useful when implementing buffers, but maybe unnecessary.
    fn load_single_element(
        &self,
        iteration: u64,
        identifier: Id,
    ) -> Result<Element, SimulationError>;

    fn load_element_history(
        &self,
        identifier: Id,
    ) -> Result<HashMap<u64, Element>, SimulationError>;

    fn load_all_elements_at_iteration(
        &self,
        iteration: u64,
    ) -> Result<HashMap<Id, Element>, SimulationError>;

    fn load_all_elements(&self) -> Result<HashMap<u64, HashMap<Id, Element>>, SimulationError>;

    fn load_all_element_histories(
        &self,
    ) -> Result<HashMap<Id, HashMap<u64, Element>>, SimulationError>;
}
