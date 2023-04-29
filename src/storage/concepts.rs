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


pub trait StorageInterface<Id, Element>
{
    fn open_or_create(location: &std::path::Path, storage_instance: u64) -> Result<Self, SimulationError>
    where
        Self: Sized;

    fn store_single_element(
        &self,
        iteration: u64,
        identifier: &Id,
        element: &Element,
    ) -> Result<(), SimulationError>
    where
        Id: Serialize,
        Element: Serialize;

    fn store_batch_elements(
        &self,
        iteration: u64,
        identifiers_elements: &[(Id, Element)],
    ) -> Result<(), SimulationError>
    where
        Id: Serialize,
        Element: Serialize,
    {
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
        identifier: &Id,
    ) -> Result<Option<Element>, SimulationError>
    where
        Id: Serialize,
        Element: for<'a>Deserialize<'a>;

    fn load_element_history(
        &self,
        identifier: &Id,
    ) -> Result<Option<HashMap<u64, Element>>, SimulationError>
    where
        Id: Serialize,
        Element: for<'a>Deserialize<'a>,
    {
        let results = self.get_all_iterations()?
            .iter()
            .filter_map(|&iteration| match self.load_single_element(iteration, identifier) {
                Ok(Some(element)) => Some(Ok((iteration, element))),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            })
            .collect::<Result<HashMap<u64,_>, SimulationError>>()?;
        if results.len() == 0 {
            return Ok(None);
        } else {
            return Ok(Some(results));
        }
    }

    fn load_all_elements_at_iteration(
        &self,
        iteration: u64,
    ) -> Result<HashMap<Id, Element>, SimulationError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a>Deserialize<'a>,
        Element: for<'a>Deserialize<'a>;

    fn get_all_iterations(&self) -> Result<Vec<u64>, SimulationError>;

    fn load_all_elements(&self) -> Result<HashMap<u64, HashMap<Id, Element>>, SimulationError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a>Deserialize<'a>,
        Element: for<'a>Deserialize<'a>,
    {
        let iterations = self.get_all_iterations()?;
        let all_elements = iterations
            .iter()
            .map(|iteration| {
                let elements = self.load_all_elements_at_iteration(*iteration)?;
                return Ok((*iteration, elements));
            })
            .collect::<Result<HashMap<_,_>, SimulationError>>()?;
        Ok(all_elements)
    }

    fn load_all_element_histories(
        &self,
    ) -> Result<HashMap<Id, HashMap<u64, Element>>, SimulationError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a>Deserialize<'a>,
        Element: for<'a>Deserialize<'a>,
    {
        let all_elements = self.load_all_elements()?;
        let reordered_elements: HashMap<Id, HashMap<u64, Element>> = all_elements
            .into_iter()
            .map(|(iteration, identifier_to_elements)| {
                identifier_to_elements
                    .into_iter()
                    .map(move |(identifier, element)| (identifier, iteration, element))
            })
            .flatten()
            .fold(
                HashMap::new(),
                |mut acc, (identifier, iteration, element)| {
                    let existing_elements = acc.entry(identifier).or_default();
                    existing_elements.insert(iteration, element);
                    acc
                },
            );
        Ok(reordered_elements)
    }
}
