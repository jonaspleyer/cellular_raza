use std::{collections::HashMap, marker::PhantomData};

use crate::concepts::errors::SimulationError;

use serde::{Deserialize, Serialize};

#[cfg(feature = "serde_json")]
use super::serde_json::JsonStorageInterface;
#[cfg(feature = "sled")]
use super::sled_database::SledStorageInterface;

// TODO implement this correctly
#[derive(Clone, Debug)]
pub enum StorageOptions {
    NoStorage,
    #[cfg(feature = "sled")]
    Sled,
    #[cfg(feature = "serde_json")]
    SerdeJson,
}

impl Default for StorageOptions {
    fn default() -> Self {
        #[cfg(feature = "sled")]
        return StorageOptions::Sled;
        #[cfg(feature = "serde_json")]
        return StorageOptions::SerdeJson;
        #[cfg(not(any(feature = "sled", feature = "serde_json")))]
        return StorageOptions::NoStorage;
    }
}

/// This manager handles if multiple storage options have been specified
/// It can load resources from one storage aspect and will
#[derive(Clone, Debug)]
pub struct StorageManager<Id, Element> {
    storage_priority: StorageOptions,

    #[cfg(feature = "sled")]
    sled_storage: SledStorageInterface<Id, Element>,
    #[cfg(feature = "serde_json")]
    json_storage: JsonStorageInterface<Id, Element>,

    phantom_id: PhantomData<Id>,
    phantom_element: PhantomData<Element>,
}

impl<Id, Element> StorageInterface<Id, Element> for StorageManager<Id, Element> {
    fn open_or_create(
        location: &std::path::Path,
        storage_instance: u64,
    ) -> Result<Self, SimulationError> {
        #[cfg(feature = "sled")]
        let sled_storage = SledStorageInterface::<Id, Element>::open_or_create(
            location.to_path_buf().join("sled"),
            storage_instance,
        )?;
        #[cfg(feature = "serde_json")]
        let json_storage = JsonStorageInterface::<Id, Element>::open_or_create(
            &location.to_path_buf().join("json"),
            storage_instance,
        )?;

        #[cfg(any(feature = "sled", feature = "serde_json"))]
        let storage_priority = StorageOptions::default();
        #[cfg(not(any(feature = "sled", feature = "serde_json")))]
        let storage_priority = StorageOptions::NoStorage;

        let manager = StorageManager {
            storage_priority: storage_priority,

            #[cfg(feature = "sled")]
            sled_storage,
            #[cfg(feature = "serde_json")]
            json_storage,

            phantom_id: PhantomData,
            phantom_element: PhantomData,
        };

        Ok(manager)
    }

    fn store_single_element(
        &self,
        iteration: u64,
        identifier: &Id,
        element: &Element,
    ) -> Result<(), SimulationError>
    where
        Id: Serialize,
        Element: Serialize,
    {
        #[cfg(feature = "sled")]
        self.sled_storage
            .store_single_element(iteration, identifier, element)?;

        #[cfg(feature = "serde_json")]
        self.json_storage
            .store_single_element(iteration, identifier, element)?;

        Ok(())
    }

    fn store_batch_elements(
        &self,
        iteration: u64,
        identifiers_elements: &[(Id, Element)],
    ) -> Result<(), SimulationError>
    where
        Id: Serialize,
        Element: Serialize,
    {
        #[cfg(feature = "sled")]
        self.sled_storage
            .store_batch_elements(iteration, identifiers_elements)?;

        #[cfg(feature = "serde_json")]
        self.json_storage
            .store_batch_elements(iteration, identifiers_elements)?;
        Ok(())
    }

    fn load_single_element(
        &self,
        iteration: u64,
        identifier: &Id,
    ) -> Result<Option<Element>, SimulationError>
    where
        Id: Serialize,
        Element: for<'a> Deserialize<'a>,
    {
        match self.storage_priority {
            #[cfg(feature = "sled")]
            StorageOptions::Sled => self.sled_storage.load_single_element(iteration, identifier),
            #[cfg(feature = "serde_json")]
            StorageOptions::SerdeJson => {
                self.json_storage.load_single_element(iteration, identifier)
            }
            StorageOptions::NoStorage => Ok(None),
        }
    }

    fn load_all_elements_at_iteration(
        &self,
        iteration: u64,
    ) -> Result<HashMap<Id, Element>, SimulationError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        match self.storage_priority {
            #[cfg(feature = "sled")]
            StorageOptions::Sled => self.sled_storage.load_all_elements_at_iteration(iteration),
            #[cfg(feature = "serde_json")]
            StorageOptions::SerdeJson => {
                self.json_storage.load_all_elements_at_iteration(iteration)
            }
            StorageOptions::NoStorage => Ok(HashMap::new()),
        }
    }

    fn get_all_iterations(&self) -> Result<Vec<u64>, SimulationError> {
        match self.storage_priority {
            #[cfg(feature = "sled")]
            StorageOptions::Sled => self.sled_storage.get_all_iterations(),
            #[cfg(feature = "serde_json")]
            StorageOptions::SerdeJson => self.json_storage.get_all_iterations(),
            _ => Ok(Vec::new()),
        }
    }
}

pub trait StorageInterface<Id, Element> {
    fn open_or_create(
        location: &std::path::Path,
        storage_instance: u64,
    ) -> Result<Self, SimulationError>
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
        Element: for<'a> Deserialize<'a>;

    fn load_element_history(
        &self,
        identifier: &Id,
    ) -> Result<Option<HashMap<u64, Element>>, SimulationError>
    where
        Id: Serialize,
        Element: for<'a> Deserialize<'a>,
    {
        let results = self
            .get_all_iterations()?
            .iter()
            .filter_map(
                |&iteration| match self.load_single_element(iteration, identifier) {
                    Ok(Some(element)) => Some(Ok((iteration, element))),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                },
            )
            .collect::<Result<HashMap<u64, _>, SimulationError>>()?;
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
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>;

    fn get_all_iterations(&self) -> Result<Vec<u64>, SimulationError>;

    fn load_all_elements(&self) -> Result<HashMap<u64, HashMap<Id, Element>>, SimulationError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        let iterations = self.get_all_iterations()?;
        let all_elements = iterations
            .iter()
            .map(|iteration| {
                let elements = self.load_all_elements_at_iteration(*iteration)?;
                return Ok((*iteration, elements));
            })
            .collect::<Result<HashMap<_, _>, SimulationError>>()?;
        Ok(all_elements)
    }

    fn load_all_element_histories(
        &self,
    ) -> Result<HashMap<Id, HashMap<u64, Element>>, SimulationError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
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
