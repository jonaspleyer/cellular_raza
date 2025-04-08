use super::concepts::StorageError;
use super::concepts::{StorageInterfaceLoad, StorageInterfaceOpen, StorageInterfaceStore};

use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, HashMap};

/// Use the [sled] database to save results to an embedded database.
// TODO use custom field for config [](https://docs.rs/sled/latest/sled/struct.Config.html) to let the user control these parameters
#[derive(Clone, Debug)]
pub struct MemoryStorageInterface<Id, Element>
where
    Id: Sized,
    Element: Sized,
{
    map: BTreeMap<u64, HashMap<Id, Element>>,
}

impl<Id, Element> StorageInterfaceOpen for MemoryStorageInterface<Id, Element> {
    fn open_or_create(
        _location: &std::path::Path,
        _storage_instance: u64,
    ) -> Result<Self, StorageError> {
        Ok(Self {
            map: BTreeMap::new(),
        })
    }

    fn clone_to_new_instance(&self, _storage_instance: u64) -> Self {
        Self {
            map: Arc::clone(&self.map),
        }
    }
}

impl<Id, Element> StorageInterfaceStore<Id, Element> for MemoryStorageInterface<Id, Element>
where
    Id: Clone + std::hash::Hash + std::cmp::Eq,
    Element: Clone,
{
    fn store_single_element(
        &mut self,
        iteration: u64,
        identifier: &Id,
        element: &Element,
    ) -> Result<(), StorageError>
    where
        Id: Serialize,
        Element: Serialize,
    {
        self.map
            .entry(iteration)
            .and_modify(|e| {
                e.entry(identifier.clone()).or_insert(element.clone());
            })
            .or_insert(HashMap::from([(identifier.clone(), element.clone())]));
        Ok(())
    }

    fn store_batch_elements<'a, I>(
        &'a mut self,
        iteration: u64,
        identifiers_elements: I,
    ) -> Result<(), StorageError>
    where
        Id: 'a + Serialize,
        Element: 'a + Serialize,
        I: Clone + IntoIterator<Item = (&'a Id, &'a Element)>,
    {
        let identifiers_elements = identifiers_elements
            .clone()
            .into_iter()
            .map(|(id, el)| (id.clone(), el.clone()));
        match self.map.get_mut(&iteration) {
            Some(e) => e.extend(identifiers_elements),
            None => {
                self.map.insert(iteration, identifiers_elements.collect());
            }
        }
        Ok(())
    }
}

impl<Id, Element> StorageInterfaceLoad<Id, Element> for MemoryStorageInterface<Id, Element>
where
    Id: core::hash::Hash + core::cmp::Eq + Clone,
    Element: Clone,
{
    fn load_single_element(
        &self,
        iteration: u64,
        identifier: &Id,
    ) -> Result<Option<Element>, StorageError>
    where
        Id: Serialize + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        Ok(self
            .map
            .get(&iteration)
            .and_then(|elements| elements.get(identifier).and_then(|x| Some(x.clone()))))
    }

    fn load_all_elements_at_iteration(
        &self,
        iteration: u64,
    ) -> Result<HashMap<Id, Element>, StorageError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        match self.map.get(&iteration) {
            Some(x) => Ok(x.clone()),
            None => Ok(HashMap::new()),
        }
    }

    fn load_all_elements(&self) -> Result<BTreeMap<u64, HashMap<Id, Element>>, StorageError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        Ok(self.map.clone())
    }

    fn get_all_iterations(&self) -> Result<Vec<u64>, StorageError> {
        Ok(self.map.keys().into_iter().map(|&k| k).collect())
    }
}
