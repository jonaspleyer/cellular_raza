use std::collections::HashMap;

use crate::concepts::errors::{SimulationError, StorageError};

use serde::{Deserialize, Serialize};

use super::quick_xml::XmlStorageInterface;
use super::serde_json::JsonStorageInterface;
use super::sled_database::SledStorageInterface;

// TODO implement this correctly
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum StorageOptions {
    Sled,
    SerdeJson,
    SerdeXml,
}

impl StorageOptions {
    pub fn default_priority() -> Vec<Self> {
        return vec![
            StorageOptions::SerdeJson,
            // TODO fix sled! This is currently not working on multiple threads
            // StorageOptions::Sled,
        ];
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CombinedSaveFormat<Id, Element> {
    pub(super) identifier: Id,
    pub(super) element: Element,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BatchSaveFormat<Id, Element> {
    pub(super) data: Vec<CombinedSaveFormat<Id, Element>>,
}

/// This manager handles if multiple storage options have been specified
/// It can load resources from one storage aspect and will
#[derive(Clone, Debug)]
pub struct StorageManager<Id, Element> {
    storage_priority: Vec<StorageOptions>,

    sled_storage: Option<SledStorageInterface<Id, Element>>,
    json_storage: Option<JsonStorageInterface<Id, Element>>,
    xml_storage: Option<XmlStorageInterface<Id, Element>>,
}

impl<Id, Element> StorageManager<Id, Element> {
    pub(crate) fn open_or_create_with_priority(
        location: &std::path::Path,
        storage_instance: u64,
        storage_priority: &Vec<StorageOptions>,
    ) -> Result<Self, SimulationError> {
        // Fill the used storage options
        let mut sled_storage = None;
        let mut json_storage = None;
        let mut xml_storage = None;
        for storage_variant in storage_priority.iter() {
            match storage_variant {
                StorageOptions::SerdeJson => {
                    json_storage = Some(JsonStorageInterface::<Id, Element>::open_or_create(
                        &location.to_path_buf().join("json"),
                        storage_instance,
                    )?);
                }
                StorageOptions::Sled => {
                    sled_storage = Some(SledStorageInterface::<Id, Element>::open_or_create(
                        &location.to_path_buf().join("sled"),
                        storage_instance,
                    )?);
                }
                StorageOptions::SerdeXml => {
                    xml_storage = Some(XmlStorageInterface::<Id, Element>::open_or_create(
                        &location.to_path_buf().join("xml"),
                        storage_instance,
                    )?);
                }
            }
        }
        let manager = StorageManager {
            storage_priority: storage_priority.clone(),

            sled_storage,
            json_storage,
            xml_storage,
        };

        Ok(manager)
    }
}

impl<Id, Element> StorageInterface<Id, Element> for StorageManager<Id, Element> {
    #[allow(unused)]
    fn open_or_create(
        location: &std::path::Path,
        storage_instance: u64,
    ) -> Result<Self, SimulationError> {
        let storage_priority = StorageOptions::default_priority();
        Self::open_or_create_with_priority(location, storage_instance, &storage_priority)
    }

    #[allow(unused)]
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
        if let Some(sled_storage) = &self.sled_storage {
            sled_storage.store_single_element(iteration, identifier, element)?;
        }

        if let Some(json_storage) = &self.json_storage {
            json_storage.store_single_element(iteration, identifier, element)?;
        }

        Ok(())
    }

    #[allow(unused)]
    fn store_batch_elements(
        &self,
        iteration: u64,
        identifiers_elements: &[(Id, Element)],
    ) -> Result<(), SimulationError>
    where
        Id: Serialize,
        Element: Serialize,
    {
        if let Some(sled_storage) = &self.sled_storage {
            sled_storage.store_batch_elements(iteration, identifiers_elements)?;
        }

        if let Some(json_storage) = &self.json_storage {
            json_storage.store_batch_elements(iteration, identifiers_elements)?;
        }

        if let Some(xml_storage) = &self.xml_storage {
            xml_storage.store_batch_elements(iteration, identifiers_elements)?;
        }
        Ok(())
    }

    #[allow(unused)]
    fn load_single_element(
        &self,
        iteration: u64,
        identifier: &Id,
    ) -> Result<Option<Element>, SimulationError>
    where
        Id: Serialize,
        Element: for<'a> Deserialize<'a>,
    {
        for priority in self.storage_priority.iter() {
            let element = match priority {
                StorageOptions::Sled => {
                    if let Some(sled_storage) = &self.sled_storage {
                        sled_storage.load_single_element(iteration, identifier)
                    } else {
                        Err(StorageError {
                            message: "Sled storage was not initialized but called".into(),
                        })?
                    }
                }
                StorageOptions::SerdeJson => {
                    if let Some(json_storage) = &self.json_storage {
                        json_storage.load_single_element(iteration, identifier)
                    } else {
                        Err(StorageError {
                            message: "SerdeJson storage was not initialized but called".into(),
                        })?
                    }
                }
                StorageOptions::SerdeXml => {
                    if let Some(xml_storage) = &self.xml_storage {
                        xml_storage.load_single_element(iteration, identifier)
                    } else {
                        Err(StorageError {
                            message: "SerdeXML storage was not initialized but called".into(),
                        })?
                    }
                }
            };
            return element;
        }
        Ok(None)
    }

    #[allow(unused)]
    fn load_all_elements_at_iteration(
        &self,
        iteration: u64,
    ) -> Result<HashMap<Id, Element>, SimulationError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        for priority in self.storage_priority.iter() {
            let elements = match priority {
                StorageOptions::Sled => {
                    if let Some(sled_storage) = &self.sled_storage {
                        sled_storage.load_all_elements_at_iteration(iteration)
                    } else {
                        Err(StorageError {
                            message: "Sled storage was not initialized but called".into(),
                        })?
                    }
                }
                StorageOptions::SerdeJson => {
                    if let Some(json_storage) = &self.json_storage {
                        json_storage.load_all_elements_at_iteration(iteration)
                    } else {
                        Err(StorageError {
                            message: "SerdeJson storage was not initialized but called".into(),
                        })?
                    }
                }
                StorageOptions::SerdeXml => {
                    if let Some(xml_storage) = &self.xml_storage {
                        xml_storage.load_all_elements_at_iteration(iteration)
                    } else {
                        Err(StorageError {
                            message: "SerdeXML storage was not initialized but called".into(),
                        })?
                    }
                }
            };
            return elements;
        }
        Ok(HashMap::new())
    }

    fn get_all_iterations(&self) -> Result<Vec<u64>, SimulationError> {
        for priority in self.storage_priority.iter() {
            let iterations = match priority {
                StorageOptions::Sled => {
                    if let Some(sled_storage) = &self.sled_storage {
                        sled_storage.get_all_iterations()
                    } else {
                        Err(StorageError {
                            message: "Sled storage was not initialized but called".into(),
                        })?
                    }
                }
                StorageOptions::SerdeJson => {
                    if let Some(json_storage) = &self.json_storage {
                        json_storage.get_all_iterations()
                    } else {
                        Err(StorageError {
                            message: "SerdeJson storage was not initialized but called".into(),
                        })?
                    }
                }
                StorageOptions::SerdeXml => {
                    if let Some(xml_storage) = &self.xml_storage {
                        xml_storage.get_all_iterations()
                    } else {
                        Err(StorageError {
                            message: "SerdeXML storage was not initialized but called".into(),
                        })?
                    }
                }
            };
            return iterations;
        }
        Ok(Vec::new())
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
