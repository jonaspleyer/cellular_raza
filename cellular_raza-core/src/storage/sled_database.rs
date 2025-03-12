use super::concepts::StorageError;
use super::concepts::{StorageInterfaceLoad, StorageInterfaceOpen, StorageInterfaceStore};

use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

/// Use the [sled] database to save results to an embedded database.
// TODO use custom field for config [](https://docs.rs/sled/latest/sled/struct.Config.html) to let the user control these parameters
#[derive(Clone, Debug)]
pub struct SledStorageInterface<Id, Element, const TEMP: bool = false> {
    db: sled::Db,
    // TODO use this buffer
    // buffer: StorageBuffer<Id, Element>,
    id_phantom: PhantomData<Id>,
    element_phantom: PhantomData<Element>,
    bincode_config: bincode::config::Configuration,
}

impl<Id, Element, const TEMP: bool> SledStorageInterface<Id, Element, TEMP> {
    /// Transform a u64 value to an iteration key which can be given to a sled tree.
    fn iteration_to_key(iteration: u64) -> [u8; 8] {
        iteration.to_le_bytes()
    }

    /// Transform the key given by the tree to the corresponding iteartion u64 value
    fn key_to_iteration(
        key: &sled::IVec,
        bincode_config: bincode::config::Configuration,
    ) -> Result<u64, StorageError> {
        let iteration = bincode::serde::decode_from_slice(key, bincode_config)?.0;
        Ok(iteration)
    }

    /// Get the correct tree of the iteration or create if not currently present.
    fn open_or_create_tree(&self, iteration: u64) -> Result<sled::Tree, StorageError> {
        let tree_key = Self::iteration_to_key(iteration);
        let tree = self.db.open_tree(&tree_key)?;
        Ok(tree)
    }

    fn open_tree(&self, iteration: u64) -> Result<Option<sled::Tree>, StorageError> {
        let tree_key = Self::iteration_to_key(iteration);
        if !self.db.tree_names().contains(&sled::IVec::from(&tree_key)) {
            Ok(None)
        } else {
            let tree = self.db.open_tree(tree_key)?;
            Ok(Some(tree))
        }
    }
}

impl<Id, Element, const TEMP: bool> StorageInterfaceOpen
    for SledStorageInterface<Id, Element, TEMP>
{
    fn open_or_create(
        location: &std::path::Path,
        _storage_instance: u64,
    ) -> Result<Self, StorageError> {
        let config = sled::Config::default()
            .mode(sled::Mode::HighThroughput)
            .cache_capacity(1024 * 1024 * 1024 * 5) // 5gb
            .path(&location)
            .temporary(TEMP)
            .use_compression(false);

        let db = config.open()?;
        let bincode_config = bincode::config::standard();

        Ok(SledStorageInterface {
            db,
            id_phantom: PhantomData,
            element_phantom: PhantomData,
            bincode_config,
        })
    }
}

impl<Id, Element, const TEMP: bool> StorageInterfaceStore<Id, Element>
    for SledStorageInterface<Id, Element, TEMP>
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
        let tree = self.open_or_create_tree(iteration)?;

        // Serialize the identifier and the element
        let identifier_serialized =
            bincode::serde::encode_to_vec(&identifier, self.bincode_config)?;
        let element_serialized = bincode::serde::encode_to_vec(&element, self.bincode_config)?;
        match tree.insert(identifier_serialized, element_serialized)? {
            None => Ok(()),
            Some(_) => Err(StorageError::InitError(format!(
                "Element already present at iteration {}",
                iteration
            ))),
        }?;
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
        let tree = self.open_or_create_tree(iteration)?;
        let mut batch = sled::Batch::default();
        for (identifier, element) in identifiers_elements.into_iter() {
            let identifier_serialized =
                bincode::serde::encode_to_vec(&identifier, self.bincode_config)?;
            let element_serialized = bincode::serde::encode_to_vec(&element, self.bincode_config)?;
            batch.insert(identifier_serialized, element_serialized)
        }
        tree.apply_batch(batch)?;
        Ok(())
    }
}

impl<Id, Element, const TEMP: bool> StorageInterfaceLoad<Id, Element>
    for SledStorageInterface<Id, Element, TEMP>
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
        let tree = match self.open_tree(iteration)? {
            Some(tree) => tree,
            None => return Ok(None),
        };
        let identifier_serialized = bincode::serde::encode_to_vec(identifier, self.bincode_config)?;
        match tree.get(&identifier_serialized)? {
            Some(element_serialized) => {
                let element: Element =
                    bincode::serde::decode_from_slice(&element_serialized, self.bincode_config)?.0;
                Ok(Some(element))
            }
            None => Ok(None),
        }
    }

    fn load_element_history(&self, identifier: &Id) -> Result<HashMap<u64, Element>, StorageError>
    where
        Id: Serialize,
        Element: for<'a> Deserialize<'a>,
    {
        // Keep track if the element has not been found in a database.
        // If so we can either get the current minimal iteration or maximal depending on where it was found else.
        let mut minimal_iteration = None;
        let mut maximal_iteration = None;
        let mut success_iteration = None;

        // Save results in this hashmap
        let mut accumulator = HashMap::new();
        // Serialize the identifier
        let identifier_serialized = bincode::serde::encode_to_vec(identifier, self.bincode_config)?;
        for iteration_serialized in self.db.tree_names() {
            // If we are above the maximal or below the minimal iteration, we skip checking
            let iteration: u64 =
                bincode::serde::decode_from_slice(&iteration_serialized, self.bincode_config)?.0;
            match minimal_iteration {
                None => (),
                Some(min_iter) => {
                    if iteration < min_iter {
                        continue;
                    }
                }
            }
            match maximal_iteration {
                None => (),
                Some(max_iter) => {
                    if max_iter < iteration {
                        continue;
                    }
                }
            }
            // Get the tree for a random iteration
            let tree = self.db.open_tree(iteration_serialized)?;
            match tree.get(&identifier_serialized)? {
                // We found and element insert it
                Some(element_serialized) => {
                    let element: Element = bincode::serde::decode_from_slice(
                        &element_serialized,
                        self.bincode_config,
                    )?
                    .0;
                    accumulator.insert(iteration, element);
                    success_iteration = Some(iteration);
                }
                // We did not find an element. Thus update the helper variables atop.
                None => match (minimal_iteration, maximal_iteration, success_iteration) {
                    (None, None, Some(suc_iter)) => {
                        if iteration > suc_iter {
                            maximal_iteration = Some(iteration);
                        }
                        if iteration < suc_iter {
                            minimal_iteration = Some(iteration);
                        }
                    }
                    (Some(min_iter), None, Some(suc_iter)) => {
                        if iteration > suc_iter {
                            maximal_iteration = Some(iteration);
                        }
                        if iteration < suc_iter && iteration > min_iter {
                            minimal_iteration = Some(iteration);
                        }
                    }
                    (None, Some(max_iter), Some(suc_iter)) => {
                        if iteration > suc_iter && iteration < max_iter {
                            maximal_iteration = Some(iteration);
                        }
                        if iteration < suc_iter {
                            minimal_iteration = Some(iteration);
                        }
                    }
                    (Some(min_iter), Some(max_iter), Some(suc_iter)) => {
                        if iteration > suc_iter && iteration < max_iter {
                            maximal_iteration = Some(iteration);
                        }
                        if iteration < suc_iter && iteration > min_iter {
                            minimal_iteration = Some(iteration);
                        }
                    }
                    (_, _, None) => (),
                },
            };
        }
        Ok(accumulator)
    }

    fn load_all_elements_at_iteration(
        &self,
        iteration: u64,
    ) -> Result<HashMap<Id, Element>, StorageError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        let tree = match self.open_tree(iteration)? {
            Some(tree) => tree,
            None => return Ok(HashMap::new()),
        };
        tree.iter()
            .map(|entry_result| {
                let (identifier_serialized, element_serialized) = entry_result?;
                let identifier: Id =
                    bincode::serde::decode_from_slice(&identifier_serialized, self.bincode_config)?
                        .0;
                let element: Element =
                    bincode::serde::decode_from_slice(&element_serialized, self.bincode_config)?.0;
                Ok((identifier, element))
            })
            .collect::<Result<HashMap<Id, Element>, StorageError>>()
    }

    fn load_all_elements(&self) -> Result<BTreeMap<u64, HashMap<Id, Element>>, StorageError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        self.db
            .tree_names()
            .iter()
            .map(|tree_name_serialized| {
                let tree = self.db.open_tree(tree_name_serialized)?;
                let iteration = Self::key_to_iteration(tree_name_serialized, self.bincode_config)?;
                let identifier_to_element = tree
                    .iter()
                    .map(|entry_result| {
                        let (identifier_serialized, element_serialized) = entry_result?;
                        let identifier: Id = bincode::serde::decode_from_slice(
                            &identifier_serialized,
                            self.bincode_config,
                        )?
                        .0;
                        let element: Element = bincode::serde::decode_from_slice(
                            &element_serialized,
                            self.bincode_config,
                        )?
                        .0;
                        Ok((identifier, element))
                    })
                    .collect::<Result<HashMap<Id, Element>, StorageError>>()?;
                Ok((iteration, identifier_to_element))
            })
            .collect::<Result<BTreeMap<u64, HashMap<Id, Element>>, StorageError>>()
    }

    fn get_all_iterations(&self) -> Result<Vec<u64>, StorageError> {
        let iterations = self
            .db
            .tree_names()
            .iter()
            // TODO this should not be here! Fix it properly (I asked on sled discord)
            .filter(|key| {
                **key
                    != sled::IVec::from(&[
                        95, 95, 115, 108, 101, 100, 95, 95, 100, 101, 102, 97, 117, 108, 116,
                    ])
            })
            .map(|tree_name_serialized| Self::key_to_iteration(tree_name_serialized, self.bincode_config))
            .collect::<Result<Vec<_>, StorageError>>()?;

        Ok(iterations)
    }
}
