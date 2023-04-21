use crate::concepts::errors::{DataBaseError, SimulationError};

use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

// TODO use this struct or similar in the future for buffering
// #[derive(Clone,Debug)]
// struct StorageBuffer<Id, Element> {
//     buffer_size: usize,
//     // Mapping from iteration to another mapping containing identifiers and elements
//     elements: HashMap<u64, HashMap<Id, Element>>,
// }

#[derive(Clone, Debug)]
pub struct SledStorageInterface<Id, Element> {
    db: sled::Db,
    // TODO use this buffer
    // buffer: StorageBuffer<Id, Element>,
    id_phantom: PhantomData<Id>,
    element_phantom: PhantomData<Element>,
}

impl<Id, Element> SledStorageInterface<Id, Element> {
    /// Transform a u64 value to an iteration key which can be given to a sled tree.
    fn iteration_to_key(iteration: u64) -> [u8; 8] {
        iteration.to_le_bytes()
    }

    /// Transform the key given by the tree to the corresponding iteartion u64 value
    fn key_to_iteration(key: &sled::IVec) -> Result<u64, SimulationError> {
        let iteration = bincode::deserialize::<u64>(key)?;
        Ok(iteration)
    }

    /// Get the correct tree of the iteration or create if not currently present.
    fn open_or_create_tree(&self, iteration: u64) -> Result<sled::Tree, SimulationError> {
        let tree_key = Self::iteration_to_key(iteration);
        let tree = self.db.open_tree(&tree_key)?;
        Ok(tree)
    }

    fn open_tree(&self, iteration: u64) -> Result<Option<sled::Tree>, SimulationError> {
        let tree_key = Self::iteration_to_key(iteration);
        if !self.db.tree_names().contains(&sled::IVec::from(&tree_key)) {
            Ok(None)
        } else {
            let tree = self.db.open_tree(tree_key)?;
            Ok(Some(tree))
        }
    }

    pub fn open_or_create(location: std::path::PathBuf) -> Result<Self, SimulationError> {
        let db = sled::open(location)?;
        Ok(SledStorageInterface {
            db,
            id_phantom: PhantomData,
            element_phantom: PhantomData,
        })
    }
}

// impl<Id, Element> StorageInterface<Id, Element> for SledStorageInterface<Id, Element>
impl<Id, Element> SledStorageInterface<Id, Element>
where
    Id: Hash + Serialize + for<'a> Deserialize<'a>,
    Element: Serialize + for<'a> Deserialize<'a>,
{
    pub fn store_single_element(
        &self,
        iteration: u64,
        identifier: Id,
        element: Element,
    ) -> Result<(), SimulationError> {
        let tree = self.open_or_create_tree(iteration)?;

        // Serialize the identifier and the element
        let identifier_serialized = bincode::serialize(&identifier)?;
        let element_serialized = bincode::serialize(&element)?;
        match tree.insert(identifier_serialized, element_serialized)? {
            None => Ok(()),
            Some(_) => Err(DataBaseError {
                message: format!("Element already present at iteration {}", iteration),
            }),
        }?;
        Ok(())
    }

    pub fn store_batch_elements(
        &self,
        iteration: u64,
        identifiers_elements: Vec<(Id, Element)>,
    ) -> Result<(), SimulationError> {
        let tree = self.open_or_create_tree(iteration)?;
        let mut batch = sled::Batch::default();
        for (identifier, element) in identifiers_elements.into_iter() {
            let identifier_serialized = bincode::serialize(&identifier)?;
            let element_serialized = bincode::serialize(&element)?;
            batch.insert(identifier_serialized, element_serialized)
        }
        tree.apply_batch(batch)?;
        Ok(())
    }

    pub fn load_single_element(
        &self,
        iteration: u64,
        identifier: Id,
    ) -> Result<Option<Element>, SimulationError> {
        let tree = match self.open_tree(iteration)? {
            Some(tree) => tree,
            None => return Ok(None),
        };
        let identifier_serialized = bincode::serialize(&identifier)?;
        match tree.get(&identifier_serialized)? {
            Some(element_serialized) => {
                let element: Element = bincode::deserialize(&element_serialized)?;
                Ok(Some(element))
            }
            None => Ok(None),
        }
    }

    pub fn load_element_history(
        &self,
        identifier: Id,
    ) -> Result<HashMap<u64, Element>, SimulationError> {
        // Keep track if the element has not been found in a database.
        // If so we can either get the current minimal iteration or maximal depending on where it was found else.
        let mut minimal_iteration = None;
        let mut maximal_iteration = None;
        let mut success_iteration = None;

        // Save results in this hashmap
        let mut accumulator = HashMap::new();
        // Serialize the identifier
        let identifier_serialized = bincode::serialize(&identifier)?;
        for iteration_serialized in self.db.tree_names() {
            // If we are above the maximal or below the minimal iteration, we skip checking
            let iteration: u64 = bincode::deserialize(&iteration_serialized)?;
            if minimal_iteration.is_some_and(|min_iter| iteration < min_iter) {
                continue;
            }
            if maximal_iteration.is_some_and(|max_iter| max_iter < iteration) {
                continue;
            }
            // Get the tree for a random iteration
            let tree = self.db.open_tree(iteration_serialized)?;
            match tree.get(&identifier_serialized)? {
                // We found and element insert it
                Some(element_serialized) => {
                    let element: Element = bincode::deserialize(&element_serialized)?;
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

    pub fn load_all_elements_at_iteration(
        &self,
        iteration: u64,
    ) -> Result<HashMap<Id, Element>, SimulationError>
    where
        Id: std::cmp::Eq,
    {
        let tree = match self.open_tree(iteration)? {
            Some(tree) => tree,
            None => return Ok(HashMap::new()),
        };
        tree.iter()
            .map(|entry_result| {
                let (identifier_serialized, element_serialized) = entry_result?;
                let identifier: Id = bincode::deserialize(&identifier_serialized)?;
                let element: Element = bincode::deserialize(&element_serialized)?;
                Ok((identifier, element))
            })
            .collect::<Result<HashMap<Id, Element>, SimulationError>>()
    }

    pub fn load_all_elements(&self) -> Result<HashMap<u64, HashMap<Id, Element>>, SimulationError>
    where
        Id: std::cmp::Eq,
    {
        self.db
            .tree_names()
            .iter()
            .map(|tree_name_serialized| {
                let tree = self.db.open_tree(tree_name_serialized)?;
                let iteration = Self::key_to_iteration(tree_name_serialized)?;
                let identifier_to_element = tree
                    .iter()
                    .map(|entry_result| {
                        let (identifier_serialized, element_serialized) = entry_result?;
                        let identifier: Id = bincode::deserialize(&identifier_serialized)?;
                        let element: Element = bincode::deserialize(&element_serialized)?;
                        Ok((identifier, element))
                    })
                    .collect::<Result<HashMap<Id, Element>, SimulationError>>()?;
                Ok((iteration, identifier_to_element))
            })
            .collect::<Result<HashMap<u64, HashMap<Id, Element>>, SimulationError>>()
    }

    pub fn load_all_element_histories(
        &self,
    ) -> Result<HashMap<Id, HashMap<u64, Element>>, SimulationError>
    where
        Id: std::cmp::Eq,
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

/* pub fn store_single_element_in_tree<Element, Func>
(
    tree: typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    element: Element,
    element_iteration_to_database_key: Func,
) -> Result<(), SimulationError>
where
    Element: Serialize + Send + Sync + 'static,
    Func: Fn(u64, &Element) -> String + Send + 'static,
{

    let name = element_iteration_to_database_key(iteration, &element);
    let element_encoded = bincode::serialize(&element).unwrap();
    match tree.insert(&name, &element_encoded) {
        Err(error) => {
            println!("An error occurred: {} when writing an element at iteration {} to database. Continue anyway ...", error, iteration);
        },
        Ok(Some(_)) => {
            println!("Could not store element at iteration {} in database: element already present. Continue anyway ...", iteration);
        },
        Ok(None) => (),
    }
    Ok(())
}


pub fn store_elements_in_tree<Element, Func>
(
    tree: typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    elements: Vec<Element>,
    element_iteration_to_database_key: Func,
) -> Result<(), SimulationError>
where
    Element: Serialize + Send + Sync + 'static,
    Func: Fn(u64, &Element) -> String + Send + 'static,
{
    // async_std::task::spawn(async move {
        let mut batch = typed_sled::Batch::<String, Vec<u8>>::default();

        elements.into_iter().for_each(|element| {
            let name = element_iteration_to_database_key(iteration, &element);
            let element_encoded = bincode::serialize(&element).unwrap();
            batch.insert(&name, &element_encoded);
        });
        match tree.apply_batch(batch) {
            Ok(()) => (),
            Err(error) => {
                print!("Storing cells: Database error: {}", error);
                println!("Continuing simulation anyway!");
            },
        }
    // });

    Ok(())
}


pub fn deserialize_single_element_from_tree<Element, Func, ReturnKey>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    key: &ReturnKey,
    iteration_and_other_to_database_key: Func
) -> Result<Element, SimulationError>
where
    Element: for<'a>Deserialize<'a> + 'static,
    Func: Fn(u64, &ReturnKey) -> String,
{
    let db_key = iteration_and_other_to_database_key(iteration, key);
    let element = tree.get(&db_key)?.ok_or(IndexError{message: format!("Cannot find element with key {} in database", db_key)})?;
    let element_deserialized = bincode::deserialize(&element)?;
    Ok(element_deserialized)
}


pub fn deserialize_all_elements_from_tree<Element, Func>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    database_key_to_iteration_and_other: Func,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<Vec<(u64, Element)>, SimulationError>
where
    Element: for<'a>Deserialize<'a>,
    Func: Fn(&String) -> Result<u64, SimulationError>,
{
    let bar = progress_style.map(|style| {
        let bar = indicatif::ProgressBar::new(tree.len() as u64);
        bar.set_style(style);
        println!("Deserializing entries in database");
        bar
    });
    let res = tree
        .iter()
        .filter_map(|opt| opt.ok())
        .filter_map(|(key, value)| {
            let cb: Option<Element> = bincode::deserialize(&value).ok();
            let res = database_key_to_iteration_and_other(&key).ok();
            match &bar {
                Some(b) => b.inc(1),
                None => (),
            }
            match (cb, res) {
                (Some(element), Some(iteration)) => Some((iteration, element)),
                _ => None,
            }
        },)
        .collect();
    bar.and_then(|bar| Some(bar.finish()));
    Ok(res)
}


pub fn deserialize_all_elements_from_tree_group<Element, Func, ReturnKey>
(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    database_key_to_iteration_and_other: Func,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<HashMap<u64, Vec<Element>>, SimulationError>
where
    Element: for<'a>Deserialize<'a> + Send + Sync,
    Func: Fn(&String) -> Result<(u64, ReturnKey), SimulationError> + Send + Sync,
{
    let bar = progress_style.map(|style| {
        let bar = indicatif::ProgressBar::new(tree.len() as u64);
        bar.set_style(style);
        println!("Deserializing entries in database");
        bar
    });
    use rayon::prelude::*;
    let key_value_pairs = tree.iter().collect::<Vec<_>>();
    let res = key_value_pairs
        .into_par_iter()
        .filter_map(|opt| opt.ok())
        .filter_map(|(key, value)| {
            let cb: Option<Element> = bincode::deserialize(&value).ok();
            let res = database_key_to_iteration_and_other(&key).ok();
            match (cb, res) {
                (Some(element), Some((iteration, _))) => Some((iteration, element)),
                _ => None,
            }
        },)
        .fold(|| HashMap::<u64, Vec<Element>>::new(), |mut acc, (iteration, element)| -> std::collections::HashMap<u64, Vec<Element>> {
            match acc.get_mut(&iteration) {
                Some(elements) => elements.push(element),
                None => {acc.insert(iteration, vec![element]);},
            }

            match &bar {
                Some(b) => b.inc(1),
                None => (),
            }
            acc
        })
        .reduce(|| HashMap::<u64, Vec<Element>>::new(), |mut h1, h2| {
            for (key, value) in h2.into_iter() {
                // h1.entry(key).and_modify(|entry| entry.extend(value)).or_insert(value);
                match h1.get_mut(&key) {
                    Some(extisting_entries) => extisting_entries.extend(value),
                    None => {h1.insert(key, value);},
                }
            }
            h1
        });
    bar.and_then(|bar| Some(bar.finish()));
    Ok(res)
}*/
