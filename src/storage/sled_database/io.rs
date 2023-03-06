use crate::concepts::errors::{SimulationError,IndexError};

use serde::{Serialize,Deserialize};

use std::collections::HashMap;


pub fn store_single_element_in_tree<Element, Func>
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
    async_std::task::spawn(async move {
        let name = element_iteration_to_database_key(iteration, &element);
        let element_encoded = bincode::serialize(&element).unwrap();
        match tree.insert(&name, &element_encoded) {
            Err(error) => {
                println!("An error occurred: {} when writing an element at iteration {} to database", error, iteration);
            },
            Ok(Some(_)) => {
                println!("Could not store element at iteration {} in database", iteration);
            },
            Ok(None) => (),
        }
    });
    
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
    async_std::task::spawn(async move {
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
    });
    
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
    Element: for<'a>Deserialize<'a>,
    Func: Fn(&String) -> Result<(u64, ReturnKey), SimulationError>,
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
            match (cb, res) {
                (Some(element), Some((iteration, _))) => Some((iteration, element)),
                _ => None,
            }
        },)
        .fold(HashMap::<u64, Vec<Element>>::new(), |mut acc, (iteration, element)| -> std::collections::HashMap<u64, Vec<Element>> {
            match acc.get_mut(&iteration) {
                Some(elements) => elements.push(element),
                None => {acc.insert(iteration, vec![element]);},
            }

            match &bar {
                Some(b) => b.inc(1),
                None => (),
            }
            acc
        });
    bar.and_then(|bar| Some(bar.finish()));
    Ok(res)
}
