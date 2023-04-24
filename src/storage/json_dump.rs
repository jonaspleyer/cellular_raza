use crate::concepts::errors::{IndexError, SimulationError};

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

pub fn store_single_element<Element, Id>(
    save_path: std::path::Path,
    iteration: u64,
    id: Id,
    element: Element,
) -> Result<(), SimulationError>
where
    Element: Serialize + Send + Sync + 'static,
    Id: Format,
{
    // TODO
    Ok(())
}

pub fn store_element_batch<Element, Func>(
    tree: typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    elements: Vec<Element>,
    element_iteration_to_database_key: Func,
) -> Result<(), SimulationError>
where
    Element: Serialize + Send + Sync + 'static,
    Func: Fn(u64, &Element) -> String + Send + 'static,
{
    // TODO
    Ok(())
}

pub fn deserialize_single_element_from_tree<Element, Func, ReturnKey>(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    iteration: u64,
    key: &ReturnKey,
    iteration_and_other_to_database_key: Func,
) -> Result<Element, SimulationError>
where
    Element: for<'a> Deserialize<'a> + 'static,
    Func: Fn(u64, &ReturnKey) -> String,
{
    Ok(element_deserialized)
}

pub fn deserialize_all_elements_from_tree<Element, Func>(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    database_key_to_iteration_and_other: Func,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<Vec<(u64, Element)>, SimulationError>
where
    Element: for<'a> Deserialize<'a>,
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
        })
        .collect();
    bar.and_then(|bar| Some(bar.finish()));
    Ok(res)
}

pub fn deserialize_all_elements_from_tree_group<Element, Func, ReturnKey>(
    tree: &typed_sled::Tree<String, Vec<u8>>,
    database_key_to_iteration_and_other: Func,
    progress_style: Option<indicatif::ProgressStyle>,
) -> Result<HashMap<u64, Vec<Element>>, SimulationError>
where
    Element: for<'a> Deserialize<'a> + Send + Sync,
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
        })
        .fold(
            || HashMap::<u64, Vec<Element>>::new(),
            |mut acc, (iteration, element)| -> std::collections::HashMap<u64, Vec<Element>> {
                match acc.get_mut(&iteration) {
                    Some(elements) => elements.push(element),
                    None => {
                        acc.insert(iteration, vec![element]);
                    }
                }

                match &bar {
                    Some(b) => b.inc(1),
                    None => (),
                }
                acc
            },
        )
        .reduce(
            || HashMap::<u64, Vec<Element>>::new(),
            |mut h1, h2| {
                for (key, value) in h2.into_iter() {
                    // h1.entry(key).and_modify(|entry| entry.extend(value)).or_insert(value);
                    match h1.get_mut(&key) {
                        Some(extisting_entries) => extisting_entries.extend(value),
                        None => {
                            h1.insert(key, value);
                        }
                    }
                }
                h1
            },
        );
    bar.and_then(|bar| Some(bar.finish()));
    Ok(res)
}
