use super::concepts::StorageError;
use super::concepts::{BatchSaveFormat, CombinedSaveFormat, StorageInterface};
use serde::{Deserialize, Serialize};

use core::marker::PhantomData;
use std::collections::HashMap;

/// Save elements as json files with [serde_json].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JsonStorageInterface<Id, Element> {
    /// Storage path.
    pub path: std::path::PathBuf,
    storage_instance: u64,
    phantom_id: PhantomData<Id>,
    phantom_element: PhantomData<Element>,
}

impl<Id, Element> JsonStorageInterface<Id, Element> {
    fn create_or_get_iteration_file_with_prefix(
        &self,
        iteration: u64,
        prefix: &str,
    ) -> Result<std::io::BufWriter<std::fs::File>, StorageError> {
        let save_path = self.get_iteration_save_path_batch_with_prefix(iteration, prefix)?;

        // Open+Create a file and wrap it inside a buffer writer
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&save_path)?;

        Ok(std::io::BufWriter::new(file))
    }

    fn get_iteration_path(&self, iteration: u64) -> std::path::PathBuf {
        self.path.join(format!("{:020.0}", iteration))
    }

    fn get_iteration_save_path_batch_with_prefix(
        &self,
        iteration: u64,
        prefix: &str,
    ) -> Result<std::path::PathBuf, StorageError> {
        // First we get the folder path of the iteration
        let iteration_path = self.get_iteration_path(iteration);
        // If this folder does not exist, we create it
        std::fs::create_dir_all(&iteration_path)?;

        // Check if other batch files are already existing
        // If this is the case increase the batch number until we find one where no batch is existing
        let save_path = iteration_path
            .join(format!("{}_{:020.0}", prefix, self.storage_instance))
            .with_extension("json");
        Ok(save_path)
    }

    fn folder_name_to_iteration(
        &self,
        file: &std::path::Path,
    ) -> Result<Option<u64>, StorageError> {
        match file.file_stem() {
            Some(filename) => match filename.to_str() {
                Some(filename_string) => Ok(Some(filename_string.parse::<u64>()?)),
                None => Ok(None),
            },
            None => Ok(None),
        }
    }
}

impl<Id, Element> StorageInterface<Id, Element> for JsonStorageInterface<Id, Element> {
    fn open_or_create(
        location: &std::path::Path,
        storage_instance: u64,
    ) -> Result<Self, StorageError>
    where
        Self: Sized,
    {
        if !location.is_dir() {
            std::fs::create_dir_all(location)?;
        }
        Ok(JsonStorageInterface {
            path: location.into(),
            storage_instance,
            phantom_id: PhantomData,
            phantom_element: PhantomData,
        })
    }

    fn store_single_element(
        &self,
        iteration: u64,
        identifier: &Id,
        element: &Element,
    ) -> Result<(), StorageError>
    where
        Id: Serialize,
        Element: Serialize,
    {
        let iteration_file = self.create_or_get_iteration_file_with_prefix(iteration, "single")?;
        let save_format = CombinedSaveFormat {
            identifier,
            element,
        };
        serde_json::to_writer_pretty(iteration_file, &save_format)?;
        Ok(())
    }

    fn store_batch_elements<'a, I>(
        &self,
        iteration: u64,
        identifiers_elements: I,
    ) -> Result<(), StorageError>
    where
        Id: 'a + Serialize,
        Element: 'a + Serialize,
        I: Clone + IntoIterator<Item = (&'a Id, &'a Element)>,
    {
        let iteration_file = self.create_or_get_iteration_file_with_prefix(iteration, "batch")?;
        let batch = BatchSaveFormat {
            data: identifiers_elements
                .into_iter()
                .map(|(id, element)| CombinedSaveFormat {
                    identifier: id,
                    element,
                })
                .collect(),
        };
        serde_json::to_writer_pretty(iteration_file, &batch)?;
        Ok(())
    }

    fn load_single_element(
        &self,
        iteration: u64,
        identifier: &Id,
    ) -> Result<Option<Element>, StorageError>
    where
        Id: Serialize + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        let iterations = self.get_all_iterations()?;
        if iterations.contains(&iteration) {
            // Get the path where the iteration folder is
            let iteration_path = self.get_iteration_path(iteration);

            // Load all elements which are inside this folder from batches and singles
            for path in std::fs::read_dir(&iteration_path)? {
                let p = path?.path();
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(&p)?;

                match p.file_stem() {
                    Some(stem) => match stem.to_str() {
                        Some(tail) => {
                            let elements = tail.split("_");
                            if elements.into_iter().next() == Some("batch") {
                                let result: BatchSaveFormat<Id, Element> =
                                    serde_json::from_reader(file)?;
                                for json_save_format in result.data.into_iter() {
                                    let id1 = bincode::serialize(&json_save_format.identifier)?;
                                    let id2 = bincode::serialize(&identifier)?;
                                    if id1 == id2 {
                                        return Ok(Some(json_save_format.element));
                                    }
                                }
                            }
                        }
                        None => (),
                    },
                    None => (),
                }
            }
            return Ok(None);
        } else {
            return Ok(None);
        }
    }

    fn load_all_elements_at_iteration(
        &self,
        iteration: u64,
    ) -> Result<HashMap<Id, Element>, StorageError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        let iterations = self.get_all_iterations()?;
        if iterations.contains(&iteration) {
            // Create a new empty hashmap
            let mut all_elements_at_iteration = HashMap::new();

            // Get the path where the iteration folder is
            let iteration_path = self.get_iteration_path(iteration);

            // Load all elements which are inside this folder from batches and singles
            for path in std::fs::read_dir(&iteration_path)? {
                let p = path?.path();
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(&p)?;

                match p.file_stem() {
                    Some(stem) => match stem.to_str() {
                        Some(tail) => {
                            let elements = tail.split("_");
                            if elements.into_iter().next() == Some("batch") {
                                let result: BatchSaveFormat<Id, Element> =
                                    serde_json::from_reader(file)?;
                                all_elements_at_iteration.extend(result.data.into_iter().map(
                                    |json_save_format| {
                                        (json_save_format.identifier, json_save_format.element)
                                    },
                                ));
                            }
                        }
                        None => (),
                    },
                    None => (),
                }
            }
            return Ok(all_elements_at_iteration);
        } else {
            return Ok(HashMap::new());
        }
    }

    fn get_all_iterations(&self) -> Result<Vec<u64>, StorageError> {
        let paths = std::fs::read_dir(&self.path)?;
        paths
            .into_iter()
            .filter_map(|path| match path {
                Ok(p) => match self.folder_name_to_iteration(&p.path()) {
                    Ok(Some(entry)) => Some(Ok(entry)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                },
                Err(_) => None,
            })
            .collect::<Result<Vec<_>, _>>()
    }
}
