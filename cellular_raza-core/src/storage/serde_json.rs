use super::concepts::StorageError;
use super::concepts::{FileBasedStorage, StorageInterfaceOpen};
use serde::{Deserialize, Serialize};

use core::marker::PhantomData;

/// Save elements as json files with [serde_json].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JsonStorageInterface<Id, Element> {
    /// Storage path.
    pub path: std::path::PathBuf,
    storage_instance: u64,
    phantom_id: PhantomData<Id>,
    phantom_element: PhantomData<Element>,
}

impl<Id, Element> FileBasedStorage<Id, Element> for JsonStorageInterface<Id, Element> {
    fn get_path(&self) -> &std::path::Path {
        &self.path
    }

    fn get_extension(&self) -> &str {
        "json"
    }

    fn get_storage_instance(&self) -> u64 {
        self.storage_instance
    }

    fn to_writer_pretty<V, W>(&self, writer: W, value: &V) -> Result<(), StorageError>
    where
        V: Serialize,
        W: std::io::Write,
    {
        Ok(serde_json::to_writer_pretty(writer, value)?)
    }
}

impl<Id, Element> StorageInterfaceOpen for JsonStorageInterface<Id, Element> {
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
}

// TODO extend this to test all functions
macro_rules! test_storage_interface(
    ($interface_name:ident, $module_name:ident) => {
        #[cfg(test)]
        mod $module_name {
            use super::*;

            #[test]
            fn store_load_all_elements() {
                use tempdir::TempDir;
                let dir = TempDir::new("tempdir").unwrap();
                let location = dir.path().join(concat!("tempdir_", stringify!($interface_name)));
                let mut interface_0 = $interface_name::open_or_create(&location, 0).unwrap();
                let mut interface_1 = $interface_name::open_or_create(&location, 1).unwrap();
                let generate_elements = |low: usize, high: usize| {
                    (low..high).map(|i| (i, i as f64))
                    .collect::<std::collections::HashMap<_, _>>()
                };
                let identifiers_elements_0 = generate_elements(0, 10);
                let identifiers_elements_1 = generate_elements(20, 30);
                let iteration = 100;
                interface_0
                    .store_batch_elements(iteration, identifiers_elements_0.iter())
                    .unwrap();
                interface_1
                    .store_batch_elements(iteration, identifiers_elements_1.iter())
                    .unwrap();
                let loaded_elements_0 = interface_0.load_all_elements_at_iteration(iteration)
                    .unwrap();
                let loaded_elements_1 = interface_1.load_all_elements_at_iteration(iteration)
                    .unwrap();
                let mut identifiers_elements = identifiers_elements_0.clone();
                identifiers_elements.extend(identifiers_elements_1);
                assert_eq!(identifiers_elements, loaded_elements_0);
                assert_eq!(identifiers_elements, loaded_elements_1);
                assert_eq!(loaded_elements_0, loaded_elements_1);
            }
        }
    }
);

test_storage_interface!(JsonStorageInterface, json_tests);
