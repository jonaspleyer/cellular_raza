use super::concepts::StorageError;
use super::concepts::{FileBasedStorage, StorageInterfaceOpen};
use serde::{Deserialize, Serialize};

use core::marker::PhantomData;

#[cfg(feature = "tracing")]
use tracing::instrument;

/// Save elements as json files with [serde_json].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JsonStorageInterface<Id, Element> {
    path: std::path::PathBuf,
    storage_instance: u64,
    phantom_id: PhantomData<Id>,
    phantom_element: PhantomData<Element>,
}

impl<Id, Element> FileBasedStorage<Id, Element> for JsonStorageInterface<Id, Element> {
    const EXTENSION: &'static str = "json";

    fn get_path(&self) -> &std::path::Path {
        &self.path
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

    fn from_reader<V, R>(&self, reader: R) -> Result<V, StorageError>
    where
        V: for<'a> Deserialize<'a>,
        R: std::io::Read,
    {
        Ok(serde_json::from_reader(reader)?)
    }
}

impl<Id, Element> StorageInterfaceOpen for JsonStorageInterface<Id, Element> {
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
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

    fn clone_to_new_instance(&self, storage_instance: u64) -> Self {
        Self {
            path: self.path.clone(),
            storage_instance,
            phantom_id: PhantomData,
            phantom_element: PhantomData,
        }
    }
}
