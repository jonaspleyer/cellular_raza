use super::concepts::StorageError;
use super::concepts::{FileBasedStorage, StorageInterfaceOpen};
use serde::{Deserialize, Serialize};

use core::marker::PhantomData;

/// Save elements as ron files with [ron].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RonStorageInterface<Id, Element> {
    path: std::path::PathBuf,
    storage_instance: u64,
    phantom_id: PhantomData<Id>,
    phantom_element: PhantomData<Element>,
}

impl<Id, Element> FileBasedStorage<Id, Element> for RonStorageInterface<Id, Element> {
    const EXTENSION: &'static str = "ron";

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
        let config = ron::ser::PrettyConfig::new()
            .depth_limit(usize::MAX)
            .struct_names(true)
            .separate_tuple_members(false)
            .compact_arrays(true)
            .indentor("  ".to_owned());
        let options = ron::Options::default();
        Ok(options.to_io_writer_pretty(writer, value, config)?)
    }

    fn from_str<V>(&self, input: &str) -> Result<V, StorageError>
    where
        V: for<'a> Deserialize<'a>,
    {
        Ok(ron::de::from_str(input)?)
    }
}

impl<Id, Element> StorageInterfaceOpen for RonStorageInterface<Id, Element> {
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
        Ok(RonStorageInterface {
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
