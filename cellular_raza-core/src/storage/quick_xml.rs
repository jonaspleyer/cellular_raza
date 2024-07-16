use super::concepts::StorageError;
use super::concepts::{FileBasedStorage, StorageInterfaceOpen};
use serde::{Deserialize, Serialize};

use core::marker::PhantomData;

/// Save elements as xml files with [quick_xml].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct XmlStorageInterface<Id, Element> {
    path: std::path::PathBuf,
    storage_instance: u64,
    phantom_id: PhantomData<Id>,
    phantom_element: PhantomData<Element>,
}

impl<Id, Element> FileBasedStorage<Id, Element> for XmlStorageInterface<Id, Element> {
    const EXTENSION: &'static str = "xml";

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
        let mut save_string = String::new();
        let mut serializer = quick_xml::se::Serializer::new(&mut save_string);
        serializer.indent(' ', 4);
        value.serialize(serializer)?;
        let mut writer = writer;
        writer.write(&save_string.as_bytes())?;
        Ok(())
    }

    fn from_reader<V, R>(&self, reader: R) -> Result<V, StorageError>
    where
        V: for<'a> Deserialize<'a>,
        R: std::io::Read,
    {
        let reader = std::io::BufReader::new(reader);
        Ok(quick_xml::de::from_reader(reader)?)
    }
}

impl<Id, Element> StorageInterfaceOpen for XmlStorageInterface<Id, Element> {
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
        Ok(XmlStorageInterface {
            path: location.into(),
            storage_instance,
            phantom_id: PhantomData,
            phantom_element: PhantomData,
        })
    }
}
