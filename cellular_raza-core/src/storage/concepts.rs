use std::collections::HashMap;
use std::error::Error;
use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::memory_storage::MemoryStorageInterface;
use super::quick_xml::XmlStorageInterface;
use super::serde_json::JsonStorageInterface;
use super::sled_database::SledStorageInterface;
use super::ron::RonStorageInterface;

/// Error related to storing and reading elements
#[derive(Debug)]
pub enum StorageError {
    /// Error related to File Io operations.
    IoError(std::io::Error),
    /// Occurs during parsing of json structs.
    SerdeJsonError(serde_json::Error),
    /// Occurs during parsing of Xml structs.
    QuickXmlError(quick_xml::Error),
    /// Occurs during parsing of Xml structs.
    FastXmlDeserializeError(quick_xml::DeError),
    /// Generic error related to serialization in the [ron] crate.
    RonError(ron::Error),
    /// Generic error related to deserialization in the [ron] crate.
    RonSpannedError(ron::error::SpannedError),
    /// Generic error related to the [sled] database.
    SledError(sled::Error),
    /// Generic serialization error thrown by the [bincode] library.
    SerializeError(Box<bincode::ErrorKind>),
    /// Initialization error mainly used for initialization of databases such as [sled].
    InitError(String),
    /// Error when parsing file/folder names.
    ParseIntError(std::num::ParseIntError),
    /// Generic Utf8 error.
    Utf8Error(std::str::Utf8Error),
}

impl From<serde_json::Error> for StorageError {
    fn from(err: serde_json::Error) -> Self {
        StorageError::SerdeJsonError(err)
    }
}

impl From<quick_xml::Error> for StorageError {
    fn from(err: quick_xml::Error) -> Self {
        StorageError::QuickXmlError(err)
    }
}

impl From<quick_xml::DeError> for StorageError {
    fn from(err: quick_xml::DeError) -> Self {
        StorageError::FastXmlDeserializeError(err)
    }
}

impl From<ron::error::SpannedError> for StorageError {
    fn from(err: ron::error::SpannedError) -> Self {
        StorageError::RonSpannedError(err)
    }
}

impl From<ron::Error> for StorageError {
    fn from(err: ron::Error) -> Self {
        StorageError::RonError(err)
    }
}

impl From<sled::Error> for StorageError {
    fn from(err: sled::Error) -> Self {
        StorageError::SledError(err)
    }
}

impl From<Box<bincode::ErrorKind>> for StorageError {
    fn from(err: Box<bincode::ErrorKind>) -> Self {
        StorageError::SerializeError(err)
    }
}

impl From<std::io::Error> for StorageError {
    fn from(err: std::io::Error) -> Self {
        StorageError::IoError(err)
    }
}

impl From<std::str::Utf8Error> for StorageError {
    fn from(err: std::str::Utf8Error) -> Self {
        StorageError::Utf8Error(err)
    }
}

impl From<std::num::ParseIntError> for StorageError {
    fn from(err: std::num::ParseIntError) -> Self {
        StorageError::ParseIntError(err)
    }
}

impl Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            StorageError::SerdeJsonError(message) => write!(f, "{}", message),
            StorageError::QuickXmlError(message) => write!(f, "{}", message),
            StorageError::FastXmlDeserializeError(message) => write!(f, "{}", message),
            StorageError::RonError(message) => write!(f, "{}", message),
            StorageError::RonSpannedError(message) => write!(f, "{}", message),
            StorageError::SledError(message) => write!(f, "{}", message),
            StorageError::SerializeError(message) => write!(f, "{}", message),
            StorageError::IoError(message) => write!(f, "{}", message),
            StorageError::InitError(message) => write!(f, "{}", message),
            StorageError::Utf8Error(message) => write!(f, "{}", message),
            StorageError::ParseIntError(message) => write!(f, "{}", message),
        }
    }
}

impl Error for StorageError {}

/// Define how to store results of the simulation.
///
/// We currently support saving results in a [sled] database, as xml files
/// via [quick_xml] or as a json file by using [serde_json].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum StorageOption {
    /// Save results as [sled] database.
    Sled,
    /// Save results as [sled] database but remove them when dropping the struct
    SledTemp,
    /// Save results as [json](https://www.json.org/json-en.html) file.
    SerdeJson,
    /// Save results as [xml](https://www.xml.org/) file.
    /// # WARNING!
    /// This option does not allow to deserializee at the moment due to a bug (probably) in
    /// quick-xml.
    #[deprecated]
    SerdeXml,
    /// Store results in the [ron] file format specifically designed for Rust structs.
    /// This format guarantees round-trips `Rust -> Ron -> Rust` and is thus preferred together
    /// with the well-established [StorageOption::SerdeJson] format.
    Ron,
    /// A [std::collections::HashMap](HashMap) based memory storage.
    Memory,
}

/// A unique vector containing only non-recurring values but in the correct order.
///
/// ```
/// # use cellular_raza_core::storage::UniqueVec;
/// let mut unique_vec = UniqueVec::new();
/// unique_vec.push(1_usize);
/// unique_vec.push(2_usize);
/// let res = unique_vec.push(1_usize);
/// assert!(res.is_some());
/// assert_eq!(*unique_vec, vec![1, 2]);
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UniqueVec<T>(Vec<T>);

impl<T> UniqueVec<T> {
    /// Creates an new empty [UniqueVec].
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Construct a new [UniqueVec] from a given vector.
    /// This function will also return the rest which was not inserted into the [UniqueVec].
    ///
    /// ```
    /// # use cellular_raza_core::storage::UniqueVec;
    /// let input = vec![1, 33, 2, 0, 33, 4, 56, 2];
    /// let (unique_vec, rest) = UniqueVec::from_iter(input);
    /// assert_eq!(*unique_vec, vec![1, 33, 2, 0, 4, 56]);
    /// assert_eq!(rest, vec![33, 2]);
    /// ```
    pub fn from_iter(iter: impl IntoIterator<Item = T>) -> (Self, Vec<T>)
    where
        T: PartialEq,
    {
        let mut new_inner = Vec::new();
        let rest = iter
            .into_iter()
            .filter_map(|element| {
                if new_inner.contains(&element) {
                    Some(element)
                } else {
                    new_inner.push(element);
                    None
                }
            })
            .collect();
        (Self(new_inner), rest)
    }

    /// Add an element to the [UniqueVec] if not already present.
    ///
    /// ```
    /// # use cellular_raza_core::storage::UniqueVec;
    /// let mut unique_vec = UniqueVec::new();
    /// assert!(unique_vec.push(1_f64).is_none());
    /// assert!(unique_vec.push(2_f64).is_none());
    /// assert!(unique_vec.push(1_f64).is_some());
    /// assert_eq!(*unique_vec, vec![1_f64, 2_f64]);
    /// ```
    pub fn push(&mut self, element: T) -> Option<T>
    where
        T: PartialEq,
    {
        if self.0.contains(&element) {
            Some(element)
        } else {
            self.0.push(element);
            None
        }
    }

    /// Empties the [UniqueVec] returning all values
    pub fn clear(&mut self) {
        self.0.clear()
    }

    /// Remove last element from [UniqueVec]
    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }
}

impl<T> core::ops::Deref for UniqueVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> From<Vec<T>> for UniqueVec<T>
where
    T: PartialEq,
{
    fn from(value: Vec<T>) -> Self {
        Self::from_iter(value).0
    }
}

impl<T> IntoIterator for UniqueVec<T> {
    type Item = T;
    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl StorageOption {
    /// Which storage option should be used by default.
    pub fn default_priority() -> UniqueVec<Self> {
        vec![
            StorageOption::SerdeJson,
            // TODO fix sled! This is currently not working on multiple threads
            // StorageOptions::Sled,
        ]
        .into()
    }
}

/// Define how elements and identifiers are saved when being serialized together.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CombinedSaveFormat<Id, Element> {
    /// Identifier of the element
    pub identifier: Id,
    /// Actual element which is being stored
    pub element: Element,
}

/// Define how batches of elements and identifiers are saved when being serialized.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BatchSaveFormat<Id, Element> {
    pub(super) data: Vec<CombinedSaveFormat<Id, Element>>,
}

/// This manager handles if multiple storage options have been specified
/// It can load resources from one storage aspect and will
#[derive(Clone, Debug)]
pub struct StorageManager<Id, Element> {
    storage_priority: UniqueVec<StorageOption>,
    builder: StorageBuilder<true>,
    instance: u64,

    sled_storage: Option<SledStorageInterface<Id, Element>>,
    sled_temp_storage: Option<SledStorageInterface<Id, Element, true>>,
    json_storage: Option<StorageWrapper<JsonStorageInterface<Id, Element>>>,
    ron_storage: Option<StorageWrapper<RonStorageInterface<Id, Element>>>,
    xml_storage: Option<StorageWrapper<XmlStorageInterface<Id, Element>>>,
    memory_storage: Option<MemoryStorageInterface<Id, Element>>,
}

/// Used to construct a [StorageManager]
///
/// This builder contains multiple options which can be used to configure the location and type in
/// which results are stored.
/// To get an overview over all possible options, we refer to the [module](crate::storage)
/// documentation.
///
/// ```
/// use cellular_raza_core::storage::{StorageBuilder, StorageOption};
///
/// let storage_priority = StorageOption::default_priority();
/// let storage_builder = StorageBuilder::new()
///     .priority(storage_priority)
///     .location("./");
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StorageBuilder<const INIT: bool = false> {
    location: std::path::PathBuf,
    priority: UniqueVec<StorageOption>,
    suffix: std::path::PathBuf,
    #[cfg(feature = "timestamp")]
    add_date: bool,
    #[cfg(feature = "timestamp")]
    date: std::path::PathBuf,
}

impl<const INIT: bool> StorageBuilder<INIT> {
    /// Define the priority of [StorageOption]. See [StorageOption::default_priority].
    pub fn priority(self, priority: impl IntoIterator<Item = StorageOption>) -> Self {
        let (priority, _) = UniqueVec::from_iter(priority);
        Self { priority, ..self }
    }

    /// Get the current priority
    pub fn get_priority(&self) -> UniqueVec<StorageOption> {
        self.priority.clone()
    }

    /// Define a suffix which will be appended to the save path
    pub fn suffix(self, suffix: impl Into<std::path::PathBuf>) -> Self {
        Self {
            suffix: suffix.into(),
            ..self
        }
    }

    /// Get the current suffix
    pub fn get_suffix(&self) -> std::path::PathBuf {
        self.suffix.clone()
    }

    /// Store results by their current date inside the specified folder path
    #[cfg(feature = "timestamp")]
    pub fn add_date(self, add_date: bool) -> Self {
        Self { add_date, ..self }
    }

    /// Get information if the current date should be appended to the storage path
    #[cfg(feature = "timestamp")]
    pub fn get_add_date(&self) -> bool {
        self.add_date
    }
}

impl StorageBuilder<false> {
    /// Constructs a new [StorageBuilder] with default settings.
    ///
    /// ```
    /// use cellular_raza_core::storage::StorageBuilder;
    /// let storage_builder = StorageBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            location: "./out".into(),
            priority: UniqueVec::from_iter([StorageOption::SerdeJson]).0,
            suffix: "".into(),
            #[cfg(feature = "timestamp")]
            add_date: true,
            #[cfg(feature = "timestamp")]
            date: "".into(),
        }
    }

    /// Initializes the [StorageBuilder] thus filling information about time.
    pub fn init(self) -> StorageBuilder<true> {
        #[cfg(feature = "timestamp")]
        let date: std::path::PathBuf = if self.add_date {
            format!("{}", chrono::Local::now().format("%Y-%m-%d-T%H-%M-%S")).into()
        } else {
            "".into()
        };
        self.init_with_date(&date)
    }

    /// Specify the time at which the results should be saved
    pub fn init_with_date(self, date: &std::path::Path) -> StorageBuilder<true> {
        StorageBuilder::<true> {
            location: self.location,
            priority: self.priority,
            suffix: self.suffix,
            #[cfg(feature = "timestamp")]
            add_date: self.add_date,
            #[cfg(feature = "timestamp")]
            date: date.into(),
        }
    }

    /// Define a folder where to store results
    ///
    /// Note that this functionality is only available as long as the [StorageBuilder] has not been
    /// initialized.
    pub fn location<P>(self, location: P) -> Self
    where
        std::path::PathBuf: From<P>,
    {
        Self {
            location: location.into(),
            ..self
        }
    }

    /// Get the current storage_location
    ///
    /// Note that this functionality is only available as long as the [StorageBuilder] has not been
    /// initialized.
    pub fn get_location(&self) -> std::path::PathBuf {
        self.location.clone()
    }
}

impl StorageBuilder<true> {
    /// Get the fully constructed path after the Builder has been initialized with the
    /// [StorageBuilder::init] function.
    pub fn get_full_path(&self) -> std::path::PathBuf {
        let mut full_path = self.location.clone();
        #[cfg(feature = "timestamp")]
        if self.add_date {
            full_path.extend(&self.date);
        }
        full_path.extend(&self.suffix);
        full_path
    }

    #[doc(hidden)]
    pub fn init(self) -> Self {
        self
    }

    /// De-initializes the StorageBuilder, making it possible to edit it again.
    pub fn de_init(self) -> StorageBuilder<false> {
        StorageBuilder {
            location: self.location,
            priority: self.priority,
            suffix: self.suffix,
            #[cfg(feature = "timestamp")]
            add_date: self.add_date,
            #[cfg(feature = "timestamp")]
            date: "".into(),
        }
    }
}

impl<Id, Element> StorageManager<Id, Element> {
    /// Constructs the [StorageManager] from the instance identifier
    /// and the settings given by the [StorageBuilder].
    ///
    /// ```
    /// use cellular_raza_core::storage::*;
    /// let builder = StorageBuilder::new()
    ///     .location("/tmp")
    ///     .init();
    ///
    /// let manager = StorageManager::<usize, f64>::open_or_create(builder, 0)?;
    /// # Ok::<(), StorageError>(())
    /// ```
    pub fn open_or_create(
        storage_builder: StorageBuilder<true>,
        instance: u64,
    ) -> Result<Self, StorageError> {
        let location = storage_builder.get_full_path();

        let mut sled_storage = None;
        let mut sled_temp_storage = None;
        let mut json_storage = None;
        let mut ron_storage = None;
        let mut xml_storage = None;
        let mut memory_storage = None;
        for storage_variant in storage_builder.priority.iter() {
            match storage_variant {
                StorageOption::SerdeJson => {
                    json_storage = Some(StorageWrapper(
                        JsonStorageInterface::<Id, Element>::open_or_create(
                            &location.to_path_buf().join("json"),
                            instance,
                        )?,
                    ));
                }
                StorageOption::Sled => {
                    sled_storage =
                        Some(SledStorageInterface::<Id, Element, false>::open_or_create(
                            &location.to_path_buf().join("sled"),
                            instance,
                        )?);
                }
                StorageOption::SledTemp => {
                    sled_temp_storage =
                        Some(SledStorageInterface::<Id, Element, true>::open_or_create(
                            &location.to_path_buf().join("sled_memory"),
                            instance,
                        )?);
                }
                StorageOption::Ron => {
                    ron_storage = Some(StorageWrapper(
                        RonStorageInterface::<Id, Element>::open_or_create(
                            &location.to_path_buf().join("xml"),
                            instance,
                        )?,
                    ));

                }
                StorageOption::SerdeXml => {
                    xml_storage = Some(StorageWrapper(
                        XmlStorageInterface::<Id, Element>::open_or_create(
                            &location.to_path_buf().join("xml"),
                            instance,
                        )?,
                    ));
                }
                StorageOption::Memory => {
                    memory_storage = Some(MemoryStorageInterface::<Id, Element>::open_or_create(
                        &location.to_path_buf().join("xml"),
                        instance,
                    )?);
                }
            }
        }
        let manager = StorageManager {
            storage_priority: storage_builder.priority.clone(),
            builder: storage_builder.clone(),
            instance,

            sled_storage,
            sled_temp_storage,
            json_storage,
            ron_storage,
            xml_storage,
            memory_storage,
        };

        Ok(manager)
    }

    /// Extracts all information given by the [StorageBuilder] when constructing
    pub fn extract_builder(&self) -> StorageBuilder<true> {
        self.builder.clone()
    }

    /// Get the instance of this object.
    ///
    /// These instances should not be overlapping, ie. there should not be two objects existing in
    /// parallel with the same instance number.
    pub fn get_instance(&self) -> u64 {
        self.instance
    }
}

macro_rules! exec_for_all_storage_options(
    (@internal $self:ident, $storage_option:ident, $field:ident, $function:ident, $($args:tt)*) => {
        {
            if let Some($field) = &$self.$field {
                $field.$function($($args)*)
            } else {
                Err(StorageError::InitError(
                    stringify!($storage_option, " storage was not initialized but called").into(),
                ))?
            }
        }
    };
    (mut $self:ident, $field:ident, $function:ident, $($args:tt)*) => {
        if let Some($field) = &mut $self.$field {
            $field.$function($($args)*)?;
        }
    };
    (all mut $self:ident, $function:ident, $($args:tt)*) => {
        exec_for_all_storage_options!(mut $self, sled_storage, $function, $($args)*);
        exec_for_all_storage_options!(mut $self, sled_temp_storage, $function, $($args)*);
        exec_for_all_storage_options!(mut $self, json_storage, $function, $($args)*);
        exec_for_all_storage_options!(mut $self, ron_storage, $function, $($args)*);
        exec_for_all_storage_options!(mut $self, xml_storage, $function, $($args)*);
        exec_for_all_storage_options!(mut $self, memory_storage, $function, $($args)*);
    };
    ($self:ident, $priority:ident, $function:ident, $($args:tt)*) => {
        match $priority {
            StorageOption::Sled => exec_for_all_storage_options!(
                @internal $self, Sled, sled_storage, $function, $($args)*
            ),
            StorageOption::SledTemp => exec_for_all_storage_options!(
                @internal $self, SledTemp, sled_temp_storage, $function, $($args)*
            ),
            StorageOption::SerdeJson => exec_for_all_storage_options!(
                @internal $self, SerdeJson, json_storage, $function, $($args)*
            ),
            StorageOption::Ron => exec_for_all_storage_options!(
                @internal $self, Ron, ron_storage, $function, $($args)*
            ),
            StorageOption::SerdeXml => exec_for_all_storage_options!(
                @internal $self, SerdeXml, xml_storage, $function, $($args)*
            ),
            StorageOption::Memory => exec_for_all_storage_options!(
                @internal $self, Memory, memory_storage, $function, $($args)*
            ),
        }
    }
);

impl<Id, Element> StorageInterfaceStore<Id, Element> for StorageManager<Id, Element>
where
    Id: core::hash::Hash + core::cmp::Eq + Clone,
    Element: Clone,
{
    #[allow(unused)]
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
        exec_for_all_storage_options!(
            all mut self,
            store_single_element,
            iteration, identifier, element
        );
        Ok(())
    }

    #[allow(unused)]
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
        exec_for_all_storage_options!(
            all mut self,
            store_batch_elements,
            iteration,
            identifiers_elements.clone()
        );
        Ok(())
    }
}

impl<Id, Element> StorageInterfaceLoad<Id, Element> for StorageManager<Id, Element>
where
    Id: core::hash::Hash + core::cmp::Eq + Clone,
    Element: Clone,
{
    #[allow(unused)]
    fn load_single_element(
        &self,
        iteration: u64,
        identifier: &Id,
    ) -> Result<Option<Element>, StorageError>
    where
        Id: Serialize + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        for priority in self.storage_priority.iter() {
            return exec_for_all_storage_options!(
                self,
                priority,
                load_single_element,
                iteration,
                identifier
            );
        }
        Ok(None)
    }

    #[allow(unused)]
    fn load_all_elements_at_iteration(
        &self,
        iteration: u64,
    ) -> Result<HashMap<Id, Element>, StorageError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        for priority in self.storage_priority.iter() {
            return exec_for_all_storage_options!(
                self,
                priority,
                load_all_elements_at_iteration,
                iteration
            );
        }
        Ok(HashMap::new())
    }

    fn get_all_iterations(&self) -> Result<Vec<u64>, StorageError> {
        for priority in self.storage_priority.iter() {
            return exec_for_all_storage_options!(self, priority, get_all_iterations,);
        }
        Ok(Vec::new())
    }
}

/// The mode in which to generate paths and store results.
pub enum StorageMode {
    /// Save one element to a single file
    Single,
    /// Save many elements in one file.
    Batch,
}

impl StorageMode {
    fn to_str(&self) -> &str {
        match self {
            Self::Single => "single",
            Self::Batch => "batch",
        }
    }
}

/// Abstraction and simplification of many file-based storage solutions
pub trait FileBasedStorage<Id, Element> {
    /// Get path where results are stored.
    fn get_path(&self) -> &std::path::Path;

    /// Get the number of this storage instance.
    /// This value may coincide with the thread number.
    fn get_storage_instance(&self) -> u64;

    /// Get the suffix which is used to distinguish this storage solution from others.
    fn get_extension(&self) -> &str;

    /// Writes either [BatchSaveFormat] or [CombinedSaveFormat] to the disk.
    fn to_writer_pretty<V, W>(&self, writer: W, value: &V) -> Result<(), StorageError>
    where
        V: Serialize,
        W: std::io::Write;

    /// Deserializes the given value type [V] from a reader.
    fn from_reader<V, R>(&self, reader: R) -> Result<V, StorageError>
    where
        V: for<'a> Deserialize<'a>,
        R: std::io::Read;

    /// Creates a new iteration file with a predefined naming scheme.
    ///
    /// The path which to use is by default determined by the
    /// [FileBasedStorage::get_iteration_save_path_batch_with_prefix] function.
    fn create_or_get_iteration_file_with_prefix(
        &self,
        iteration: u64,
        mode: StorageMode,
    ) -> Result<std::io::BufWriter<std::fs::File>, StorageError> {
        let save_path = self.get_iteration_save_path_batch_with_prefix(iteration, mode)?;

        // Open+Create a file and wrap it inside a buffer writer
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&save_path)?;

        Ok(std::io::BufWriter::new(file))
    }

    /// Get the path which holds saved entries if the given iteration.
    ///
    /// By default this function joins the path generated by [FileBasedStorage::get_path]
    /// with a 0-delimited number according to the iteration number.
    fn get_iteration_path(&self, iteration: u64) -> std::path::PathBuf {
        self.get_path().join(format!("{:020.0}", iteration))
    }

    /// Creates the path used by the [FileBasedStorage::create_or_get_iteration_file_with_prefix]
    /// function.
    fn get_iteration_save_path_batch_with_prefix(
        &self,
        iteration: u64,
        mode: StorageMode,
    ) -> Result<std::path::PathBuf, StorageError> {
        // First we get the folder path of the iteration
        let iteration_path = self.get_iteration_path(iteration);
        // If this folder does not exist, we create it
        std::fs::create_dir_all(&iteration_path)?;

        // Check if other batch files are already existing
        // If this is the case increase the batch number until we find one where no batch is existing
        let prefix = mode.to_str();
        let create_save_path = |i: usize| -> std::path::PathBuf {
            iteration_path
                .join(format!(
                    "{}_{:020.0}_{:020.0}",
                    prefix,
                    self.get_storage_instance(),
                    i
                ))
                .with_extension(self.get_extension())
        };
        let mut counter = 0;
        let mut save_path;
        while {
            save_path = create_save_path(counter);
            save_path.exists()
        } {
            counter += 1
        }
        Ok(save_path)
    }

    /// Converts a given path of a folder to a iteration number.
    ///
    /// This function is used for loading results
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct StorageWrapper<T>(pub(crate) T);

impl<T, Id, Element> StorageInterfaceStore<Id, Element> for StorageWrapper<T>
where
    T: FileBasedStorage<Id, Element>,
{
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
        let iteration_file = self
            .0
            .create_or_get_iteration_file_with_prefix(iteration, StorageMode::Batch)?;
        let batch = BatchSaveFormat {
            data: identifiers_elements
                .into_iter()
                .map(|(id, element)| CombinedSaveFormat {
                    identifier: id,
                    element,
                })
                .collect(),
        };
        self.0.to_writer_pretty(iteration_file, &batch)?;
        Ok(())
    }

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
        let iteration_file = self
            .0
            .create_or_get_iteration_file_with_prefix(iteration, StorageMode::Single)?;
        let save_format = CombinedSaveFormat {
            identifier,
            element,
        };
        self.0.to_writer_pretty(iteration_file, &save_format)?;
        Ok(())
    }
}

/// Open or create a new instance of the Storage controller.
pub trait StorageInterfaceOpen {
    /// Initializes the current storage device.
    ///
    /// In the case of databases, this may already result in an IO operation
    /// while when saving as files such as json or xml folders might be created.
    fn open_or_create(
        location: &std::path::Path,
        storage_instance: u64,
    ) -> Result<Self, StorageError>
    where
        Self: Sized;
}

/// Handles storing of elements
pub trait StorageInterfaceStore<Id, Element> {
    /// Saves a single element at given iteration.
    fn store_single_element(
        &mut self,
        iteration: u64,
        identifier: &Id,
        element: &Element,
    ) -> Result<(), StorageError>
    where
        Id: Serialize,
        Element: Serialize;

    /// Stores a batch of multiple elements with identifiers all at the same iteration.
    fn store_batch_elements<'a, I>(
        &'a mut self,
        iteration: u64,
        identifiers_elements: I,
    ) -> Result<(), StorageError>
    where
        Id: 'a + Serialize,
        Element: 'a + Serialize,
        I: Clone + IntoIterator<Item = (&'a Id, &'a Element)>;
}

/// Handles loading of elements
pub trait StorageInterfaceLoad<Id, Element> {
    // TODO decide if these functions should be &mut self instead of &self
    // This could be useful when implementing buffers, but maybe unnecessary.
    /// Loads a single element from the storage solution if the element exists.
    fn load_single_element(
        &self,
        iteration: u64,
        identifier: &Id,
    ) -> Result<Option<Element>, StorageError>
    where
        Id: Eq + Serialize + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>;

    /// Loads the elements history, meaning every occurrence of the element in the storage.
    /// This function by default provides the results in ordered fashion such that the time
    /// direction is retained.
    /// Furthermore this function assumes that a given index occurs over the course of a complete
    /// time segment with no interceptions.
    /// ```
    /// // All elements (given by Strings) occur over a period of time
    /// // but do not appear afterwards.
    /// use std::collections::HashMap;
    /// let valid_state = HashMap::from([
    ///     (, vec!["E1", "E2", "E3"])
    ///     (, vec!["E1", "E2", "E3", "E4"]),
    ///     (, vec!["E1", "E2", "E3", "E4"]),
    ///     (, vec!["E1", "E2", "E4"]),
    ///     (, vec!["E2", "E4"]),
    ///     (, vec!["E2", "E4", "E5"]),
    ///     (, vec!["E4", "E5"]),
    /// ]);
    /// // The entry "E2" is missing in iteration 1 but present afterwards.
    /// // This is an invalid state but will not be caught.
    /// // The backend is responsible to avoid this state.
    /// let invalid_state = HashMap::from([
    ///     (0, vec!["E1", "E2"]),
    ///     (1, vec!["E1"]),
    ///     (2, vec!["E1", "E2"]),
    /// ]);
    /// ```
    fn load_element_history(&self, identifier: &Id) -> Result<HashMap<u64, Element>, StorageError>
    where
        Id: Eq + Serialize + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        let mut iterations = self.get_all_iterations()?;
        iterations.sort();
        let mut started_gathering = false;
        let mut stop_gathering = false;
        let results = iterations
            .iter()
            .filter_map(|&iteration| {
                if stop_gathering {
                    None
                } else {
                    match self.load_single_element(iteration, identifier) {
                        Ok(Some(element)) => {
                            started_gathering = true;
                            Some(Ok((iteration, element)))
                        }
                        Ok(None) => {
                            if started_gathering {
                                stop_gathering = true;
                            }
                            None
                        }
                        Err(e) => Some(Err(e)),
                    }
                }
            })
            .collect::<Result<HashMap<u64, _>, StorageError>>()?;
        Ok(results)
    }

    /// Gets a snapshot of all elements at a given iteration.
    ///
    /// This function might be useful when implementing how simulations can be restored from saved
    /// results.
    fn load_all_elements_at_iteration(
        &self,
        iteration: u64,
    ) -> Result<HashMap<Id, Element>, StorageError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>;

    /// Get all iteration values which have been saved.
    fn get_all_iterations(&self) -> Result<Vec<u64>, StorageError>;

    /// Loads all elements for every iteration.
    /// This will yield the complete storage and may result in extremely large allocations of
    /// memory.
    fn load_all_elements(&self) -> Result<HashMap<u64, HashMap<Id, Element>>, StorageError>
    where
        Id: std::hash::Hash + std::cmp::Eq + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        let iterations = self.get_all_iterations()?;
        let all_elements = iterations
            .iter()
            .map(|iteration| {
                let elements = self.load_all_elements_at_iteration(*iteration)?;
                Ok((*iteration, elements))
            })
            .collect::<Result<HashMap<_, _>, StorageError>>()?;
        Ok(all_elements)
    }

    /// Similarly to the [load_all_elements](StorageInterfaceLoad::load_all_elements) function,
    /// but this function returns all elements as their histories.
    fn load_all_element_histories(&self) -> Result<HashMap<Id, HashMap<u64, Element>>, StorageError>
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

impl<T, Id, Element> StorageInterfaceLoad<Id, Element> for StorageWrapper<T>
where
    T: FileBasedStorage<Id, Element>,
{
    fn load_single_element(
        &self,
        iteration: u64,
        identifier: &Id,
    ) -> Result<Option<Element>, StorageError>
    where
        Id: Eq + Serialize + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        let iterations = self.get_all_iterations()?;
        if iterations.contains(&iteration) {
            // Get the path where the iteration folder is
            let iteration_path = self.0.get_iteration_path(iteration);

            // Load all elements which are inside this folder from batches and singles
            for path in std::fs::read_dir(&iteration_path)? {
                let p = path?.path();
                let file = std::fs::OpenOptions::new().read(true).open(&p)?;

                match p.file_stem() {
                    Some(stem) => match stem.to_str() {
                        Some(tail) => {
                            let first_name_segment = tail.split("_").into_iter().next();
                            if first_name_segment == Some("batch") {
                                let result: BatchSaveFormat<Id, Element> =
                                    self.0.from_reader(file)?;
                                for json_save_format in result.data.into_iter() {
                                    if &json_save_format.identifier == identifier {
                                        return Ok(Some(json_save_format.element));
                                    }
                                }
                            } else if first_name_segment == Some("single") {
                                let result: CombinedSaveFormat<Id, Element> =
                                    self.0.from_reader(file)?;
                                if &result.identifier == identifier {
                                    return Ok(Some(result.element));
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
            let iteration_path = self.0.get_iteration_path(iteration);

            // Load all elements which are inside this folder from batches and singles
            for path in std::fs::read_dir(&iteration_path)? {
                let p = path?.path();
                let file = std::fs::OpenOptions::new().read(true).open(&p)?;

                match p.file_stem() {
                    Some(stem) => match stem.to_str() {
                        Some(tail) => {
                            let first_name_segment = tail.split("_").into_iter().next();
                            if first_name_segment == Some("batch") {
                                let result: BatchSaveFormat<Id, Element> =
                                    self.0.from_reader(file)?;
                                all_elements_at_iteration.extend(result.data.into_iter().map(
                                    |json_save_format| {
                                        (json_save_format.identifier, json_save_format.element)
                                    },
                                ));
                            } else if first_name_segment == Some("single") {
                                let result: CombinedSaveFormat<Id, Element> =
                                    self.0.from_reader(file)?;
                                all_elements_at_iteration
                                    .extend([(result.identifier, result.element)]);
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
        let paths = std::fs::read_dir(&self.0.get_path())?;
        paths
            .into_iter()
            .filter_map(|path| match path {
                Ok(p) => match self.0.folder_name_to_iteration(&p.path()) {
                    Ok(Some(entry)) => Some(Ok(entry)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                },
                Err(_) => None,
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

/// Provide methods to initialize, store and load single and multiple elements at iterations.
pub trait StorageInterface<Id, Element>:
    StorageInterfaceOpen + StorageInterfaceLoad<Id, Element> + StorageInterfaceStore<Id, Element>
{
}

impl<Id, Element, T> StorageInterface<Id, Element> for T
where
    T: StorageInterfaceLoad<Id, Element>,
    T: StorageInterfaceStore<Id, Element>,
    T: StorageInterfaceOpen,
{
}
