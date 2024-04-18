use std::collections::HashMap;
use std::error::Error;
use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::quick_xml::XmlStorageInterface;
use super::serde_json::JsonStorageInterface;
use super::sled_database::SledStorageInterface;

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

// TODO implement this correctly
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
    SerdeXml,
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
    pub(super) identifier: Id,
    pub(super) element: Element,
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
    json_storage: Option<JsonStorageInterface<Id, Element>>,
    xml_storage: Option<XmlStorageInterface<Id, Element>>,
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
        let mut xml_storage = None;
        for storage_variant in storage_builder.priority.iter() {
            match storage_variant {
                StorageOption::SerdeJson => {
                    json_storage = Some(JsonStorageInterface::<Id, Element>::open_or_create(
                        &location.to_path_buf().join("json"),
                        instance,
                    )?);
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
                StorageOption::SerdeXml => {
                    xml_storage = Some(XmlStorageInterface::<Id, Element>::open_or_create(
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
            xml_storage,
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
    ($self:ident, $priority:ident, $function:ident, $($args:tt)*) => {
        match $priority {
            StorageOption::Sled => exec_for_all_storage_options!(@internal $self, Sled, sled_storage, $function, $($args)*),
            StorageOption::SledTemp => exec_for_all_storage_options!(@internal $self, SledTemp, sled_temp_storage, $function, $($args)*),
            StorageOption::SerdeJson => exec_for_all_storage_options!(@internal $self, SerdeJson, json_storage, $function, $($args)*),
            StorageOption::SerdeXml => exec_for_all_storage_options!(@internal $self, SerdeXml, xml_storage, $function, $($args)*),
        }
    }
);

impl<Id, Element> StorageInterfaceStore<Id, Element> for StorageManager<Id, Element> {
    #[allow(unused)]
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
        if let Some(sled_storage) = &self.sled_storage {
            sled_storage.store_single_element(iteration, identifier, element)?;
        }

        if let Some(json_storage) = &self.json_storage {
            json_storage.store_single_element(iteration, identifier, element)?;
        }

        if let Some(xml_storage) = &self.xml_storage {
            xml_storage.store_single_element(iteration, identifier, element)?;
        }
        Ok(())
    }

    #[allow(unused)]
    fn store_batch_elements<'a, I>(
        &'a self,
        iteration: u64,
        identifiers_elements: I,
    ) -> Result<(), StorageError>
    where
        Id: 'a + Serialize,
        Element: 'a + Serialize,
        I: Clone + IntoIterator<Item = (&'a Id, &'a Element)>,
    {
        if let Some(sled_storage) = &self.sled_storage {
            sled_storage.store_batch_elements(iteration, identifiers_elements.clone())?;
        }

        if let Some(json_storage) = &self.json_storage {
            json_storage.store_batch_elements(iteration, identifiers_elements.clone())?;
        }

        if let Some(xml_storage) = &self.xml_storage {
            xml_storage.store_batch_elements(iteration, identifiers_elements)?;
        }
        Ok(())
    }
}

impl<Id, Element> StorageInterfaceLoad<Id, Element> for StorageManager<Id, Element> {
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

/// Open or create a new instance of the Storage controller.
pub trait StorageInterfaceOpen<Id, Element> {
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
        &self,
        iteration: u64,
        identifier: &Id,
        element: &Element,
    ) -> Result<(), StorageError>
    where
        Id: Serialize,
        Element: Serialize;

    /// Stores a batch of multiple elements with identifiers all at the same iteration.
    fn store_batch_elements<'a, I>(
        &'a self,
        iteration: u64,
        identifiers_elements: I,
    ) -> Result<(), StorageError>
    where
        Id: 'a + Serialize,
        Element: 'a + Serialize,
        I: Clone + IntoIterator<Item = (&'a Id, &'a Element)>,
    {
        identifiers_elements
            .into_iter()
            .map(|(id, element)| self.store_single_element(iteration, id, element))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }
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
        Id: Serialize + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>;

    /// Loads the elements history, meaning every occurrence of the element in the storage.
    /// This function should provide the results in ordered fashion such that the time
    /// direction is retained.
    fn load_element_history(
        &self,
        identifier: &Id,
    ) -> Result<Option<HashMap<u64, Element>>, StorageError>
    where
        Id: Serialize + for<'a> Deserialize<'a>,
        Element: for<'a> Deserialize<'a>,
    {
        let results = self
            .get_all_iterations()?
            .iter()
            .filter_map(
                |&iteration| match self.load_single_element(iteration, identifier) {
                    Ok(Some(element)) => Some(Ok((iteration, element))),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                },
            )
            .collect::<Result<HashMap<u64, _>, StorageError>>()?;
        Ok(if results.len() == 0 {
            None
        } else {
            Some(results)
        })
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

    /// Similarly to the [load_all_elements](StorageInterface::load_all_elements) function,
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

/// Provide methods to initialize, store and load single and multiple elements at iterations.
pub trait StorageInterface<Id, Element>:
    StorageInterfaceOpen<Id, Element>
    + StorageInterfaceLoad<Id, Element>
    + StorageInterfaceStore<Id, Element>
{
}

impl<Id, Element, T> StorageInterface<Id, Element> for T
where
    T: StorageInterfaceLoad<Id, Element>,
    T: StorageInterfaceStore<Id, Element>,
    T: StorageInterfaceOpen<Id, Element>,
{
}
