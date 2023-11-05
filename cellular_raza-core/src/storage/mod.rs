/// Common interface for all storage solutions.
pub mod concepts;

/// Save elements as xml files via [quick_xml].
pub mod quick_xml;

/// Save elements as json files via [serde_json].
pub mod serde_json;

/// Save elements in the embedded database [sled].
pub mod sled_database;
