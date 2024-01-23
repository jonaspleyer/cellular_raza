/// Common interface for all storage solutions.
mod concepts;

/// Save elements as xml files via [quick_xml].
mod quick_xml;

/// Save elements as json files via [serde_json].
mod serde_json;

/// Save elements in the embedded database [sled].
mod sled_database;

pub use concepts::*;
pub use quick_xml::*;
pub use serde_json::*;
pub use sled_database::*;
