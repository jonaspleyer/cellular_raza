//! Interface and methods to store and load simulation aspects.
//!
//! # Overview
//! In general, the storage solutions used can be configured with the [StorageBuilder] struct.
//! Head there to view a list of all supported options.
//! ```
//! use cellular_raza_core::storage::*;
//! let builder = StorageBuilder::new()
//!     .priority([StorageOption::SerdeJson])
//!     .location("/tmp")
//!     .add_date(true)
//!     .suffix("my_awesome_sim");
//! ```
//! Afterwards, we can provide this builder to our chosen backend which will take the information
//! contained in it and then construct a [StorageManager] to actually handle loading and storing.
//!
//! # Storage Solutions
//! We provide multiple storage options to choose from.
//!
//! ## Json
//! Relies on the [serde_json](https://docs.rs/serde_json/latest/serde_json/) crate to serialize
//! elements and store them as plain `.json` files.
//! See [JsonStorageInterface].
//!
//! ## Sled
//! Builds an embedded database at the specified location. This database is a key-value storage and
//! can be accessed via the [sled](https://docs.rs/sled/latest/sled/) crate.
//! See [SledStorageInterface]
//!
//! ## Sled (Temp)
//! Identical to the previous item but will remove the database after it has dropped.
//! This options is mostly required when performing analysis steps afterwards without saving the
//! full simulation results.
//! See [SledStorageInterface]

mod concepts;
mod memory_storage;
mod ron;
mod serde_json;
mod sled_database;

mod test;

pub use concepts::*;
pub use memory_storage::*;
pub use ron::*;
pub use serde_json::*;
pub use sled_database::*;
