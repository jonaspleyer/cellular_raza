/// This module contains a ModularCell which is used to define cell types from individual cellular
/// properties such as mechanics, reactions, etc.
///
/// The [ModularCell](modular_cell::ModularCell) is a struct with fields that implement the various
/// [concepts](crate::concepts). The concepts are afterwards derived automatically for the
/// [ModularCell](modular_cell::ModularCell) struct.
pub mod modular_cell;

#[cfg(feature = "pyo3")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "pyo3")))]
pub mod pool_bacteria;
