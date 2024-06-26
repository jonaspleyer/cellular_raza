mod modular_cell;
#[cfg(feature = "pyo3")]
#[cfg_attr(docsrs, doc(cfg(feature = "pyo3")))]
pub mod pool_bacteria;

pub use modular_cell::*;
