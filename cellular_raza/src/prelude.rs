pub use cellular_raza_building_blocks::*;
pub use cellular_raza_concepts::*;
pub use cellular_raza_core::storage::*;
pub use cellular_raza_core::time::*;
pub use cellular_raza_core::*;

#[cfg(feature = "chili")]
#[cfg_attr(docsrs, doc(cfg(feature = "chili")))]
pub use cellular_raza_core::backend::chili::*;

/// See [cellular_raza_core::backend::cpu_os_threads]
///
/// Note that this backend is *NOT* compatible with the [cellular_raza_core::backend::chili].
/// In order to use this backend, import it with
/// ```
/// use cellular_raza::prelude::cpu_os_threads::*;
/// ```
#[cfg(feature = "cpu_os_threads")]
#[cfg_attr(docsrs, doc(cfg(feature = "cpu_os_threads")))]
pub mod cpu_os_threads {
    pub use cellular_raza_core::backend::cpu_os_threads::*;

    pub use cellular_raza_building_blocks::*;
    pub use cellular_raza_concepts::*;
    pub use cellular_raza_core::storage::*;
    pub use cellular_raza_core::time::*;
    pub use cellular_raza_core::*;
}

/// See [cellular_raza_core::backend::cara]
///
/// In order to use this backend, import it with
/// ```
/// use cellular_raza::prelude::cara::*;
/// ```
#[cfg(feature = "cara")]
#[cfg_attr(docsrs, doc(cfg(feature = "cara")))]
pub mod cara {
    pub use cellular_raza_core::backend::cara;

    pub use cellular_raza_building_blocks::*;
    pub use cellular_raza_concepts::*;
    pub use cellular_raza_core::storage::*;
    pub use cellular_raza_core::time::*;
    pub use cellular_raza_core::*;
}

/// See [cellular_raza_core::backend::elli]
///
/// In order to use this backend, import it with
/// ```
/// use cellular_raza::prelude::elli::*;
/// ```
#[cfg(feature = "elli")]
#[cfg_attr(docsrs, doc(cfg(feature = "elli")))]
pub mod elli {
    pub use cellular_raza_core::backend::elli;

    pub use cellular_raza_building_blocks::*;
    pub use cellular_raza_concepts::*;
    pub use cellular_raza_core::storage::*;
    pub use cellular_raza_core::time::*;
    pub use cellular_raza_core::*;
}
