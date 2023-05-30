pub use cellular_raza_building_blocks;
/// Implementations of various concepts that can be readily used. Contains cellular properties and domains.
// pub mod building_blocks;
/// Abstract concepts to describe cell properties, the domain and possible errors.
///
/// These concepts should be implemented by the user and then be used by a simulation
/// [backends](crate::backend) which actually integrates the defined cellular properties.
/// Some predefined implementations of concepts can be found in the [implementations] module.
pub use cellular_raza_concepts;
pub use cellular_raza_core;

pub mod building_blocks {
    pub use cellular_raza_building_blocks::prelude::*;
}

pub mod concepts {
    pub use cellular_raza_concepts::prelude::*;
}

pub mod core {
    pub mod backend {
        pub use cellular_raza_core::backend::cpu_os_threads;
    }
    pub mod storage {
        pub use cellular_raza_core::storage::*;
    }
}

pub mod prelude;
