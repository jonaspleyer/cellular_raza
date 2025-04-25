mod cartesian_2d_diffusion;
mod cartesian_cuboid_n;

/// Contains deprecated cartesian cuboid implementations for an older vertex model
// TODO #[allow(deprecated)]
pub mod cartesian_cuboid_2_vertex_old;

/// Contains deprecated cartesian cuboid implementations
// TODO #[allow(deprecated)]
pub mod cartesian_cuboid_n_old;

pub use cartesian_cuboid_n::*;

pub use cartesian_2d_diffusion::*;
