mod cartesian_2d_diffusion;
mod cartesian_cuboid_n;

/// Contains deprecated cartesian cuboid implementations
// TODO #[allow(deprecated)]
#[cfg(feature = "plotters")]
#[cfg_attr(docsrs, doc(cfg(feature = "plotters")))]
pub mod cartesian_cuboid_n_old;

pub use cartesian_cuboid_n::*;

pub use cartesian_2d_diffusion::*;
