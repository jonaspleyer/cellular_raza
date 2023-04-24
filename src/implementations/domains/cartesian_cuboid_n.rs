// Imports from this crate
use crate::concepts::domain::*;
use crate::concepts::errors::*;

use crate::plotting::spatial::CreatePlottingRoot;

// Imports from std and core
use core::cmp::{max, min};

// Imports from other crates
use itertools::Itertools;
use nalgebra::SVector;

use serde::{Deserialize, Serialize};

use plotters::backend::BitMapBackend;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::DrawingArea;

/// Helper function to calculate the decomposition of a large number N into n as evenly-sized chunks as possible
/// Examples:
/// N   n   decomp
/// 10  3    1 *  4  +  3 *  3
/// 13  4    1 *  5  +  3 *  4
/// 100 13   4 * 13  +  4 * 12
/// 225 16   1 * 15  + 15 * 14
/// 225 17   4 * 14  + 13 * 13
pub(super) fn get_decomp_res(n_voxel: usize, n_regions: usize) -> Option<(usize, usize, usize)> {
    // We calculate how many times we need to drain how many voxels
    // Example:
    //      n_voxels    = 59
    //      n_regions   = 6
    //      average_len = (59 / 8).ceil() = (9.833 ...).ceil() = 10
    //
    // try to solve this equation:
    //      n_voxels = average_len * n + (average_len-1) * m
    //      where n,m are whole positive numbers
    //
    // We start with    n = n_regions = 6
    // and with         m = min(0, n_voxel - average_len.pow(2)) = min(0, 59 - 6^2) = 23
    let mut average_len: i64 = (n_voxel as f64 / n_regions as f64).ceil() as i64;

    let residue = |n: i64, m: i64, avg: i64| n_voxel as i64 - avg * n - (avg - 1) * m;

    let mut n = n_regions as i64;
    let mut m = 0;

    for _ in 0..n_regions {
        let r = residue(n, m, average_len);
        if r == 0 {
            return Some((n as usize, m as usize, average_len as usize));
        } else if r > 0 {
            if n == n_regions as i64 {
                // Start from the beginning again but with different value for average length
                average_len += 1;
                n = n_regions as i64;
                m = 0;
            } else {
                n += 1;
                m -= 1;
            }
        // Residue is negative. This means we have subtracted too much and we just decrease n and increase m
        } else {
            n -= 1;
            m += 1;
        }
    }
    None
}

// TODO use const generics instead of macros

#[macro_export]
macro_rules! define_and_implement_cartesian_cuboid {
    ($d: expr, $name: ident, $($k: expr),+) => {
        #[doc = "Cuboid Domain with regular cartesian coordinates in `"]
        #[doc = stringify!($d)]
        #[doc = "` dimensions"]
        #[derive(Clone,Debug,Serialize,Deserialize)]
        pub struct $name {
            min: [f64; $d],
            max: [f64; $d],
            n_vox: [i64; $d],
            voxel_sizes: [f64; $d],
        }


        impl $name {
            fn check_min_max(min: [f64; $d], max: [f64; $d]) -> Result<(), CalcError> {
                for i in 0..$d {
                    match max[i] > min[i] {
                        false => Err(CalcError { message: format!("Min {:?} must be smaller than Max {:?} for domain boundaries!", min, max)}),
                        true => Ok(()),
                    }?;
                }
                Ok(())
            }

            fn check_positive<F>(interaction_ranges: [F; $d]) -> Result<(), CalcError>
            where
                F: PartialOrd + num::Zero + core::fmt::Debug,
            {
                for i in 0..$d {
                    match interaction_ranges[i] > F::zero() {
                        false => Err(CalcError { message: format!("Interaction range must be positive and non-negative! Got value {:?}", interaction_ranges[i])}),
                        true => Ok(())
                    }?;
                }
                Ok(())
            }

            // TODO write this nicely!
            #[doc = "Builds a new `"]
            #[doc = stringify!($name)]
            #[doc = "` from given boundaries and maximum interaction ranges of the containing cells."]
            pub fn from_boundaries_and_interaction_ranges(min: [f64; $d], max: [f64; $d], interaction_ranges: [f64; $d]) -> Result<$name, CalcError> {
                $name::check_min_max(min, max)?;
                $name::check_positive(interaction_ranges)?;
                let mut n_vox = [0; $d];
                let mut voxel_sizes = [0.0; $d];
                for i in 0..$d {
                    n_vox[i] = ((max[i] - min[i]) / interaction_ranges[i] * 0.5).ceil() as i64;
                    voxel_sizes[i] = (max[i]-min[i])/n_vox[i] as f64;
                }
                Ok($name {
                    min,
                    max,
                    n_vox,
                    voxel_sizes,
                })
            }

            #[doc = "Builds a new `"]
            #[doc = stringify!($name)]
            #[doc = "` from given boundaries and the number of voxels per dimension specified."]
            pub fn from_boundaries_and_n_voxels(min: [f64; $d], max: [f64; $d], n_vox: [usize; $d]) -> Result<$name, CalcError> {
                $name::check_min_max(min, max)?;
                $name::check_positive(n_vox)?;
                let mut voxel_sizes = [0.0; $d];
                for i in 0..$d {
                    voxel_sizes[i] = (max[i] - min[i]) / n_vox[i] as f64;
                }
                Ok($name {
                    min,
                    max,
                    n_vox: [$(n_vox[$k] as i64),+],
                    voxel_sizes,
                })
            }
        }
    }
}

macro_rules! define_and_implement_cartesian_cuboid_voxel{
    ($d: expr, $n_reactions:expr, $name: ident, $voxel_name: ident, $($k: expr),+) => {
        // Define the struct for the voxel
        #[doc = "Cuboid Voxel for `"]
        #[doc = stringify!($name)]
        #[doc = "` in `"]
        #[doc = stringify!($d)]
        #[doc = "` dimensions"]
        #[derive(Clone,Debug,Serialize,Deserialize)]
        pub struct $voxel_name {
                min: [f64; $d],
                max: [f64; $d],
                middle: [f64; $d],
                dx: [f64; $d],
                index: [i64; $d],

                pub extracellular_concentrations: SVector<f64, $n_reactions>,
                pub extracellular_gradient: SVector<SVector<f64, $d>, $n_reactions>,
                pub diffusion_constant: SVector<f64, $n_reactions>,
                pub production_rate: SVector<f64, $n_reactions>,
                pub degradation_rate: SVector<f64, $n_reactions>,
                domain_boundaries: Vec<([i64; $d], BoundaryCondition<SVector<f64, $n_reactions>>)>,
        }

        impl $voxel_name {
            pub(crate) fn new(min: [f64; $d], max: [f64; $d], index: [i64; $d], domain_boundaries: Vec<([i64; $d], BoundaryCondition<SVector<f64, $n_reactions>>)>) -> $voxel_name {
                let middle = [$((max[$k] + min[$k])/2.0),+];
                let dx = [$(max[$k]-min[$k]),+];
                $voxel_name {
                    min,
                    max,
                    middle,
                    dx,
                    index,
                    extracellular_concentrations: SVector::<f64, $n_reactions>::from_element(0.0),
                    extracellular_gradient: SVector::<SVector<f64, $d>, $n_reactions>::from_element(SVector::<f64, $d>::from_element(0.0)),
                    diffusion_constant: SVector::<f64, $n_reactions>::from_element(0.0),
                    production_rate: SVector::<f64, $n_reactions>::from_element(0.0),
                    degradation_rate: SVector::<f64, $n_reactions>::from_element(0.0),
                    domain_boundaries,
                }
            }

            pub fn get_min(&self) -> [f64; $d] {self.min}
            pub fn get_max(&self) -> [f64; $d] {self.max}
            pub fn get_middle(&self) -> [f64; $d] {self.middle}
            pub fn get_dx(&self) -> [f64; $d] {self.dx}

            fn position_is_in_domain(&self, pos: &SVector<f64, $d>) -> Result<(), RequestError> {
                match pos.iter().enumerate().any(|(i, p)| !(self.min[i] <= *p && *p <= self.max[i])) {
                    true => Err(RequestError{ message: format!("point {:?} is not in requested voxel with boundaries {:?} {:?}", pos, self.min, self.max)}),
                    false => Ok(()),
                }
            }

            fn index_to_distance_squared(&self, index: &[i64; $d]) -> f64 {
                let mut diffs = [0; $d];
                for i in 0..$d {
                    diffs[i] = (index[i] as i32 - self.index[i] as i32).abs()
                }
                diffs.iter().enumerate().map(|(i, d)| self.dx[i].powf(2.0)* (*d as f64)).sum::<f64>()
            }
        }

        // Implement the Voxel trait for our n-dim voxel
        impl Voxel<[i64; $d], SVector<f64, $d>, SVector<f64, $d>> for $voxel_name {
            fn get_index(&self) -> [i64; $d] {
                self.index
            }
        }

        impl ExtracellularMechanics<[i64; $d], SVector<f64, $d>, SVector<f64, $n_reactions>, SVector<SVector<f64, $d>, $n_reactions>, SVector<f64, $n_reactions>, SVector<f64, $n_reactions>> for $voxel_name {
            fn get_extracellular_at_point(&self, pos: &SVector<f64, $d>) -> Result<SVector<f64, $n_reactions>, SimulationError> {
                self.position_is_in_domain(pos)?;
                Ok(self.extracellular_concentrations)
            }

            fn get_total_extracellular(&self) -> SVector<f64, $n_reactions> {
                self.extracellular_concentrations
            }

            #[cfg(feature = "gradients")]
            fn update_extracellular_gradient(&mut self, boundaries: &[([i64; $d], BoundaryCondition<SVector<f64, $n_reactions>>)]) -> Result<(), SimulationError> {
                let mut new_gradient = SVector::<SVector<f64, $d>, $n_reactions>::from_element(SVector::<f64, $d>::from_element(0.0));
                boundaries.iter()
                    .for_each(|(index, boundary_condition)| {
                        let extracellular_difference = match boundary_condition {
                            BoundaryCondition::Neumann(value) => {*value},
                            BoundaryCondition::Dirichlet(value) => {self.extracellular_concentrations-value},
                            BoundaryCondition::Value(value) => {self.extracellular_concentrations-value},
                        };
                        let pointer = SVector::from([$(self.index[$k] as f64 - index[$k] as f64),+]);
                        let dist = pointer.norm();
                        let gradient = pointer.normalize()/dist;
                        new_gradient.iter_mut().zip(extracellular_difference.into_iter()).for_each(|(component, diff)| *component += *diff*gradient);
                        // let total_gradient = SVector::<SVector<f64,$d>,$n_reactions>::from_iterator(extracellular_difference.into_iter().map(|diff| *diff*gradient));
                        // gradient += total_gradient;
                    });
                self.extracellular_gradient = new_gradient;
                Ok(())
            }

            #[cfg(feature = "gradients")]
            fn get_extracellular_gradient_at_point(&self, _pos: &SVector<f64, $d>) -> Result<SVector<SVector<f64, $d>, $n_reactions>, SimulationError> {
                Ok(self.extracellular_gradient)
            }

            fn set_total_extracellular(&mut self, concentrations: &SVector<f64, $n_reactions>) -> Result<(), CalcError> {
                Ok(self.extracellular_concentrations = *concentrations)
            }

            fn calculate_increment(&self, total_extracellular: &SVector<f64, $n_reactions>, point_sources: &[(SVector<f64, $d>, SVector<f64, $n_reactions>)], boundaries: &[([i64; $d], BoundaryCondition<SVector<f64, $n_reactions>>)]) -> Result<SVector<f64, $n_reactions>, CalcError> {
                let mut inc = SVector::<f64, $n_reactions>::from_element(0.0);

                self.domain_boundaries
                    .iter()
                    .for_each(|(index, boundary)| match boundary {
                        BoundaryCondition::Neumann(value) => inc += value / self.index_to_distance_squared(index).sqrt(),
                        BoundaryCondition::Dirichlet(value) => inc += (value-total_extracellular) / self.index_to_distance_squared(index),
                        BoundaryCondition::Value(value) => inc += (value-total_extracellular) / self.index_to_distance_squared(index),
                    });

                boundaries.iter()
                    .for_each(|(index, boundary)| match boundary {
                        BoundaryCondition::Neumann(value) => inc += value / self.index_to_distance_squared(&index).sqrt(),
                        BoundaryCondition::Dirichlet(value) => inc += (value-total_extracellular) / self.index_to_distance_squared(&index),
                        BoundaryCondition::Value(value) => inc += (value-total_extracellular) / self.index_to_distance_squared(&index),
                    });
                inc = inc.component_mul(&self.diffusion_constant);

                point_sources.iter()
                    .for_each(|(_, value)| inc += value);

                // Also calculate internal reactions. Here it is very simple only given by degradation and production.
                inc += self.production_rate - self.degradation_rate.component_mul(&total_extracellular);
                Ok(inc)
            }

            fn boundary_condition_to_neighbor_voxel(&self, _neighbor_index: &[i64; $d]) -> Result<BoundaryCondition<SVector<f64, $n_reactions>>, IndexError> {
                Ok(BoundaryCondition::Value(self.extracellular_concentrations))
            }
        }

        // Implement the cartesian cuboid
        // Index is an array of size 3 with elements of type usize
        impl<C> Domain<C, [i64; $d], $voxel_name> for $name
        // Position, Force and Velocity are all Vector$d supplied by the Nalgebra crate
        where C: crate::concepts::mechanics::Mechanics<SVector<f64, $d>, SVector<f64, $d>, SVector<f64, $d>>,
        {
            fn apply_boundary(&self, cell: &mut C) -> Result<(),BoundaryError> {
                let mut pos = cell.pos();
                let mut velocity = cell.velocity();

                // For each dimension
                for i in 0..$d {
                    // Check if the particle is below lower edge
                    if pos[i] < self.min[i] {
                        pos[i] = 2.0 * self.min[i] - pos[i];
                        velocity[i] = velocity[i].abs();
                    }
                    // Check if the particle is over the edge
                    if pos[i] > self.max[i] {
                        pos[i] = 2.0 * self.max[i] - pos[i];
                        velocity[i] = - velocity[i].abs();
                    }
                }
                // Set new position and velocity of particle
                cell.set_pos(&pos);
                cell.set_velocity(&velocity);

                // If new position is still out of boundary return error
                for i in 0..$d {
                    if pos[i] < self.min[i] || pos[i] > self.max[i] {
                        return Err(BoundaryError { message: format!("Particle is out of domain at position {:?}", pos) });
                    }
                }
                Ok(())
            }

            fn get_voxel_index(&self, cell: &C) -> [i64; $d] {
                let p = cell.pos();
                let mut out = [0; $d];

                for i in 0..$d {
                    out[i] = ((p[i] - self.min[0]) / self.voxel_sizes[i]) as i64;
                    out[i] = out[i].min(self.n_vox[i]-1).max(0);
                }
                return out;
            }

            fn get_all_indices(&self) -> Vec<[i64; $d]> {
                [$($k),+].iter()
                    .map(|i| (0..self.n_vox[*i]))
                    .multi_cartesian_product()
                    .map(|ind_v| [$(ind_v[$k]),+])
                    .collect()
            }

            fn get_neighbor_voxel_indices(&self, index: &[i64; $d]) -> Vec<[i64; $d]> {
                // Create the bounds for the following creation of all the voxel indices
                let bounds: [[i64; 2]; $d] = [$(
                    [
                        max(index[$k] as i32 - 1, 0) as i64,
                        min(index[$k]+2, self.n_vox[$k])
                    ]
                ),+];

                // Create voxel indices
                let v: Vec<[i64; $d]> = [$($k),+].iter()      // indices supplied in macro invokation
                    .map(|i| (bounds[*i][0]..bounds[*i][1]))    // ranges from bounds
                    .multi_cartesian_product()                  // all possible combinations
                    .map(|ind_v| [$(ind_v[$k]),+])              // multi_cartesian_product gives us vector elements. We map them to arrays.
                    .filter(|ind| ind!=index)                   // filter the elements such that the current index is not included.
                    .collect();                                 // collect into the correct type

                return v;
            }

            fn generate_contiguous_multi_voxel_regions(&self, n_regions: usize) -> Result<(usize, Vec<Vec<([i64; $d], $voxel_name)>>), CalcError> {
                // Get all voxel indices
                let indices: Vec<[i64; $d]> = [$($k),+]
                    .iter()                                     // indices supplied in macro invokation
                    .map(|i| (0..self.n_vox[*i]))               // ranges from self.n_vox
                    .multi_cartesian_product()                  // all possible combinations
                    .map(|ind_v| [$(ind_v[$k]),+])              // multi_cartesian_product gives us vector elements. We map them to arrays.
                    .collect();

                let (n, _m, average_len);
                match get_decomp_res(indices.len(), n_regions) {
                    Some(res) => (n, _m, average_len) = res,
                    None => return Err(CalcError {message: "Could not find a suiting decomposition".to_owned(), }),
                };

                // Now we drain the indices vector
                let mut index_voxel_combinations: Vec<([i64; $d], $voxel_name)> = indices
                    .into_iter()
                    .map(|ind| {
                        let min = [$(self.min[$k] +    ind[$k]  as f64*self.voxel_sizes[$k]),+];
                        let max = [$(self.min[$k] + (1+ind[$k]) as f64*self.voxel_sizes[$k]),+];
                        // TODO FIXUP we need to insert boundary conditions here as last argument
                        let domain_boundaries = (0..$d)
                            .map(|_| (-1_i64..2_i64))
                            .multi_cartesian_product()
                            .map(|v| [$(ind[$k] + v[$k]),+])
                            .filter(|new_index| *new_index != ind)
                            .filter(|new_index| new_index.iter().zip(self.n_vox.iter()).any(|(i1, i2)| *i1<0 || i2<=i1))
                            .map(|new_index| (new_index, BoundaryCondition::Neumann(SVector::<f64, $n_reactions>::from_element(0.0))))
                            .collect::<Vec<_>>();
                        (ind, $voxel_name::new(min, max, ind, domain_boundaries))
                    })
                    .collect();

                // TODO optimize this!
                // Currently we are not splitting the voxels apart efficiently
                let mut ind_n: Vec<Vec<_>> = index_voxel_combinations
                    .drain(0..(average_len*n) as usize)
                    .into_iter()
                    .chunks(average_len as usize)
                    .into_iter()
                    .map(|chunk| chunk.collect::<Vec<_>>())
                    .collect();

                let mut ind_m: Vec<Vec<_>> = index_voxel_combinations
                    .drain(..)
                    .into_iter()
                    .chunks((max(average_len-1, 1)) as usize)
                    .into_iter()
                    .map(|chunk| chunk.collect::<Vec<_>>())
                    .collect();

                ind_n.append(&mut ind_m);

                Ok((n_regions as usize, ind_n))
            }
        }
    }
}

// TODO reformulate definition with const generics
// TODO make them only visible if correspoding feature (eg. fluid_mechanics or gradients) is active
define_and_implement_cartesian_cuboid!(1, CartesianCuboid1, 0);
define_and_implement_cartesian_cuboid_voxel!(
    1,
    1,
    CartesianCuboid1,
    CartesianCuboidVoxel1Reactions1,
    0
);
define_and_implement_cartesian_cuboid_voxel!(
    1,
    2,
    CartesianCuboid1,
    CartesianCuboidVoxel1Reactions2,
    0
);
define_and_implement_cartesian_cuboid_voxel!(
    1,
    3,
    CartesianCuboid1,
    CartesianCuboidVoxel1Reactions3,
    0
);
define_and_implement_cartesian_cuboid_voxel!(
    1,
    4,
    CartesianCuboid1,
    CartesianCuboidVoxel1Reactions4,
    0
);
define_and_implement_cartesian_cuboid_voxel!(
    1,
    5,
    CartesianCuboid1,
    CartesianCuboidVoxel1Reactions5,
    0
);
define_and_implement_cartesian_cuboid_voxel!(
    1,
    6,
    CartesianCuboid1,
    CartesianCuboidVoxel1Reactions6,
    0
);
define_and_implement_cartesian_cuboid_voxel!(
    1,
    7,
    CartesianCuboid1,
    CartesianCuboidVoxel1Reactions7,
    0
);
define_and_implement_cartesian_cuboid_voxel!(
    1,
    8,
    CartesianCuboid1,
    CartesianCuboidVoxel1Reactions8,
    0
);
define_and_implement_cartesian_cuboid_voxel!(
    1,
    9,
    CartesianCuboid1,
    CartesianCuboidVoxel1Reactions9,
    0
);

define_and_implement_cartesian_cuboid!(2, CartesianCuboid2, 0, 1);
define_and_implement_cartesian_cuboid_voxel!(
    2,
    1,
    CartesianCuboid2,
    CartesianCuboidVoxel2Reactions1,
    0,
    1
);
define_and_implement_cartesian_cuboid_voxel!(
    2,
    2,
    CartesianCuboid2,
    CartesianCuboidVoxel2Reactions2,
    0,
    1
);
define_and_implement_cartesian_cuboid_voxel!(
    2,
    3,
    CartesianCuboid2,
    CartesianCuboidVoxel2Reactions3,
    0,
    1
);
define_and_implement_cartesian_cuboid_voxel!(
    2,
    4,
    CartesianCuboid2,
    CartesianCuboidVoxel2Reactions4,
    0,
    1
);
define_and_implement_cartesian_cuboid_voxel!(
    2,
    5,
    CartesianCuboid2,
    CartesianCuboidVoxel2Reactions5,
    0,
    1
);
define_and_implement_cartesian_cuboid_voxel!(
    2,
    6,
    CartesianCuboid2,
    CartesianCuboidVoxel2Reactions6,
    0,
    1
);
define_and_implement_cartesian_cuboid_voxel!(
    2,
    7,
    CartesianCuboid2,
    CartesianCuboidVoxel2Reactions7,
    0,
    1
);
define_and_implement_cartesian_cuboid_voxel!(
    2,
    8,
    CartesianCuboid2,
    CartesianCuboidVoxel2Reactions8,
    0,
    1
);
define_and_implement_cartesian_cuboid_voxel!(
    2,
    9,
    CartesianCuboid2,
    CartesianCuboidVoxel2Reactions9,
    0,
    1
);

define_and_implement_cartesian_cuboid!(3, CartesianCuboid3, 0, 1, 2);
define_and_implement_cartesian_cuboid_voxel!(
    3,
    1,
    CartesianCuboid3,
    CartesianCuboidVoxel3Reactions1,
    0,
    1,
    2
);
define_and_implement_cartesian_cuboid_voxel!(
    3,
    2,
    CartesianCuboid3,
    CartesianCuboidVoxel3Reactions2,
    0,
    1,
    2
);
define_and_implement_cartesian_cuboid_voxel!(
    3,
    3,
    CartesianCuboid3,
    CartesianCuboidVoxel3Reactions3,
    0,
    1,
    2
);
define_and_implement_cartesian_cuboid_voxel!(
    3,
    4,
    CartesianCuboid3,
    CartesianCuboidVoxel3Reactions4,
    0,
    1,
    2
);
define_and_implement_cartesian_cuboid_voxel!(
    3,
    5,
    CartesianCuboid3,
    CartesianCuboidVoxel3Reactions5,
    0,
    1,
    2
);
define_and_implement_cartesian_cuboid_voxel!(
    3,
    6,
    CartesianCuboid3,
    CartesianCuboidVoxel3Reactions6,
    0,
    1,
    2
);
define_and_implement_cartesian_cuboid_voxel!(
    3,
    7,
    CartesianCuboid3,
    CartesianCuboidVoxel3Reactions7,
    0,
    1,
    2
);
define_and_implement_cartesian_cuboid_voxel!(
    3,
    8,
    CartesianCuboid3,
    CartesianCuboidVoxel3Reactions8,
    0,
    1,
    2
);
define_and_implement_cartesian_cuboid_voxel!(
    3,
    9,
    CartesianCuboid3,
    CartesianCuboidVoxel3Reactions9,
    0,
    1,
    2
);

impl CreatePlottingRoot for CartesianCuboid2 {
    fn create_bitmap_root<'a, T>(
        &self,
        image_size: u32,
        filename: &'a T,
    ) -> Result<
        DrawingArea<BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
        DrawingError,
    >
    where
        T: AsRef<std::path::Path> + ?Sized,
    {
        use plotters::drawing::IntoDrawingArea;
        // let root = plotters::backend::BitMapBackend::new(filename, (image_size, image_size)).into_drawing_area();
        let dx = (self.max[0] - self.min[0]).abs();
        let dy = (self.max[1] - self.min[1]).abs();
        let q = dx.min(dy);
        let image_size_x = (image_size as f64 * dx / q).round() as u32;
        let image_size_y = (image_size as f64 * dy / q).round() as u32;
        let root = BitMapBackend::new(filename, (image_size_x, image_size_y)).into_drawing_area();
        root.fill(&plotters::prelude::full_palette::WHITE).unwrap();

        let label_space = (0.05 * image_size as f64).round() as u32;
        use plotters::prelude::IntoFont;
        let pos = plotters::style::text_anchor::Pos::new(
            plotters::style::text_anchor::HPos::Center,
            plotters::style::text_anchor::VPos::Center,
        );
        let label_style = plotters::prelude::TextStyle::from(
            ("sans-serif", (0.02 * image_size as f64).round() as u32).into_font(),
        )
        .color(&plotters::prelude::BLACK)
        .pos(pos);

        // Draw legend
        let voxel_pixel_size_x =
            ((image_size_x - 2 * label_space) as f64 / self.n_vox[0] as f64).round() as i32;
        let voxel_pixel_size_y =
            ((image_size_y - 2 * label_space) as f64 / self.n_vox[1] as f64).round() as i32;
        let xy0 = (label_space as f64 * 0.5).round() as i32;

        let create_element = |index: usize, i: usize, pos: (i32, i32)| {
            plotters::prelude::Text::new(
                format!(
                    "{:.0}",
                    self.min[index] + i as f64 * self.voxel_sizes[index]
                ),
                pos,
                label_style.clone(),
            )
        };

        let step_x = max(1, ((self.n_vox[0] + 1) as f64 / 10.0).floor() as usize);
        let step_y = max(1, ((self.n_vox[1] + 1) as f64 / 10.0).floor() as usize);
        // Draw descriptions along x axis
        (0..self.n_vox[0] as usize + 1)
            .filter(|i| i % step_x == 0)
            .for_each(|i| {
                let element_top = create_element(
                    0,
                    i,
                    (label_space as i32 + i as i32 * voxel_pixel_size_x, xy0),
                );
                let element_bot = create_element(
                    0,
                    i,
                    (
                        label_space as i32 + i as i32 * voxel_pixel_size_x,
                        image_size_y as i32 - xy0,
                    ),
                );

                root.draw(&element_top).unwrap();
                root.draw(&element_bot).unwrap();
            });

        // Draw descriptions along y axis
        (0..self.n_vox[1] as usize + 1)
            .filter(|j| j % step_y == 0)
            .for_each(|j| {
                let element_left = create_element(
                    1,
                    j,
                    (xy0, label_space as i32 + j as i32 * voxel_pixel_size_y),
                );
                let element_right = create_element(
                    1,
                    j,
                    (
                        image_size_x as i32 - xy0,
                        label_space as i32 + j as i32 * voxel_pixel_size_y,
                    ),
                );

                root.draw(&element_left).unwrap();
                root.draw(&element_right).unwrap();
            });

        let mut chart = plotters::prelude::ChartBuilder::on(&root)
            .margin(label_space)
            // Finally attach a coordinate on the drawing area and make a chart context
            .build_cartesian_2d(self.min[0]..self.max[0], self.min[1]..self.max[1])
            .unwrap();

        chart
            .configure_mesh()
            // we do not want to draw any mesh lines automatically but do this manually below
            .disable_mesh()
            .draw()
            .unwrap();

        // Draw vertical lines manually
        for i in 0..self.n_vox[0] + 1 {
            let element = plotters::prelude::LineSeries::new(
                [
                    (self.min[0] + i as f64 * self.voxel_sizes[0], self.min[1]),
                    (self.min[0] + i as f64 * self.voxel_sizes[0], self.max[1]),
                ],
                plotters::prelude::BLACK,
            );
            chart.draw_series(element).unwrap();
        }

        // Draw horizontal lines manually
        for i in 0..self.n_vox[1] + 1 {
            let element = plotters::prelude::LineSeries::new(
                [
                    (self.min[0], self.min[1] + i as f64 * self.voxel_sizes[1]),
                    (self.max[0], self.min[1] + i as f64 * self.voxel_sizes[1]),
                ],
                plotters::prelude::BLACK,
            );
            chart.draw_series(element)?;
        }

        Ok(chart.plotting_area().clone())
    }
}

#[cfg(test)]
mod test {
    use super::get_decomp_res;
    use rayon::prelude::*;

    #[test]
    fn test_get_demomp_res() {
        #[cfg(not(feature = "test_exhaustive"))]
        let max = 5_000;
        #[cfg(feature = "test_exhaustive")]
        let max = 5_000_000;

        (1..max)
            .into_par_iter()
            .map(|n_voxel| {
                for n_regions in 1..1_000 {
                    match get_decomp_res(n_voxel, n_regions) {
                        Some(res) => {
                            let (n, m, average_len) = res;
                            assert_eq!(n + m, n_regions);
                            assert_eq!(n * average_len + m * (average_len - 1), n_voxel);
                        }
                        None => panic!(
                            "No result for inputs n_voxel: {} n_regions: {}",
                            n_voxel, n_regions
                        ),
                    }
                }
            })
            .collect::<Vec<()>>();
    }
}
