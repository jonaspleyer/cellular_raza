// Imports from this crate
use cellular_raza_concepts::domain::*;
use cellular_raza_concepts::errors::*;
use cellular_raza_concepts::plotting::*;

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
// This can only be done once serde supports deriving Serialize and Deserialize for structs with const generics
// Otherwise we would have to implement it by hand.
// Sadly this is currently not possible ...
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
                        false => Err(CalcError(format!("Min {:?} must be smaller than Max {:?} for domain boundaries!", min, max))),
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
                        false => Err(CalcError(format!("Interaction range must be positive and non-negative! Got value {:?}", interaction_ranges[i]))),
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

macro_rules! implement_cartesian_cuboid_voxel_fluid_mechanics{
    ($d: literal, $name: ident, $voxel_name: ident, $($k: expr),+) => {
        // Define the struct for the voxel
        #[doc = "Cuboid Voxel for `"]
        #[doc = stringify!($name)]
        #[doc = "` in `"]
        #[doc = stringify!($d)]
        #[doc = "` dimensions"]
        #[derive(Clone,Debug,Serialize,Deserialize)]
        pub struct $voxel_name<const N: usize> {
                min: [f64; $d],
                max: [f64; $d],
                middle: [f64; $d],
                dx: [f64; $d],
                index: [i64; $d],

                pub extracellular_concentrations: SVector<f64, N>,
                #[cfg(feature = "gradients")]
                pub extracellular_gradient: SVector<SVector<f64, $d>, N>,
                pub diffusion_constant: SVector<f64, N>,
                pub production_rate: SVector<f64, N>,
                pub degradation_rate: SVector<f64, N>,
                domain_boundaries: Vec<([i64; $d], BoundaryCondition<SVector<f64, N>>)>,
        }

        impl<const N: usize> $voxel_name<N> {
            pub(crate) fn new(min: [f64; $d], max: [f64; $d], index: [i64; $d], domain_boundaries: Vec<([i64; $d], BoundaryCondition<SVector<f64, N>>)>) -> $voxel_name<N> {
                let middle = [$((max[$k] + min[$k])/2.0),+];
                let dx = [$(max[$k]-min[$k]),+];
                $voxel_name::<N> {
                    min,
                    max,
                    middle,
                    dx,
                    index,
                    extracellular_concentrations: SVector::<f64, N>::from_element(0.0),
                    #[cfg(feature = "gradients")]
                    extracellular_gradient: SVector::<SVector<f64, $d>, N>::from_element(SVector::<f64, $d>::from_element(0.0)),
                    diffusion_constant: SVector::<f64, N>::from_element(0.0),
                    production_rate: SVector::<f64, N>::from_element(0.0),
                    degradation_rate: SVector::<f64, N>::from_element(0.0),
                    domain_boundaries,
                }
            }

            pub fn get_min(&self) -> [f64; $d] {self.min}
            pub fn get_max(&self) -> [f64; $d] {self.max}
            pub fn get_middle(&self) -> [f64; $d] {self.middle}
            pub fn get_dx(&self) -> [f64; $d] {self.dx}

            fn position_is_in_domain(&self, pos: &SVector<f64, $d>) -> Result<(), RequestError> {
                match pos.iter().enumerate().any(|(i, p)| !(self.min[i] <= *p && *p <= self.max[i])) {
                    true => Err(RequestError(format!("point {:?} is not in requested voxel with boundaries {:?} {:?}", pos, self.min, self.max))),
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
        impl<const N: usize> Voxel<[i64; $d], SVector<f64, $d>, SVector<f64, $d>, SVector<f64, $d>> for $voxel_name<N> {
            fn get_index(&self) -> [i64; $d] {
                self.index
            }
        }

        impl<const N: usize> ExtracellularMechanics<[i64; $d], SVector<f64, $d>, SVector<f64, N>, SVector<SVector<f64, $d>, N>, SVector<f64, N>, SVector<f64, N>> for $voxel_name<N> {
            fn get_extracellular_at_point(&self, pos: &SVector<f64, $d>) -> Result<SVector<f64, N>, RequestError> {
                self.position_is_in_domain(pos)?;
                Ok(self.extracellular_concentrations)
            }

            fn get_total_extracellular(&self) -> SVector<f64, N> {
                self.extracellular_concentrations
            }

            #[cfg(feature = "gradients")]
            fn update_extracellular_gradient(&mut self, boundaries: &[([i64; $d], BoundaryCondition<SVector<f64, N>>)]) -> Result<(), CalcError> {
                let mut new_gradient = SVector::<SVector<f64, $d>, N>::from_element(SVector::<f64, $d>::from_element(0.0));
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
                        // let total_gradient = SVector::<SVector<f64,$d>,N>::from_iterator(extracellular_difference.into_iter().map(|diff| *diff*gradient));
                        // gradient += total_gradient;
                    });
                self.extracellular_gradient = new_gradient;
                Ok(())
            }

            #[cfg(feature = "gradients")]
            fn get_extracellular_gradient_at_point(&self, _pos: &SVector<f64, $d>) -> Result<SVector<SVector<f64, $d>, N>, RequestError> {
                Ok(self.extracellular_gradient)
            }

            fn set_total_extracellular(&mut self, concentrations: &SVector<f64, N>) -> Result<(), CalcError> {
                Ok(self.extracellular_concentrations = *concentrations)
            }

            fn calculate_increment(&self, total_extracellular: &SVector<f64, N>, point_sources: &[(SVector<f64, $d>, SVector<f64, N>)], boundaries: &[([i64; $d], BoundaryCondition<SVector<f64, N>>)]) -> Result<SVector<f64, N>, CalcError> {
                let mut inc = SVector::<f64, N>::from_element(0.0);

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

            fn boundary_condition_to_neighbor_voxel(&self, _neighbor_index: &[i64; $d]) -> Result<BoundaryCondition<SVector<f64, N>>, IndexError> {
                Ok(BoundaryCondition::Value(self.extracellular_concentrations))
            }
        }

        // Implement the cartesian cuboid
        // Index is an array of size 3 with elements of type usize
        impl<Cel, const N: usize> Domain<Cel, [i64; $d], $voxel_name<N>> for $name
        // Position, Force and Velocity are all Vector$d supplied by the Nalgebra crate
        where Cel: cellular_raza_concepts::mechanics::Mechanics<SVector<f64, $d>, SVector<f64, $d>, SVector<f64, $d>>,
        {
            fn apply_boundary(&self, cell: &mut Cel) -> Result<(),BoundaryError> {
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
                        return Err(BoundaryError(format!("Particle is out of domain at position {:?}", pos)));
                    }
                }
                Ok(())
            }

            fn get_voxel_index(&self, cell: &Cel) -> [i64; $d] {
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

            fn generate_contiguous_multi_voxel_regions(&self, n_regions: usize) -> Result<(usize, Vec<Vec<([i64; $d], $voxel_name<N>)>>), CalcError> {
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
                    None => return Err(CalcError("Could not find a suiting decomposition".to_owned())),
                };

                // Now we drain the indices vector
                let mut index_voxel_combinations: Vec<([i64; $d], $voxel_name<N>)> = indices
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
                            .map(|new_index| (new_index, BoundaryCondition::Neumann(SVector::<f64, N>::from_element(0.0))))
                            .collect::<Vec<_>>();
                        (ind, $voxel_name::<N>::new(min, max, ind, domain_boundaries))
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

macro_rules! implement_cartesian_cuboid_domain_new {
    ($d: literal, $domain_name: ident, $subdomain_name: ident, $voxel_name: ident, $($k: expr),+) => {
        // TODO
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $domain_name {
            pub min: [f64; $d],
            pub max: [f64; $d],
            pub n_voxels: [i64; $d],
            pub dx_voxels: [f64; $d],
            pub rng_seed: u64,
        }

        impl $domain_name {
            fn check_min_max(min: [f64; $d], max: [f64; $d]) -> Result<(), CalcError> {
                for i in 0..$d {
                    match max[i] > min[i] {
                        false => Err(CalcError(format!("Min {:?} must be smaller than Max {:?} for domain boundaries!", min, max))),
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
                        false => Err(CalcError(format!("Interaction range must be positive and non-negative! Got value {:?}", interaction_ranges[i]))),
                        true => Ok(())
                    }?;
                }
                Ok(())
            }

            pub fn from_boundaries_and_interaction_ranges(
                min: [f64; $d],
                max: [f64; $d],
                interaction_ranges: [f64; $d]
            ) -> Result<$domain_name, CalcError>
            {
                Self::check_min_max(min, max)?;
                Self::check_positive(interaction_ranges)?;
                let mut n_voxels = [0; $d];
                let mut dx_voxels = [0.0; $d];
                for i in 0..$d {
                    n_voxels[i] = ((max[i] - min[i]) / interaction_ranges[i] * 0.5).ceil() as i64;
                    dx_voxels[i] = (max[i]-min[i])/n_voxels[i] as f64;
                }
                Ok(Self {
                    min,
                    max,
                    n_voxels,
                    dx_voxels,
                    rng_seed: 0,
                })
            }

            pub fn from_boundaries_and_n_voxels(
                min: [f64; $d],
                max: [f64; $d],
                n_vox: [usize; $d]
            ) -> Result<$domain_name, CalcError>
            {
                Self::check_min_max(min, max)?;
                Self::check_positive(n_vox)?;
                let mut dx_voxels = [0.0; $d];
                for i in 0..$d {
                    dx_voxels[i] = (max[i] - min[i]) / n_vox[i] as f64;
                }
                Ok(Self {
                    min,
                    max,
                    n_voxels: [$(n_vox[$k] as i64),+],
                    dx_voxels,
                    rng_seed: 0,
                })
            }

            fn get_voxel_index(
                &self,
                position: &nalgebra::SVector<f64, $d>,
            ) -> Result<[i64; $d], BoundaryError> {
                let mut percent: nalgebra::SVector<f64, $d> = self.max.into();
                percent -= nalgebra::SVector::<f64, $d>::from(self.min);
                percent = position.component_div(&percent);
                let vox = [$(
                    (percent[$k] * self.n_voxels[$k] as f64).floor() as i64,
                )+];

                // If the returned voxel is not positive and smaller than the maximum
                // number of voxel indices this function needs to return an error.
                if vox
                    .iter()
                    .enumerate()
                    .any(|(i, &p)| p<0 && self.n_voxels[i]<p) {
                        return Err(
                            BoundaryError(format!("Cell with position {:?} could not find index in domain with size min: {:?} max: {:?}", position, self.min, self.max))
                        );
                } else {
                    return Ok(vox);
                }
            }

            fn get_neighbor_voxel_indices(&self, index: &[i64; $d]) -> Vec<[i64; $d]> {
                // Create the bounds for the following creation of all the voxel indices
                let bounds: [[i64; 2]; $d] = [$(
                    [
                        max(index[$k] as i32 - 1, 0) as i64,
                        min(index[$k]+2, self.n_voxels[$k])
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
        }

        #[doc ="Subdomain of ["]
        #[doc = stringify!($domain_name)]
        #[doc = "]"]
        ///
        /// The subdomain contains voxels
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $subdomain_name {
            pub voxels: Vec<$voxel_name>,
        }

        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $voxel_name {
            pub min: [f64; $d],
            pub max: [f64; $d],
            pub ind: [i64; $d],
        }

        impl<C> cellular_raza_concepts::domain_new::Domain<C, $subdomain_name> for $domain_name
        where
            C: cellular_raza_concepts::mechanics::Mechanics<nalgebra::SVector<f64, $d>, nalgebra::SVector<f64, $d>, nalgebra::SVector<f64, $d>>,
        {
            // TODO THINK VERY HARD ABOUT THESE TYPES! THEY MIGHT BE CHOSEN STUPIDLY!
            type SubDomainIndex = i64;
            type VoxelIndex = [i64; $d];

            fn get_all_voxel_indices(&self) -> Vec<Self::VoxelIndex> {
                [$($k),+]
                    .iter()                                     // indices supplied in macro invokation
                    .map(|i| (0..self.n_voxels[*i]))            // ranges from self.n_vox
                    .multi_cartesian_product()                  // all possible combinations
                    .map(|ind_v| [$(ind_v[$k]),+])              // multi_cartesian_product gives us vector elements. We map them to arrays.
                    .collect()
            }

            /// Much more research must be done to effectively write this function.
            /// We should be using more sophisticated functionality based on common known facts for
            /// minimizing surface area and number of neighbors.
            /// For more information also see
            /// - [Wikipedia](https://en.wikipedia.org/wiki/Plateau%27s_laws)
            /// - [Math StackExchange](https://math.stackexchange.com/questions/3488409/dividing-a-square-into-n-equal-size-parts-with-minimal-fence)
            fn decompose(
                self,
                n_subdomains: core::num::NonZeroUsize,
                cells: Vec<C>,
            ) -> Result<cellular_raza_concepts::domain_new::DecomposedDomain<Self::SubDomainIndex, $subdomain_name, C>, DecomposeError> {
                let mut indices = <Self as cellular_raza_concepts::domain_new::Domain<C, $subdomain_name>>::get_all_voxel_indices(&self);

                let (n, m, average_len);
                match get_decomp_res(indices.len(), n_subdomains.into()) {
                    Some(res) => (n, m, average_len) = res,
                    None => return Err(DecomposeError::Generic("Could not find a suiting decomposition".to_owned())),
                };

                // TODO optimize this!
                // Currently we are not splitting the voxels apart efficiently
                // These are subdomains which contain n voxels
                let mut ind_n: Vec<Vec<_>> = indices
                    .drain(0..(average_len*n) as usize)
                    .into_iter()
                    .chunks(average_len as usize)
                    .into_iter()
                    .map(|chunk| chunk.collect::<Vec<_>>())
                    .collect();

                // These are subdomains that contain m indices
                let mut ind_m: Vec<Vec<_>> = indices
                    .drain(..)
                    .into_iter()
                    .chunks((max(average_len-1, 1)) as usize)
                    .into_iter()
                    .map(|chunk| chunk.collect::<Vec<_>>())
                    .collect();

                // Combine them into one Vector
                ind_n.append(&mut ind_m);

                // Construct a map from voxel_index to plain_index
                let voxel_index_to_plain_index: std::collections::HashMap::<Self::VoxelIndex,u128> = ind_n.clone()
                    .into_iter()
                    .map(|indices| indices.into_iter())
                    .flatten()
                    .enumerate()
                    .map(|(i, voxel_index)| (voxel_index, i as u128))
                    .collect();

                // We construct all Voxels which are grouped in their according subdomains
                // Then we construct the subdomain
                let mut index_subdomain_cells: std::collections::HashMap<Self::SubDomainIndex, (_, Vec<C>)> = ind_n
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|(i, indices)| {
                        let voxels = indices
                            .into_iter()
                            .map(|ind| {
                                let min = [$(self.min[$k] +    ind[$k]  as f64*self.dx_voxels[$k]),+];
                                let max = [$(self.min[$k] + (1+ind[$k]) as f64*self.dx_voxels[$k]),+];

                                $voxel_name {
                                    min,
                                    max,
                                    ind,
                                }
                            }).collect::<Vec<_>>();
                            (i as Self::SubDomainIndex, ($subdomain_name {voxels,}, Vec::<C>::new()))
                        }
                    ).collect();

                // Construct a map from voxel_index to subdomain_index
                let voxel_index_to_subdomain_index = ind_n
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|(subdomain_index, voxel_indices)| voxel_indices
                        .into_iter()
                        .map(move |voxel_index| (voxel_index, subdomain_index as i64))
                    )
                    .flatten()
                    .collect::<std::collections::HashMap<Self::VoxelIndex, Self::SubDomainIndex>>();

                // Sort the cells into the correct voxels
                cells
                    .into_iter()
                    .map(|cell| {
                        // Get the voxel index of the cell
                        let voxel_index = self.get_voxel_index(&cell.pos())?;
                        // Now get the subdomain index of the voxel
                        let subdomain_index = voxel_index_to_subdomain_index.get(&voxel_index).ok_or(
                            DecomposeError::IndexError(IndexError(format!("Could not cell with position {:?} in domain {:?}", cell.pos(), self)))
                        )?;
                        // Then add the cell to the subdomains cells.
                        index_subdomain_cells.get_mut(&subdomain_index).ok_or(
                            DecomposeError::IndexError(IndexError(format!("Could not find subdomain index {:?} internally which should have been there.", subdomain_index)))
                        )?.1.push(cell);
                        Ok(())

                    }).collect::<Result<Vec<_>, DecomposeError>>()?;

                //
                let index_subdomain_cells: Vec<(Self::SubDomainIndex, _, _)> = index_subdomain_cells
                    .into_iter()
                    .map(|(index, (subdomain, cells))| (index, subdomain, cells))
                    .collect();

                let neighbor_map = ind_n
                    .into_iter()
                    .enumerate()
                    .map(|(subdomain_index, voxel_indices)| {
                        let neighbor_voxels = voxel_indices
                            .into_iter()
                            .map(|voxel_index| self.get_neighbor_voxel_indices(&voxel_index))
                            .flatten();
                        let neighbor_subdomains = neighbor_voxels
                            .map(|neighbor_voxel_index| voxel_index_to_subdomain_index
                                .get(&neighbor_voxel_index)
                                .and_then(|v| Some(v.clone()))
                                .ok_or(
                                    DecomposeError::IndexError(
                                        IndexError(format!("Could not find neighboring voxel index {:?} internally which should have been initialized.", neighbor_voxel_index))
                                )
                            ))
                            .collect::<Result<Vec<i64>, _>>()
                            .and_then(|neighbors| Ok(neighbors
                                .into_iter()
                                .unique()
                                .filter(|neighbor_index| *neighbor_index!=subdomain_index as i64)
                                .collect::<Vec<_>>()))?;
                        Ok((subdomain_index.clone() as i64, neighbor_subdomains))
                    })
                    .collect::<Result<_, DecomposeError>>()?;

                Ok(cellular_raza_concepts::domain_new::DecomposedDomain {
                    n_subdomains: n+m,
                    index_subdomain_cells,
                    neighbor_map,
                    rng_seed: self.rng_seed.clone(),
                    voxel_index_to_plain_index,
                })
            }
        }

        impl<C> cellular_raza_concepts::domain_new::SubDomain<C> for $subdomain_name {
            type VoxelIndex = [i64; $d];

            fn get_voxel_index_of(&self, cell: &C) -> Result<Self::VoxelIndex, BoundaryError> {
                todo!()
            }

            fn get_neighbor_voxel_indices(&self, index: &Self::VoxelIndex) -> Vec<Self::VoxelIndex> {
                // Create the bounds for the following creation of all the voxel indices
                /* let bounds: [[i64; 2]; $d] = [$(
                    [
                        max(index[$k] as i32 - 1, 0) as i64,
                        min(index[$k]+2, self.n_voxels[$k])
                    ]
                ),+];

                // Create voxel indices
                let v: Vec<[i64; $d]> = [$($k),+].iter()      // indices supplied in macro invokation
                    .map(|i| (bounds[*i][0]..bounds[*i][1]))    // ranges from bounds
                    .multi_cartesian_product()                  // all possible combinations
                    .map(|ind_v| [$(ind_v[$k]),+])              // multi_cartesian_product gives us vector elements. We map them to arrays.
                    .filter(|ind| ind!=index)                   // filter the elements such that the current index is not included.
                    .collect();                                 // collect into the correct type

                return v;*/
                todo!()
            }

            fn apply_boundary(&self, cell: &mut C) -> Result<(), BoundaryError> {
                todo!()
            }

            fn get_all_indices(&self) -> Vec<Self::VoxelIndex> {
                todo!()
            }
        }
    }
}

implement_cartesian_cuboid_domain_new!(
    2,
    CartesianCuboid2_New,
    CartesianSubDomain2,
    CartesianVoxel2,
    0,
    1
);

// TODO make them only visible if correspoding feature (eg. fluid_mechanics or gradients) is active
define_and_implement_cartesian_cuboid!(1, CartesianCuboid1, 0);
define_and_implement_cartesian_cuboid!(2, CartesianCuboid2, 0, 1);
define_and_implement_cartesian_cuboid!(3, CartesianCuboid3, 0, 1, 2);
implement_cartesian_cuboid_voxel_fluid_mechanics!(1, CartesianCuboid1, CartesianCuboidVoxel1, 0);
implement_cartesian_cuboid_voxel_fluid_mechanics!(2, CartesianCuboid2, CartesianCuboidVoxel2, 0, 1);
implement_cartesian_cuboid_voxel_fluid_mechanics!(
    3,
    CartesianCuboid3,
    CartesianCuboidVoxel3,
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
        // Calculate the images dimensions by the dimensions of the simulation domain
        let dx = (self.max[0] - self.min[0]).abs();
        let dy = (self.max[1] - self.min[1]).abs();
        let q = dx.min(dy);
        let image_size_x = (image_size as f64 * dx / q).round() as u32;
        let image_size_y = (image_size as f64 * dy / q).round() as u32;

        // Create a domain with the correct size and fill it white.
        use plotters::drawing::IntoDrawingArea;
        let root = BitMapBackend::new(filename, (image_size_x, image_size_y)).into_drawing_area();
        root.fill(&plotters::prelude::full_palette::WHITE).unwrap();

        // Build a chart on the domain such that plotting later will be simplified
        let mut chart = plotters::prelude::ChartBuilder::on(&root)
            // Finally attach a coordinate on the drawing area and make a chart context
            .build_cartesian_2d(self.min[0]..self.max[0], self.min[1]..self.max[1])
            .unwrap();

        let root = chart.plotting_area().clone();

        chart
            .configure_mesh()
            // we do not want to draw any mesh lines automatically
            .disable_mesh()
            .draw()
            .unwrap();

        Ok(root)
    }
}

#[cfg(test)]
mod test {
    use super::get_decomp_res;
    use rayon::prelude::*;

    #[test]
    fn test_get_demomp_res() {
        let max = 5_000;

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
