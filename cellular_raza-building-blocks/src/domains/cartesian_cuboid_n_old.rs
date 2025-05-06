use cellular_raza_concepts::domain_old::*;
use cellular_raza_concepts::reactions_old::Volume;
use cellular_raza_concepts::{
    BoundaryError, CalcError, CreatePlottingRoot, IndexError, RequestError,
};

use super::get_decomp_res;

use serde::{Deserialize, Serialize};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use itertools::Itertools;

// Imports from std and core
use core::cmp::{max, min};

use plotters::backend::BitMapBackend;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::DrawingArea;

use nalgebra::SVector;

macro_rules! define_and_implement_cartesian_cuboid {
    ($d: expr, $name: ident, $($k: expr),+) => {
        /// Cuboid Domain with regular cartesian coordinates in
        #[doc = concat!(" `", stringify!($d), "D`")]
        #[derive(Clone, Debug, Serialize, Deserialize)]
        #[cfg_attr(feature = "pyo3", pyclass)]
        #[cfg_attr(feature = "pyo3", pyo3(get_all, set_all))]
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
                        false => Err(CalcError(format!(
                            "Min {:?} must be smaller than Max {:?} for domain boundaries!",
                            min,
                            max
                        ))),
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
                        false => Err(CalcError(format!(
                            "Interaction range must be positive and non-negative! Got value {:?}",
                            interaction_ranges[i]
                        ))),
                        true => Ok(())
                    }?;
                }
                Ok(())
            }

            // TODO write this nicely!
            #[doc = "Builds a new `"]
            #[doc = stringify!($name)]
            #[doc = "` from given boundaries and maximum interaction ranges of the containing
                cells."]
            pub fn from_boundaries_and_interaction_ranges(
                min: [f64; $d],
                max: [f64; $d],
                interaction_ranges: [f64; $d]
            ) -> Result<$name, CalcError> {
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
            pub fn from_boundaries_and_n_voxels(
                min: [f64; $d],
                max: [f64; $d],
                n_vox: [usize; $d]
            ) -> Result<$name, CalcError> {
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
        /// Cuboid Voxel for [
        #[doc = stringify!($name)]
        /// ] in
        #[doc = concat!(" `", stringify!($d), "D`")]
        #[derive(Clone,Debug,Serialize,Deserialize)]
        pub struct $voxel_name<const N: usize> {
                min: [f64; $d],
                max: [f64; $d],
                middle: [f64; $d],
                dx: [f64; $d],
                index: [i64; $d],

                /// Concentrations of the different diffusables
                pub extracellular_concentrations: SVector<f64, N>,
                #[cfg(feature = "gradients")]
                /// The gradient of diffusables at this voxel
                pub extracellular_gradient: SVector<SVector<f64, $d>, N>,
                /// Local diffusion constant
                pub diffusion_constant: SVector<f64, N>,
                /// Local production rate of diffusables
                pub production_rate: SVector<f64, N>,
                /// Local degradation rate of diffusables
                pub degradation_rate: SVector<f64, N>,
                domain_boundaries: Vec<([i64; $d], BoundaryCondition<SVector<f64, N>>)>,
        }

        impl<const N: usize> Volume for $voxel_name<N> {
            fn get_volume(&self) -> f64 {
                self.min.iter().zip(self.max.iter()).map(|(x, y)| y-x).product()
            }
        }

        impl<const N: usize> $voxel_name<N> {
            pub(crate) fn new(
                min: [f64; $d],
                max: [f64; $d],
                index: [i64; $d],
                domain_boundaries: Vec<([i64; $d], BoundaryCondition<SVector<f64, N>>)>
            ) -> $voxel_name<N> {
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
                    extracellular_gradient: SVector::<SVector<f64, $d>, N>::from_element(
                        SVector::<f64, $d>::from_element(0.0)
                    ),
                    diffusion_constant: SVector::<f64, N>::from_element(0.0),
                    production_rate: SVector::<f64, N>::from_element(0.0),
                    degradation_rate: SVector::<f64, N>::from_element(0.0),
                    domain_boundaries,
                }
            }

            /// Get lower boundary of voxel
            pub fn get_min(&self) -> [f64; $d] {self.min}
            /// Get upper boundary of voxel
            pub fn get_max(&self) -> [f64; $d] {self.max}
            /// Get middle of voxel
            pub fn get_middle(&self) -> [f64; $d] {self.middle}
            /// Get side lengths of voxel
            pub fn get_dx(&self) -> [f64; $d] {self.dx}

            fn position_is_in_domain(&self, pos: &SVector<f64, $d>) -> Result<(), RequestError> {
                match pos.iter().enumerate().any(|(i, p)| !(self.min[i] <= *p && *p <= self.max[i])) {
                    true => Err(RequestError(format!(
                        "point {:?} is not in requested voxel with boundaries {:?} {:?}",
                        pos,
                        self.min,
                        self.max
                    ))),
                    false => Ok(()),
                }
            }

            fn index_to_distance_squared(&self, index: &[i64; $d]) -> f64 {
                let mut diffs = [0; $d];
                for i in 0..$d {
                    diffs[i] = (index[i] as i32 - self.index[i] as i32).abs()
                }
                diffs
                    .iter()
                    .enumerate()
                    .map(|(i, d)| self.dx[i].powf(2.0)* (*d as f64))
                    .sum::<f64>()
            }
        }

        // Implement the Voxel trait for our n-dim voxel
        impl<const N: usize> Voxel<[i64; $d], SVector<f64, $d>, SVector<f64, $d>, SVector<f64, $d>>
            for $voxel_name<N> {
            fn get_index(&self) -> [i64; $d] {
                self.index
            }
        }

        impl<const N: usize> ExtracellularMechanics<
            [i64; $d],
            SVector<f64, $d>,
            SVector<f64, N>,
            SVector<SVector<f64, $d>, N>,
            SVector<f64, N>,
            SVector<f64, N>
        > for $voxel_name<N> {
            fn get_extracellular_at_point(
                &self,
                pos: &SVector<f64, $d>
            ) -> Result<SVector<f64, N>, RequestError> {
                self.position_is_in_domain(pos)?;
                Ok(self.extracellular_concentrations)
            }

            fn get_total_extracellular(&self) -> SVector<f64, N> {
                self.extracellular_concentrations
            }

            #[cfg(feature = "gradients")]
            fn update_extracellular_gradient(
                &mut self,
                boundaries: &[([i64; $d], BoundaryCondition<SVector<f64, N>>)]
            ) -> Result<(), CalcError> {
                let mut new_gradient = SVector::<SVector<f64, $d>, N>::from_element(
                    SVector::<f64, $d>::from_element(0.0)
                );
                boundaries.iter()
                    .for_each(|(index, boundary_condition)| {
                        let extracellular_difference = match boundary_condition {
                            BoundaryCondition::Neumann(value) => {*value},
                            BoundaryCondition::Dirichlet(value) => {
                                self.extracellular_concentrations-value
                            },
                            BoundaryCondition::Value(value) => {
                                self.extracellular_concentrations-value
                            },
                        };
                        let pointer = SVector::from(
                            [$(self.index[$k] as f64 - index[$k] as f64),+]
                        );
                        let dist = pointer.norm();
                        let gradient = pointer.normalize()/dist;
                        new_gradient
                            .iter_mut()
                            .zip(extracellular_difference.into_iter())
                            .for_each(|(component, diff)| *component += *diff*gradient);
                    });
                self.extracellular_gradient = new_gradient;
                Ok(())
            }

            #[cfg(feature = "gradients")]
            fn get_extracellular_gradient_at_point(
                &self,
                _pos: &SVector<f64, $d>
            ) -> Result<SVector<SVector<f64, $d>, N>, RequestError> {
                Ok(self.extracellular_gradient)
            }

            fn set_total_extracellular(
                &mut self,
                concentrations: &SVector<f64, N>
            ) -> Result<(), CalcError> {
                Ok(self.extracellular_concentrations = *concentrations)
            }

            fn calculate_increment(
                &self,
                total_extracellular: &SVector<f64, N>,
                point_sources: &[(SVector<f64, $d>, SVector<f64, N>)],
                boundaries: &[([i64; $d], BoundaryCondition<SVector<f64, N>>)]
            ) -> Result<SVector<f64, N>, CalcError> {
                let mut inc = SVector::<f64, N>::from_element(0.0);

                self.domain_boundaries
                    .iter()
                    .for_each(|(index, boundary)| match boundary {
                        BoundaryCondition::Neumann(value) =>
                            inc += value / self.index_to_distance_squared(index).sqrt(),
                        BoundaryCondition::Dirichlet(value) =>
                            inc += (value-total_extracellular)
                                / self.index_to_distance_squared(index),
                        BoundaryCondition::Value(value) =>
                            inc += (value-total_extracellular)
                                / self.index_to_distance_squared(index),
                    });

                boundaries.iter()
                    .for_each(|(index, boundary)| match boundary {
                        BoundaryCondition::Neumann(value) =>
                            inc += value
                                / self.index_to_distance_squared(&index).sqrt(),
                        BoundaryCondition::Dirichlet(value) =>
                            inc += (value-total_extracellular)
                                / self.index_to_distance_squared(&index),
                        BoundaryCondition::Value(value) =>
                            inc += (value-total_extracellular)
                                / self.index_to_distance_squared(&index),
                    });
                inc = inc.component_mul(&self.diffusion_constant);

                point_sources.iter()
                    .for_each(|(_, value)| inc += value);

                // Also calculate internal reactions. Here it is very simple only given by
                // degradation and production.
                inc += self.production_rate
                    - self.degradation_rate.component_mul(&total_extracellular);
                Ok(inc)
            }

            fn boundary_condition_to_neighbor_voxel(
                &self,
                _neighbor_index: &[i64; $d]
            ) -> Result<BoundaryCondition<SVector<f64, N>>, IndexError> {
                Ok(BoundaryCondition::Value(self.extracellular_concentrations))
            }
        }

        // Implement the cartesian cuboid
        // Index is an array of size 3 with elements of type usize
        impl<Cel, const N: usize> Domain<Cel, [i64; $d], $voxel_name<N>> for $name
        // Position, Force and Velocity are all Vector$d supplied by the Nalgebra crate
            where
                Cel: cellular_raza_concepts::Position<SVector<f64, $d>>,
                Cel: cellular_raza_concepts::Velocity<SVector<f64, $d>>,
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
                        return Err(BoundaryError(format!(
                            "Particle is out of domain at position {:?}",
                            pos
                        )));
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

            fn generate_contiguous_multi_voxel_regions(&self, n_regions: usize) -> Result<Vec<Vec<([i64; $d], $voxel_name<N>)>>, CalcError> {
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

                Ok(ind_n)
            }
        }
    }
}

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
        cellular_raza_concepts::DrawingError,
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
