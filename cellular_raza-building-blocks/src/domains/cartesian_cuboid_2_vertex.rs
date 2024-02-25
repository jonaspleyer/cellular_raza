// Imports from this crate
use cellular_raza_concepts::*;

use super::cartesian_cuboid_n::get_decomp_res;
use crate::cell_building_blocks::VertexVector2;

// Imports from std and core
use core::cmp::{max, min};
use nalgebra::SVector;

// Imports from other crates
use itertools::Itertools;

use serde::{Deserialize, Serialize};

use plotters::backend::BitMapBackend;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::DrawingArea;

/// Cuboid Domain with coordinates specialized for vertex systems in 2 dimensions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CartesianCuboid2Vertex {
    min: [f64; 2],
    max: [f64; 2],
    n_vox: [i64; 2],
    voxel_sizes: [f64; 2],
}

impl CartesianCuboid2Vertex {
    fn check_min_max(min: [f64; 2], max: [f64; 2]) -> Result<(), CalcError> {
        for i in 0..2 {
            match max[i] > min[i] {
                false => Err(CalcError(format!(
                    "Min {:?} must be smaller than Max {:?} for domain boundaries!",
                    min, max
                ))),
                true => Ok(()),
            }?;
        }
        Ok(())
    }

    fn check_positive<F>(interaction_ranges: [F; 2]) -> Result<(), CalcError>
    where
        F: PartialOrd + num::Zero + core::fmt::Debug,
    {
        for i in 0..2 {
            match interaction_ranges[i] > F::zero() {
                false => Err(CalcError(format!(
                    "Interaction range must be positive and non-negative! Got value {:?}",
                    interaction_ranges[i]
                ))),
                true => Ok(()),
            }?;
        }
        Ok(())
    }

    // TODO write this nicely!
    /// Builds a new [CartesianCuboid2Vertex]
    /// from given boundaries and maximum interaction ranges of the containing cells.
    pub fn from_boundaries_and_interaction_ranges(
        min: [f64; 2],
        max: [f64; 2],
        interaction_ranges: [f64; 2],
    ) -> Result<CartesianCuboid2Vertex, CalcError> {
        CartesianCuboid2Vertex::check_min_max(min, max)?;
        CartesianCuboid2Vertex::check_positive(interaction_ranges)?;
        let mut n_vox = [0; 2];
        let mut voxel_sizes = [0.0; 2];
        for i in 0..2 {
            n_vox[i] = ((max[i] - min[i]) / interaction_ranges[i] * 0.5).ceil() as i64;
            voxel_sizes[i] = (max[i] - min[i]) / n_vox[i] as f64;
        }
        Ok(CartesianCuboid2Vertex {
            min,
            max,
            n_vox,
            voxel_sizes,
        })
    }

    /// Builds a new [CartesianCuboid2Vertex] from given boundaries and the number of voxels per dimension specified
    pub fn from_boundaries_and_n_voxels(
        min: [f64; 2],
        max: [f64; 2],
        n_vox: [usize; 2],
    ) -> Result<CartesianCuboid2Vertex, CalcError> {
        CartesianCuboid2Vertex::check_min_max(min, max)?;
        CartesianCuboid2Vertex::check_positive(n_vox)?;
        let mut voxel_sizes = [0.0; 2];
        for i in 0..2 {
            voxel_sizes[i] = (max[i] - min[i]) / n_vox[i] as f64;
        }
        Ok(CartesianCuboid2Vertex {
            min,
            max,
            n_vox: [n_vox[0] as i64, n_vox[1] as i64],
            voxel_sizes,
        })
    }
}

/// Cuboid Voxel for [CartesianCuboid2Vertex] in 2 dimensions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CartesianCuboidVoxel2Vertex<const D: usize, const N: usize> {
    min: [f64; 2],
    max: [f64; 2],
    middle: [f64; 2],
    dx: [f64; 2],
    index: [i64; 2],

    /// Concentrations of the different diffusables
    pub extracellular_concentrations: SVector<f64, N>,
    /// The gradient of diffusables at this voxel
    pub extracellular_gradient: SVector<SVector<f64, 2>, N>,
    /// Local diffusion constant
    pub diffusion_constant: SVector<f64, N>,
    /// Local production rate of diffusables
    pub production_rate: SVector<f64, N>,
    /// Local degradation rate of diffusables
    pub degradation_rate: SVector<f64, N>,
    domain_boundaries: Vec<([i64; 2], BoundaryCondition<SVector<f64, N>>)>,
}

impl<const D: usize, const N: usize> CartesianCuboidVoxel2Vertex<D, N> {
    pub(crate) fn new(
        min: [f64; 2],
        max: [f64; 2],
        index: [i64; 2],
        domain_boundaries: Vec<([i64; 2], BoundaryCondition<SVector<f64, N>>)>,
    ) -> CartesianCuboidVoxel2Vertex<D, N> {
        let middle = [(max[0] + min[0]) / 2.0, (max[1] + min[1]) / 2.0];
        let dx = [max[0] - min[0], max[1] - min[1]];
        CartesianCuboidVoxel2Vertex {
            min,
            max,
            middle,
            dx,
            index,
            extracellular_concentrations: SVector::<f64, N>::from_element(0.0),
            extracellular_gradient: SVector::<SVector<f64, 2>, N>::from_element(
                SVector::<f64, 2>::from_element(0.0),
            ),
            diffusion_constant: SVector::<f64, N>::from_element(0.0),
            production_rate: SVector::<f64, N>::from_element(0.0),
            degradation_rate: SVector::<f64, N>::from_element(0.0),
            domain_boundaries,
        }
    }

    /// Get lower boundary of voxel
    pub fn get_min(&self) -> [f64; 2] {
        self.min
    }
    /// Get upper boundary of voxel
    pub fn get_max(&self) -> [f64; 2] {
        self.max
    }
    /// Get middle of voxel
    pub fn get_middle(&self) -> [f64; 2] {
        self.middle
    }
    /// Get side lengths of voxel
    pub fn get_dx(&self) -> [f64; 2] {
        self.dx
    }

    fn position_is_in_domain(&self, pos: &VertexVector2<D>) -> Result<(), RequestError> {
        let middle = pos.row_sum() / pos.shape().0 as f64;
        match middle
            .iter()
            .enumerate()
            .any(|(i, p)| !(self.min[i] <= *p && *p <= self.max[i]))
        {
            true => Err(RequestError(format!(
                "point {:?} is not in requested voxel with boundaries {:?} {:?}",
                pos, self.min, self.max
            ))),
            false => Ok(()),
        }?;
        Ok(())
    }

    fn index_to_distance_squared(&self, index: &[i64; 2]) -> f64 {
        let mut diffs = [0; 2];
        for i in 0..2 {
            diffs[i] = (index[i] as i32 - self.index[i] as i32).abs()
        }
        diffs
            .iter()
            .enumerate()
            .map(|(i, d)| self.dx[i].powf(2.0) * (*d as f64))
            .sum::<f64>()
    }
}

// Implement the Voxel trait for our n-dim voxel
impl<const D: usize, const N: usize>
    Voxel<[i64; 2], VertexVector2<D>, VertexVector2<D>, VertexVector2<D>>
    for CartesianCuboidVoxel2Vertex<D, N>
{
    fn get_index(&self) -> [i64; 2] {
        self.index
    }
}

impl<const D: usize, const N: usize>
    ExtracellularMechanics<
        [i64; 2],
        VertexVector2<D>,
        SVector<f64, N>,
        SVector<SVector<f64, 2>, N>,
        SVector<f64, N>,
        SVector<f64, N>,
    > for CartesianCuboidVoxel2Vertex<D, N>
{
    fn get_extracellular_at_point(
        &self,
        pos: &VertexVector2<D>,
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
        boundaries: &[([i64; 2], BoundaryCondition<SVector<f64, N>>)],
    ) -> Result<(), CalcError> {
        let mut new_gradient =
            SVector::<SVector<f64, 2>, N>::from_element(SVector::from_element(0.0));
        boundaries.iter().for_each(|(index, boundary_condition)| {
            let extracellular_difference = match boundary_condition {
                BoundaryCondition::Neumann(value) => *value,
                BoundaryCondition::Dirichlet(value) => self.extracellular_concentrations - value,
                BoundaryCondition::Value(value) => self.extracellular_concentrations - value,
            };
            let pointer = SVector::from([
                self.index[0] as f64 - index[0] as f64,
                self.index[1] as f64 - index[1] as f64,
            ]);
            let dist = pointer.norm();
            let gradient = pointer.normalize() / dist;
            new_gradient
                .iter_mut()
                .zip(extracellular_difference.into_iter())
                .for_each(|(component, diff)| *component += *diff * gradient);
        });
        self.extracellular_gradient = new_gradient;
        Ok(())
    }

    #[cfg(feature = "gradients")]
    fn get_extracellular_gradient_at_point(
        &self,
        _pos: &VertexVector2<D>,
    ) -> Result<SVector<SVector<f64, 2>, N>, RequestError> {
        Ok(self.extracellular_gradient)
    }

    fn set_total_extracellular(
        &mut self,
        concentrations: &SVector<f64, N>,
    ) -> Result<(), CalcError> {
        Ok(self.extracellular_concentrations = *concentrations)
    }

    fn calculate_increment(
        &self,
        total_extracellular: &SVector<f64, N>,
        point_sources: &[(VertexVector2<D>, SVector<f64, N>)],
        boundaries: &[([i64; 2], BoundaryCondition<SVector<f64, N>>)],
    ) -> Result<SVector<f64, N>, CalcError> {
        let mut inc = SVector::<f64, N>::from_element(0.0);

        self.domain_boundaries
            .iter()
            .for_each(|(index, boundary)| match boundary {
                BoundaryCondition::Neumann(value) => {
                    inc += value / self.index_to_distance_squared(index).sqrt()
                }
                BoundaryCondition::Dirichlet(value) => {
                    inc += (value - total_extracellular) / self.index_to_distance_squared(index)
                }
                BoundaryCondition::Value(value) => {
                    inc += (value - total_extracellular) / self.index_to_distance_squared(index)
                }
            });

        boundaries
            .iter()
            .for_each(|(index, boundary)| match boundary {
                BoundaryCondition::Neumann(value) => {
                    inc += value / self.index_to_distance_squared(&index).sqrt()
                }
                BoundaryCondition::Dirichlet(value) => {
                    inc += (value - total_extracellular) / self.index_to_distance_squared(&index)
                }
                BoundaryCondition::Value(value) => {
                    inc += (value - total_extracellular) / self.index_to_distance_squared(&index)
                }
            });
        inc = inc.component_mul(&self.diffusion_constant);

        point_sources.iter().for_each(|(_, value)| inc += value);

        // Also calculate internal reactions. Here it is very simple only given by degradation and production.
        inc += self.production_rate - self.degradation_rate.component_mul(&total_extracellular);
        Ok(inc)
    }

    fn boundary_condition_to_neighbor_voxel(
        &self,
        _neighbor_index: &[i64; 2],
    ) -> Result<BoundaryCondition<SVector<f64, N>>, IndexError> {
        Ok(BoundaryCondition::Value(self.extracellular_concentrations))
    }
}

impl<const D: usize, const N: usize> Volume for CartesianCuboidVoxel2Vertex<D, N> {
    fn get_volume(&self) -> f64 {
        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(x, y)| y - x)
            .product()
    }
}

// Implement the cartesian cuboid
// Index is an array of size 3 with elements of type usize
impl<Cel, const D: usize, const N: usize> Domain<Cel, [i64; 2], CartesianCuboidVoxel2Vertex<D, N>>
    for CartesianCuboid2Vertex
// Position, Force and Velocity are all Vector2 supplied by the Nalgebra crate
where
    Cel: Mechanics<VertexVector2<D>, VertexVector2<D>, VertexVector2<D>>,
{
    fn apply_boundary(&self, cell: &mut Cel) -> Result<(), BoundaryError> {
        let mut pos_single = cell.pos();
        let mut velocity_single = cell.velocity();

        for (mut pos, mut velocity) in pos_single
            .row_iter_mut()
            .zip(velocity_single.row_iter_mut())
        {
            // For each dimension
            for i in 0..2 {
                // Check if the particle is below lower edge
                if pos[i] < self.min[i] {
                    pos[i] = 2.0 * self.min[i] - pos[i];
                    velocity[i] = velocity[i].abs();
                }
                // Check if the particle is over the edge
                if pos[i] > self.max[i] {
                    pos[i] = 2.0 * self.max[i] - pos[i];
                    velocity[i] = -velocity[i].abs();
                }
            }
        }

        // Set new position and velocity of particle
        cell.set_pos(&pos_single);
        cell.set_velocity(&velocity_single);

        for pos in pos_single.row_iter() {
            // If new position is still out of boundary return error
            for i in 0..2 {
                if pos[i] < self.min[i] || pos[i] > self.max[i] {
                    return Err(BoundaryError(format!(
                        "Particle is out of domain at position {:?}",
                        pos
                    )));
                }
            }
        }
        Ok(())
    }

    fn get_voxel_index(&self, cell: &Cel) -> [i64; 2] {
        // Calculate middle
        let p = cell.pos().row_sum() / cell.pos().shape().0 as f64;
        let mut out = [0; 2];

        for i in 0..2 {
            out[i] = ((p[i] - self.min[0]) / self.voxel_sizes[i]) as i64;
            out[i] = out[i].min(self.n_vox[i] - 1).max(0);
        }
        return out;
    }

    fn get_all_indices(&self) -> Vec<[i64; 2]> {
        (0..2)
            .map(|i| (0..self.n_vox[i]))
            .multi_cartesian_product()
            .map(|ind_v| [ind_v[0], ind_v[1]])
            .collect()
    }

    fn get_neighbor_voxel_indices(&self, index: &[i64; 2]) -> Vec<[i64; 2]> {
        // Create the bounds for the following creation of all the voxel indices
        let bounds: [[i64; 2]; 2] = [
            [
                max(index[0] as i32 - 1, 0) as i64,
                min(index[0] + 2, self.n_vox[0]),
            ],
            [
                max(index[1] as i32 - 1, 0) as i64,
                min(index[1] + 2, self.n_vox[1]),
            ],
        ];

        // Create voxel indices
        let v: Vec<[i64; 2]> = (0..2) // indices supplied in macro invokation
            .map(|i| (bounds[i][0]..bounds[i][1])) // ranges from bounds
            .multi_cartesian_product() // all possible combinations
            .map(|ind_v| [ind_v[0], ind_v[1]]) // multi_cartesian_product gives us vector elements. We map them to arrays.
            .filter(|ind| ind != index) // filter the elements such that the current index is not included.
            .collect(); // collect into the correct type

        return v;
    }

    fn generate_contiguous_multi_voxel_regions(
        &self,
        n_regions: usize,
    ) -> Result<Vec<Vec<([i64; 2], CartesianCuboidVoxel2Vertex<D, N>)>>, CalcError> {
        // Get all voxel indices
        let indices: Vec<[i64; 2]> = (0..2) // indices supplied in macro invokation
            .map(|i| (0..self.n_vox[i])) // ranges from self.n_vox
            .multi_cartesian_product() // all possible combinations
            .map(|ind_v| [ind_v[0], ind_v[1]]) // multi_cartesian_product gives us vector elements. We map them to arrays.
            .collect();

        let (n, _m, average_len);
        match get_decomp_res(indices.len(), n_regions) {
            Some(res) => (n, _m, average_len) = res,
            None => {
                return Err(CalcError(
                    "Could not find a suiting decomposition".to_owned(),
                ))
            }
        };

        // Now we drain the indices vector
        let mut index_voxel_combinations: Vec<([i64; 2], CartesianCuboidVoxel2Vertex<D, N>)> =
            indices
                .into_iter()
                .map(|ind| {
                    let min = [
                        self.min[0] + ind[0] as f64 * self.voxel_sizes[0],
                        self.min[1] + ind[1] as f64 * self.voxel_sizes[1],
                    ];
                    let max = [
                        self.min[0] + (1 + ind[0]) as f64 * self.voxel_sizes[0],
                        self.min[1] + (1 + ind[1]) as f64 * self.voxel_sizes[1],
                    ];
                    // TODO FIXUP we need to insert boundary conditions here as last argument
                    let domain_boundaries = (0..2)
                        .map(|_| (-1_i64..2_i64))
                        .multi_cartesian_product()
                        .map(|v| [ind[0] + v[0], ind[1] + v[1]])
                        .filter(|new_index| *new_index != ind)
                        .filter(|new_index| {
                            new_index
                                .iter()
                                .zip(self.n_vox.iter())
                                .any(|(i1, i2)| *i1 < 0 || i2 <= i1)
                        })
                        .map(|new_index| {
                            (
                                new_index,
                                BoundaryCondition::Neumann(SVector::<f64, N>::from_element(0.0)),
                            )
                        })
                        .collect::<Vec<_>>();
                    (
                        ind,
                        CartesianCuboidVoxel2Vertex::<D, N>::new(min, max, ind, domain_boundaries),
                    )
                })
                .collect();

        // TODO optimize this!
        // Currently we are not splitting the voxels apart efficiently
        let mut ind_n: Vec<Vec<_>> = index_voxel_combinations
            .drain(0..(average_len * n) as usize)
            .into_iter()
            .chunks(average_len as usize)
            .into_iter()
            .map(|chunk| chunk.collect::<Vec<_>>())
            .collect();

        let mut ind_m: Vec<Vec<_>> = index_voxel_combinations
            .drain(..)
            .into_iter()
            .chunks((max(average_len - 1, 1)) as usize)
            .into_iter()
            .map(|chunk| chunk.collect::<Vec<_>>())
            .collect();

        ind_n.append(&mut ind_m);

        Ok(ind_n)
    }
}

impl CreatePlottingRoot for CartesianCuboid2Vertex {
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

        let mut chart = plotters::prelude::ChartBuilder::on(&root)
            // Finally attach a coordinate on the drawing area and make a chart context
            .build_cartesian_2d(self.min[0]..self.max[0], self.min[1]..self.max[1])
            .unwrap();

        chart
            .configure_mesh()
            // we do not want to draw any mesh lines automatically but do this manually below
            .disable_mesh()
            .draw()
            .unwrap();

        Ok(chart.plotting_area().clone())
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
