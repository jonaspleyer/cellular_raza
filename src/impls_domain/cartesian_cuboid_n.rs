// Imports from this crate
use crate::concepts::errors::*;
use crate::concepts::domain::*;
use crate::concepts::cell::*;

use crate::plotting::spatial::CreatePlottingRoot;

// Imports from std and core
use core::cmp::{min,max};

// Imports from other crates
use nalgebra::SVector;
use itertools::Itertools;

use serde::{Serialize,Deserialize};

use plotters::prelude::DrawingArea;
use plotters::backend::BitMapBackend;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::types::RangedCoordf64;


/// Helper function to calculate the decomposition of a large number N into n as evenly-sized chunks as possible
/// Examples:
/// N   n   decomp
/// 10  3    1 *  4  +  3 *  3
/// 13  4    1 *  5  +  3 *  4
/// 100 13   4 * 13  +  4 * 12
/// 225 16   1 * 15  + 15 * 14
/// 225 17   4 * 14  + 13 * 13
fn get_decomp_res(n_voxel: usize, n_regions: usize) -> Option<(usize, usize, usize)> {
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

    let residue = |n: i64, m: i64, avg: i64| {n_voxel as i64 - avg*n - (avg-1)*m};

    let mut n = n_regions as i64;
    let mut m = 0;

    for _ in 0..n_regions {
        let r = residue(n, m, average_len);
        if r == 0 {
            return Some((n as usize, m as usize, average_len as usize));
        } else if r > 0 {
            if n==n_regions as i64 {
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


macro_rules! define_and_implement_cartesian_cuboid {
    ($d: expr, $name: ident, $voxel_name: ident, $($k: expr),+) => {
        #[doc = "Cuboid Domain with regular cartesian coordinates in `"]
        #[doc = stringify!($d)]
        #[doc = "` dimensions"]
        #[derive(Clone,Debug,Serialize,Deserialize)]
        pub struct $name {
            pub min: [f64; $d],
            pub max: [f64; $d],
            pub n_vox: [usize; $d],
            pub voxel_sizes: [f64; $d],
        }

        // Define the struct for the voxel
        #[doc = "Cuboid Voxel for `"]
        #[doc = stringify!($name)]
        #[doc = "` in `"]
        #[doc = stringify!($d)]
        #[doc = "` dimensions"]
        #[derive(Clone,Debug,Serialize,Deserialize)]
        pub struct $voxel_name {
                pub min: [f64; $d],
                pub max: [f64; $d],
                pub index: [usize; $d],
        }

        // Implement the Voxel trait for our n-dim voxel
        impl Voxel<[usize; $d], SVector<f64, $d>, SVector<f64, $d>> for $voxel_name {
            fn get_index(&self) -> [usize; $d] {
                self.index
            }
        }

        // Implement the cartesian cuboid
        // Index is an array of size 3 with elements of type usize
        impl<C> Domain<C, [usize; $d], $voxel_name> for $name
        // Position, Force and Velocity are all Vector$d supplied by the Nalgebra crate
        where C: CellAgent<SVector<f64, $d>, SVector<f64, $d>, SVector<f64, $d>>,
        {
            fn apply_boundary(&self, cell: &mut C) -> Result<(),BoundaryError> {
                let mut pos = cell.pos();
                let mut velocity = cell.velocity();
        
                // For each dimension
                for i in 0..$d {
                    // Check if the particle is below lower edge
                    if pos[i] < self.min[i] {
                        pos[i] = 2.0 * self.min[i] - pos[i];
                        velocity[i] *= -1.0;
                    }
                    // Check if the particle is over the edge
                    if pos[i] > self.max[i] {
                        pos[i] = 2.0 * self.max[i] - pos[i];
                        velocity[i] *= -1.0;
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
        
            fn get_voxel_index(&self, cell: &C) -> [usize; $d] {
                let p = cell.pos();
                let mut out = [0; $d];

                for i in 0..$d {
                    out[i] = ((p[i] - self.min[0]) / self.voxel_sizes[i]) as usize;
                }
                return out;
            }
        
            fn get_neighbor_voxel_indices(&self, index: &[usize; $d]) -> Vec<[usize; $d]> {
                // Create the bounds for the following creation of all the voxel indices
                let bounds: [[usize; 2]; $d] = [$(
                    [
                        max(index[$k] as i32 - 1, 0) as usize,
                        min(index[$k]+2, self.n_vox[$k])
                    ]
                ),+];

                // Create voxel indices
                let v: Vec<[usize; $d]> = [$($k),+].iter()      // indices supplied in macro invokation
                    .map(|i| (bounds[*i][0]..bounds[*i][1]))    // ranges from bounds
                    .multi_cartesian_product()                  // all possible combinations
                    .map(|ind_v| [$(ind_v[$k]),+])              // multi_cartesian_product gives us vector elements. We map them to arrays.
                    .filter(|ind| ind!=index)                   // filter the elements such that the current index is not included.
                    .collect();                                 // collect into the correct type

                return v;
            }

            /// # Create Voxels and sort them into threads
            /// The procedure depends on if we have an even or odd amount of threads
            ///    
            /// ## Case 1)
            /// It is clear that n_regions >= n_voxel
            /// otherwise we set n_regions = n_voxel
            ///
            /// ## Case 2)
            /// For the problem where
            ///     $$n_\text{voxel} = 2^k$$
            /// we can split the cuboid along the longest edge
            /// multiple times:
            /// ```text
            /// ┌───────┐       ┌───────┐       ┌───────┐       ┌───┬───┐
            /// │       │       │       │       │       │       │   │   │       
            /// │       │       │       │       ├───────┤       ├───┼───┤       
            /// │       │       │       │       │       │       │   │   │       
            /// │       │  ==>  ├───────┤  ==>  ├───────┤  ==>  ├───┼───┤  ==>  ...
            /// │       │       │       │       │       │       │   │   │
            /// │       │       │       │       ├───────┤       ├───┼───┤
            /// │       │       │       │       │       │       │   │   │
            /// └───────┘       └───────┘       └───────┘       └───┴───┘
            /// ```
            /// ## Case 3) - General Case
            /// Algorithm
            /// 1. Initialize: For each group A,B,C,... place an initial voxel at the border of the cuboid.
            /// ```text
            /// ┌───┬───┬───┬───┬───┐       ┌───┬───┬───┬───┬───┐       ┌───┬───┬───┬───┬───┐
            /// │   │   │   │   │   │       │ B │   │   │   │   │       │ B │   │   │ C │   │
            /// ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤
            /// │   │   │   │   │   │       │   │   │   │   │   │       │   │   │   │   │   │
            /// ├───┼───┼───┼───┼───┤  ==>  ├───┼───┼───┼───┼───┤  ==>  ├───┼───┼───┼───┼───┤  ==> ...
            /// │   │   │   │   │   │       │   │   │   │   │   │       │   │   │   │   │   │
            /// ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤
            /// │ A │   │   │   │   │       │ A │   │   │   │   │       │ A │   │   │   │   │
            /// └───┴───┴───┴───┴───┘       └───┴───┴───┴───┴───┘       └───┴───┴───┴───┴───┘
            /// ```
            /// 2. Repeat the following commands until all voxels have been picked
            ///     1. Calculate a ranking value for each voxel adjacent to the group.
            ///        In this picture, each voxel (which is not at the border) has 8 neighbors.
            ///         1. For each empty voxel adjacent add `+1` to the ranking value.
            ///         2. For each adjacent voxel of another Group add `2^dim - 1` to the ranking value.
            ///         3. For each adjacent voxel of the same Group subtract `2^dim - 1`.
            ///         4. Pick the voxel with the lowest score.
            ///            If there are multiple voxels matching, pick the voxel with the most amount of empty neighbors.
            ///            If there are still multiple voxels matching, pick the first in the list.
            /// ```text
            /// Steps 1-3                   Step 4                      Steps 1-3
            /// ┌───┬───┬───┬───┬───┐       ┌───┬───┬───┬───┬───┐       ┌───┬───┬───┬───┬───┐
            /// │ B │   │   │ C │   │       │ B │   │   │ C │   │       │ B │ 1 │   │ C │   │
            /// ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤
            /// │   │   │   │   │   │       │   │   │   │   │   │       │ 3 │ 4 │   │   │   │
            /// ├───┼───┼───┼───┼───┤  ==>  ├───┼───┼───┼───┼───┤  ==>  ├───┼───┼───┼───┼───┤  ==> ...
            /// │ 1 │ 4 │   │   │ D │       │ A │   │   │   │ D │       │ A │   │   │   │ D │
            /// ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤
            /// │ A │ 3 │ E │   │   │       │ A │   │ E │   │   │       │ A │   │ E │   │   │
            /// └───┴───┴───┴───┴───┘       └───┴───┴───┴───┴───┘       └───┴───┴───┴───┴───┘
            /// Step 4                      Steps 1-3                   Step 4
            /// ┌───┬───┬───┬───┬───┐       ┌───┬───┬───┬───┬───┐       ┌───┬───┬───┬───┬───┐
            /// │ B │ B │   │ C │   │       │ B │ B │ 3 │ C │-1 │       │ B │ B │   │ C │ C │
            /// ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤
            /// │   │   │   │   │   │       │   │   │ 5 │ 5 │ 2 │       │   │   │   │   │   │
            /// ├───┼───┼───┼───┼───┤  ==>  ├───┼───┼───┼───┼───┤  ==>  ├───┼───┼───┼───┼───┤  ==> ...
            /// │ A │   │   │   │ D │       │ A │   │   │   │ D │       │ A │   │   │   │ D │
            /// ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤
            /// │ A │   │ E │   │   │       │ A │   │ E │   │   │       │ A │   │ E │   │   │
            /// └───┴───┴───┴───┴───┘       └───┴───┴───┴───┴───┘       └───┴───┴───┴───┴───┘
            /// Steps 1-3                   Step 4                          Final
            /// ┌───┬───┬───┬───┬───┐       ┌───┬───┬───┬───┬───┐           ┌───┬───┬───┬───┬───┐
            /// │ B │ B │   │ C │ C │       │ B │ B │   │ C │ C │           │ B │ B │ B │ C │ C │
            /// ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤           ├───┼───┼───┼───┼───┤
            /// │   │   │   │ 4 │ 4 │       │   │   │   │   │   │           │ A │ E │ B │ C │ C │
            /// ├───┼───┼───┼───┼───┤  ==>  ├───┼───┼───┼───┼───┤  ======>  ├───┼───┼───┼───┼───┤
            /// │ A │   │   │ 5 │ D │       │ A │   │   │   │ D │           │ A │ A │ E │ D │ D │
            /// ├───┼───┼───┼───┼───┤       ├───┼───┼───┼───┼───┤           ├───┼───┼───┼───┼───┤
            /// │ A │   │ E │ 3 │-1 │       │ A │   │ E │   │ D │           │ A │ E │ E │ D │ D │
            /// └───┴───┴───┴───┴───┘       └───┴───┴───┴───┴───┘           └───┴───┴───┴───┴───┘
            /// ```
            /// THIS ALGORITHM IS NOT WORKING AND NOT A GOOD IDEA! MAYBE WE NEED TO MODIFY IT.
            /// THE CURRENT IMPLEMENTATION IS VERY STUDPID AND SIMPLY PRODUCES AN ITERATOR OVER ALL
            /// AVAILABLE INDICES AND THEN YIELDS CHUNKS FOR THE VOXEL CONTAINERS
            fn generate_contiguous_multi_voxel_regions(&self, n_regions: usize) -> Result<(usize, Vec<Vec<([usize; $d], $voxel_name)>>), CalcError> {
                // Get all voxel indices
                let indices: Vec<[usize; $d]> = [$($k),+]
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

                let mut index_voxel_combinations: Vec<([usize; $d], $voxel_name)> = indices
                    .into_iter()
                    .map(|ind| {
                        (ind, $voxel_name {
                            min: [$(self.min[$k] +    ind[$k]  as f64*self.voxel_sizes[$k]),+],
                            max: [$(self.min[$k] + (1+ind[$k]) as f64*self.voxel_sizes[$k]),+],
                            index: ind,
                        })
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

define_and_implement_cartesian_cuboid!(1, CartesianCuboid1, CartesianCuboidVoxel1, 0);
define_and_implement_cartesian_cuboid!(2, CartesianCuboid2, CartesianCuboidVoxel2, 0, 1);
define_and_implement_cartesian_cuboid!(3, CartesianCuboid3, CartesianCuboidVoxel3, 0, 1, 2);


impl<'a> CreatePlottingRoot<'a, BitMapBackend<'a>> for CartesianCuboid2
{
    fn create_plotting_root(&self, image_size: u32, filename: &'a String) -> DrawingArea<BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>> {
        use plotters::drawing::IntoDrawingArea;
        // let root = plotters::backend::BitMapBackend::new(filename, (image_size, image_size)).into_drawing_area();
        let dx = (self.max[0]-self.min[0]).abs();
        let dy = (self.max[1]-self.min[1]).abs();
        let q = dx.min(dy);
        let image_size_x = (image_size as f64 * dx / q).round() as u32;
        let image_size_y = (image_size as f64 * dy / q).round() as u32;
        let root = BitMapBackend::new(filename, (image_size_x, image_size_y)).into_drawing_area();
        root.fill(&plotters::prelude::full_palette::WHITE).unwrap();

        let label_space = (0.05 * image_size as f64).round() as u32;
        use plotters::prelude::IntoFont;
        let pos = plotters::style::text_anchor::Pos::new(plotters::style::text_anchor::HPos::Center, plotters::style::text_anchor::VPos::Center);
        let label_style = plotters::prelude::TextStyle::from(("sans-serif", (0.02 * image_size as f64).round() as u32)
            .into_font())
            .color(&plotters::prelude::BLACK)
            .pos(pos);

        // Draw legend
        let voxel_pixel_size_x = ((image_size_x - 2 * label_space) as f64 / (self.n_vox[0]) as f64).round() as i32;
        let voxel_pixel_size_y = ((image_size_y - 2 * label_space) as f64 / (self.n_vox[1]) as f64).round() as i32;
        let xy0 = (label_space as f64 * 0.5).round() as i32;

        let create_element = |index: usize, i: usize, pos: (i32, i32)| {
            plotters::prelude::Text::new(
                format!("{:.0}", self.min[index] + i as f64 * self.voxel_sizes[index]),
                pos,
                label_style.clone(),
            )
        };

        for i in 0..self.n_vox[0]+1 {
            let element_top = create_element(0, i, (label_space as i32 + i as i32 * voxel_pixel_size_x,                       xy0));
            let element_bot = create_element(0, i, (label_space as i32 + i as i32 * voxel_pixel_size_x, image_size_y as i32 - xy0));

            root.draw(&element_top).unwrap();
            root.draw(&element_bot).unwrap();
        }
        for j in 0..self.n_vox[1]+1 {
            let element_left  = create_element(1, j, (                      xy0, label_space as i32 + j as i32 * voxel_pixel_size_y));
            let element_right = create_element(1, j, (image_size_x as i32 - xy0, label_space as i32 + j as i32 * voxel_pixel_size_y));

            root.draw(&element_left).unwrap();
            root.draw(&element_right).unwrap();
        }

        let mut chart = plotters::prelude::ChartBuilder::on(&root)
            .margin(label_space)
            // Finally attach a coordinate on the drawing area and make a chart context
            .build_cartesian_2d(
                self.min[0]..self.max[0],
                self.min[1]..self.max[1],
            ).unwrap();

        chart
            .configure_mesh()
            // we do not want to draw any mesh lines automatically but do this manually below
            .disable_mesh()
            .draw()
            .unwrap();

        // Draw vertical lines manually
        for i in 0..self.n_vox[0]+1 {
            let element = plotters::prelude::LineSeries::new(
                [
                    (self.min[0] + i as f64 * self.voxel_sizes[0], self.min[1]),
                    (self.min[0] + i as f64 * self.voxel_sizes[0], self.max[1])
                ],
                plotters::prelude::BLACK
            );
            chart.draw_series(element).unwrap();
        }

        // Draw horizontal lines manually
        for i in 0..self.n_vox[1]+1 {
            let element = plotters::prelude::LineSeries::new(
                [
                    (self.min[0], self.min[1] + i as f64 * self.voxel_sizes[1]),
                    (self.max[0], self.min[1] + i as f64 * self.voxel_sizes[1])
                ],
                plotters::prelude::BLACK
            );
            chart.draw_series(element).unwrap();
        }

        chart.plotting_area().clone()
    }
}




#[cfg(test)]
mod test {
    use super::get_decomp_res;
    use rayon::prelude::*;

    #[test]
    fn test_get_demomp_res() {
        #[cfg(not(feature = "test_exhaustive"))]
        let max = 50_000;
        #[cfg(feature = "test_exhaustive")]
        let max = 5_000_000;

        (1..max).into_par_iter().map(|n_voxel| {
            for n_regions in 1..1_000 {
                match get_decomp_res(n_voxel, n_regions) {
                    Some(res) => {
                        let (n, m, average_len) = res;
                        assert_eq!(n + m, n_regions);
                        assert_eq!(n*average_len + m*(average_len-1), n_voxel);
                    },
                    None => panic!("No result for inputs n_voxel: {} n_regions: {}", n_voxel, n_regions),
                }
            }
        }).collect::<Vec<()>>();
    }
}
