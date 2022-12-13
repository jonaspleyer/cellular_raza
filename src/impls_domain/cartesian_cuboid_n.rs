// Imports from this crate
use crate::concepts::errors::*;
use crate::concepts::domain::*;
use crate::concepts::cell::*;

// Imports from std and core
use core::cmp::{min,max};

// Imports from other crates
use nalgebra::SVector;
use itertools::Itertools;


#[macro_export]
macro_rules! define_and_implement_cartesian_cuboid {
    ($d: expr, $name: ident, $voxel_name: ident, $($k: expr),+) => {
        #[doc = "Cuboid Domain with regular cartesian coordinates in `"]
        #[doc = stringify!($d)]
        #[doc = "` dimensions"]
        #[derive(Clone,Debug)]
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
        #[derive(Clone,Debug)]
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
        where C: Cell<SVector<f64, $d>, SVector<f64, $d>, SVector<f64, $d>>,
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
                        return Err(BoundaryError { message: format!("Particle with id {} is out of domain at position {:?}", cell.get_uuid(), pos) });
                    } else {
                        return Ok(());
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
                        min(index[$k]+2, self.n_vox[$k  ])
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
                let average_len = (indices.len() as f64 / n_regions as f64).ceil() as i64;
                let n_voxel = indices.len() as i64;
                let mut n = n_regions as i64;
                let mut m = max(0, n_voxel - average_len.pow(2));

                // Check that we have more voxels than threads. Otherwise assign one voxel to each thread.
                if n_regions >= n_voxel as usize {
                    let voxels = indices.iter().map(|ind| {
                        vec![(*ind, $voxel_name {
                            min: [$(self.min[$k] +    ind[$k]  as f64*self.voxel_sizes[$k]),+],
                            max: [$(self.min[$k] + (1+ind[$k]) as f64*self.voxel_sizes[$k]),+],
                            index: *ind,
                        })]
                    }).collect();
                    return Ok((n_voxel as usize, voxels));
                }

                // Define a closure to calculate the residue
                let residue = |n, m| {n_voxel - average_len*n - (average_len-1)*m};

                // We give the first iterations of this loop for the previous example
                // 1. Calculate residue r = ...
                // TODO continue documentation
                for _ in 0..n_voxel {
                    let r = residue(n, m);
                    if r.abs() > average_len as i64 {
                        n += r.signum() * (r.abs() as f64 / average_len as f64).floor() as i64;
                    } else {
                        n += r.signum();
                        m -= r.signum();
                    }

                    if residue(n, m) == 0 {
                        break;
                    }
                }
                if residue(n, m) != 0 {
                    return Err(CalcError {message: "Could not find a suiting decomposition".to_owned(), });
                }

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
                    .chunks((average_len-1) as usize)
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
