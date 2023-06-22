use super::concepts::*;

use cellular_raza_concepts::errors::DecomposeError;

use serde::{Deserialize, Serialize};

/// Struct containing all necessary information to construct a fully working simulation and run it.
#[derive(Clone, Deserialize, Serialize)]
pub struct SimulationSetup<C, D> {
    /// The initial cells contained in the simulation. In the future we hope to provide an abstract type
    /// in order to allow for pure iterators to be stored here
    pub cells: Vec<C>,
    /// The physical simulation domain which is specified by the [Domain] trait.
    pub domain: D,
}

impl<C, D> SimulationSetup<C, D> {
    /// Insert more cells into the setup after having already initialized the setup.
    pub fn insert_cells<I>(&mut self, cells: I)
    where
        I: IntoIterator<Item = C>,
    {
        self.cells.extend(cells.into_iter());
    }

    /// Decomposes the struct into a [DecomposedDomain] which can be taken by the backend and turned into multiple subdomains.
    pub fn decompose<S>(
        self,
        n_subdomains: usize,
    ) -> Result<DecomposedDomain<D::SubDomainIndex, S, C>, DecomposeError>
    where
        D: Domain<C, S>,
    {
        self.domain.decompose(n_subdomains, self.cells)
    }

    // TODO add a funtion which will automatically generate the correct number of subdomains
    // and simply checks beforehand how many cpu threads are available
    /// Similar to [decompose](SimulationSetup::decompose) method but does not require to specify
    /// how many subdomains should be chosen. It will attempt to retrieve resources available to the system
    /// and spawn threads which are either pre-calculated, read from an existing file or acquired by auto-tuning.
    pub fn decompose_auto_tune<S>(self) -> Result<DecomposedDomain<D::SubDomainIndex, S, C>, DecomposeError>
    where
        D: Domain<C, S>
    {
        todo!();
        let max_n_threads = std::thread::available_parallelism()?;
        self.decompose(max_n_threads.into())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_config_insert_cells() {
        let mut config = SimulationSetup {
            cells: vec![1, 2, 3, 4],
            domain: "hello".to_owned(),
        };

        let new_cells = vec![5, 6, 7];
        config.insert_cells(new_cells);
        assert_eq!(config.cells, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    struct TestDomain {
        min: f64,
        max: f64,
        n_voxels: usize,
        rng_seed: u64,
    }

    struct VoxelIndex(usize);

    struct TestSubDomain {
        min: f64,
        max: f64,
        voxels: std::collections::BTreeMap<usize, [f64; 2]>,
        reflect_at_boundary: (bool, bool),
    }

    impl Domain<f64, TestSubDomain> for TestDomain {
        type SubDomainIndex = usize;
        type VoxelIndex = VoxelIndex;

        fn get_neighbor_subdomains(
            &self,
            index: &Self::SubDomainIndex,
        ) -> Vec<Self::SubDomainIndex> {
            if self.n_voxels >= 1 && *index == 0 {
                vec![1]
            } else if self.n_voxels >= 1 && *index == self.n_voxels - 1 {
                vec![self.n_voxels - 1]
            } else if self.n_voxels >= 1 {
                vec![index - 1, index + 1]
            } else {
                Vec::new()
            }
        }

        fn get_all_voxel_indices(&self) -> Vec<Self::VoxelIndex> {
            (0..self.n_voxels).map(|i| VoxelIndex(i)).collect()
        }

        fn decompose(
            self,
            n_subdomains: usize,
            cells: Vec<f64>,
        ) -> Result<DecomposedDomain<Self::SubDomainIndex, TestSubDomain, f64>, DecomposeError> {
            let mut cells = cells;
            let mut index_subdomain_cells = Vec::new();
            let n_return_subdomains = n_subdomains.min(self.n_voxels);
            let dx = (self.max - self.min) / (self.n_voxels as f64);

            let voxel_distributions = (0..self.n_voxels)
                .map(|i| (i, (i * n_return_subdomains).div_euclid(self.n_voxels)))
                .fold(std::collections::BTreeMap::new(), |mut acc, x| {
                    let entry = acc.entry(x.1).or_insert(Vec::new());
                    entry.push(x.0);
                    acc
                });

            let mut n_total_voxels = 0;
            for (subdomain_index, voxel_indices) in voxel_distributions {
                let lower = if subdomain_index == 0 {
                    self.min
                } else {
                    self.min + n_total_voxels as f64 * dx
                };
                let upper = if subdomain_index == n_subdomains - 1 {
                    self.max
                } else {
                    self.min + (n_total_voxels + voxel_indices.len()) as f64 * dx
                };
                n_total_voxels += voxel_indices.len();
                let (cells_in_subdomain, other_cells): (Vec<_>, Vec<_>) =
                    cells.into_iter().partition(|&x| lower <= x && x < upper);
                cells = other_cells;

                index_subdomain_cells.push((
                    subdomain_index,
                    TestSubDomain {
                        min: lower,
                        max: upper,
                        voxels: voxel_indices
                            .into_iter()
                            .map(|voxel_index| {
                                (
                                    voxel_index,
                                    [
                                        self.min + voxel_index as f64 * dx,
                                        self.min + (voxel_index + 1) as f64 * dx,
                                    ],
                                )
                            })
                            .collect(),
                        reflect_at_boundary: (
                            subdomain_index == 0,
                            subdomain_index == n_subdomains - 1,
                        ),
                    },
                    cells_in_subdomain,
                ));
            }
            let n_subdomains = index_subdomain_cells.len();
            let decomposed_domain = DecomposedDomain {
                n_subdomains,
                index_subdomain_cells,
                neighbor_map: (0..n_subdomains)
                    .map(|i| (i, vec![if i == 0 { n_subdomains } else { i - 1 }, i + 1]))
                    .collect(),
                rng_seed: self.rng_seed,
            };
            Ok(decomposed_domain)
        }
    }

    impl SubDomain<f64> for TestSubDomain {
        type VoxelIndex = usize;

        fn apply_boundary(
            &self,
            cell: &mut f64,
        ) -> Result<(), cellular_raza_concepts::prelude::BoundaryError> {
            if self.reflect_at_boundary.0 && *cell < self.min {
                *cell = self.min;
            } else if self.reflect_at_boundary.1 && *cell > self.max {
                *cell = self.max;
            }
            Ok(())
        }

        fn get_voxel_index_of(
            &self,
            cell: &f64,
        ) -> Result<Self::VoxelIndex, cellular_raza_concepts::prelude::BoundaryError> {
            for (index, voxel) in self.voxels.iter() {
                if cell >= &voxel[0] && cell <= &voxel[1] {
                    return Ok(*index);
                }
            }
            Err(cellular_raza_concepts::prelude::BoundaryError {
                message: "Could not find voxel which contains cell".into(),
            })
        }

        fn get_all_indices(&self) -> Vec<Self::VoxelIndex> {
            self.voxels.iter().map(|(&i, _)| i).collect()
        }
    }

    #[test]
    fn test_config_to_subdomains() {
        // Define the testdomain with cells
        let min = 0.0;
        let max = 100.0;
        let config = SimulationSetup {
            cells: vec![1.0, 20.0, 26.0, 41.0, 56.0, 84.0, 95.0],
            domain: TestDomain {
                min,
                max,
                n_voxels: 8,
                rng_seed: 0,
            },
        };

        let n_subdomains = 4;
        let decomposed_domain = config.decompose(n_subdomains).unwrap();
        for (_, subdomain, cells) in decomposed_domain.index_subdomain_cells.into_iter() {
            assert!(cells.len() > 0);
            // Test if each cell is really contained in the subdomain it is supposed to be.
            for cell in cells {
                assert!(cell >= subdomain.min);
                assert!(cell <= subdomain.max);
            }
        }
    }

    #[test]
    fn test_apply_boundary() {
        // Define the testdomain with cells
        let min = 0.0;
        let max = 100.0;
        let config = SimulationSetup {
            cells: vec![1.0, 20.0, 25.0, 50.0, 88.0],
            domain: TestDomain {
                min,
                max,
                n_voxels: 8,
                rng_seed: 0,
            },
        };

        let n_subdomains = 4;
        let decomposed_domain = config.decompose(n_subdomains).unwrap();
        let mut cell_outside = -10.0;
        for (_, subdomain, cells) in decomposed_domain.index_subdomain_cells.into_iter() {
            for mut cell in cells.into_iter() {
                let cell_prev = cell.clone();
                subdomain.apply_boundary(&mut cell).unwrap();
                assert_eq!(cell_prev, cell);
            }
            subdomain.apply_boundary(&mut cell_outside).unwrap();
            assert!(cell_outside >= min);
            assert!(cell_outside <= max);
        }
    }

    #[test]
    fn test_neighbors() {
        let min = 0.0;
        let max = 100.0;
        let domain = TestDomain {
            min,
            max,
            n_voxels: 8,
            rng_seed: 0,
        };

        for i in 0..domain.n_voxels {
            let neighbor_subdomain_indices = domain.get_neighbor_subdomains(&i);
            if i == 0 {
                assert_eq!(neighbor_subdomain_indices, vec![1]);
            } else if i == domain.n_voxels - 1 {
                assert_eq!(neighbor_subdomain_indices, vec![domain.n_voxels - 1]);
            } else {
                assert_eq!(neighbor_subdomain_indices, vec![i - 1, i + 1]);
            }
        }
    }
}
