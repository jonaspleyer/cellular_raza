use cellular_raza_concepts::domain_new::*;
use cellular_raza_concepts::{BoundaryError, DecomposeError};

#[allow(unused)]
struct Agent {
    pos: f32,
}

struct MyDomain {
    x_min: f32,
    x_max: f32,
}

struct MySubDomain {
    x_min: f32,
    x_max: f32,
}

impl SubDomain for MySubDomain {
    type VoxelIndex = u8;

    fn get_neighbor_voxel_indices(&self, _voxel_index: &Self::VoxelIndex) -> Vec<Self::VoxelIndex> {
        Vec::new()
    }

    fn get_all_indices(&self) -> Vec<Self::VoxelIndex> {
        vec![1]
    }
}

impl Domain<Agent, MySubDomain> for MyDomain {
    type VoxelIndex = u8;
    type SubDomainIndex = usize;

    fn decompose(
        self,
        _: core::num::NonZeroUsize,
        cells: Vec<Agent>,
    ) -> Result<
        DecomposedDomain<Self::SubDomainIndex, MySubDomain, Agent>,
        cellular_raza_concepts::DecomposeError,
    >
    where
        MySubDomain: SubDomain,
    {
        Ok(DecomposedDomain {
            n_subdomains: 1.try_into().unwrap(),
            index_subdomain_cells: vec![(
                1,
                MySubDomain {
                    x_min: self.x_min,
                    x_max: self.x_max,
                },
                cells,
            )],
            neighbor_map: std::collections::HashMap::from([(1, Vec::new())]),
            rng_seed: 1,
        })
    }
}

#[allow(unused)]
#[derive(Domain)]
struct DerivedDomain {
    #[Base]
    my_domain: MyDomain,
}

#[test]
fn derive_domain() {
    let domain = DerivedDomain {
        my_domain: MyDomain {
            x_min: 3000.0,
            x_max: 3001.0,
        },
    };
    let decomposed_domain = domain.decompose(2.try_into().unwrap(), vec![]).unwrap();
    assert_eq!(decomposed_domain.rng_seed, 1);
}

impl SortCells<Agent, MySubDomain> for MyDomain {
    type Index = u8;

    fn sort_cells<'a>(
        &self,
        cells: impl IntoIterator<Item = Agent>,
        _sub_units: impl IntoIterator<Item = &'a MySubDomain>,
    ) -> Result<impl IntoIterator<Item = (Self::Index, Agent)>, BoundaryError> {
        Ok(cells.into_iter().map(|cell| (1, cell)))
    }
}

#[derive(Domain)]
struct DerivedDomain2 {
    #[SortCells]
    my_domain: MyDomain,
}

#[test]
fn derive_sort_cells() {
    let domain = DerivedDomain2 {
        my_domain: MyDomain {
            x_min: 0.0,
            x_max: 3000000.0,
        },
    };
    let subdomains = vec![MySubDomain {
        x_min: 0.0,
        x_max: 3000000.0,
    }];
    let agents = vec![Agent { pos: 3.0 }];
    let index_cells = domain.sort_cells(agents, subdomains.iter()).unwrap();
    for (index, cell) in index_cells.into_iter() {
        assert_eq!(index, 1);
        assert_eq!(cell.pos, 3.0);
    }
}

impl DomainRngSeed for MyDomain {
    fn get_rng_seed(&self) -> u64 {
        0
    }
}

#[derive(Domain)]
struct DerivedDomain3 {
    #[DomainRngSeed]
    my_domain: MyDomain,
}

#[test]
fn derive_rng_seed() {
    let domain = DerivedDomain3 {
        my_domain: MyDomain {
            x_min: -3.0,
            x_max: 0.01,
        },
    };
    assert_eq!(0, domain.get_rng_seed());
}

impl DomainCreateSubDomains<MySubDomain> for MyDomain {
    type VoxelIndex = u8;
    type SubDomainIndex = usize;

    fn create_subdomains(
        &self,
        n_subdomains: core::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<Item = (Self::SubDomainIndex, MySubDomain, Vec<Self::VoxelIndex>)>,
        DecomposeError,
    > {
        let dx = (self.x_max - self.x_min)
            / <usize as From<core::num::NonZeroUsize>>::from(n_subdomains) as f32;
        Ok((0..n_subdomains.into()).map(move |n| {
            (
                0,
                MySubDomain {
                    x_min: self.x_min.clone() + n as f32 * dx,
                    x_max: self.x_min.clone() + (n + 1) as f32 * dx,
                },
                Vec::new(),
            )
        }))
    }
}
